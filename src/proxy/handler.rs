//! Request handlers — parse, optimize, forward, log.
//!
//! # Bridging sync optimization code with async handlers
//!
//! `tokio::task::spawn_blocking(|| { ... })` moves CPU-bound sync work
//! (our pipeline from Weeks 1-2) onto a dedicated thread pool so it
//! doesn't block tokio's async I/O threads. The async handler `.await`s
//! the result without starving other concurrent requests.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::analyzer::health::assign_attention_zones;
use crate::core::context::{AttentionZone, Chunk, Context, Message};
use crate::core::tokenizer::Tokenizer;
use crate::pipeline::executor::{truncate_to_budget, Pipeline};
use crate::pipeline::{make_strategy, PipelineConfig};

use super::metrics::Metrics;
use super::upstream::UpstreamClient;

/// Shared application state.
pub struct AppState {
    pub upstream: UpstreamClient,
    pub metrics: Metrics,
    pub strategy_names: Vec<String>,
    pub pipeline_config: Arc<PipelineConfig>,
    pub budget: Option<usize>,
    pub dry_run: bool,
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, Ordering::Relaxed);

    let start = Instant::now();
    let has_strategies = !state.strategy_names.is_empty() || state.budget.is_some();

    // ── Try to optimize ───────────────────────────────────────────────────
    let opt_result = if has_strategies {
        optimize_request(
            &body,
            &state.strategy_names,
            &state.pipeline_config,
            state.budget,
        )
        .await
    } else {
        Ok(OptimizeResult {
            body: body.to_vec(),
            original_tokens: 0,
            optimized_tokens: 0,
            budget_applied: false,
        })
    };

    let (forward_body, original_tokens, optimized_tokens, log_suffix) = match opt_result {
        Ok(result) => {
            let suffix = if result.budget_applied {
                " [BUDGET]".to_string()
            } else {
                String::new()
            };

            if state.dry_run && has_strategies {
                // Dry-run: log the optimization result but forward the ORIGINAL body.
                (body.to_vec(), result.original_tokens, result.optimized_tokens, format!("{} [DRY RUN]", suffix))
            } else {
                (result.body, result.original_tokens, result.optimized_tokens, suffix)
            }
        }
        Err(e) => {
            eprintln!("[{}] [warn] optimization failed, passthrough: {}", now_str(), e);
            (body.to_vec(), 0, 0, " [FALLBACK]".to_string())
        }
    };

    // ── Update metrics ────────────────────────────────────────────────────
    if original_tokens > 0 {
        state
            .metrics
            .tokens_original_total
            .fetch_add(original_tokens, Ordering::Relaxed);
        // In dry-run, the "optimized" tokens still reflect what WOULD happen.
        // Metrics track potential savings even in dry-run.
        state
            .metrics
            .tokens_optimized_total
            .fetch_add(optimized_tokens, Ordering::Relaxed);
    }

    // ── Forward to upstream ───────────────────────────────────────────────
    let response = match state
        .upstream
        .forward("/v1/chat/completions", headers, forward_body)
        .await
    {
        Ok((status, resp_headers, resp_body)) => {
            let mut builder = axum::http::Response::builder().status(status);
            for (name, value) in &resp_headers {
                builder = builder.header(name, value);
            }
            builder
                .body(axum::body::Body::from(resp_body))
                .unwrap()
                .into_response()
        }
        Err(e) => {
            let error = serde_json::json!({
                "error": {
                    "message": format!("cctx proxy: upstream error: {}", e),
                    "type": "proxy_error",
                    "code": "upstream_unavailable"
                }
            });
            (StatusCode::BAD_GATEWAY, Json(error)).into_response()
        }
    };

    // ── Log ───────────────────────────────────────────────────────────────
    let elapsed_ms = start.elapsed().as_millis();
    if original_tokens > 0 {
        let saved_pct = ((original_tokens as f64 - optimized_tokens as f64)
            / original_tokens as f64)
            * 100.0;
        eprintln!(
            "[{}] POST /v1/chat/completions | {} -> {} tokens ({:+.1}%) | {}ms{}",
            now_str(),
            original_tokens,
            optimized_tokens,
            -saved_pct,
            elapsed_ms,
            log_suffix,
        );
    } else {
        eprintln!(
            "[{}] POST /v1/chat/completions | passthrough | {}ms",
            now_str(),
            elapsed_ms,
        );
    }

    response
}

/// GET /cctx/health
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// GET /cctx/metrics
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.metrics.snapshot()).unwrap())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimization bridge
// ═══════════════════════════════════════════════════════════════════════════════

struct OptimizeResult {
    body: Vec<u8>,
    original_tokens: u64,
    optimized_tokens: u64,
    budget_applied: bool,
}

async fn optimize_request(
    body: &[u8],
    strategy_names: &[String],
    pipeline_config: &Arc<PipelineConfig>,
    budget: Option<usize>,
) -> anyhow::Result<OptimizeResult> {
    let mut request: serde_json::Value = serde_json::from_slice(body)?;

    let messages_value = request
        .get("messages")
        .ok_or_else(|| anyhow::anyhow!("no 'messages' field"))?
        .clone();

    let messages: Vec<Message> = serde_json::from_value(messages_value)?;
    if messages.is_empty() {
        return Ok(OptimizeResult {
            body: body.to_vec(),
            original_tokens: 0,
            optimized_tokens: 0,
            budget_applied: false,
        });
    }

    let names = strategy_names.to_vec();
    let config = Arc::clone(pipeline_config);

    let (optimized_messages, original_tokens, optimized_tokens, budget_applied) =
        tokio::task::spawn_blocking(move || {
            run_pipeline_sync(messages, &names, &config, budget)
        })
        .await??;

    request["messages"] = serde_json::to_value(&optimized_messages)?;
    let modified_body = serde_json::to_vec(&request)?;

    Ok(OptimizeResult {
        body: modified_body,
        original_tokens,
        optimized_tokens,
        budget_applied,
    })
}

/// Sync optimization — runs on a blocking thread.
fn run_pipeline_sync(
    messages: Vec<Message>,
    strategy_names: &[String],
    config: &PipelineConfig,
    budget: Option<usize>,
) -> anyhow::Result<(Vec<Message>, u64, u64, bool)> {
    let n = messages.len();

    // ── Build Context ─────────────────────────────────────────────────────
    let mut chunks: Vec<Chunk> = messages
        .into_iter()
        .enumerate()
        .map(|(i, msg)| {
            let relevance = msg
                .relevance_score
                .map(|s| s.clamp(0.0, 1.0))
                .unwrap_or_else(|| {
                    if msg.role == "system" {
                        1.0
                    } else if n <= 1 {
                        0.5
                    } else {
                        0.1 + (i as f64 / (n - 1) as f64) * 0.8
                    }
                });
            Chunk {
                index: i,
                role: msg.role,
                content: msg.content.clone(),
                token_count: config.tokenizer.count(&msg.content),
                relevance_score: relevance,
                attention_zone: AttentionZone::Strong,
            }
        })
        .collect();

    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);
    let context = Context::new(chunks);
    let original_tokens = context.total_tokens as u64;

    // ── Run strategy pipeline ─────────────────────────────────────────────
    let optimized = if strategy_names.is_empty() {
        context
    } else {
        let pipeline_config = PipelineConfig {
            query: config.query.clone(),
            tokenizer: Tokenizer::new()?,
        };
        let mut pipeline = Pipeline::new(pipeline_config);
        for name in strategy_names {
            pipeline.add(make_strategy(name)?);
        }
        pipeline.run(context)?
    };

    // ── Budget enforcement ────────────────────────────────────────────────
    let (final_ctx, budget_applied) = if let Some(b) = budget {
        if optimized.total_tokens > b {
            let (truncated, warnings) = truncate_to_budget(&optimized.chunks, b);
            for w in &warnings {
                eprintln!("[{}] [BUDGET] {}", now_str(), w);
            }
            (Context::new(truncated), true)
        } else {
            (optimized, false)
        }
    } else {
        (optimized, false)
    };

    let optimized_tokens = final_ctx.total_tokens as u64;
    let out: Vec<Message> = final_ctx.chunks.iter().map(|c| c.to_message()).collect();

    Ok((out, original_tokens, optimized_tokens, budget_applied))
}

fn now_str() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}
