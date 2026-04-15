//! Request handlers — parse, optimize, forward (streaming or buffered), log.
//!
//! Edge case handling:
//!   - Invalid JSON body → 400
//!   - No "messages" field → passthrough (might be embeddings/completions)
//!   - Upstream non-200 → forward error response unchanged
//!   - Upstream unreachable → 502
//!   - Upstream timeout → 504
//!   - Unknown model → still optimize with default tokenizer
//!   - Any other path → catch-all passthrough to upstream

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::analyzer::health::assign_attention_zones;
use crate::core::context::{AttentionZone, Chunk, Context, Message};
use crate::core::tokenizer::Tokenizer;
use crate::pipeline::executor::{truncate_to_budget, Pipeline};
use crate::pipeline::{make_strategy, PipelineConfig};

use super::metrics::Metrics;
use super::upstream::{classify_error, UpstreamClient};

/// Shared application state.
pub struct AppState {
    pub upstream: UpstreamClient,
    pub metrics: Metrics,
    pub strategy_names: Vec<String>,
    pub pipeline_config: Arc<PipelineConfig>,
    pub budget: Option<usize>,
    pub dry_run: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// POST /v1/chat/completions — the main handler
// ═══════════════════════════════════════════════════════════════════════════════

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, Ordering::Relaxed);

    // ── Parse body to detect stream mode and validate JSON ────────────────
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Request body is not valid JSON",
                "invalid_request",
            );
        }
    };

    let has_messages = parsed
        .get("messages")
        .and_then(|m| m.as_array())
        .map_or(false, |a| !a.is_empty());
    let is_streaming = parsed
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    if is_streaming {
        state
            .metrics
            .streaming_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    // If no "messages" field → passthrough without optimization.
    // This handles non-chat API calls or empty message arrays.
    if !has_messages {
        eprintln!(
            "[{}] POST /v1/chat/completions | no messages, passthrough | stream={}",
            now_str(),
            is_streaming
        );
        return forward_raw(&state, headers, body.to_vec(), "/v1/chat/completions", is_streaming).await;
    }

    if is_streaming {
        handle_streaming(state, headers, body).await
    } else {
        handle_non_streaming(state, headers, body).await
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Catch-all: forward ANY other path to upstream unchanged
// ═══════════════════════════════════════════════════════════════════════════════
//
// This makes cctx a transparent proxy for ALL OpenAI endpoints:
//   /v1/embeddings, /v1/models, /v1/audio/*, /v1/images/*, etc.
// Only /v1/chat/completions gets optimization. Everything else passes through.

pub async fn catchall(
    State(state): State<Arc<AppState>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    eprintln!(
        "[{}] {} {} | passthrough",
        now_str(),
        method,
        path,
    );

    match state
        .upstream
        .passthrough_request(method, path, headers, body.to_vec())
        .await
    {
        Ok(resp) => response_from_reqwest(resp).await,
        Err(e) => make_upstream_error(e),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// /cctx/* endpoints
// ═══════════════════════════════════════════════════════════════════════════════

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.metrics.snapshot()).unwrap())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-streaming handler
// ═══════════════════════════════════════════════════════════════════════════════

async fn handle_non_streaming(state: Arc<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let start = Instant::now();
    let has_strategies = !state.strategy_names.is_empty() || state.budget.is_some();

    let opt_result = if has_strategies {
        optimize_request(&body, &state.strategy_names, &state.pipeline_config, state.budget).await
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
            let suffix = if result.budget_applied { " [BUDGET]" } else { "" }.to_string();
            if state.dry_run && has_strategies {
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

    update_metrics(&state.metrics, original_tokens, optimized_tokens);

    let response = match state.upstream.forward("/v1/chat/completions", headers, forward_body).await {
        Ok((status, resp_headers, resp_body)) => {
            let mut builder = axum::http::Response::builder().status(status);
            for (name, value) in &resp_headers {
                builder = builder.header(name, value);
            }
            builder.body(Body::from(resp_body)).unwrap().into_response()
        }
        Err(e) => make_upstream_error(e),
    };

    log_request(start, original_tokens, optimized_tokens, &log_suffix, false);
    response
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streaming handler
// ═══════════════════════════════════════════════════════════════════════════════

async fn handle_streaming(state: Arc<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let start = Instant::now();
    let has_strategies = !state.strategy_names.is_empty() || state.budget.is_some();

    let opt_result = if has_strategies {
        optimize_request(&body, &state.strategy_names, &state.pipeline_config, state.budget).await
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
            let suffix = if result.budget_applied { " [BUDGET]" } else { "" }.to_string();
            if state.dry_run && has_strategies {
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

    update_metrics(&state.metrics, original_tokens, optimized_tokens);

    let response = match state
        .upstream
        .forward_streaming("/v1/chat/completions", headers, forward_body)
        .await
    {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut builder = axum::http::Response::builder().status(status);

            for (name, value) in resp.headers() {
                if matches!(name.as_str(), "transfer-encoding" | "connection" | "keep-alive") {
                    continue;
                }
                if let Ok(n) = HeaderName::from_bytes(name.as_str().as_bytes()) {
                    if let Ok(v) = axum::http::HeaderValue::from_bytes(value.as_bytes()) {
                        builder = builder.header(n, v);
                    }
                }
            }
            builder = builder
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache");

            builder
                .body(Body::from_stream(resp.bytes_stream()))
                .unwrap()
                .into_response()
        }
        Err(e) => make_upstream_error(e),
    };

    log_request(start, original_tokens, optimized_tokens, &format!("{} [STREAM]", log_suffix), true);
    response
}

// ═══════════════════════════════════════════════════════════════════════════════
// Raw passthrough (no optimization)
// ═══════════════════════════════════════════════════════════════════════════════

async fn forward_raw(
    state: &AppState,
    headers: HeaderMap,
    body: Vec<u8>,
    path: &str,
    is_streaming: bool,
) -> Response {
    if is_streaming {
        match state.upstream.forward_streaming(path, headers, body).await {
            Ok(resp) => {
                let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                let mut builder = axum::http::Response::builder().status(status);
                builder = builder
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache");
                builder
                    .body(Body::from_stream(resp.bytes_stream()))
                    .unwrap()
                    .into_response()
            }
            Err(e) => make_upstream_error(e),
        }
    } else {
        match state.upstream.forward(path, headers, body).await {
            Ok((status, resp_headers, resp_body)) => {
                let mut builder = axum::http::Response::builder().status(status);
                for (name, value) in &resp_headers {
                    builder = builder.header(name, value);
                }
                builder.body(Body::from(resp_body)).unwrap().into_response()
            }
            Err(e) => make_upstream_error(e),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn error_response(status: StatusCode, message: &str, code: &str) -> Response {
    let body = serde_json::json!({
        "error": {"message": message, "type": "proxy_error", "code": code}
    });
    (status, Json(body)).into_response()
}

fn make_upstream_error(e: reqwest::Error) -> Response {
    let (status, code) = classify_error(&e);
    let message = if e.is_timeout() {
        "Upstream request timed out".to_string()
    } else if e.is_connect() {
        "Upstream unreachable".to_string()
    } else {
        format!("cctx proxy: upstream error: {}", e)
    };
    error_response(status, &message, code)
}

/// Convert a raw reqwest::Response into an axum Response (for catch-all).
async fn response_from_reqwest(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = axum::http::Response::builder().status(status);
    for (name, value) in resp.headers() {
        if matches!(name.as_str(), "transfer-encoding" | "connection" | "keep-alive") {
            continue;
        }
        if let Ok(n) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = axum::http::HeaderValue::from_bytes(value.as_bytes()) {
                builder = builder.header(n, v);
            }
        }
    }
    match resp.bytes().await {
        Ok(body) => builder.body(Body::from(body)).unwrap().into_response(),
        Err(e) => make_upstream_error(e),
    }
}

fn update_metrics(metrics: &Metrics, original: u64, optimized: u64) {
    if original > 0 {
        metrics.tokens_original_total.fetch_add(original, Ordering::Relaxed);
        metrics.tokens_optimized_total.fetch_add(optimized, Ordering::Relaxed);
    }
}

fn log_request(start: Instant, original: u64, optimized: u64, suffix: &str, _streaming: bool) {
    let elapsed_ms = start.elapsed().as_millis();
    if original > 0 {
        let saved_pct = ((original as f64 - optimized as f64) / original as f64) * 100.0;
        eprintln!(
            "[{}] POST /v1/chat/completions | {} -> {} tokens ({:+.1}%) | {}ms{}",
            now_str(), original, optimized, -saved_pct, elapsed_ms, suffix,
        );
    } else {
        eprintln!(
            "[{}] POST /v1/chat/completions | passthrough | {}ms{}",
            now_str(), elapsed_ms, suffix,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimization bridge: async handler → sync pipeline
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
        return Ok(OptimizeResult { body: body.to_vec(), original_tokens: 0, optimized_tokens: 0, budget_applied: false });
    }

    let names = strategy_names.to_vec();
    let config = Arc::clone(pipeline_config);
    let (optimized_messages, original_tokens, optimized_tokens, budget_applied) =
        tokio::task::spawn_blocking(move || run_pipeline_sync(messages, &names, &config, budget))
            .await??;

    request["messages"] = serde_json::to_value(&optimized_messages)?;
    let modified_body = serde_json::to_vec(&request)?;
    Ok(OptimizeResult { body: modified_body, original_tokens, optimized_tokens, budget_applied })
}

fn run_pipeline_sync(
    messages: Vec<Message>,
    strategy_names: &[String],
    config: &PipelineConfig,
    budget: Option<usize>,
) -> anyhow::Result<(Vec<Message>, u64, u64, bool)> {
    let n = messages.len();
    let mut chunks: Vec<Chunk> = messages
        .into_iter()
        .enumerate()
        .map(|(i, msg)| {
            let relevance = msg.relevance_score.map(|s| s.clamp(0.0, 1.0)).unwrap_or_else(|| {
                if msg.role == "system" { 1.0 }
                else if n <= 1 { 0.5 }
                else { 0.1 + (i as f64 / (n - 1) as f64) * 0.8 }
            });
            Chunk {
                index: i, role: msg.role, content: msg.content.clone(),
                token_count: config.tokenizer.count(&msg.content),
                relevance_score: relevance, attention_zone: AttentionZone::Strong,
            }
        })
        .collect();

    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);
    let context = Context::new(chunks);
    let original_tokens = context.total_tokens as u64;

    let optimized = if strategy_names.is_empty() {
        context
    } else {
        let pc = PipelineConfig { query: config.query.clone(), tokenizer: Tokenizer::new()? };
        let mut pipeline = Pipeline::new(pc);
        for name in strategy_names { pipeline.add(make_strategy(name)?); }
        pipeline.run(context)?
    };

    let (final_ctx, budget_applied) = if let Some(b) = budget {
        if optimized.total_tokens > b {
            let (truncated, warnings) = truncate_to_budget(&optimized.chunks, b);
            for w in &warnings { eprintln!("[{}] [BUDGET] {}", now_str(), w); }
            (Context::new(truncated), true)
        } else { (optimized, false) }
    } else { (optimized, false) };

    let optimized_tokens = final_ctx.total_tokens as u64;
    let out: Vec<Message> = final_ctx.chunks.iter().map(|c| c.to_message()).collect();
    Ok((out, original_tokens, optimized_tokens, budget_applied))
}

fn now_str() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    format!("{:02}:{:02}:{:02}", (secs % 86400) / 3600, (secs % 3600) / 60, secs % 60)
}
