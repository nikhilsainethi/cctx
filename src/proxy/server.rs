//! axum server setup — routes, shared state, dashboard task, and startup.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::core::tokenizer::Tokenizer;
use crate::pipeline::PipelineConfig;

use super::config::ProxyConfig;
use super::dashboard;
use super::handler::{self, AppState};
use super::metrics::Metrics;
use super::upstream::UpstreamClient;

pub async fn run(config: ProxyConfig) -> anyhow::Result<()> {
    let embedding_provider = build_embedding_provider(config.embedding_provider.as_deref())?;

    let pipeline_config = Arc::new(PipelineConfig {
        query: None,
        tokenizer: Tokenizer::new()?,
        embedding_provider,
        dedup_threshold: config.dedup_threshold,
        prune_threshold: 0.3,
        llm_provider: None, // proxy doesn't support LLM summarization yet
    });

    let strategy_label = if config.strategy_names.is_empty() {
        "none (passthrough)".to_string()
    } else {
        config.strategy_names.join(", ")
    };
    let budget_label = match config.budget {
        Some(b) => format!("{} tokens", b),
        None => "none".to_string(),
    };
    let mode_label = if config.dry_run { "DRY RUN" } else { "live" };

    let metrics = Arc::new(Metrics::default());

    let state = Arc::new(AppState {
        upstream: UpstreamClient::new(&config.upstream_url, config.timeout_secs),
        metrics: Arc::clone(&metrics),
        strategy_names: config.strategy_names,
        pipeline_config,
        budget: config.budget,
        dry_run: config.dry_run,
        dashboard: config.dashboard,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/cctx/health", get(handler::health))
        .route("/cctx/metrics", get(handler::get_metrics))
        .route("/cctx/metrics/reset", get(handler::reset_metrics))
        .fallback(handler::catchall)
        .with_state(state);

    // ── Startup banner ────────────────────────────────────────────────────
    let version = env!("CARGO_PKG_VERSION");
    let w = 50;
    eprintln!("╭{}╮", "─".repeat(w));
    eprintln!(
        "│  {:<width$}│",
        format!("cctx proxy v{}", version),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Listening:   {}", config.listen_addr),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Upstream:    {}", config.upstream_url),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Strategies:  {}", strategy_label),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Budget:      {}", budget_label),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Timeout:     {}s", config.timeout_secs),
        width = w - 2
    );
    eprintln!(
        "│  {:<width$}│",
        format!("Mode:        {}", mode_label),
        width = w - 2
    );
    if config.dashboard {
        eprintln!("│  {:<width$}│", "Dashboard:   enabled", width = w - 2);
    }
    eprintln!("╰{}╯", "─".repeat(w));
    eprintln!();
    eprintln!("export OPENAI_BASE_URL=http://{}", config.listen_addr);
    eprintln!();

    // ── Dashboard refresh task ────────────────────────────────────────────
    //
    // tokio::spawn launches a concurrent task — it runs alongside the HTTP
    // server on the same async runtime. The task sleeps 5 seconds between
    // renders so it barely uses any CPU.
    if config.dashboard {
        let dash_metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            let mut first = true;
            loop {
                dashboard::render(&dash_metrics, first);
                first = false;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Cannot bind to {}: {}", config.listen_addr, e))?;

    axum::serve(listener, app).await?;

    Ok(())
}

fn build_embedding_provider(
    name: Option<&str>,
) -> anyhow::Result<Option<Arc<dyn crate::embeddings::EmbeddingProvider>>> {
    match name {
        None => Ok(None),
        Some("tfidf") => Ok(Some(Arc::new(crate::embeddings::tfidf::TfIdfEmbedder))),
        #[cfg(feature = "embeddings")]
        Some("ollama") => Ok(Some(Arc::new(
            crate::embeddings::ollama::OllamaEmbedder::default_local(),
        ))),
        #[cfg(feature = "embeddings")]
        Some("openai") => Ok(Some(Arc::new(
            crate::embeddings::openai::OpenAIEmbedder::from_env()?,
        ))),
        #[cfg(not(feature = "embeddings"))]
        Some(name) if name == "ollama" || name == "openai" => {
            anyhow::bail!("Provider '{}' requires --features embeddings", name)
        }
        Some(other) => anyhow::bail!("Unknown embedding provider: {}", other),
    }
}
