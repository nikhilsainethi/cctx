//! axum server setup — routes, shared state, and startup.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::core::tokenizer::Tokenizer;
use crate::pipeline::PipelineConfig;

use super::config::ProxyConfig;
use super::handler::{self, AppState};
use super::metrics::Metrics;
use super::upstream::UpstreamClient;

/// Build the router and start the server.
pub async fn run(config: ProxyConfig) -> anyhow::Result<()> {
    let pipeline_config = Arc::new(PipelineConfig {
        query: None,
        tokenizer: Tokenizer::new()?,
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
    let mode_label = if config.dry_run {
        "DRY RUN (log only, forward original)"
    } else {
        "live"
    };

    let state = Arc::new(AppState {
        upstream: UpstreamClient::new(&config.upstream_url),
        metrics: Metrics::default(),
        strategy_names: config.strategy_names,
        pipeline_config,
        budget: config.budget,
        dry_run: config.dry_run,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/cctx/health", get(handler::health))
        .route("/cctx/metrics", get(handler::get_metrics))
        .with_state(state);

    // ── Startup banner ────────────────────────────────────────────────────
    let version = env!("CARGO_PKG_VERSION");
    let w = 50;
    eprintln!("╭{}╮", "─".repeat(w));
    eprintln!("│  {:<width$}│", format!("cctx proxy v{}", version), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Listening:   {}", config.listen_addr), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Upstream:    {}", config.upstream_url), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Strategies:  {}", strategy_label), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Budget:      {}", budget_label), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Mode:        {}", mode_label), width = w - 2);
    eprintln!("╰{}╯", "─".repeat(w));
    eprintln!();
    eprintln!("export OPENAI_BASE_URL=http://{}", config.listen_addr);
    eprintln!();

    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Cannot bind to {}: {}", config.listen_addr, e))?;

    axum::serve(listener, app).await?;

    Ok(())
}
