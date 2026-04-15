//! axum server setup — routes, shared state, and startup.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use super::config::ProxyConfig;
use super::handler::{self, AppState};
use super::metrics::Metrics;
use super::upstream::UpstreamClient;

/// Build the router and start the server.
pub async fn run(config: ProxyConfig) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        upstream: UpstreamClient::new(&config.upstream_url),
        metrics: Metrics::default(),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/cctx/health", get(handler::health))
        .route("/cctx/metrics", get(handler::get_metrics))
        .with_state(state);

    // ── Startup banner ────────────────────────────────────────────────────
    let version = env!("CARGO_PKG_VERSION");
    let w = 44;
    eprintln!("╭{}╮", "─".repeat(w));
    eprintln!("│  {:<width$}│", format!("cctx proxy v{}", version), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Listening: {}", config.listen_addr), width = w - 2);
    eprintln!("│  {:<width$}│", format!("Upstream:  {}", config.upstream_url), width = w - 2);
    eprintln!("│  {:<width$}│", "Strategies: none (passthrough)", width = w - 2);
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
