//! Request handlers — the functions axum calls when a route matches.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::metrics::Metrics;
use super::upstream::UpstreamClient;

/// Shared application state, accessible from every handler via `State(...)`.
pub struct AppState {
    pub upstream: UpstreamClient,
    pub metrics: Metrics,
}

/// POST /v1/chat/completions — forward to upstream, return response unchanged.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, Ordering::Relaxed);

    match state
        .upstream
        .forward("/v1/chat/completions", headers, body.to_vec())
        .await
    {
        Ok((status, resp_headers, resp_body)) => {
            let mut builder = axum::http::Response::builder().status(status);
            for (name, value) in &resp_headers {
                builder = builder.header(name, value);
            }
            // Response::builder().body() only fails if the builder itself is
            // in an error state (e.g., invalid status code). Our status comes
            // from a valid upstream response, so unwrap is safe here.
            builder
                .body(axum::body::Body::from(resp_body))
                .unwrap()
                .into_response()
        }
        Err(e) => {
            // Upstream unreachable, timed out, or connection refused.
            // Return a structured error that matches OpenAI's error format
            // so client SDKs can parse it.
            let error = serde_json::json!({
                "error": {
                    "message": format!("cctx proxy: upstream error: {}", e),
                    "type": "proxy_error",
                    "code": "upstream_unavailable"
                }
            });
            (StatusCode::BAD_GATEWAY, Json(error)).into_response()
        }
    }
}

/// GET /cctx/health — simple liveness check.
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// GET /cctx/metrics — return current proxy metrics.
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.metrics.snapshot()).unwrap())
}
