//! HTTP client that forwards requests to the upstream LLM API.
//!
//! # Async functions
//!
//! Every function marked `async fn` returns a *future* — a value that
//! represents work that hasn't happened yet. The actual HTTP call only
//! executes when the future is `.await`-ed inside a tokio task.
//!
//! This is fundamentally different from our sync CLI code:
//!   - Sync: `let resp = client.post(url).send();` — blocks the thread
//!   - Async: `let resp = client.post(url).send().await;` — yields the thread
//!     back to tokio so it can run other tasks while waiting for the network
//!
//! A single tokio thread can handle thousands of concurrent requests because
//! it never blocks — it just switches between futures as they become ready.

use axum::http::{HeaderMap, HeaderName, StatusCode};
use reqwest::Client;

/// Reusable HTTP client for upstream requests.
///
/// `Client` holds a connection pool internally — creating one per request
/// (like we did with httpx in the Python example) is wasteful. A single
/// Client shared via `Arc<AppState>` reuses TCP connections across requests.
pub struct UpstreamClient {
    client: Client,
    base_url: String,
}

impl UpstreamClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| Client::new());
        UpstreamClient {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Forward a request to the upstream API and return the raw response.
    ///
    /// Preserves the original headers (especially Authorization) but strips
    /// hop-by-hop headers that shouldn't be forwarded between proxies.
    pub async fn forward(
        &self,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>), reqwest::Error> {
        let url = format!("{}{}", self.base_url, path);

        // ── Build forwarded headers ───────────────────────────────────────
        // Clone the incoming headers and remove hop-by-hop headers.
        // These are specific to the client↔proxy connection, not proxy↔upstream.
        let mut fwd_headers = headers;
        let hop_by_hop: &[HeaderName] = &[
            HeaderName::from_static("host"),
            HeaderName::from_static("content-length"),
            HeaderName::from_static("transfer-encoding"),
            HeaderName::from_static("connection"),
        ];
        for h in hop_by_hop {
            fwd_headers.remove(h);
        }

        // ── Send to upstream ──────────────────────────────────────────────
        let resp = self
            .client
            .post(&url)
            .headers(fwd_headers)
            .body(body)
            .send()
            .await?;

        // ── Collect response ──────────────────────────────────────────────
        let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let resp_headers = convert_headers(resp.headers());
        let resp_body = resp.bytes().await?.to_vec();

        Ok((status, resp_headers, resp_body))
    }
}

/// Convert reqwest's HeaderMap to axum's HeaderMap.
///
/// Both use the `http` crate's HeaderMap under the hood, but reqwest
/// re-exports its own version. We iterate and rebuild to avoid version
/// mismatch issues between the `http` crate versions.
fn convert_headers(src: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut dst = HeaderMap::new();
    for (name, value) in src.iter() {
        // Skip hop-by-hop headers from the upstream response.
        if matches!(
            name.as_str(),
            "transfer-encoding" | "connection" | "keep-alive"
        ) {
            continue;
        }
        if let Ok(name) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(val) = axum::http::HeaderValue::from_bytes(value.as_bytes()) {
                dst.insert(name, val);
            }
        }
    }
    dst
}
