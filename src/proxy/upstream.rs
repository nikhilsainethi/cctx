//! HTTP client that forwards requests to the upstream LLM API.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderName, Method, StatusCode};
use reqwest::Client;

/// Async HTTP client used by the proxy to forward requests to the upstream
/// LLM API. Configured with connection + overall timeouts.
pub struct UpstreamClient {
    client: Client,
    base_url: String,
}

impl UpstreamClient {
    /// Build a client targeting `base_url` with the given overall timeout.
    ///
    /// Connection timeout is fixed at 10 seconds; falls back to the default
    /// client if the configured one fails to build.
    pub fn new(base_url: &str, timeout_secs: u64) -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_else(|_| Client::new());
        UpstreamClient {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Forward a POST request, buffer the entire response body, and return
    /// the status, response headers, and bytes. Used for non-streaming chat.
    ///
    /// # Errors
    ///
    /// Returns the underlying `reqwest` error on network failure, timeout,
    /// or malformed response.
    pub async fn forward(
        &self,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>), reqwest::Error> {
        let resp = self.send_post(path, headers, body).await?;
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let resp_headers = convert_response_headers(resp.headers());
        let resp_body = resp.bytes().await?.to_vec();
        Ok((status, resp_headers, resp_body))
    }

    /// Forward a POST request and return the raw `reqwest::Response`, so the
    /// caller can stream the body chunk-by-chunk (SSE).
    ///
    /// # Errors
    ///
    /// Returns the underlying `reqwest` error on network failure.
    pub async fn forward_streaming(
        &self,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        self.send_post(path, headers, body).await
    }

    /// Forward a request of any HTTP method to any upstream path. Used by
    /// the catch-all route to proxy unrelated OpenAI endpoints.
    ///
    /// # Errors
    ///
    /// Returns the underlying `reqwest` error on network failure.
    pub async fn passthrough_request(
        &self,
        method: Method,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.base_url, path);
        let fwd_headers = strip_hop_by_hop(headers);

        // Convert http::Method to reqwest::Method via string.
        let reqwest_method =
            reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);

        self.client
            .request(reqwest_method, &url)
            .headers(fwd_headers)
            .body(body)
            .send()
            .await
    }

    async fn send_post(
        &self,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.base_url, path);
        let fwd_headers = strip_hop_by_hop(headers);
        self.client
            .post(&url)
            .headers(fwd_headers)
            .body(body)
            .send()
            .await
    }
}

fn strip_hop_by_hop(mut headers: HeaderMap) -> HeaderMap {
    let remove: &[HeaderName] = &[
        HeaderName::from_static("host"),
        HeaderName::from_static("content-length"),
        HeaderName::from_static("transfer-encoding"),
        HeaderName::from_static("connection"),
    ];
    for h in remove {
        headers.remove(h);
    }
    headers
}

fn convert_response_headers(src: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut dst = HeaderMap::new();
    for (name, value) in src.iter() {
        if matches!(
            name.as_str(),
            "transfer-encoding" | "connection" | "keep-alive"
        ) {
            continue;
        }
        if let Ok(n) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = axum::http::HeaderValue::from_bytes(value.as_bytes()) {
                dst.insert(n, v);
            }
        }
    }
    dst
}

/// Map a `reqwest` failure to an HTTP status + short error tag.
///
/// Maps timeouts to `504 Gateway Timeout`, connection failures to
/// `502 Bad Gateway`, and everything else to `500 Internal Server Error`.
pub fn classify_error(e: &reqwest::Error) -> (StatusCode, &'static str) {
    if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
    } else if e.is_connect() {
        (StatusCode::BAD_GATEWAY, "upstream_unavailable")
    } else {
        (StatusCode::BAD_GATEWAY, "upstream_error")
    }
}
