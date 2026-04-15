//! HTTP client that forwards requests to the upstream LLM API.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderName, Method, StatusCode};
use reqwest::Client;

pub struct UpstreamClient {
    client: Client,
    base_url: String,
}

impl UpstreamClient {
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

    /// Forward a POST and buffer the entire response (non-streaming chat).
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

    /// Forward a POST and return the raw response for streaming.
    pub async fn forward_streaming(
        &self,
        path: &str,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        self.send_post(path, headers, body).await
    }

    /// Forward any HTTP method to any path (catch-all passthrough).
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

/// Classify a reqwest error for appropriate HTTP status codes.
pub fn classify_error(e: &reqwest::Error) -> (StatusCode, &'static str) {
    if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
    } else if e.is_connect() {
        (StatusCode::BAD_GATEWAY, "upstream_unavailable")
    } else {
        (StatusCode::BAD_GATEWAY, "upstream_error")
    }
}
