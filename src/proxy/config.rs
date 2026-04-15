//! Proxy configuration — parsed from CLI flags.

/// Configuration for the proxy server.
pub struct ProxyConfig {
    /// Socket address to bind to, e.g. "127.0.0.1:8080".
    pub listen_addr: String,
    /// Base URL of the upstream LLM API, e.g. "https://api.openai.com".
    pub upstream_url: String,
}
