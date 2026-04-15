//! Proxy configuration — parsed from CLI flags.

/// Configuration for the proxy server.
pub struct ProxyConfig {
    /// Socket address to bind to, e.g. "127.0.0.1:8080".
    pub listen_addr: String,
    /// Base URL of the upstream LLM API, e.g. "https://api.openai.com".
    pub upstream_url: String,
    /// Strategy names to apply to every request (e.g. ["bookend", "structural"]).
    /// Empty = passthrough (no optimization).
    pub strategy_names: Vec<String>,
    /// Optional token budget. After strategies run, if tokens exceed this,
    /// oldest non-critical messages are dropped.
    pub budget: Option<usize>,
    /// Dry-run mode: optimize and log savings, but forward the ORIGINAL
    /// unmodified request. Safe for testing in production.
    pub dry_run: bool,
}
