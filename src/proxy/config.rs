//! Proxy configuration — parsed from CLI flags.

/// Configuration for the proxy server.
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_url: String,
    pub strategy_names: Vec<String>,
    pub budget: Option<usize>,
    pub dry_run: bool,
    /// Upstream request timeout in seconds.
    pub timeout_secs: u64,
}
