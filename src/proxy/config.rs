//! Proxy configuration — parsed from CLI flags.

/// Configuration for the proxy server.
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_url: String,
    pub strategy_names: Vec<String>,
    pub budget: Option<usize>,
    pub dry_run: bool,
    pub timeout_secs: u64,
    /// Embedding provider name: "ollama", "openai", or None.
    pub embedding_provider: Option<String>,
    /// Cosine similarity threshold for semantic dedup.
    pub dedup_threshold: f64,
    /// Show live-updating dashboard on stderr.
    pub dashboard: bool,
    /// Source of the merged configuration (e.g. ".cctx.toml (project)").
    /// Displayed in the startup banner. `None` hides the line.
    pub config_source: Option<String>,
}
