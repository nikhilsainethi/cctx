//! Proxy configuration — parsed from CLI flags.

/// Runtime configuration for the proxy server, built from CLI flags +
/// `.cctx.toml` values in `main.rs`.
pub struct ProxyConfig {
    /// Socket address to bind to (e.g. `"127.0.0.1:8080"`).
    pub listen_addr: String,
    /// Upstream LLM API base URL (e.g. `"https://api.openai.com"`).
    pub upstream_url: String,
    /// Ordered list of strategies to apply to every request.
    pub strategy_names: Vec<String>,
    /// Optional post-pipeline token budget. `None` = unbounded.
    pub budget: Option<usize>,
    /// When `true`, optimize and log savings but forward the original request.
    pub dry_run: bool,
    /// Upstream request timeout in seconds.
    pub timeout_secs: u64,
    /// Embedding provider name: `"ollama"`, `"openai"`, or `None` for exact-match dedup.
    pub embedding_provider: Option<String>,
    /// Cosine similarity threshold for semantic dedup.
    pub dedup_threshold: f64,
    /// Show live-updating dashboard on stderr.
    pub dashboard: bool,
    /// Human-readable config source shown in the startup banner
    /// (e.g. `".cctx.toml (project)"`). `None` hides the banner line.
    pub config_source: Option<String>,
}
