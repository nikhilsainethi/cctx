//! cctx configuration: `.cctx.toml` and `~/.config/cctx/config.toml`.
//!
//! Precedence (highest → lowest):
//!   1. CLI flags
//!   2. .cctx.toml in the current directory
//!   3. ~/.config/cctx/config.toml
//!   4. Built-in defaults
//!
//! Every field is Option so we can cleanly distinguish "set" vs "unset"
//! — the merge happens in main.rs where CLI values take priority.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

// ── Schema ────────────────────────────────────────────────────────────────────

/// Root config. `#[serde(default)]` on each section means a missing
/// `[section]` block parses as the section's Default impl (all None).
#[derive(Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub default: DefaultSection,
    #[serde(default)]
    pub optimize: OptimizeSection,
    #[serde(default)]
    pub proxy: ProxySection,
    #[serde(default)]
    pub dedup: DedupSection,
    #[serde(default)]
    pub prune: PruneSection,
    #[serde(default)]
    pub summarize: SummarizeSection,
}

#[derive(Debug, Default, Deserialize)]
pub struct DefaultSection {
    pub model: Option<String>,
    pub format: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OptimizeSection {
    pub strategies: Option<Vec<String>>,
    pub budget: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ProxySection {
    pub listen: Option<String>,
    pub upstream: Option<String>,
    pub strategies: Option<Vec<String>>,
    /// 0 in TOML means "no budget" — normalized to None at load time.
    pub budget: Option<usize>,
    pub embedding_provider: Option<String>,
    pub dashboard: Option<bool>,
    pub timeout: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DedupSection {
    pub threshold: Option<f64>,
    pub embedding_provider: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct PruneSection {
    pub threshold: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SummarizeSection {
    pub llm_provider: Option<String>,
    pub llm_model: Option<String>,
    pub recent_turns: Option<usize>,
}

// ── Loading ───────────────────────────────────────────────────────────────────

/// Where the loaded config came from, for reporting in banners / help output.
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// .cctx.toml in the current directory.
    Project(PathBuf),
    /// ~/.config/cctx/config.toml.
    User(PathBuf),
    /// No file found — using built-in defaults.
    Defaults,
}

impl ConfigSource {
    /// Short label suitable for banners, e.g. `".cctx.toml (project)"`.
    pub fn label(&self) -> String {
        match self {
            ConfigSource::Project(p) => format!("{} (project)", p.display()),
            ConfigSource::User(p) => format!("{} (user)", p.display()),
            ConfigSource::Defaults => "built-in defaults".to_string(),
        }
    }
}

/// Load config from the standard locations, returning the merged config and its source.
/// Never fails unless a file exists but is unparseable — missing files are fine.
pub fn load() -> Result<(Config, ConfigSource)> {
    let project = PathBuf::from(".cctx.toml");
    if project.is_file() {
        let cfg = load_from_path(&project)?;
        return Ok((normalize(cfg), ConfigSource::Project(project)));
    }

    if let Some(user) = user_config_path()
        && user.is_file()
    {
        let cfg = load_from_path(&user)?;
        return Ok((normalize(cfg), ConfigSource::User(user)));
    }

    Ok((Config::default(), ConfigSource::Defaults))
}

/// Parse a specific config file. Exposed for tests and `cctx --config path`.
pub fn load_from_path(path: &Path) -> Result<Config> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read config file '{}'", path.display()))?;
    let cfg: Config =
        toml::from_str(&raw).with_context(|| format!("Invalid TOML in '{}'", path.display()))?;
    Ok(cfg)
}

fn user_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".config")
            .join("cctx")
            .join("config.toml")
    })
}

/// Apply small normalizations: treat `[proxy].budget = 0` as "no budget" (None).
fn normalize(mut cfg: Config) -> Config {
    if matches!(cfg.proxy.budget, Some(0)) {
        cfg.proxy.budget = None;
    }
    if matches!(cfg.optimize.budget, Some(0)) {
        cfg.optimize.budget = None;
    }
    cfg
}

// ── Init template ─────────────────────────────────────────────────────────────

/// Template written by `cctx init`. Every setting is commented out — users
/// uncomment what they need. This keeps a fresh config non-opinionated.
pub const INIT_TEMPLATE: &str = r#"# cctx configuration.
# Precedence: CLI flags > .cctx.toml > ~/.config/cctx/config.toml > defaults.
# Uncomment the settings you want to customize.

[default]
# model  = "gpt-4o"      # tokenizer model (informational)
# format = "terminal"    # "terminal" or "json"

[optimize]
# strategies = ["bookend", "structural"]
# budget     = 32000

[proxy]
# listen             = "127.0.0.1:8080"
# upstream           = "https://api.openai.com"
# strategies         = ["bookend", "structural", "dedup"]
# budget             = 0          # 0 means no budget limit
# embedding_provider = "ollama"   # "tfidf" | "ollama" | "openai"
# dashboard          = true
# timeout            = 120        # seconds

[dedup]
# threshold          = 0.85       # cosine similarity cutoff (0.0–1.0)
# embedding_provider = "ollama"

[prune]
# threshold = 0.3                 # importance cutoff (0.0–1.0)

[summarize]
# llm_provider = "ollama"
# llm_model    = "llama3.2:3b"
# recent_turns = 6
"#;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_template_parses() {
        // The template we hand users must itself be valid TOML.
        let cfg: Config = toml::from_str(INIT_TEMPLATE).expect("template must parse");
        // Every field is commented out, so everything is None.
        assert!(cfg.proxy.listen.is_none());
        assert!(cfg.optimize.strategies.is_none());
    }

    #[test]
    fn parses_full_example() {
        let raw = r#"
            [default]
            model = "gpt-4o"

            [proxy]
            listen = "127.0.0.1:9000"
            upstream = "https://api.openai.com"
            strategies = ["bookend", "structural"]
            budget = 0

            [dedup]
            threshold = 0.9
        "#;
        let cfg = normalize(toml::from_str::<Config>(raw).unwrap());
        assert_eq!(cfg.proxy.listen.as_deref(), Some("127.0.0.1:9000"));
        assert_eq!(
            cfg.proxy.strategies.as_deref(),
            Some(&["bookend".into(), "structural".into()][..])
        );
        // budget = 0 should normalize to None.
        assert!(cfg.proxy.budget.is_none());
        assert_eq!(cfg.dedup.threshold, Some(0.9));
    }

    #[test]
    fn missing_sections_default_to_none() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.proxy.listen.is_none());
        assert!(cfg.default.model.is_none());
        assert!(cfg.summarize.llm_provider.is_none());
    }

    #[test]
    fn source_label() {
        let p = ConfigSource::Project(PathBuf::from(".cctx.toml"));
        assert_eq!(p.label(), ".cctx.toml (project)");
        assert_eq!(ConfigSource::Defaults.label(), "built-in defaults");
    }
}
