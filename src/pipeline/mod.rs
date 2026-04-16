//! Strategy pipeline — composable, ordered strategy execution.

pub mod executor;

use std::sync::Arc;

use anyhow::Result;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;
use crate::embeddings::EmbeddingProvider;
use crate::llm::LlmProvider;
use crate::strategies::{bookend, dedup, prune, structural, summarize};

// ── Strategy trait ────────────────────────────────────────────────────────────

/// A context transformation that can be composed in a [`executor::Pipeline`].
///
/// Strategies are stateless — they take an immutable [`Context`] plus shared
/// [`PipelineConfig`] and return a new chunk list. The pipeline re-wraps
/// those chunks in a fresh `Context` between passes.
pub trait Strategy {
    /// Stable identifier used in log output and error messages
    /// (e.g. `"bookend"`, `"dedup"`).
    fn name(&self) -> &str;
    /// Apply the transformation to `context` and return the new chunk list.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the strategy needs an external resource (e.g. an
    /// LLM or embedding provider) that's unavailable or fails mid-call.
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>>;
}

// ── Pipeline config ───────────────────────────────────────────────────────────

/// Shared state every strategy receives when running in a pipeline.
///
/// The tokenizer is held by value (not `Arc`) because it's cheap to create
/// and strategies don't share it across threads. The provider fields are
/// `Arc<dyn Trait>` so the proxy can clone the config onto each request
/// handler without rebuilding the underlying HTTP client.
pub struct PipelineConfig {
    /// Optional query string. `bookend` scores relevance against it;
    /// `structural` uses it to decide which markdown sections to keep.
    pub query: Option<String>,
    /// BPE tokenizer, used by strategies that need accurate token counts
    /// (`structural`, `dedup`, `prune`, `summarize`).
    pub tokenizer: Tokenizer,
    /// Embedding provider for semantic dedup. `None` ⇒ dedup falls back to
    /// exact-match comparison.
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Cosine similarity threshold for semantic dedup (typical 0.85).
    pub dedup_threshold: f64,
    /// Importance-score threshold for sentence pruning (typical 0.3).
    pub prune_threshold: f64,
    /// LLM provider for summarization. `None` ⇒ `summarize` falls back to
    /// dropping older turns without an LLM call.
    pub llm_provider: Option<Arc<dyn LlmProvider>>,
}

// ── Strategy wrappers ─────────────────────────────────────────────────────────

/// Attention-aware reordering strategy (bookend placement).
///
/// Sorts chunks by relevance and interleaves them at the window edges —
/// highest relevance at positions 0, N-1, 1, N-2, …  Zero token delta; the
/// goal is purely to combat the U-shaped recall curve.
pub struct BookendStrategy;

impl Strategy for BookendStrategy {
    fn name(&self) -> &str {
        "bookend"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        Ok(bookend::apply(context, config.query.as_deref()))
    }
}

/// Inline compression of JSON payloads, code blocks, and markdown.
///
/// Strips structural redundancy (timestamps, UUIDs, deep nesting, code bodies)
/// without dropping any chunks. Typically -25–50% on structured content,
/// no-op on plain prose.
pub struct StructuralStrategy;

impl Strategy for StructuralStrategy {
    fn name(&self) -> &str {
        "structural"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        Ok(structural::apply(
            context,
            config.query.as_deref(),
            &config.tokenizer,
        ))
    }
}

/// Near-duplicate removal via exact-match or cosine similarity.
///
/// When an embedding provider is configured in [`PipelineConfig`], uses
/// semantic similarity; otherwise falls back to exact-match comparison.
/// Keeps the longer chunk from each duplicate pair.
pub struct DeduplicateStrategy;

impl Strategy for DeduplicateStrategy {
    fn name(&self) -> &str {
        "dedup"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        // If an embedding provider is configured, use semantic dedup.
        // Otherwise, fall back to exact-match dedup (no external dependencies).
        if let Some(ref provider) = config.embedding_provider {
            dedup::apply_semantic(
                context,
                provider.as_ref(),
                config.dedup_threshold,
                &config.tokenizer,
            )
        } else {
            Ok(dedup::apply(context))
        }
    }
}

/// Importance-aware sentence pruning.
///
/// Scores sentences by stop-word ratio, repetition, structural markers, and
/// filler-phrase detection; removes anything below the threshold in
/// [`PipelineConfig::prune_threshold`]. System messages and the last two
/// user messages are always preserved.
pub struct PruneStrategy;

impl Strategy for PruneStrategy {
    fn name(&self) -> &str {
        "prune"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        Ok(prune::apply(
            context,
            config.prune_threshold,
            &config.tokenizer,
        ))
    }
}

/// Hierarchical LLM-powered summarization.
///
/// Three tiers: the last 6 turns stay verbatim, the next 6 are bulletized
/// by the configured LLM, older turns are merged into a single paragraph.
/// Falls back to dropping archived turns when no LLM provider is configured.
pub struct SummarizeStrategy;

impl Strategy for SummarizeStrategy {
    fn name(&self) -> &str {
        "summarize"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        summarize::apply(
            context,
            config.llm_provider.as_deref(),
            &config.tokenizer,
            6, // default: keep last 6 turns verbatim
        )
    }
}

// ── Factory + presets ─────────────────────────────────────────────────────────

/// Build a strategy by its stable name.
///
/// # Errors
///
/// Returns `Err` if `name` is not one of the supported strategies:
/// `"bookend"`, `"structural"`, `"dedup"`, `"prune"`, `"summarize"`.
///
/// # Examples
///
/// ```
/// use cctx::pipeline::make_strategy;
///
/// let s = make_strategy("bookend").unwrap();
/// assert_eq!(s.name(), "bookend");
/// assert!(make_strategy("nonexistent").is_err());
/// ```
pub fn make_strategy(name: &str) -> Result<Box<dyn Strategy>> {
    match name {
        "bookend" => Ok(Box::new(BookendStrategy)),
        "structural" => Ok(Box::new(StructuralStrategy)),
        "dedup" => Ok(Box::new(DeduplicateStrategy)),
        "prune" => Ok(Box::new(PruneStrategy)),
        "summarize" => Ok(Box::new(SummarizeStrategy)),
        other => anyhow::bail!(
            "Unknown strategy '{}'. Supported: bookend, structural, dedup, prune, summarize",
            other
        ),
    }
}

/// Return the strategy list for a named preset.
///
/// - `"safe"` → `["bookend"]`
/// - `"balanced"` → `["bookend", "structural"]`
/// - `"aggressive"` → `["bookend", "structural", "dedup", "prune", "summarize"]`
///
/// # Errors
///
/// Returns `Err` if `preset` isn't one of the three names above.
///
/// # Examples
///
/// ```
/// use cctx::pipeline::preset_strategies;
/// assert_eq!(preset_strategies("safe").unwrap(), vec!["bookend"]);
/// assert_eq!(preset_strategies("balanced").unwrap(), vec!["bookend", "structural"]);
/// ```
pub fn preset_strategies(preset: &str) -> Result<Vec<&'static str>> {
    match preset {
        "safe" => Ok(vec!["bookend"]),
        "balanced" => Ok(vec!["bookend", "structural"]),
        "aggressive" => Ok(vec!["bookend", "structural", "dedup", "prune", "summarize"]),
        other => anyhow::bail!(
            "Unknown preset '{}'. Supported: safe, balanced, aggressive",
            other
        ),
    }
}
