//! Strategy pipeline — composable, ordered strategy execution.

pub mod executor;

use std::sync::Arc;

use anyhow::Result;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;
use crate::embeddings::EmbeddingProvider;
use crate::strategies::{bookend, dedup, structural};

// ── Strategy trait ────────────────────────────────────────────────────────────

pub trait Strategy {
    fn name(&self) -> &str;
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>>;
}

// ── Pipeline config ───────────────────────────────────────────────────────────

pub struct PipelineConfig {
    pub query: Option<String>,
    pub tokenizer: Tokenizer,
    /// Embedding provider for semantic dedup. None = fallback to exact-match.
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Cosine similarity threshold for semantic dedup (default 0.85).
    pub dedup_threshold: f64,
}

// ── Strategy wrappers ─────────────────────────────────────────────────────────

pub struct BookendStrategy;

impl Strategy for BookendStrategy {
    fn name(&self) -> &str {
        "bookend"
    }
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>> {
        Ok(bookend::apply(context, config.query.as_deref()))
    }
}

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

// ── Factory + presets ─────────────────────────────────────────────────────────

pub fn make_strategy(name: &str) -> Result<Box<dyn Strategy>> {
    match name {
        "bookend" => Ok(Box::new(BookendStrategy)),
        "structural" => Ok(Box::new(StructuralStrategy)),
        "dedup" => Ok(Box::new(DeduplicateStrategy)),
        other => anyhow::bail!(
            "Unknown strategy '{}'. Supported: bookend, structural, dedup",
            other
        ),
    }
}

pub fn preset_strategies(preset: &str) -> Result<Vec<&'static str>> {
    match preset {
        "safe" => Ok(vec!["bookend"]),
        "balanced" => Ok(vec!["bookend", "structural"]),
        "aggressive" => Ok(vec!["bookend", "structural", "dedup"]),
        other => anyhow::bail!(
            "Unknown preset '{}'. Supported: safe, balanced, aggressive",
            other
        ),
    }
}
