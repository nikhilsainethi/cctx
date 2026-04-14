//! Strategy pipeline — composable, ordered strategy execution.
//!
//! The pipeline module defines:
//!   - `Strategy` trait: common interface for all optimization strategies
//!   - `PipelineConfig`: shared configuration (query, tokenizer)
//!   - Wrapper structs that adapt each strategy to the trait
//!   - Preset definitions (safe, balanced, aggressive)

pub mod executor;

use anyhow::Result;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;
use crate::strategies::{bookend, dedup, structural};

// ── Strategy trait ────────────────────────────────────────────────────────────
//
// `dyn Strategy` is a *trait object* — Rust's runtime polymorphism.
// A `Box<dyn Strategy>` is a fat pointer: one pointer to the data, one to a
// vtable of function pointers (name, apply). This lets us store different
// strategy types in the same Vec without generics.

/// Common interface for all optimization strategies.
pub trait Strategy {
    /// Human-readable name shown in pipeline logs.
    fn name(&self) -> &str;

    /// Transform a context: reorder, compress, or deduplicate chunks.
    /// Returns the new chunk list. The pipeline rebuilds the Context.
    fn apply(&self, context: &Context, config: &PipelineConfig) -> Result<Vec<Chunk>>;
}

// ── Pipeline config ───────────────────────────────────────────────────────────

/// Shared configuration passed to every strategy in the pipeline.
pub struct PipelineConfig {
    /// Optional query for TF-IDF scoring / markdown collapse.
    pub query: Option<String>,
    /// Shared tokenizer for token recounting after compression.
    pub tokenizer: Tokenizer,
}

// ── Strategy wrappers ─────────────────────────────────────────────────────────
//
// Each wrapper is a zero-size struct that implements Strategy by delegating
// to the existing functions in src/strategies/. This keeps the strategy
// algorithms independent of the trait system.

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
    fn apply(&self, context: &Context, _config: &PipelineConfig) -> Result<Vec<Chunk>> {
        Ok(dedup::apply(context))
    }
}

// ── Factory + presets ─────────────────────────────────────────────────────────

/// Build a boxed Strategy from a name string (used by --strategy flag).
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

/// Expand a preset name into an ordered list of strategy names.
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
