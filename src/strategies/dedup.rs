//! Exact-match deduplication — removes chunks with identical content.
//!
//! This is a stub for the full semantic dedup (Week 3, needs embeddings).
//! For now it catches verbatim duplicates: copy-pasted messages, repeated
//! system prompts, or RAG chunks retrieved more than once.

use std::collections::HashSet;

use crate::core::context::{Chunk, Context};

/// Remove chunks whose content is an exact duplicate of an earlier chunk.
/// First occurrence is kept; subsequent duplicates are dropped.
pub fn apply(context: &Context) -> Vec<Chunk> {
    // HashSet::insert returns false if the value was already present.
    // We use that as a one-line dedup filter.
    let mut seen = HashSet::new();
    context
        .chunks
        .iter()
        .filter(|c| seen.insert(c.content.clone()))
        .cloned()
        .collect()
}
