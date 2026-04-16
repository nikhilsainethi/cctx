//! Deduplication strategies — exact-match and semantic.
//!
//! Exact-match catches verbatim duplicates (copy-paste, repeated prompts).
//! Semantic dedup uses embeddings to find chunks that say the same thing
//! in different words — common in multi-turn conversations and RAG retrieval.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;
use crate::embeddings::{cosine_similarity, EmbeddingProvider};

/// Exact-match deduplication — drop chunks whose `content` has already appeared.
///
/// The first occurrence is always kept; every subsequent chunk with byte-
/// identical content is removed. This is the fallback path when no embedding
/// provider is configured; for semantic deduplication, use [`apply_semantic`].
pub fn apply(context: &Context) -> Vec<Chunk> {
    let mut seen = HashSet::new();
    context
        .chunks
        .iter()
        .filter(|c| seen.insert(c.content.clone()))
        .cloned()
        .collect()
}

/// Semantic deduplication — merge chunks whose embeddings are nearly identical.
///
/// Algorithm:
///
/// 1. Embed every chunk's content in a single batch call.
/// 2. Compute pairwise cosine similarity.
/// 3. For each pair above `threshold`:
///    - Keep the longer chunk (more detail = more value).
///    - Split the shorter chunk into sentences, embed each, and append
///      sentences whose similarity against the longer chunk is below 0.7
///      (they carry unique information).
///    - Mark the shorter chunk for removal.
/// 4. Recount tokens for any chunk whose content grew.
///
/// # Errors
///
/// Returns `Err` if the embedding provider fails or returns a different
/// number of embeddings than chunks — typically a provider configuration
/// or network issue.
pub fn apply_semantic(
    context: &Context,
    provider: &dyn EmbeddingProvider,
    threshold: f64,
    tokenizer: &Tokenizer,
) -> Result<Vec<Chunk>> {
    if context.chunks.len() < 2 {
        return Ok(context.chunks.clone());
    }

    // ── 1. Embed all chunks ───────────────────────────────────────────────
    let texts: Vec<String> = context.chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = provider.embed(&texts)?;

    if embeddings.len() != context.chunks.len() {
        anyhow::bail!(
            "embedding count ({}) doesn't match chunk count ({})",
            embeddings.len(),
            context.chunks.len()
        );
    }

    let n = context.chunks.len();
    let mut removed: HashSet<usize> = HashSet::new();
    let mut appended: HashMap<usize, Vec<String>> = HashMap::new();

    // ── 2. Pairwise comparison ────────────────────────────────────────────
    for i in 0..n {
        if removed.contains(&i) {
            continue;
        }
        for j in (i + 1)..n {
            if removed.contains(&j) {
                continue;
            }

            let sim = cosine_similarity(&embeddings[i], &embeddings[j]) as f64;
            if sim < threshold {
                continue;
            }

            // ── 3. Keep longer, merge unique sentences from shorter ────────
            let (keep, drop) = if context.chunks[i].token_count >= context.chunks[j].token_count {
                (i, j)
            } else {
                (j, i)
            };

            // Split shorter chunk into sentences and find unique ones.
            let sentences = split_sentences(&context.chunks[drop].content);
            if sentences.len() > 1 {
                // Embed individual sentences to check for unique content.
                let sent_texts: Vec<String> = sentences.iter().map(|s| s.to_string()).collect();
                if let Ok(sent_embeddings) = provider.embed(&sent_texts) {
                    for (s_idx, sent_emb) in sent_embeddings.iter().enumerate() {
                        let sent_sim = cosine_similarity(sent_emb, &embeddings[keep]) as f64;
                        // Sentence similarity < 0.7 means it has unique info
                        // not captured by the longer chunk.
                        if sent_sim < 0.7 && !sentences[s_idx].is_empty() {
                            appended
                                .entry(keep)
                                .or_default()
                                .push(sentences[s_idx].to_string());
                        }
                    }
                }
            }

            removed.insert(drop);
        }
    }

    // ── 4. Build result with merged content ───────────────────────────────
    let result: Vec<Chunk> = context
        .chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(i, chunk)| {
            if let Some(unique_sentences) = appended.get(&i) {
                let merged = format!(
                    "{}\n\n[merged from duplicate]: {}",
                    chunk.content,
                    unique_sentences.join(". ")
                );
                Chunk {
                    content: merged.clone(),
                    token_count: tokenizer.count(&merged),
                    ..chunk.clone()
                }
            } else {
                chunk.clone()
            }
        })
        .collect();

    Ok(result)
}

/// Split text into sentences at ". " and newline boundaries.
fn split_sentences(text: &str) -> Vec<&str> {
    text.split('\n')
        .flat_map(|line| line.split(". "))
        .map(|s| s.trim())
        .filter(|s| s.len() > 5) // skip tiny fragments
        .collect()
}
