use std::collections::HashSet;

use serde::Serialize;

use crate::core::context::Chunk;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A detected pair of near-duplicate chunks with their Jaccard similarity.
#[derive(Serialize)]
pub struct DuplicatePair {
    pub chunk_a: usize,
    pub chunk_b: usize,
    /// Jaccard similarity coefficient: 0.0 = no overlap, 1.0 = identical word sets.
    pub similarity: f64,
}

// ── Stop words ────────────────────────────────────────────────────────────────
//
// Common English words that inflate similarity without indicating real
// content duplication. Filtering these makes Jaccard focus on meaningful overlap.

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "it", "its", "are", "was", "be", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "can", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "my", "your", "his", "her", "our", "their", "me", "him", "us", "them", "what",
    "which", "who", "whom", "if", "not", "no", "so", "as", "from", "about", "into", "through",
    "also", "just", "more", "very", "too", "than",
];

// ── Word extraction ───────────────────────────────────────────────────────────

/// Extract meaningful words from text: lowercase, strip punctuation, remove stop words.
///
/// `HashSet<String>` is a set of unique strings — used for set intersection/union.
/// HashSet gives O(1) lookups and automatically deduplicates.
fn extract_words(text: &str) -> HashSet<String> {
    // Build a HashSet of stop words for O(1) lookup.
    // iter().copied() turns &&str into &str (peels off one layer of reference).
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    text.split_whitespace()
        // to_lowercase() returns a new String — we own it, original &str untouched.
        .map(|w| w.to_lowercase())
        // trim_matches strips characters from both ends of a string.
        // The closure |c: char| is a predicate — strip if true.
        // We keep alphanumeric chars and $ (for "$3,000" style tokens).
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '$')
                .to_string()
        })
        // Filter: keep words longer than 1 char and not in stop list.
        // .as_str() borrows the String as &str so we can look it up in the stop set.
        .filter(|w| w.len() > 1 && !stop.contains(w.as_str()))
        .collect()
}

// ── Jaccard similarity ────────────────────────────────────────────────────────

/// Jaccard index: |A ∩ B| / |A ∪ B|.
///
/// This is a classic set-similarity metric. Two identical sets → 1.0,
/// completely disjoint → 0.0. Works well for detecting topical overlap
/// in short text (chat messages) without needing any ML model.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    // .intersection() returns an iterator of elements in both sets.
    let intersection = a.intersection(b).count();
    // .union() returns an iterator of elements in either set (no duplicates).
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Detect near-duplicate chunk pairs.
///
/// Compares every pair of chunks (O(n^2) — fine for conversations with < 200 turns).
/// Returns (pairs above threshold, estimated duplicate token count).
///
/// `&[Chunk]` is a *slice* — a borrowed view of a contiguous sequence.
/// It works with Vec<Chunk>, arrays, or any contiguous collection.
pub fn detect_duplicates(chunks: &[Chunk], threshold: f64) -> (Vec<DuplicatePair>, usize) {
    // Pre-compute word sets so each chunk is tokenized only once.
    let word_sets: Vec<HashSet<String>> =
        chunks.iter().map(|c| extract_words(&c.content)).collect();

    let mut pairs = Vec::new();
    let mut dup_tokens_est = 0.0f64;

    for i in 0..chunks.len() {
        for j in (i + 1)..chunks.len() {
            let sim = jaccard(&word_sets[i], &word_sets[j]);
            if sim >= threshold {
                pairs.push(DuplicatePair {
                    chunk_a: chunks[i].index,
                    chunk_b: chunks[j].index,
                    similarity: sim,
                });
                // Estimate: the smaller chunk's tokens × similarity ≈ redundant tokens.
                let smaller = chunks[i].token_count.min(chunks[j].token_count) as f64;
                dup_tokens_est += sim * smaller;
            }
        }
    }

    // Sort most-similar pairs first (descending).
    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    (pairs, dup_tokens_est.round() as usize)
}
