use std::collections::{HashMap, HashSet};

use crate::core::context::{Chunk, Context};

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply bookend reordering with optional query-based relevance scoring.
///
/// When `query` is Some: rescore every chunk using TF-IDF against the query
/// so chunks that answer the user's question get placed at the bookends.
///
/// When `query` is None: use an improved heuristic — system messages and the
/// last 3 user messages get highest priority (they represent the most recent
/// intent), everything else scored by recency.
///
/// Then apply the alternating-placement algorithm (Liu et al., TACL 2024):
///   rank 1 → position 0, rank 2 → position N-1, rank 3 → position 1, …
///
/// `Option<&str>` is Rust's idiomatic "maybe a string":
///   - Some("how do I handle refunds?") → query provided
///   - None → no query, use heuristic
pub fn apply(context: &Context, query: Option<&str>) -> Vec<Chunk> {
    let mut chunks = context.chunks.clone();

    // Rescore based on query or heuristic.
    match query {
        Some(q) => score_by_query(&mut chunks, q),
        None => score_by_heuristic(&mut chunks),
    }

    // Sort descending by the (now-updated) relevance scores.
    chunks.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = chunks.len();
    if n == 0 {
        return vec![];
    }

    // Alternating placement into result slots.
    let mut result: Vec<Option<Chunk>> = vec![None; n];
    let mut front = 0usize;
    let mut back = n - 1;

    for (rank, chunk) in chunks.into_iter().enumerate() {
        if front > back {
            break;
        }
        if rank % 2 == 0 {
            result[front] = Some(chunk);
            front += 1;
        } else {
            result[back] = Some(chunk);
            if back == 0 {
                break;
            }
            back -= 1;
        }
    }

    result.into_iter().flatten().collect()
}

// ── TF-IDF query scoring ──────────────────────────────────────────────────────
//
// TF-IDF (Term Frequency × Inverse Document Frequency) scores how relevant
// a chunk is to a query. Two components:
//
//   TF = (times word appears in chunk) / (total words in chunk)
//        → measures local importance: a chunk that mentions "refund" 5 times
//          is more relevant to a refund query than one mentioning it once.
//
//   IDF = ln(N / (1 + df))   where df = chunks containing this word
//        → measures global rarity: "refund" appearing in 2 of 20 chunks is
//          more discriminating than "the" appearing in all 20.
//
// Final score = Σ(TF × IDF) for each query word.

fn score_by_query(chunks: &mut [Chunk], query: &str) {
    let query_words = tokenize(query);
    if query_words.is_empty() {
        score_by_heuristic(chunks);
        return;
    }

    let n = chunks.len() as f64;

    // Tokenize each chunk's content once (avoids repeated work).
    let chunk_word_lists: Vec<Vec<String>> = chunks.iter().map(|c| tokenize(&c.content)).collect();

    // Document frequency: how many chunks contain each query word.
    // HashMap<&str, usize> maps borrowed string slices to counts.
    // The keys borrow from query_words which outlives this HashMap.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for qw in &query_words {
        let count = chunk_word_lists
            .iter()
            .filter(|words| words.iter().any(|w| w == qw))
            .count();
        df.insert(qw.as_str(), count);
    }

    // Compute TF-IDF score per chunk.
    let mut scores: Vec<f64> = chunk_word_lists
        .iter()
        .map(|words| {
            if words.is_empty() {
                return 0.0;
            }
            let word_count = words.len() as f64;
            query_words
                .iter()
                .map(|qw| {
                    // TF: fraction of chunk words that match this query word.
                    let tf = words.iter().filter(|w| *w == qw).count() as f64 / word_count;
                    // IDF: log-scaled rarity. Clamp to 0 — if every chunk has the word,
                    // it's not discriminating and shouldn't contribute.
                    let raw_idf =
                        (n / (1.0 + *df.get(qw.as_str()).unwrap_or(&0) as f64)).ln();
                    tf * raw_idf.max(0.0)
                })
                .sum::<f64>()
        })
        .collect();

    // Normalize to 0.0–1.0 so scores are comparable across different queries.
    let max = scores.iter().copied().fold(0.0f64, f64::max);
    if max > 0.0 {
        for s in &mut scores {
            *s /= max;
        }
    }

    // Apply scores. System messages always keep max relevance.
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.relevance_score = if chunk.role == "system" {
            1.0
        } else {
            scores[i]
        };
    }
}

// ── Heuristic scoring (no query) ──────────────────────────────────────────────
//
// When the user doesn't specify a query, we use a smarter heuristic than pure
// recency. The intuition: system prompts define the task, the last few user
// messages represent the current intent, and everything else fades by age.

fn score_by_heuristic(chunks: &mut [Chunk]) {
    let n = chunks.len();
    if n == 0 {
        return;
    }

    // Collect the Vec-position indices of the last 3 user messages.
    //
    // .rev() on a filtered iterator walks backward through the original
    // sequence — no intermediate collection needed. This is a zero-cost
    // abstraction: the compiler fuses the iterator chain into a single loop.
    let last_3_user: HashSet<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.role == "user")
        .map(|(i, _)| i)
        .rev()
        .take(3)
        .collect();

    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.relevance_score = if chunk.role == "system" {
            1.0 // System prompts are always critical — define the LLM's role.
        } else if last_3_user.contains(&i) {
            0.9 // Recent user messages represent the current question/intent.
        } else if n <= 1 {
            0.5
        } else {
            // Everything else: linear recency score 0.1–0.7.
            // Later messages score higher but cap below the 0.9 "last user" tier.
            0.1 + (i as f64 / (n - 1) as f64) * 0.6
        };
    }
}

// ── Tokenization ──────────────────────────────────────────────────────────────

/// Split text into lowercase words with punctuation stripped.
/// Shared by TF-IDF scoring and duplication detection would use a similar approach.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        // trim_matches takes a closure predicate — strip chars where the closure returns true.
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '$')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}
