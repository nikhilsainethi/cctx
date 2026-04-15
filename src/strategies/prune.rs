//! Importance-aware token pruning — removes low-information sentences.
//!
//! Research basis: LLMLingua (Microsoft), ATACompressor (SIGIR 2025).
//!
//! This is the Tier 0 (heuristic) version — no ML model needed.
//! Each sentence is scored 0.0–1.0 by combining:
//!   1. Stop-word ratio (high ratio = filler)
//!   2. Repetition detection (content seen in earlier messages)
//!   3. Structural importance (role, code blocks, headings)
//!   4. Filler phrase detection ("Sure!", "Great question!", etc.)
//!   5. Length penalty (very short acknowledgments)
//!
//! Sentences below `--prune-threshold` (default 0.3) are removed.
//! Messages that become empty are replaced with "[earlier context summarized]".
//! System messages and the last 2 user messages are NEVER pruned.

use std::collections::HashSet;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;

// ── Stop words ────────────────────────────────────────────────────────────────

const STOP_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "it",
    "its",
    "are",
    "was",
    "be",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "me",
    "him",
    "us",
    "them",
    "what",
    "which",
    "who",
    "whom",
    "if",
    "not",
    "no",
    "so",
    "as",
    "from",
    "about",
    "into",
    "through",
    "also",
    "just",
    "more",
    "very",
    "too",
    "than",
    "then",
    "here",
    "there",
    "when",
    "where",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "some",
    "any",
    "most",
    "other",
    "new",
    "old",
    "been",
    "being",
    "get",
    "got",
    "going",
    "go",
    "come",
    "came",
    "make",
    "made",
    "take",
    "took",
    "know",
    "known",
    "think",
    "see",
    "want",
    "give",
    "use",
    "tell",
    "try",
    "like",
    "well",
    "way",
    "now",
    "even",
    "still",
    "actually",
    "really",
    "basically",
    "right",
    "sure",
    "yes",
    "yeah",
    "ok",
    "okay",
];

// ── Filler phrases ────────────────────────────────────────────────────────────

const FILLER_PHRASES: &[&str] = &[
    "sure, i can help",
    "great question",
    "good question",
    "that's a great question",
    "absolutely",
    "of course",
    "let me help",
    "i'd be happy to",
    "no problem",
    "sure thing",
    "thanks for asking",
    "thanks for sharing",
    "thanks for the context",
    "thank you for",
    "i understand",
    "i see",
    "got it",
    "makes sense",
    "let me explain",
    "let me break this down",
    "here's the thing",
    "so basically",
    "to be honest",
    "in my opinion",
    "as i mentioned",
    "as you know",
    "as we discussed",
];

// ── Public API ────────────────────────────────────────────────────────────────

/// Prune low-importance sentences from each message.
///
/// Protected chunks (never pruned):
///   - System messages (define the LLM's role)
///   - Last 2 user messages (current intent)
///
/// Returns new chunks with trimmed content and recounted tokens.
pub fn apply(context: &Context, threshold: f64, tokenizer: &Tokenizer) -> Vec<Chunk> {
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let n = context.chunks.len();

    // Find the last 2 user message positions (protected from pruning).
    let protected_users: HashSet<usize> = context
        .chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.role == "user")
        .map(|(i, _)| i)
        .rev()
        .take(2)
        .collect();

    // Build a set of "content fingerprints" from earlier messages for
    // repetition detection. Each fingerprint is the set of non-stop words.
    let mut seen_content: Vec<HashSet<String>> = Vec::new();

    context
        .chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            // ── Protected chunks pass through unchanged ───────────────────
            let is_protected = chunk.role == "system" || protected_users.contains(&i);
            if is_protected {
                seen_content.push(content_fingerprint(&chunk.content, &stop));
                return chunk.clone();
            }

            // ── Score and filter lines ─────────────────────────────────────
            // Split on newlines only (preserves original formatting).
            // Each line is scored; lines below threshold are dropped.
            let lines: Vec<&str> = chunk.content.split('\n').collect();
            let mut kept: Vec<&str> = Vec::new();
            let mut in_code_block = false;

            for line in &lines {
                let trimmed = line.trim();
                // Code block fences and their contents are always kept.
                if trimmed.starts_with("```") {
                    in_code_block = !in_code_block;
                    kept.push(line);
                    continue;
                }
                if in_code_block {
                    kept.push(line);
                    continue;
                }
                // Empty lines pass through (preserve paragraph breaks).
                if trimmed.is_empty() {
                    kept.push(line);
                    continue;
                }

                let score = score_sentence(line, &chunk.role, &stop, &seen_content, i, n);

                if score >= threshold {
                    kept.push(line);
                }
            }

            // Track this chunk's content for future repetition detection.
            seen_content.push(content_fingerprint(&chunk.content, &stop));

            // ── Rebuild content ───────────────────────────────────────────
            // Join with "\n" — lossless since we split on "\n".
            let new_content = kept.join("\n");
            let new_content = if new_content.trim().is_empty() {
                "[earlier context summarized]".to_string()
            } else {
                new_content
            };

            Chunk {
                content: new_content.clone(),
                token_count: tokenizer.count(&new_content),
                ..chunk.clone()
            }
        })
        .collect()
}

// ── Sentence scoring ──────────────────────────────────────────────────────────

/// Score a sentence 0.0–1.0. Higher = more important, keep it.
fn score_sentence(
    sentence: &str,
    role: &str,
    stop: &HashSet<&str>,
    seen_content: &[HashSet<String>],
    chunk_idx: usize,
    total_chunks: usize,
) -> f64 {
    let words = tokenize_words(sentence, stop);
    let all_words: Vec<String> = sentence
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if all_words.is_empty() {
        return 0.0;
    }

    // ── 1. Stop-word ratio (weight: 0.25) ─────────────────────────────────
    // High stop-word ratio = likely filler. >60% is a bad sign.
    let stop_ratio = 1.0 - (words.len() as f64 / all_words.len() as f64);
    let stop_score = if stop_ratio > 0.7 {
        0.1 // almost all stop words
    } else if stop_ratio > 0.6 {
        0.3
    } else {
        0.8
    };

    // ── 2. Repetition detection (weight: 0.25) ───────────────────────────
    // If these content words appeared in an earlier chunk → redundant.
    let word_set: HashSet<String> = words.iter().cloned().collect();
    let mut max_overlap = 0.0f64;
    for earlier in seen_content {
        if earlier.is_empty() || word_set.is_empty() {
            continue;
        }
        let overlap = word_set.intersection(earlier).count() as f64 / word_set.len().max(1) as f64;
        if overlap > max_overlap {
            max_overlap = overlap;
        }
    }
    // overlap > 0.7 means most content words were said before.
    let repetition_score = if max_overlap > 0.7 {
        0.1
    } else if max_overlap > 0.5 {
        0.4
    } else {
        0.9
    };

    // ── 3. Structural importance (weight: 0.20) ──────────────────────────
    let trimmed = sentence.trim();
    let structural_score = if trimmed.starts_with('#') {
        1.0 // Markdown heading
    } else if trimmed.starts_with("```") || trimmed.starts_with("    ") {
        0.85 // Code block
    } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("1.") {
        0.7 // List item
    } else if role == "assistant" {
        0.6 // Assistant body text
    } else {
        0.65 // User body text
    };

    // ── 4. Filler phrase detection (weight: 0.15) ─────────────────────────
    let lower = sentence.to_lowercase();
    let is_filler = FILLER_PHRASES
        .iter()
        .any(|phrase| lower.starts_with(phrase) || lower.contains(phrase));
    let filler_score = if is_filler { 0.1 } else { 0.9 };

    // ── 5. Length penalty (weight: 0.15) ──────────────────────────────────
    // Very short sentences (<5 words) that aren't headings or code → likely
    // just acknowledgments like "Thanks!" or "OK".
    let length_score = if all_words.len() < 4 && !trimmed.starts_with('#') {
        0.2
    } else if all_words.len() < 8 {
        0.5
    } else {
        0.9
    };

    // ── Recency bonus ─────────────────────────────────────────────────────
    // Later messages are more likely to contain the current conversation state.
    let recency = if total_chunks <= 1 {
        0.5
    } else {
        0.3 + (chunk_idx as f64 / (total_chunks - 1) as f64) * 0.7
    };

    // ── Compound penalty ───────────────────────────────────────────────
    // When multiple low signals coincide, the line is almost certainly noise.
    // Short + filler → definitely prunable.
    // Short + high stop ratio → "So basically yeah" → prunable.
    let compound: f64 = if is_filler && all_words.len() < 8 {
        0.0 // filler AND short → guaranteed prune
    } else if stop_score <= 0.3 && all_words.len() < 6 {
        0.1 // almost all stop words AND very short
    } else {
        1.0 // no compound penalty
    };

    // ── Weighted combination ──────────────────────────────────────────────
    let raw = stop_score * 0.20
        + repetition_score * 0.20
        + structural_score * 0.10
        + filler_score * 0.20
        + length_score * 0.15
        + recency * 0.10;

    // Apply compound penalty as a multiplier.
    (raw * compound.max(0.1)).clamp(0.0, 1.0)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract meaningful (non-stop) words from text, lowercased.
fn tokenize_words(text: &str, stop: &HashSet<&str>) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| w.len() > 1 && !stop.contains(w.as_str()))
        .collect()
}

/// Content fingerprint for repetition detection — set of non-stop words.
fn content_fingerprint(text: &str, stop: &HashSet<&str>) -> HashSet<String> {
    tokenize_words(text, stop).into_iter().collect()
}
