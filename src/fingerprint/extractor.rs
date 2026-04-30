//! Three-layer extraction pipeline for the Tier-0 fingerprint engine.
//!
//! Each layer adds candidate items into a shared bag, deduping at the
//! `(chunk_idx, sentence)` level so the same sentence can't be captured
//! twice from the same chunk:
//!
//! 1. **Regex layer** — pulls technical facts (ports, paths, URLs, IPs,
//!    env vars, version strings, dollar amounts, time durations) out of
//!    every chunk and emits the *containing sentence* as the item content.
//!    Items are tagged [`ItemCategory::TechnicalFact`].
//! 2. **Keyword layer** — splits each chunk into sentences and captures
//!    sentences containing trigger phrases for constraints, decisions,
//!    debug insights, or progress markers.
//! 3. **RAKE layer** — runs RAKE over the concatenated transcript text,
//!    takes the top-N keyphrases, and adds sentences containing them
//!    only if no earlier layer already captured that sentence.
//!
//! System chunks are intentionally skipped — they're prompt boilerplate
//! that repeats across sessions and adds no fingerprintable signal.

use std::collections::HashSet;

use keyword_extraction::rake::{Rake, RakeParams};
use regex::Regex;

#[cfg(test)]
use crate::core::context::Chunk;
use crate::core::context::Context;

use super::types::{make_id, FingerprintItem, ItemCategory, ItemScores};

// ── Stop words for RAKE + sentence splitting ──────────────────────────────────

/// Compact English stopword list used by the layer-3 RAKE pass.
/// Hardcoded so we don't depend on a separate `stop-words` crate.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "doing", "will", "would",
    "could", "should", "may", "might", "must", "can", "shall", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their", "its", "what", "which", "who",
    "whom", "as", "also", "just", "like",
];

// ── Public entry point ────────────────────────────────────────────────────────

/// Run all three extraction layers over `context` and return the raw
/// candidate items (unscored, undeduplicated cross-item).
///
/// Layer-internal dedup at the `(chunk_idx, sentence)` granularity is
/// already applied — same sentence in same chunk yields one item even if
/// multiple regexes or trigger words match. Cross-item Jaccard dedup is
/// the responsibility of [`super::dedup`].
pub fn extract(context: &Context) -> Vec<FingerprintItem> {
    let regexes = compile_regexes();

    // Collect (chunk_idx, sentence_text) pairs once — every layer reuses them.
    let sentences = collect_sentences(context);

    // Dedup memory shared across layers so the RAKE pass can skip
    // sentences already captured by earlier passes.
    let mut seen: HashSet<(usize, String)> = HashSet::new();
    let mut items: Vec<FingerprintItem> = Vec::new();

    // Layer ordering matters: keyword triggers categorize semantically
    // ("Constraint", "Decision", …) and outweigh raw pattern hits, so we
    // run them first. Regex catches what the keyword pass left over —
    // ports, paths, URLs that happen to sit in plain technical
    // sentences. RAKE last as the catch-all.
    extract_keyword_layer(&sentences, &mut seen, &mut items);
    extract_regex_layer(&sentences, &regexes, &mut seen, &mut items);
    extract_rake_layer(&sentences, &mut seen, &mut items);

    items
}

// ── Regex layer (Tier-0 layer 1) ──────────────────────────────────────────────

/// One regex pattern. The `name` is purely informational so debug output
/// can tell which pattern caught a sentence.
struct PatternRule {
    #[allow(dead_code)] // kept for diagnostics + future telemetry
    name: &'static str,
    regex: Regex,
}

fn compile_regexes() -> Vec<PatternRule> {
    // Patterns adapted from the Day 22 brief. Each is anchored with
    // word boundaries where appropriate so we don't pick up
    // sub-fragments of identifiers.
    let raw: &[(&str, &str)] = &[
        // "port 8443" or ":8443" — capture 2-5 digit groups
        ("port", r"(?i)(?:port\s+|:)(\d{2,5})\b"),
        // Absolute paths with at least two segments (avoid bare "/etc")
        ("path", r"(?:/[\w.-]+){2,}"),
        // http(s) URLs (use multi-hash raw string so we can include `"`)
        ("url", r#"https?://[^\s)\]>"']+"#),
        // IPv4 addresses
        ("ipv4", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        // Env var assignments: SCREAMING_SNAKE=value
        ("env", r"\b[A-Z][A-Z0-9_]{2,}=\S+"),
        // Version strings: "1.2.3", "v1.2", "2.0.0-beta1"
        ("version", r"\bv?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b"),
        // Dollar amounts: $50, $50K, $1.5M, $1,000
        ("money", r"\$[\d,.]+[KMBkmb]?"),
        // Durations: "30 seconds", "2 weeks", "10 mins"
        (
            "duration",
            r"(?i)\b\d+\s*(?:hours?|hrs?|days?|weeks?|months?|minutes?|mins?|seconds?|secs?)\b",
        ),
    ];

    raw.iter()
        .filter_map(|(name, pat)| {
            Regex::new(pat)
                .ok()
                .map(|regex| PatternRule { name, regex })
        })
        .collect()
}

fn extract_regex_layer(
    sentences: &[(usize, String)],
    regexes: &[PatternRule],
    seen: &mut HashSet<(usize, String)>,
    items: &mut Vec<FingerprintItem>,
) {
    for (chunk_idx, sentence) in sentences {
        // Pass cheaply if the sentence has no regex hit at all.
        if !regexes.iter().any(|r| r.regex.is_match(sentence)) {
            continue;
        }
        let key = (*chunk_idx, sentence.clone());
        if !seen.insert(key) {
            continue;
        }
        items.push(make_item(
            sentence,
            ItemCategory::TechnicalFact,
            *chunk_idx,
            "regex",
        ));
    }
}

// ── Keyword layer (Tier-0 layer 2) ────────────────────────────────────────────

/// Trigger lists — each (category, &[phrases]) tuple defines what to
/// classify a matching sentence as. Lookup is case-insensitive substring
/// matching against the sentence.
fn keyword_triggers() -> &'static [(ItemCategory, &'static [&'static str])] {
    &[
        (
            ItemCategory::Constraint,
            &[
                "must",
                "should not",
                "shouldn't",
                "requirement",
                "cannot",
                "can't",
                "budget",
                "deadline",
                "limit",
                "no more than",
                "at least",
                "maximum",
                "minimum",
                "not allowed",
                "forbidden",
                "restrict",
                "comply",
                "mandatory",
            ],
        ),
        (
            ItemCategory::Decision,
            &[
                "chose",
                "decided",
                "went with",
                "agreed on",
                "let's go with",
                "let's use",
                "instead of",
                "picked",
                "selected",
                "opting for",
                "trade-off",
                " over ",
                "because we",
                "the reason",
                "we'll use",
            ],
        ),
        (
            ItemCategory::DebugInsight,
            &[
                "the issue was",
                "root cause",
                "fixed by",
                "caused by",
                "the problem was",
                "turns out",
                "the fix",
                "the error was",
                "solved by",
                "the bug was",
                "resolved by",
                "workaround",
            ],
        ),
        (
            ItemCategory::ProgressMarker,
            &[
                "done",
                "completed",
                "remaining",
                "todo",
                "next step",
                "finished",
                "still need to",
                "left to do",
                "in progress",
                "blocked on",
                "shipped",
                "deployed",
            ],
        ),
    ]
}

fn extract_keyword_layer(
    sentences: &[(usize, String)],
    seen: &mut HashSet<(usize, String)>,
    items: &mut Vec<FingerprintItem>,
) {
    let triggers = keyword_triggers();

    for (chunk_idx, sentence) in sentences {
        if !is_meaningful_sentence(sentence) {
            continue;
        }
        let lower = sentence.to_lowercase();

        // First-trigger-wins. Categories are listed Constraint → Decision
        // → DebugInsight → ProgressMarker so generic terms ("done") don't
        // shadow specific ones ("budget", "root cause") on a sentence
        // that happens to contain both.
        let category = triggers
            .iter()
            .find(|(_, phrases)| phrases.iter().any(|p| lower.contains(p)))
            .map(|(cat, _)| cat.clone());

        let Some(category) = category else { continue };

        let key = (*chunk_idx, sentence.clone());
        if !seen.insert(key) {
            continue;
        }
        items.push(make_item(sentence, category, *chunk_idx, "keyword"));
    }
}

// ── RAKE layer (Tier-0 layer 3) ───────────────────────────────────────────────

/// Run RAKE over the concatenated transcript text, take top-N keyphrases,
/// and emit one [`ItemCategory::Other`] item per keyphrase whose
/// containing sentence wasn't already captured by an earlier layer.
fn extract_rake_layer(
    sentences: &[(usize, String)],
    seen: &mut HashSet<(usize, String)>,
    items: &mut Vec<FingerprintItem>,
) {
    if sentences.is_empty() {
        return;
    }

    // RAKE ingests one big string. Sentences are joined with ". " so
    // RAKE's punctuation-aware phrase splitter still respects boundaries.
    let blob: String = sentences
        .iter()
        .map(|(_, s)| s.as_str())
        .collect::<Vec<_>>()
        .join(". ");
    if blob.trim().is_empty() {
        return;
    }

    // RakeParams::WithDefaults expects `&[String]` for stopwords.
    let stopwords: Vec<String> = STOP_WORDS.iter().map(|s| s.to_string()).collect();
    let params = RakeParams::WithDefaults(blob.as_str(), &stopwords);
    let rake = Rake::new(params);
    let ranked = rake.get_ranked_phrases_scores(30);
    if ranked.is_empty() {
        return;
    }

    let max_score = ranked
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_score = if max_score.is_finite() && max_score > 0.0 {
        max_score
    } else {
        1.0
    };

    for (phrase, score) in ranked.iter() {
        // Phrases too short to be useful (single stop-words slip through
        // sometimes). Skip them.
        if phrase.trim().len() < 4 {
            continue;
        }
        let phrase_lower = phrase.to_lowercase();

        for (chunk_idx, sentence) in sentences {
            if !is_meaningful_sentence(sentence) {
                continue;
            }
            if !sentence.to_lowercase().contains(&phrase_lower) {
                continue;
            }
            let key = (*chunk_idx, sentence.clone());
            if !seen.insert(key) {
                continue;
            }
            // Normalize RAKE score to [0, 1] using the batch maximum.
            let normalized = (score / max_score).clamp(0.0, 1.0) as f64;
            let mut item = make_item(sentence, ItemCategory::Other, *chunk_idx, "rake");
            item.scores.textrank_boost = normalized;
            items.push(item);
        }
    }
}

// ── Sentence collection + helpers ─────────────────────────────────────────────

/// Walk the context once and produce one `(chunk_idx, sentence)` tuple
/// per non-trivial sentence found in any non-system chunk.
///
/// System chunks are skipped — they're typically prompt boilerplate that
/// repeats across sessions and isn't fingerprintable signal. Tool
/// interaction chunks are kept because they often contain the
/// highest-density technical facts (file paths, ports, IPs).
fn collect_sentences(context: &Context) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    for (idx, chunk) in context.chunks.iter().enumerate() {
        if chunk.role == "system" {
            continue;
        }
        for sentence in split_sentences(&chunk.content) {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }
            out.push((idx, trimmed.to_string()));
        }
    }
    out
}

/// Cheap English sentence splitter — line-breaks are hard boundaries,
/// then `. ! ?` followed by whitespace. Good enough for chat / code
/// dumps where Real NLP-grade splitting isn't worth the dependency.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        // Walk byte-by-byte tracking a sentence buffer.
        let mut buf = String::new();
        let mut prev_terminal = false;
        for ch in line.chars() {
            buf.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                prev_terminal = true;
                continue;
            }
            if prev_terminal && ch.is_whitespace() {
                let s = buf.trim().to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                buf.clear();
                prev_terminal = false;
            } else {
                prev_terminal = false;
            }
        }
        let tail = buf.trim();
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
    }
    out
}

/// Filter for the keyword + RAKE layers. Single-word fragments and
/// novel-length essays are both noise — keep the middle 5–200 words.
fn is_meaningful_sentence(sentence: &str) -> bool {
    let words = sentence.split_whitespace().count();
    (5..=200).contains(&words)
}

/// Construct a [`FingerprintItem`] with placeholder scores. Scoring is a
/// separate pass; this seeds occurrence_count = 1 and source_positions
/// pointed at the chunk we extracted from.
fn make_item(
    content: &str,
    category: ItemCategory,
    chunk_idx: usize,
    method: &str,
) -> FingerprintItem {
    let content = content.to_string();
    FingerprintItem {
        id: make_id(&content),
        category,
        content,
        // Token count is filled in by the engine after scoring (we want
        // to amortize the tokenizer across all items, not init it here).
        tokens: 0,
        occurrence_count: 1,
        source_positions: vec![chunk_idx],
        priority_score: 0.0,
        scores: ItemScores {
            uniqueness: 1.0,
            recency: 1.0,
            position_risk: 0.0,
            textrank_boost: 0.0,
        },
        extraction_method: method.to_string(),
        confidence: None,
    }
}

/// Test helper — exposes [`Chunk`] without leaking it into the public surface.
#[cfg(test)]
pub(super) fn _chunk_for_test(role: &str, content: &str) -> Chunk {
    use crate::core::context::AttentionZone;
    Chunk {
        index: 0,
        role: role.to_string(),
        content: content.to_string(),
        token_count: 0,
        relevance_score: 0.5,
        attention_zone: AttentionZone::Strong,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(chunks: Vec<Chunk>) -> Context {
        Context::new(chunks)
    }

    #[test]
    fn regex_layer_captures_port_and_path_and_money() {
        let context = ctx(vec![_chunk_for_test(
            "user",
            "We deployed the auth service on port 8443. Config lives at /etc/myapp/prod.yml. \
                 The contract caps spending at $50K total.",
        )]);
        let items = extract(&context);
        let bodies: Vec<_> = items.iter().map(|i| i.content.as_str()).collect();

        assert!(bodies.iter().any(|b| b.contains("port 8443")));
        assert!(bodies.iter().any(|b| b.contains("/etc/myapp/prod.yml")));
        assert!(bodies.iter().any(|b| b.contains("$50K")));
        assert!(items.iter().any(|i| i.extraction_method == "regex"));
    }

    #[test]
    fn keyword_layer_classifies_constraint_decision_debug_progress() {
        let context = ctx(vec![
            _chunk_for_test(
                "user",
                "Reminder, our budget should not exceed forty thousand dollars this quarter.",
            ),
            _chunk_for_test(
                "user",
                "After more thought we decided to go with Postgres because it gives us ACID.",
            ),
            _chunk_for_test(
                "user",
                "I tracked it down — the root cause was a missing CORS header on staging.",
            ),
            _chunk_for_test(
                "user",
                "First two endpoints are completed, the rest remain blocked on review.",
            ),
        ]);
        let items = extract(&context);
        assert!(items.iter().any(|i| i.category == ItemCategory::Constraint
            && i.content.to_lowercase().contains("budget")));
        assert!(items.iter().any(|i| i.category == ItemCategory::Decision
            && i.content.to_lowercase().contains("postgres")));
        assert!(items
            .iter()
            .any(|i| i.category == ItemCategory::DebugInsight
                && i.content.to_lowercase().contains("cors")));
        assert!(items
            .iter()
            .any(|i| i.category == ItemCategory::ProgressMarker
                && i.content.to_lowercase().contains("blocked on")));
    }

    #[test]
    fn rake_layer_does_not_double_capture() {
        // Sentence already caught by regex (port 8443) must not be
        // re-emitted by RAKE.
        let context = ctx(vec![_chunk_for_test(
            "user",
            "The auth service runs on port 8443 inside the mesh.",
        )]);
        let items = extract(&context);
        let rake_count = items
            .iter()
            .filter(|i| i.extraction_method == "rake")
            .count();
        let regex_count = items
            .iter()
            .filter(|i| i.extraction_method == "regex")
            .count();
        assert!(regex_count >= 1);
        assert_eq!(
            rake_count, 0,
            "RAKE must skip sentences already captured by an earlier layer"
        );
    }

    #[test]
    fn system_chunks_are_skipped() {
        let context = ctx(vec![
            _chunk_for_test("system", "You are an expert engineer running on port 9000."),
            _chunk_for_test("user", "Hi."),
        ]);
        let items = extract(&context);
        // The "port 9000" inside the system message must NOT produce an item.
        assert!(items.iter().all(|i| !i.content.contains("port 9000")));
    }

    #[test]
    fn empty_context_yields_no_items() {
        let items = extract(&Context::new(vec![]));
        assert!(items.is_empty());
    }
}
