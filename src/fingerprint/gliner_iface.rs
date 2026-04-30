//! Provider-agnostic interface for the Tier-1 GLiNER pass.
//!
//! Defining [`GlinerEngine`] as a trait lets us:
//!
//! 1. Compile the merge logic in [`super::engine`] regardless of feature
//!    flags — the real `gline-rs`-backed engine and the test `MockEngine`
//!    both implement the same trait.
//! 2. Keep the `gline-rs` dep gated behind the `gliner` feature so a
//!    default `cargo install cctx` doesn't pull a 188 MB ONNX runtime.
//! 3. Unit-test the merge / pre-filter logic without a real ONNX model.

use crate::core::context::{Chunk, Context};

use super::types::ItemCategory;

// ── Public types ──────────────────────────────────────────────────────────────

/// Custom GLiNER entity labels used by the Tier-1 pass.
///
/// Order matters: the merge logic prefers earlier labels when an entity
/// overlaps multiple categories. Constraints + decisions outrank
/// technical specs because they encode user intent rather than passing
/// detail.
pub const ENTITY_LABELS: &[&str] = &[
    "user constraint",
    "architecture decision",
    "technical specification",
    "debug conclusion",
    "action item",
    "user preference",
];

/// Map a GLiNER label string to the cctx [`ItemCategory`]. Falls back
/// to [`ItemCategory::Other`] for unknown labels — keeps us forward-
/// compatible with future label sets.
pub fn label_to_category(label: &str) -> ItemCategory {
    match label {
        "user constraint" | "user preference" => ItemCategory::Constraint,
        "architecture decision" => ItemCategory::Decision,
        "technical specification" => ItemCategory::TechnicalFact,
        "debug conclusion" => ItemCategory::DebugInsight,
        "action item" => ItemCategory::ProgressMarker,
        _ => ItemCategory::Other,
    }
}

/// One detected entity span. Mirrors `gline_rs::text::span::Span` but
/// without leaking the upstream type into trait signatures.
#[derive(Debug, Clone)]
pub struct GlinerEntity {
    /// Index of the source chunk (line in the original transcript).
    pub source_chunk_idx: usize,
    /// GLiNER label (one of [`ENTITY_LABELS`] in production runs).
    pub label: String,
    /// The detected span text.
    pub text: String,
    /// Start byte offset in the source text.
    pub start: usize,
    /// End byte offset (exclusive) in the source text.
    pub end: usize,
    /// Probability `[0.0, 1.0]` from the model's sigmoid head.
    pub confidence: f64,
}

/// The runtime contract for any GLiNER backend.
///
/// `predict` takes the pre-filtered chunks and returns one
/// [`GlinerEntity`] per detected span, tagged with the chunk index it
/// came from. Implementations are expected to honor the trait's
/// `threshold` parameter — `extract_with_engine` filters again
/// defensively but won't recover a span the model already dropped.
pub trait GlinerEngine {
    /// Run the model on `chunks` (already pre-filtered for relevance).
    /// `chunks` is `&[(usize, &Chunk)]` so callers preserve the
    /// original chunk indices through the gather step.
    ///
    /// # Errors
    ///
    /// Implementations may surface model-load, tensor, or tokenizer
    /// errors. The real backend additionally returns `Err` if the
    /// downloaded ONNX files are missing or unreadable.
    fn predict(
        &self,
        chunks: &[(usize, &Chunk)],
        labels: &[&str],
        threshold: f64,
    ) -> anyhow::Result<Vec<GlinerEntity>>;
}

// ── Pre-filter ────────────────────────────────────────────────────────────────

/// Decide which chunks are worth running through GLiNER.
///
/// A chunk is "interesting" when **any** of these holds:
///
/// - Tier-0 already extracted at least one item from it (the chunk
///   landed regex / keyword / RAKE signal already).
/// - The chunk is from the user (`role == "user"`) — user statements
///   are the primary source of constraints / decisions / requirements.
/// - The chunk is a tool interaction (`role == "tool_interaction"`) —
///   tool outputs frequently carry technical specs (paths, ports, IPs).
///
/// Chunks shorter than 10 words are always skipped, regardless of the
/// rules above — too short to give the NER head useful context.
pub fn select_interesting<'a>(
    context: &'a Context,
    tier0_items: &[super::types::FingerprintItem],
) -> Vec<(usize, &'a Chunk)> {
    use std::collections::HashSet;
    let tier0_positions: HashSet<usize> = tier0_items
        .iter()
        .flat_map(|i| i.source_positions.iter().copied())
        .collect();

    context
        .chunks
        .iter()
        .enumerate()
        .filter(|(idx, chunk)| {
            // Word count gate — too-short messages can't host an entity.
            if chunk.content.split_whitespace().count() < 10 {
                return false;
            }
            tier0_positions.contains(idx)
                || chunk.role == "user"
                || chunk.role == "tool_interaction"
        })
        .collect()
}

// ── Item construction ─────────────────────────────────────────────────────────

/// Convert a list of [`GlinerEntity`]s into [`super::types::FingerprintItem`]s,
/// expanding each span to its containing sentence (±1 sentence of
/// readability context) so the resulting items are useful in re-injection.
pub fn entities_to_items(
    entities: &[GlinerEntity],
    chunks: &[crate::core::context::Chunk],
) -> Vec<super::types::FingerprintItem> {
    use super::types::{make_id, FingerprintItem, ItemScores};

    let mut items: Vec<FingerprintItem> = Vec::new();

    for entity in entities {
        let chunk = match chunks.get(entity.source_chunk_idx) {
            Some(c) => c,
            None => continue,
        };
        let content = expand_to_sentence(&chunk.content, entity.start, entity.end);
        if content.is_empty() {
            continue;
        }
        items.push(FingerprintItem {
            id: make_id(&content),
            category: label_to_category(&entity.label),
            content,
            tokens: 0,
            occurrence_count: 1,
            source_positions: vec![entity.source_chunk_idx],
            priority_score: 0.0,
            scores: ItemScores {
                uniqueness: 1.0,
                recency: 1.0,
                position_risk: 0.0,
                textrank_boost: 0.0,
            },
            extraction_method: "gliner".to_string(),
            confidence: Some(entity.confidence),
        });
    }

    items
}

/// Expand a span at byte offsets `[start, end)` into the full sentence
/// it sits inside, plus one sentence of context on either side.
///
/// Falls back to the entire chunk content when sentence boundaries
/// can't be found within reasonable distance — better to over-quote than
/// to crop a constraint mid-clause.
fn expand_to_sentence(text: &str, start: usize, end: usize) -> String {
    if text.is_empty() || start >= text.len() {
        return String::new();
    }
    let end = end.min(text.len());

    // Find the start of the containing sentence: walk left to a `.`/`!`/
    // `?`/`\n` boundary, then back up one more to grab a sentence of
    // leading context.
    let left_anchor = sentence_boundary_left(text, start, 2);
    let right_anchor = sentence_boundary_right(text, end, 2);

    let slice = &text[left_anchor..right_anchor];
    slice.trim().to_string()
}

fn sentence_boundary_left(text: &str, pos: usize, hops: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.min(bytes.len());
    let mut hops_left = hops;

    while i > 0 {
        i -= 1;
        if matches!(bytes[i], b'.' | b'!' | b'?' | b'\n') {
            if hops_left == 0 {
                return (i + 1).min(bytes.len());
            }
            hops_left -= 1;
        }
    }
    0
}

fn sentence_boundary_right(text: &str, pos: usize, hops: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.min(bytes.len());
    let mut hops_left = hops;

    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?' | b'\n') {
            if hops_left == 0 {
                return (i + 1).min(bytes.len());
            }
            hops_left -= 1;
        }
        i += 1;
    }
    bytes.len()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::AttentionZone;

    fn chunk(role: &str, content: &str) -> Chunk {
        Chunk {
            index: 0,
            role: role.into(),
            content: content.into(),
            token_count: 0,
            relevance_score: 0.5,
            attention_zone: AttentionZone::Strong,
        }
    }

    #[test]
    fn label_to_category_maps_known_labels() {
        assert_eq!(
            label_to_category("user constraint"),
            ItemCategory::Constraint
        );
        assert_eq!(
            label_to_category("user preference"),
            ItemCategory::Constraint
        );
        assert_eq!(
            label_to_category("architecture decision"),
            ItemCategory::Decision
        );
        assert_eq!(
            label_to_category("technical specification"),
            ItemCategory::TechnicalFact
        );
        assert_eq!(
            label_to_category("debug conclusion"),
            ItemCategory::DebugInsight
        );
        assert_eq!(
            label_to_category("action item"),
            ItemCategory::ProgressMarker
        );
        assert_eq!(label_to_category("unknown"), ItemCategory::Other);
    }

    #[test]
    fn pre_filter_keeps_user_chunks_skips_short_ones() {
        let ctx = Context::new(vec![
            chunk("system", "You are a helpful assistant for software design."),
            chunk("user", "Hi"), // too short
            chunk(
                "user",
                "I want the budget capped at fifty thousand for the project.",
            ),
            chunk(
                "assistant",
                "Sure, that's a reasonable target for an internal tool.",
            ),
            chunk(
                "tool_interaction",
                "[Tool: Bash] Input: {} Output: file_a.txt file_b.txt and other files here",
            ),
        ]);
        let interesting = select_interesting(&ctx, &[]);
        let kept_indices: Vec<usize> = interesting.iter().map(|(i, _)| *i).collect();
        assert!(kept_indices.contains(&2), "user message should be kept");
        assert!(kept_indices.contains(&4), "tool_interaction should be kept");
        assert!(!kept_indices.contains(&0), "system should be skipped");
        assert!(
            !kept_indices.contains(&1),
            "short user msg should be skipped"
        );
        assert!(
            !kept_indices.contains(&3),
            "assistant chunk without tier-0 hit should be skipped"
        );
    }

    #[test]
    fn pre_filter_includes_assistant_chunks_with_tier0_hits() {
        use super::super::types::{make_id, FingerprintItem, ItemScores};

        let ctx = Context::new(vec![
            chunk(
                "user",
                "Hi, what stack should I pick for the API tier we discussed?",
            ),
            chunk(
                "assistant",
                "Run the auth service on port 8443, with config at /etc/auth.yml.",
            ),
        ]);
        // Tier-0 already hit chunk 1 via the regex layer.
        let tier0 = vec![FingerprintItem {
            id: make_id("port 8443"),
            category: ItemCategory::TechnicalFact,
            content: "port 8443".into(),
            tokens: 1,
            occurrence_count: 1,
            source_positions: vec![1],
            priority_score: 0.5,
            scores: ItemScores {
                uniqueness: 1.0,
                recency: 1.0,
                position_risk: 0.0,
                textrank_boost: 0.0,
            },
            extraction_method: "regex".into(),
            confidence: None,
        }];
        let interesting = select_interesting(&ctx, &tier0);
        let kept: Vec<usize> = interesting.iter().map(|(i, _)| *i).collect();
        assert!(kept.contains(&1), "assistant chunk with tier-0 hit kept");
    }

    #[test]
    fn expand_to_sentence_grabs_context_around_span() {
        let text = "We're scoping. The auth service runs on port 8443 in the mesh. Anything else?";
        // Span over "port 8443"
        let start = text.find("port 8443").unwrap();
        let end = start + "port 8443".len();
        let out = expand_to_sentence(text, start, end);
        assert!(out.contains("port 8443"));
        // Should include surrounding context.
        assert!(out.contains("auth service"));
    }
}
