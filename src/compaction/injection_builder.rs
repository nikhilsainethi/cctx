//! Build the recovery payload that gets emitted to stdout on
//! `SessionStart` after compaction.
//!
//! The format is designed to be scannable by both the LLM (so it can
//! act on the recovered facts) and the human (so they can sanity-
//! check what was recovered). See architecture §3.5.2 for the exact
//! shape.

use crate::core::tokenizer::Tokenizer;
use crate::fingerprint::{FingerprintItem, ItemCategory};

use super::budget::BudgetManager;
use super::ClassifiedItem;

/// Default token budget for re-injection — 4096 ≈ 2 % of a 200K window.
pub const DEFAULT_INJECTION_BUDGET: usize = 4096;

/// Confidence at-or-above which a GLiNER-extracted item is tagged
/// `[HIGH CONFIDENCE]` in the rendered payload.
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 0.8;

// ── Public payload type ───────────────────────────────────────────────────────

/// Result of [`build_injection`]. The text in `payload` is what the
/// SessionStart hook prints to stdout; the metadata fields are useful
/// for the compaction-log entry and the on-screen summary.
#[derive(Debug, Clone)]
pub struct InjectionPayload {
    /// The full formatted text — empty string when there are no lost
    /// items (so the SessionStart hook can simply skip emitting).
    pub payload: String,
    /// Number of items included in `payload`.
    pub item_count: usize,
    /// Approximate token count of `payload` (cl100k_base).
    pub token_count: usize,
}

impl InjectionPayload {
    /// True when there's nothing to inject — a clean exit signal for
    /// the SessionStart handler.
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty() || self.item_count == 0
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Build the SessionStart recovery payload from a list of classified
/// items.
///
/// Steps:
///
/// 1. Filter to `Lost` items (the rest are by definition still present).
/// 2. Sort by descending priority (the input is already in fingerprint
///    priority order so this is essentially a stable confirmation).
/// 3. Greedily allocate against `budget`. If the highest-priority item
///    alone exceeds `budget`, truncate its content to fit.
/// 4. Format each selected item with its category + provenance and
///    wrap them in a structured header/footer block.
pub fn build_injection(items: &[ClassifiedItem], budget: usize) -> InjectionPayload {
    use super::LossClassification;

    if items.is_empty() {
        return InjectionPayload {
            payload: String::new(),
            item_count: 0,
            token_count: 0,
        };
    }

    let mut lost: Vec<&ClassifiedItem> = items
        .iter()
        .filter(|c| c.classification == LossClassification::Lost)
        .collect();
    if lost.is_empty() {
        return InjectionPayload {
            payload: String::new(),
            item_count: 0,
            token_count: 0,
        };
    }

    // Stable sort by descending priority. Equal priorities preserve
    // their input order — the input is already priority-sorted by the
    // fingerprint engine, so this is a defensive guarantee.
    lost.sort_by(|a, b| {
        b.fingerprint_item
            .priority_score
            .partial_cmp(&a.fingerprint_item.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let tokenizer = Tokenizer::new().ok();
    let mut budgeter = BudgetManager::new(budget);
    let mut formatted_items: Vec<String> = Vec::new();

    for (idx, item) in lost.iter().enumerate() {
        let line = format_item(&item.fingerprint_item);
        let cost = approx_tokens(&line, tokenizer.as_ref());

        if budgeter.allocate(cost).is_ok() {
            formatted_items.push(line);
            continue;
        }

        // Cannot fit this whole item. Two cases:
        //
        // (a) Nothing has been included yet AND the item is bigger
        //     than the entire budget — truncate this top-priority
        //     item to fit and stop. Per the architecture, we'd rather
        //     surface a partial top-priority constraint than nothing.
        // (b) We've already filled the budget with smaller items —
        //     stop here; subsequent items only have lower priority,
        //     so it's not worth continuing.
        if idx == 0 && formatted_items.is_empty() {
            let remaining = budgeter.remaining().max(50);
            let truncated = truncate_to_token_budget(&line, remaining, tokenizer.as_ref());
            let truncated_cost = approx_tokens(&truncated, tokenizer.as_ref());
            // Force-allocate even if it slightly overshoots — we
            // already proved we can't fit; this is the explicit
            // truncate-to-fit path the brief describes.
            let _ = budgeter.allocate(truncated_cost.min(budgeter.remaining()));
            formatted_items.push(truncated);
        }
        break;
    }

    if formatted_items.is_empty() {
        return InjectionPayload {
            payload: String::new(),
            item_count: 0,
            token_count: 0,
        };
    }

    let body = formatted_items.join("\n\n");
    let header = format!(
        "[cctx] Context items recovered from pre-compaction analysis ({} items, {} tokens):",
        formatted_items.len(),
        budgeter.allocated()
    );
    let footer = "These items were verified present before compaction and absent after. \
         Treat as authoritative context.";
    let payload = format!("{}\n\n{}\n\n{}\n", header, body, footer);
    let token_count = approx_tokens(&payload, tokenizer.as_ref());

    InjectionPayload {
        payload,
        item_count: formatted_items.len(),
        token_count,
    }
}

// ── Formatting helpers ────────────────────────────────────────────────────────

/// Render one fingerprint item as its multi-line block:
///
/// ```text
/// [HIGH CONFIDENCE] CONSTRAINT: Budget should not exceed $50K total.
/// (Originally in messages [3, 11], not found in compaction summary)
/// ```
///
/// The `[HIGH CONFIDENCE]` prefix only appears for GLiNER-attributed
/// items with `confidence >= 0.8`. Items extracted by Tier 0 or
/// GLiNER items with `0.4 ≤ confidence < 0.8` get no prefix.
fn format_item(item: &FingerprintItem) -> String {
    let category_label = category_label(&item.category);
    let high_conf = item
        .confidence
        .is_some_and(|c| c >= HIGH_CONFIDENCE_THRESHOLD);
    let prefix = if high_conf { "[HIGH CONFIDENCE] " } else { "" };

    let positions = format_positions(&item.source_positions);
    format!(
        "{}{}: {}\n(Originally in {}, not found in compaction summary)",
        prefix,
        category_label,
        item.content.trim(),
        positions
    )
}

fn category_label(category: &ItemCategory) -> &'static str {
    match category {
        ItemCategory::Constraint => "CONSTRAINT",
        ItemCategory::Decision => "DECISION",
        ItemCategory::TechnicalFact => "TECHNICAL",
        ItemCategory::DebugInsight => "DEBUG",
        ItemCategory::ProgressMarker => "PROGRESS",
        ItemCategory::Other => "ITEM",
    }
}

fn format_positions(positions: &[usize]) -> String {
    match positions.len() {
        0 => "the conversation".to_string(),
        1 => format!("message {}", positions[0]),
        _ => {
            let inner = positions
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("messages [{}]", inner)
        }
    }
}

/// Best-effort token count using the BPE tokenizer when available;
/// falls back to char/4 estimate if the tokenizer failed to init.
fn approx_tokens(text: &str, tokenizer: Option<&Tokenizer>) -> usize {
    match tokenizer {
        Some(t) => t.count(text),
        None => text.chars().count().div_ceil(4),
    }
}

/// Truncate `text` to roughly `max_tokens` tokens, appending an
/// ellipsis marker so readers know it was clipped. Used only on the
/// special "single oversized top-priority item" path.
fn truncate_to_token_budget(
    text: &str,
    max_tokens: usize,
    tokenizer: Option<&Tokenizer>,
) -> String {
    let current = approx_tokens(text, tokenizer);
    if current <= max_tokens {
        return text.to_string();
    }
    // 4 chars/token is a coarse-but-stable approximation. Refine by
    // checking the actual tokenizer in a single pass after the cut.
    let target_chars = (max_tokens.saturating_sub(8) * 4).max(40);
    let mut buf: String = text.chars().take(target_chars).collect();
    while approx_tokens(&buf, tokenizer) > max_tokens && !buf.is_empty() {
        buf.pop();
    }
    buf.push_str(" … [truncated]");
    buf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::LossClassification;
    use crate::fingerprint::{make_id, ItemScores};

    fn classified(
        category: ItemCategory,
        content: &str,
        priority: f64,
        positions: Vec<usize>,
        confidence: Option<f64>,
        cls: LossClassification,
    ) -> ClassifiedItem {
        ClassifiedItem {
            fingerprint_item: FingerprintItem {
                id: make_id(content),
                category,
                content: content.into(),
                tokens: 0,
                occurrence_count: 1,
                source_positions: positions,
                priority_score: priority,
                scores: ItemScores {
                    uniqueness: 1.0,
                    recency: 1.0,
                    position_risk: 0.0,
                    textrank_boost: 0.0,
                },
                extraction_method: "test".into(),
                confidence,
            },
            classification: cls,
            overlap_score: 0.1,
        }
    }

    #[test]
    fn empty_lost_list_yields_empty_payload() {
        let payload = build_injection(&[], 4096);
        assert!(payload.is_empty());
    }

    #[test]
    fn only_preserved_items_yields_empty_payload() {
        let items = vec![classified(
            ItemCategory::Constraint,
            "some preserved item",
            0.5,
            vec![0],
            None,
            LossClassification::Preserved,
        )];
        let payload = build_injection(&items, 4096);
        assert!(payload.is_empty(), "no Lost items → no injection");
    }

    #[test]
    fn injection_respects_budget() {
        // Three items, each ~50 tokens. Budget 100 → first two land,
        // third must NOT.
        let mk = |i: usize, prio: f64| {
            classified(
                ItemCategory::Constraint,
                &format!(
                    "item {} with enough words to be roughly fifty tokens worth of content for testing alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma",
                    i
                ),
                prio,
                vec![i],
                None,
                LossClassification::Lost,
            )
        };
        let items = vec![mk(1, 0.9), mk(2, 0.8), mk(3, 0.7)];
        let payload = build_injection(&items, 100);
        assert!(
            payload.token_count <= 110,
            "budget overshoot too large: {}",
            payload.token_count
        );
    }

    #[test]
    fn injection_orders_by_priority_descending() {
        let mk = |label: &str, prio: f64| {
            classified(
                ItemCategory::Constraint,
                label,
                prio,
                vec![0],
                None,
                LossClassification::Lost,
            )
        };
        let items = vec![
            mk("LOWPRI item content here at low priority", 0.3),
            mk("HIGHPRI item content here at high priority", 0.9),
            mk("MIDPRI item content here at mid priority", 0.6),
        ];
        let payload = build_injection(&items, 4096);
        let high = payload.payload.find("HIGHPRI").unwrap();
        let mid = payload.payload.find("MIDPRI").unwrap();
        let low = payload.payload.find("LOWPRI").unwrap();
        assert!(high < mid, "HIGHPRI should appear before MIDPRI");
        assert!(mid < low, "MIDPRI should appear before LOWPRI");
    }

    #[test]
    fn high_confidence_gliner_items_get_marker() {
        let items = vec![
            classified(
                ItemCategory::Constraint,
                "high-confidence GLiNER constraint item content",
                0.9,
                vec![5],
                Some(0.9),
                LossClassification::Lost,
            ),
            classified(
                ItemCategory::Decision,
                "medium-confidence GLiNER decision item content",
                0.8,
                vec![6],
                Some(0.5),
                LossClassification::Lost,
            ),
            classified(
                ItemCategory::TechnicalFact,
                "tier-zero technical fact item content",
                0.7,
                vec![7],
                None,
                LossClassification::Lost,
            ),
        ];
        let payload = build_injection(&items, 4096);
        assert!(
            payload.payload.contains("[HIGH CONFIDENCE] CONSTRAINT"),
            "0.9 confidence GLiNER item should carry [HIGH CONFIDENCE]"
        );
        assert!(
            !payload.payload.contains("[HIGH CONFIDENCE] DECISION"),
            "0.5 confidence should NOT carry [HIGH CONFIDENCE]"
        );
        assert!(
            !payload.payload.contains("[HIGH CONFIDENCE] TECHNICAL"),
            "non-GLiNER (None confidence) should NOT carry [HIGH CONFIDENCE]"
        );
    }

    #[test]
    fn oversized_top_priority_gets_truncated() {
        // Item is way over budget — should be truncated and surface
        // alone rather than being skipped.
        let huge_content: String = "alpha beta gamma delta epsilon ".repeat(400);
        let items = vec![classified(
            ItemCategory::Constraint,
            &huge_content,
            1.0,
            vec![1],
            None,
            LossClassification::Lost,
        )];
        let payload = build_injection(&items, 100);
        assert!(
            payload.item_count == 1,
            "truncated item should still appear"
        );
        assert!(
            payload.payload.contains("[truncated]"),
            "should mark truncation"
        );
    }

    #[test]
    fn injection_format_includes_header_and_footer() {
        let items = vec![classified(
            ItemCategory::Constraint,
            "the budget should not exceed fifty thousand dollars total",
            0.9,
            vec![3],
            None,
            LossClassification::Lost,
        )];
        let payload = build_injection(&items, 4096);
        assert!(payload
            .payload
            .starts_with("[cctx] Context items recovered"));
        assert!(payload.payload.contains("Treat as authoritative context"));
        assert!(payload.payload.contains("CONSTRAINT:"));
        assert!(payload.payload.contains("Originally in message 3"));
    }

    #[test]
    fn category_labels_render_correctly() {
        for (cat, expected) in [
            (ItemCategory::Constraint, "CONSTRAINT"),
            (ItemCategory::Decision, "DECISION"),
            (ItemCategory::TechnicalFact, "TECHNICAL"),
            (ItemCategory::DebugInsight, "DEBUG"),
            (ItemCategory::ProgressMarker, "PROGRESS"),
            (ItemCategory::Other, "ITEM"),
        ] {
            assert_eq!(category_label(&cat), expected);
        }
    }
}
