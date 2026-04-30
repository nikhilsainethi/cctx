//! Top-level orchestrator: extract → dedup → score → rank → truncate.
//!
//! Producing a [`Fingerprint`] is a pure function of `(Context, FingerprintConfig)`:
//! no I/O, no mutation of the input, fully deterministic. All side effects
//! (saving to `.cctx/fingerprints/`, hook stdout) live in callers.

use crate::core::context::Context;
use crate::core::tokenizer::Tokenizer;

use super::dedup::merge_duplicates;
use super::extractor::extract;
use super::scorer::score;
use super::types::{Fingerprint, FingerprintConfig};

/// Build a fingerprint from a normalized [`Context`].
///
/// Steps:
/// 1. Layered extraction (regex / keyword / RAKE) — see [`super::extractor::extract`].
/// 2. Cross-item Jaccard merge — see [`super::dedup::merge_duplicates`].
/// 3. Token counting on each surviving item using cl100k_base.
/// 4. Priority scoring — see [`super::scorer::score`].
/// 5. Filter by `min_priority_score`, sort descending, truncate to `max_items`.
///
/// `session_id` and `created_at` are passed in rather than auto-generated
/// so the caller controls correlation across pre/post-compaction artifacts
/// and tests can pin the timestamp for snapshot stability.
pub fn fingerprint(
    context: &Context,
    config: &FingerprintConfig,
    session_id: &str,
    created_at: &str,
) -> Fingerprint {
    let total_messages = context.chunk_count();
    let total_tokens = context.total_tokens;

    // ── Step 1+2: extract candidates and merge near-duplicates ────────────
    let raw = extract(context);
    let mut items = merge_duplicates(raw);

    // ── Step 3: count tokens per item ─────────────────────────────────────
    if !items.is_empty() {
        // Best-effort tokenizer — failure here is rare (only the embedded
        // BPE table being broken). Fall back to char/4 estimate so we
        // still produce a fingerprint instead of erroring out.
        let tokenizer = Tokenizer::new().ok();
        for item in items.iter_mut() {
            item.tokens = match &tokenizer {
                Some(t) => t.count(&item.content),
                None => item.content.len().div_ceil(4),
            };
        }
    }

    // ── Step 4: score ─────────────────────────────────────────────────────
    score(&mut items, total_messages, config);

    // ── Step 5: filter, sort, truncate ────────────────────────────────────
    items.retain(|i| i.priority_score >= config.min_priority_score);
    items.sort_by(|a, b| {
        b.priority_score
            .partial_cmp(&a.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if items.len() > config.max_items {
        items.truncate(config.max_items);
    }

    Fingerprint {
        session_id: session_id.to_string(),
        created_at: created_at.to_string(),
        total_tokens,
        total_items: items.len(),
        items,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::extractor::_chunk_for_test;
    use super::super::types::ItemCategory;
    use super::*;

    fn ctx(chunks: Vec<crate::core::context::Chunk>) -> Context {
        Context::new(chunks)
    }

    #[test]
    fn empty_transcript_produces_empty_fingerprint() {
        let fp = fingerprint(
            &Context::new(vec![]),
            &FingerprintConfig::default(),
            "sess_x",
            "2026-04-30T00:00:00Z",
        );
        assert_eq!(fp.total_items, 0);
        assert!(fp.items.is_empty());
    }

    #[test]
    fn only_system_messages_produces_empty_fingerprint() {
        let context = ctx(vec![
            _chunk_for_test("system", "You are an expert. Run on port 9000."),
            _chunk_for_test("system", "Working dir is /etc/myapp."),
        ]);
        let fp = fingerprint(
            &context,
            &FingerprintConfig::default(),
            "sess_x",
            "2026-04-30T00:00:00Z",
        );
        assert_eq!(fp.total_items, 0);
    }

    #[test]
    fn fingerprint_is_sorted_by_priority_descending() {
        let context = ctx(vec![
            _chunk_for_test("user", "Budget should not exceed $50K total."),
            _chunk_for_test("user", "We decided to use PostgreSQL because of ACID."),
            _chunk_for_test("user", "The auth service runs on port 8443."),
            _chunk_for_test("user", "The CORS error was caused by a missing header."),
        ]);
        let fp = fingerprint(
            &context,
            &FingerprintConfig::default(),
            "sess_x",
            "2026-04-30T00:00:00Z",
        );
        assert!(fp.total_items >= 1);
        for w in fp.items.windows(2) {
            assert!(
                w[0].priority_score >= w[1].priority_score,
                "items must be sorted by priority desc"
            );
        }
    }

    #[test]
    fn deduplication_merges_repeated_facts() {
        let context = ctx(vec![
            _chunk_for_test("user", "Budget should not exceed $50K. That's the cap."),
            _chunk_for_test("assistant", "Acknowledged."),
            _chunk_for_test(
                "user",
                "Reminder: budget should not exceed $50K. Hard limit.",
            ),
            _chunk_for_test("assistant", "Got it."),
            _chunk_for_test(
                "user",
                "And again: budget should not exceed $50K. Please respect this.",
            ),
        ]);
        let fp = fingerprint(
            &context,
            &FingerprintConfig::default(),
            "sess_x",
            "2026-04-30T00:00:00Z",
        );
        // The "budget should not exceed $50K" fact mentioned thrice should
        // collapse to one item with occurrence_count >= 2 (Jaccard at 0.7
        // is permissive enough for this paraphrase set).
        let budget = fp
            .items
            .iter()
            .find(|i| i.content.to_lowercase().contains("budget"))
            .expect("budget item should exist");
        assert!(budget.occurrence_count >= 2);
    }

    #[test]
    fn unique_constraint_outranks_repeated_debug_message() {
        // Sanity check on the priority formula: a unique high-impact
        // constraint should land above a debug message that's been repeated
        // four times.
        let context = ctx(vec![
            _chunk_for_test(
                "user",
                "Budget shouldn't exceed forty thousand for the year.",
            ),
            _chunk_for_test("user", "the issue was a missing CORS header."),
            _chunk_for_test("user", "the issue was a missing CORS header again."),
            _chunk_for_test("user", "the issue was a missing CORS header still."),
            _chunk_for_test("user", "the issue was a missing CORS header once more."),
        ]);
        let fp = fingerprint(
            &context,
            &FingerprintConfig::default(),
            "sess_x",
            "2026-04-30T00:00:00Z",
        );
        let budget_score = fp
            .items
            .iter()
            .find(|i| i.category == ItemCategory::Constraint)
            .map(|i| i.priority_score);
        let cors_score = fp
            .items
            .iter()
            .find(|i| {
                i.content.to_lowercase().contains("cors")
                    && i.category == ItemCategory::DebugInsight
            })
            .map(|i| i.priority_score);
        if let (Some(b), Some(c)) = (budget_score, cors_score) {
            assert!(b > c, "unique constraint ({}) > repeated debug ({})", b, c);
        } else {
            // If categorization shifted, at least confirm the budget item
            // ended up with a higher score than any CORS item.
            let budget_max = fp
                .items
                .iter()
                .filter(|i| i.content.to_lowercase().contains("budget"))
                .map(|i| i.priority_score)
                .fold(0.0_f64, f64::max);
            let cors_max = fp
                .items
                .iter()
                .filter(|i| i.content.to_lowercase().contains("cors"))
                .map(|i| i.priority_score)
                .fold(0.0_f64, f64::max);
            assert!(budget_max > cors_max);
        }
    }
}
