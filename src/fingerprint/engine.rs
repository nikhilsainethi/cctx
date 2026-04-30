//! Top-level orchestrator: extract → dedup → (optional Tier-1 GLiNER) →
//! merge → score → rank → truncate.
//!
//! Producing a [`Fingerprint`] is a pure function of `(Context, FingerprintConfig)`:
//! no I/O, no mutation of the input. The Tier-1 path goes through a
//! [`crate::fingerprint::gliner_iface::GlinerEngine`] trait object so
//! the merge logic can be unit-tested with a mock and the real
//! `gline-rs` backend stays gated behind the `gliner` feature.
//!
//! All side effects (saving to `.cctx/fingerprints/`, hook stdout) live
//! in callers.

use crate::core::context::Context;
use crate::core::tokenizer::Tokenizer;

use super::dedup::merge_duplicates;
use super::extractor::extract;
use super::gliner_iface::{entities_to_items, select_interesting, GlinerEngine, ENTITY_LABELS};
use super::scorer::score;
use super::types::{Fingerprint, FingerprintConfig, FingerprintItem};

/// Build a fingerprint from a normalized [`Context`].
///
/// At Tier 0 (default): regex + keyword + RAKE → Jaccard merge → score → rank.
///
/// At Tier 1 with `--features gliner` and a downloaded model: Tier 0
/// runs first, then GLiNER runs on the pre-filtered "interesting"
/// chunks, and the two result sets are merged. Items present in both
/// tiers get a configurable priority boost (default 1.5×) when GLiNER
/// confidence is high.
///
/// At Tier 1 *without* the `gliner` feature, the engine falls back to
/// Tier 0 with a stderr warning. This keeps `cargo install cctx` from
/// the default-feature path always-functional even if a config asks
/// for Tier 1.
pub fn fingerprint(
    context: &Context,
    config: &FingerprintConfig,
    session_id: &str,
    created_at: &str,
) -> Fingerprint {
    let total_messages = context.chunk_count();
    let total_tokens = context.total_tokens;

    // ── Step 1+2: extract candidates and merge near-duplicates (Tier 0) ───
    let raw = extract(context);
    let mut items = merge_duplicates(raw);

    // ── Step 3: optional Tier-1 GLiNER pass ───────────────────────────────
    if config.tier >= 1 {
        items = run_tier1_or_warn(context, items, config);
    }

    // ── Step 4: count tokens per item ─────────────────────────────────────
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

    // ── Step 5: score ─────────────────────────────────────────────────────
    score(&mut items, total_messages, config);

    // ── Step 6: filter, sort, truncate ────────────────────────────────────
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

/// Tier-1 entry point with feature-gating and a soft fallback.
///
/// When `gliner` is enabled at compile time, this constructs the real
/// `gline-rs` backend and merges its output with the Tier-0 items. When
/// the feature is off, we log a one-line warning to stderr and return
/// the Tier-0 items unchanged so the caller still gets *something*
/// useful.
#[cfg(feature = "gliner")]
fn run_tier1_or_warn(
    context: &Context,
    tier0: Vec<FingerprintItem>,
    config: &FingerprintConfig,
) -> Vec<FingerprintItem> {
    match super::gliner::GlineRsEngine::from_default_dir() {
        Ok(engine) => run_tier1_with_engine(context, tier0, config, &engine),
        Err(e) => {
            eprintln!("[cctx] Tier 1 unavailable, falling back to Tier 0: {:#}", e);
            tier0
        }
    }
}

#[cfg(not(feature = "gliner"))]
fn run_tier1_or_warn(
    _context: &Context,
    tier0: Vec<FingerprintItem>,
    _config: &FingerprintConfig,
) -> Vec<FingerprintItem> {
    eprintln!(
        "[cctx] Tier 1 requested but `gliner` feature is not compiled — \
         rebuild with: cargo install cctx --features gliner"
    );
    tier0
}

/// The actual Tier-0/Tier-1 merge, factored out so it's reachable from
/// both the production path and unit tests with a mock engine.
///
/// `dead_code` is allowed because in a default (non-`gliner`, non-test)
/// build the production wrapper [`run_tier1_or_warn`] short-circuits
/// before this is reached, but the function still needs to exist so
/// the tests + the gliner-feature path compile.
#[allow(dead_code)]
pub(crate) fn run_tier1_with_engine<E: GlinerEngine + ?Sized>(
    context: &Context,
    tier0: Vec<FingerprintItem>,
    config: &FingerprintConfig,
    engine: &E,
) -> Vec<FingerprintItem> {
    // 1. Pre-filter chunks worth running through the model.
    let interesting = select_interesting(context, &tier0);
    if interesting.is_empty() {
        return tier0;
    }

    // 2. Run inference. On failure, log + degrade to Tier 0.
    let entities = match engine.predict(&interesting, ENTITY_LABELS, config.gliner_threshold) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "[cctx] GLiNER prediction failed, keeping Tier 0 only: {:#}",
                err
            );
            return tier0;
        }
    };

    // 3. Build candidate items from the entities.
    let gliner_items = entities_to_items(&entities, &context.chunks);
    if gliner_items.is_empty() {
        return tier0;
    }

    // 4. Merge: any item appearing in both layers gets the agreement
    //    boost; pure GLiNER items get added; pure Tier-0 items
    //    survive untouched. Implementation: concatenate then run the
    //    same Jaccard dedup that Tier 0 already uses, but with a
    //    post-merge agreement-boost pass.
    merge_with_agreement_boost(tier0, gliner_items, config)
}

/// Merge Tier-0 items with GLiNER items, boosting agreed-upon items.
///
/// Strategy:
/// 1. Tag GLiNER items as "from gliner" before merging so the dedup
///    survivor's `extraction_method` reflects who got there first
///    (preserving the existing dedup invariant).
/// 2. Run the same Jaccard merge as Tier 0 — this collapses
///    near-duplicate sentences across the two sources.
/// 3. After merging, scan the result and apply an *additional* boost
///    to any item whose `confidence` is `Some(_)` AND
///    `extraction_method` is one of the Tier-0 layers. That's the
///    "found by both" signal: the item came from Tier 0 but absorbed
///    a GLiNER entity carrying a confidence score.
#[allow(dead_code)] // see `run_tier1_with_engine` for why
fn merge_with_agreement_boost(
    tier0: Vec<FingerprintItem>,
    gliner: Vec<FingerprintItem>,
    config: &FingerprintConfig,
) -> Vec<FingerprintItem> {
    let mut combined = tier0;
    combined.extend(gliner);

    let merged = merge_duplicates(combined);

    let boost = config.gliner_agreement_boost.max(1.0);
    let mut out = merged;
    for item in out.iter_mut() {
        let from_tier0 = matches!(
            item.extraction_method.as_str(),
            "regex" | "keyword" | "rake"
        );
        let high_confidence = item.confidence.is_some_and(|c| c >= 0.6);
        if from_tier0 && high_confidence {
            // Encode the boost as a textrank_boost bump; the scorer
            // adds this term to the priority directly so the boost
            // survives the multiplicative formula.
            item.scores.textrank_boost = (item.scores.textrank_boost + boost - 1.0).clamp(0.0, 5.0);
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::extractor::_chunk_for_test;
    use super::super::gliner_iface::{GlinerEngine, GlinerEntity};
    use super::super::types::ItemCategory;
    use super::*;

    fn ctx(chunks: Vec<crate::core::context::Chunk>) -> Context {
        Context::new(chunks)
    }

    /// Hand-coded engine that emits canned entities when asked. Lets
    /// us unit-test the merge logic without an ONNX model.
    struct MockEngine {
        entities: Vec<GlinerEntity>,
    }

    impl GlinerEngine for MockEngine {
        fn predict(
            &self,
            _chunks: &[(usize, &crate::core::context::Chunk)],
            _labels: &[&str],
            _threshold: f64,
        ) -> anyhow::Result<Vec<GlinerEntity>> {
            Ok(self.entities.clone())
        }
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
        let budget = fp
            .items
            .iter()
            .find(|i| i.content.to_lowercase().contains("budget"))
            .expect("budget item should exist");
        assert!(budget.occurrence_count >= 2);
    }

    #[test]
    fn unique_constraint_outranks_repeated_debug_message() {
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

    // ── Tier-1 merge logic (mock-driven) ──────────────────────────────────

    #[test]
    fn tier1_adds_pure_gliner_items() {
        // Tier 0 finds nothing in this short transcript (sentences too
        // brief for keyword triggers). The mock GLiNER nonetheless
        // emits a high-confidence entity that should land in the
        // final fingerprint with extraction_method = "gliner".
        let context = ctx(vec![_chunk_for_test(
            "user",
            "I don't want any new managed services this fiscal year. \
             Keep us on the existing PostgreSQL cluster only.",
        )]);
        let config = FingerprintConfig {
            tier: 1,
            ..FingerprintConfig::default()
        };

        let entities = vec![GlinerEntity {
            source_chunk_idx: 0,
            label: "user constraint".into(),
            text: "no new managed services this fiscal year".into(),
            start: 0,
            end: 50,
            confidence: 0.85,
        }];
        let engine = MockEngine { entities };

        let tier0 = merge_duplicates(extract(&context));
        let merged = run_tier1_with_engine(&context, tier0, &config, &engine);
        assert!(
            merged
                .iter()
                .any(|i| i.extraction_method == "gliner" && i.confidence.is_some()),
            "GLiNER-only item must appear in merged set"
        );
    }

    #[test]
    fn tier1_boosts_agreed_items() {
        // Both Tier-0 (keyword on "should not") and the mock GLiNER
        // see the same sentence. After merge, the survivor should
        // carry a boost — its priority should outscore an
        // identical-but-non-GLiNER-confirmed item.
        let agreed = ctx(vec![_chunk_for_test(
            "user",
            "We should not exceed our forty-thousand-dollar engineering budget this year.",
        )]);
        let solo = ctx(vec![_chunk_for_test(
            "user",
            "We should not exceed our forty-thousand-dollar engineering budget this year.",
        )]);
        let config = FingerprintConfig {
            tier: 1,
            ..FingerprintConfig::default()
        };

        // Mock entity covering the same sentence.
        let entities = vec![GlinerEntity {
            source_chunk_idx: 0,
            label: "user constraint".into(),
            text: "should not exceed our forty-thousand-dollar engineering budget".into(),
            start: 3,
            end: 65,
            confidence: 0.85,
        }];
        let engine = MockEngine { entities };

        let tier0_a = merge_duplicates(extract(&agreed));
        let merged = run_tier1_with_engine(&agreed, tier0_a, &config, &engine);

        let tier0_b = merge_duplicates(extract(&solo));
        // No engine call → simulate Tier 0 only via direct return.
        let solo_only = tier0_b;

        let agreed_boost = merged
            .iter()
            .find(|i| i.content.to_lowercase().contains("budget"))
            .map(|i| i.scores.textrank_boost)
            .unwrap_or(0.0);
        let solo_boost = solo_only
            .iter()
            .find(|i| i.content.to_lowercase().contains("budget"))
            .map(|i| i.scores.textrank_boost)
            .unwrap_or(0.0);

        assert!(
            agreed_boost > solo_boost,
            "agreed item boost ({}) should exceed solo Tier-0 ({})",
            agreed_boost,
            solo_boost
        );
    }

    #[test]
    fn tier1_pre_filter_skips_short_chunks() {
        // GLiNER would happily extract "8443" from "port 8443" but
        // the pre-filter requires >= 10 words. Construct a chunk
        // with 5 words and a port; expect Tier-1 to NOT call the
        // engine on it. We verify via a mock that records calls.
        struct CountingEngine {
            calls: std::cell::RefCell<usize>,
        }
        impl GlinerEngine for CountingEngine {
            fn predict(
                &self,
                chunks: &[(usize, &crate::core::context::Chunk)],
                _labels: &[&str],
                _threshold: f64,
            ) -> anyhow::Result<Vec<GlinerEntity>> {
                *self.calls.borrow_mut() += chunks.len();
                Ok(vec![])
            }
        }

        let context = ctx(vec![
            _chunk_for_test("user", "Use port 8443."), // 3 words → skip
        ]);
        let config = FingerprintConfig {
            tier: 1,
            ..FingerprintConfig::default()
        };
        let engine = CountingEngine {
            calls: std::cell::RefCell::new(0),
        };
        let tier0 = merge_duplicates(extract(&context));
        let _ = run_tier1_with_engine(&context, tier0, &config, &engine);
        assert_eq!(*engine.calls.borrow(), 0, "short chunk must be filtered");
    }
}
