//! IDF-weighted token-overlap loss detection.
//!
//! Naive Jaccard / token-overlap classifies an item as PRESERVED when
//! the summary echoes its filler words ("the", "is", "to") even after
//! the *important* tokens — the rationale, the constraint value, the
//! port number — have been dropped. We can't afford that false negative
//! on the loss side: missing a lost item means the model never gets
//! it back via re-injection.
//!
//! IDF weighting fixes this. Tokens that are RARE in the source
//! conversation carry MORE weight in the overlap score. So when a
//! summary preserves "PostgreSQL" (frequent) but drops "ACID
//! compliance" (mentioned once), the overlap drops below the
//! Paraphrased threshold and we correctly flag the item as Lost.

use std::collections::{HashMap, HashSet};

use crate::core::context::Context;
use crate::fingerprint::FingerprintItem;

use super::{ClassifiedItem, LossClassification, LossReport};

// ── Thresholds ────────────────────────────────────────────────────────────────

/// Below this overlap → `Lost`.
pub const DEFAULT_LOST_THRESHOLD: f64 = 0.3;
/// At-or-above this overlap → `Preserved`. Between the two → `Paraphrased`.
pub const DEFAULT_PRESERVED_THRESHOLD: f64 = 0.7;

// ── Tokenization ──────────────────────────────────────────────────────────────

/// Lowercase + strip leading / trailing non-alphanumerics, keep internal
/// punctuation (so `8443` and `auth-service` survive). Tokens of length
/// `< 2` are dropped — single chars are noise.
fn tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Same shape as [`tokens`] but emitting a set rather than a `Vec`.
/// Used for membership tests where duplicate counts don't matter.
fn token_set(text: &str) -> HashSet<String> {
    tokens(text).into_iter().collect()
}

// ── IDF weighting ─────────────────────────────────────────────────────────────

/// Compute inverse document frequency over the chunks of `context`.
///
/// `idf(t) = ln(total_messages / messages_containing_t)`. A token
/// appearing in every message gets `0.0` (zero discriminative power);
/// a token appearing in one message gets `ln(N)`. The map omits
/// tokens that never appear, so callers should `unwrap_or(1.0)` for
/// summary-only tokens.
///
/// # Examples
///
/// ```
/// use cctx::compaction::loss_detector::compute_idf;
/// use cctx::core::context::{Chunk, Context, AttentionZone};
///
/// let mk = |s: &str| Chunk {
///     index: 0, role: "user".into(), content: s.into(),
///     token_count: 0, relevance_score: 0.5,
///     attention_zone: AttentionZone::Strong,
/// };
/// let ctx = Context::new(vec![
///     mk("budget is fifty thousand"),
///     mk("budget should not exceed fifty"),
///     mk("port is 8443"),
/// ]);
/// let idf = compute_idf(&ctx);
/// // "budget" appears in 2 of 3 messages: ln(3/2) ≈ 0.405
/// assert!(*idf.get("budget").unwrap() < 0.5);
/// // "8443" appears in 1 of 3 messages: ln(3/1) ≈ 1.099
/// assert!(*idf.get("8443").unwrap() > 1.0);
/// ```
pub fn compute_idf(context: &Context) -> HashMap<String, f64> {
    let total = context.chunk_count();
    if total == 0 {
        return HashMap::new();
    }

    // For each chunk, compute its token SET (dedup within chunk so a
    // token mentioned 3× in one message still only contributes 1 to df).
    let mut df: HashMap<String, usize> = HashMap::new();
    for chunk in &context.chunks {
        for tok in token_set(&chunk.content) {
            *df.entry(tok).or_insert(0) += 1;
        }
    }

    let total_f = total as f64;
    df.into_iter()
        .map(|(tok, doc_freq)| (tok, (total_f / doc_freq as f64).ln()))
        .collect()
}

// ── Per-item detection ────────────────────────────────────────────────────────

/// IDF-weighted overlap of `item.content` against `summary`.
///
/// Algorithm (matches §3.4 of the architecture, with IDF weighting on top):
///
/// 1. Tokenize the item content into a unique-token set `A`.
/// 2. Tokenize the summary into `B`.
/// 3. For each `t ∈ A`: lookup `weight = idf(t)` (default `1.0`).
///    Accumulate `total_weight += weight`. If `t ∈ B`,
///    `overlap += weight`.
/// 4. Return `overlap / total_weight`. Empty `A` ⇒ `0.0`.
///
/// All-zero IDF (token shows up in every chunk, weight=0) on every
/// item-token would zero out total_weight; treat that case as "no
/// signal worth comparing" and return `0.0` (Lost).
pub fn weighted_overlap(item_content: &str, summary: &str, idf: &HashMap<String, f64>) -> f64 {
    let item_tokens = token_set(item_content);
    if item_tokens.is_empty() {
        return 0.0;
    }
    let summary_tokens = token_set(summary);

    let mut overlap = 0.0_f64;
    let mut total = 0.0_f64;
    for tok in &item_tokens {
        // Floor the weight at a small positive value so an item entirely
        // composed of "everywhere" tokens doesn't divide by zero — we
        // still want SOME signal from "did the surface form survive".
        let weight = idf.get(tok).copied().unwrap_or(1.0).max(0.05);
        total += weight;
        if summary_tokens.contains(tok) {
            overlap += weight;
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    overlap / total
}

/// Naive (unweighted) Jaccard-style overlap. Kept around for the
/// "weighted vs naive" comparison test in [`tests`] and as a reference
/// implementation in commit history. Not used in production.
#[allow(dead_code)]
pub(crate) fn naive_overlap(item_content: &str, summary: &str) -> f64 {
    let item = token_set(item_content);
    if item.is_empty() {
        return 0.0;
    }
    let summary = token_set(summary);
    let intersection = item.intersection(&summary).count();
    intersection as f64 / item.len() as f64
}

/// Classify a single fingerprint item against `summary`.
///
/// `overlap >= preserved_threshold` ⇒ `Preserved`; in the gap ⇒
/// `Paraphrased`; below `lost_threshold` ⇒ `Lost`.
pub fn classify_item(
    item: &FingerprintItem,
    summary: &str,
    idf: &HashMap<String, f64>,
    preserved_threshold: f64,
    lost_threshold: f64,
) -> ClassifiedItem {
    let overlap = weighted_overlap(&item.content, summary, idf);
    let classification = if overlap >= preserved_threshold {
        LossClassification::Preserved
    } else if overlap >= lost_threshold {
        LossClassification::Paraphrased
    } else {
        LossClassification::Lost
    };
    ClassifiedItem {
        fingerprint_item: item.clone(),
        classification,
        overlap_score: overlap,
    }
}

// ── Top-level loss detection ──────────────────────────────────────────────────

/// Run loss detection over every fingerprint item, returning a fully-
/// populated [`LossReport`].
///
/// `pre_compaction_tokens` and `post_compaction_tokens` are surfaced
/// in the report directly — callers (the PostCompact hook) compute
/// these via the tokenizer outside this module.
pub fn detect_loss(
    items: &[FingerprintItem],
    summary: &str,
    pre_context: &Context,
    session_id: &str,
    compaction_trigger: &str,
    pre_compaction_tokens: usize,
    post_compaction_tokens: usize,
) -> LossReport {
    let idf = compute_idf(pre_context);

    let classified: Vec<ClassifiedItem> = items
        .iter()
        .map(|item| {
            classify_item(
                item,
                summary,
                &idf,
                DEFAULT_PRESERVED_THRESHOLD,
                DEFAULT_LOST_THRESHOLD,
            )
        })
        .collect();

    let preserved_count = classified
        .iter()
        .filter(|c| c.classification == LossClassification::Preserved)
        .count();
    let paraphrased_count = classified
        .iter()
        .filter(|c| c.classification == LossClassification::Paraphrased)
        .count();
    let lost_count = classified
        .iter()
        .filter(|c| c.classification == LossClassification::Lost)
        .count();

    let total = classified.len();
    let compression_ratio = if pre_compaction_tokens > 0 {
        post_compaction_tokens as f64 / pre_compaction_tokens as f64
    } else {
        0.0
    };
    let preservation_ratio = if total > 0 {
        preserved_count as f64 / total as f64
    } else {
        0.0
    };

    LossReport {
        session_id: session_id.to_string(),
        compaction_trigger: compaction_trigger.to_string(),
        pre_compaction_tokens,
        post_compaction_tokens,
        compression_ratio,
        total_fingerprinted: total,
        preserved_count,
        paraphrased_count,
        lost_count,
        preservation_ratio,
        items: classified,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::{AttentionZone, Chunk};
    use crate::fingerprint::{make_id, ItemCategory, ItemScores};

    fn chunk(s: &str) -> Chunk {
        Chunk {
            index: 0,
            role: "user".into(),
            content: s.into(),
            token_count: 0,
            relevance_score: 0.5,
            attention_zone: AttentionZone::Strong,
        }
    }

    fn item(content: &str) -> FingerprintItem {
        FingerprintItem {
            id: make_id(content),
            category: ItemCategory::Decision,
            content: content.into(),
            tokens: 0,
            occurrence_count: 1,
            source_positions: vec![0],
            priority_score: 0.5,
            scores: ItemScores {
                uniqueness: 1.0,
                recency: 1.0,
                position_risk: 0.0,
                textrank_boost: 0.0,
            },
            extraction_method: "test".into(),
            confidence: None,
        }
    }

    #[test]
    fn preserved_item_when_summary_carries_all_high_weight_tokens() {
        // Build a context where everything's roughly equal-weight, then
        // make the summary contain every item token. Should be Preserved.
        let ctx = Context::new(vec![
            chunk("alpha beta gamma delta"),
            chunk("epsilon zeta eta theta"),
        ]);
        let idf = compute_idf(&ctx);
        let it = item("alpha beta gamma delta");
        let summary = "alpha beta gamma delta and other stuff";
        let res = classify_item(&it, summary, &idf, 0.7, 0.3);
        assert_eq!(res.classification, LossClassification::Preserved);
        assert!(res.overlap_score >= 0.7);
    }

    #[test]
    fn lost_when_summary_omits_high_idf_tokens() {
        let ctx = Context::new(vec![
            chunk("auth service runs on port 8443 inside the mesh"),
            chunk("we will deploy via helm to the gke cluster"),
        ]);
        let idf = compute_idf(&ctx);
        let it = item("Auth runs on port 8443 inside the mesh");
        let summary = "we are deploying through the cluster"; // no port, no auth
        let res = classify_item(&it, summary, &idf, 0.7, 0.3);
        assert_eq!(res.classification, LossClassification::Lost);
    }

    #[test]
    fn empty_summary_classifies_all_as_lost() {
        let ctx = Context::new(vec![chunk("budget is fifty thousand dollars")]);
        let idf = compute_idf(&ctx);
        let it = item("budget is fifty thousand dollars");
        let res = classify_item(&it, "", &idf, 0.7, 0.3);
        assert_eq!(res.classification, LossClassification::Lost);
        assert!(res.overlap_score < 0.001);
    }

    #[test]
    fn identical_summary_classifies_all_as_preserved() {
        let ctx = Context::new(vec![chunk("budget is fifty thousand dollars")]);
        let idf = compute_idf(&ctx);
        let it = item("budget is fifty thousand dollars");
        let res = classify_item(&it, &it.content, &idf, 0.7, 0.3);
        assert_eq!(res.classification, LossClassification::Preserved);
    }

    /// **The critical test the brief calls out.**
    ///
    /// Pre-compaction conversation talks about PostgreSQL many times.
    /// Item is the classic decision sentence; the summary preserves
    /// the surface-form noun + a few common words but drops every
    /// high-IDF rationale token (`chose`, `over`, `mongodb`, `acid`,
    /// `compliance`).
    ///
    /// - Naive (set) overlap counts every shared token equally — the
    ///   shared common words keep us above the Lost threshold, so the
    ///   summary looks Paraphrased.
    /// - Weighted overlap discounts those common tokens — the rare
    ///   rationale tokens dominate the denominator, and missing them
    ///   pushes us decisively into Lost.
    ///
    /// This is the entire reason we IDF-weight: don't let frequent
    /// tokens mask that the rationale itself was dropped.
    #[test]
    fn weighted_beats_naive_on_dropped_rationale() {
        // Pre-conversation: "postgresql" / "we" / "for" / "the" appear
        // in most chunks (low IDF). "acid" / "mongodb" / "chose" /
        // "over" / "compliance" each appear once (high IDF).
        let chunks: Vec<Chunk> = [
            "we use postgresql for the api tier",
            "postgresql holds session state for the project",
            "postgresql runs in the same vpc for performance",
            "postgresql has nightly backups for the team",
            "postgresql is the primary store for the project",
            "we considered mongodb chose over postgresql for acid compliance reasons",
            "the team is comfortable with sql so postgresql it is for now",
            "we use postgresql for everything in the project",
        ]
        .iter()
        .map(|s| chunk(s))
        .collect();
        let ctx = Context::new(chunks);
        let idf = compute_idf(&ctx);

        // Item with both common-word filler AND rare rationale words.
        let it = item("We chose PostgreSQL over MongoDB for ACID compliance");
        // Summary keeps the common words + the surface noun, drops
        // every high-IDF rationale token.
        let summary = "We use PostgreSQL for the project";

        let weighted = weighted_overlap(&it.content, summary, &idf);
        let naive = naive_overlap(&it.content, summary);

        assert!(
            (0.3..0.7).contains(&naive),
            "naive overlap should land in Paraphrased band [0.3, 0.7), got {}",
            naive
        );
        assert!(
            weighted < naive,
            "weighted ({}) must drop below naive ({}) when high-IDF tokens missing",
            weighted,
            naive
        );
        assert!(
            weighted < 0.3,
            "weighted ({}) should classify item as LOST — IDF-weighting must surface dropped rationale",
            weighted
        );
    }

    #[test]
    fn detect_loss_aggregates_counts_and_ratios() {
        let ctx = Context::new(vec![
            chunk("budget is fifty thousand dollars"),
            chunk("port 8443 for auth service"),
            chunk("we use postgresql for the database"),
        ]);
        let items = vec![
            item("budget is fifty thousand dollars"),
            item("port 8443 for auth service"),
            item("we use postgresql for the database"),
        ];
        let summary = "budget is fifty thousand dollars and we use postgresql"; // missing port
        let report = detect_loss(&items, summary, &ctx, "sess", "auto", 1000, 200);
        assert_eq!(report.total_fingerprinted, 3);
        assert_eq!(report.items.len(), 3);
        assert!(report.preservation_ratio > 0.0);
        assert_eq!(
            report.preserved_count + report.paraphrased_count + report.lost_count,
            3
        );
        assert!((report.compression_ratio - 0.2).abs() < 1e-6);
    }
}
