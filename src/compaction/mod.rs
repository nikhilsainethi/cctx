//! Compaction loss detection + injection payload types.
//!
//! Day 21 stubbed the data types so the state store could read/write
//! `LossReport`. Day 24 fills in:
//!
//! - [`loss_detector`] — IDF-weighted token-overlap classification of
//!   each fingerprint item against the post-compaction summary.
//! - [`injection_builder`] — greedy budget-constrained selection of
//!   the lost items to surface back to the model on `SessionStart`.
//! - [`budget`] — small helper that tracks token allocation across
//!   selected items.
//!
//! Schema differs from the architecture document's §3.4.2 in one way:
//! the report stores **every** fingerprint item with its classification
//! (`Preserved` / `Paraphrased` / `Lost`), not just the lost ones. The
//! richer schema lets `cctx loss-report` show preservation breakdown
//! at a glance and lets later phases re-derive subsets without re-
//! running detection.

pub mod budget;
pub mod injection_builder;
pub mod loss_detector;

use serde::{Deserialize, Serialize};

use crate::fingerprint::FingerprintItem;

/// Bucket each [`FingerprintItem`] is sorted into after the loss
/// detector compares it against the compaction summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossClassification {
    /// `weighted_overlap >= preserved_threshold` (0.7 by default).
    /// Most tokens — and the high-IDF tokens specifically — survived.
    Preserved,
    /// `lost_threshold <= weighted_overlap < preserved_threshold`
    /// (0.3 ≤ overlap < 0.7). Survives, but the precise phrasing or
    /// the high-IDF tokens may be missing — partial information loss.
    Paraphrased,
    /// `weighted_overlap < lost_threshold` (0.3). Effectively dropped;
    /// candidate for re-injection on the next SessionStart.
    Lost,
}

impl LossClassification {
    /// Human-readable label used in the CLI's loss-report rendering.
    pub fn label(&self) -> &'static str {
        match self {
            LossClassification::Preserved => "preserved",
            LossClassification::Paraphrased => "paraphrased",
            LossClassification::Lost => "lost",
        }
    }
}

/// A single fingerprint item paired with its loss-detection verdict.
///
/// Stored in [`LossReport::items`] in the order produced by the loss
/// detector (which mirrors the input fingerprint's priority order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedItem {
    /// Full clone of the source fingerprint item — keeps the loss
    /// report self-contained and replayable without re-loading the
    /// original fingerprint.
    pub fingerprint_item: FingerprintItem,
    /// Bucket the item landed in.
    pub classification: LossClassification,
    /// Weighted token-overlap score in `[0.0, 1.0]`. The thresholds
    /// in [`loss_detector::DEFAULT_PRESERVED_THRESHOLD`] /
    /// [`loss_detector::DEFAULT_LOST_THRESHOLD`] map this to a class.
    pub overlap_score: f64,
}

/// Result of comparing a pre-compaction fingerprint against the
/// generated compaction summary.
///
/// Schema:
///
/// - aggregate counts (`preserved_count` / `paraphrased_count` /
///   `lost_count`) are derived but stored explicitly so consumers
///   don't need to re-tally on read,
/// - `items` holds every classified item — a downstream caller
///   (e.g. the injection builder) filters to `Lost` only,
/// - `compression_ratio` is `post_compaction_tokens / pre_compaction_tokens`
///   for at-a-glance summary-quality reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossReport {
    /// Claude Code session id this report belongs to.
    pub session_id: String,
    /// `"auto"` (auto-compact at ~95%) or `"manual"` (`/compact` command).
    pub compaction_trigger: String,
    /// Token count of the transcript right before compaction.
    pub pre_compaction_tokens: usize,
    /// Token count of the compaction summary.
    pub post_compaction_tokens: usize,
    /// `post_compaction_tokens / pre_compaction_tokens`.
    pub compression_ratio: f64,
    /// Total items in the fingerprint (denominator for ratios below).
    pub total_fingerprinted: usize,
    /// Items classified as PRESERVED.
    pub preserved_count: usize,
    /// Items classified as PARAPHRASED.
    pub paraphrased_count: usize,
    /// Items classified as LOST.
    pub lost_count: usize,
    /// `preserved_count / total_fingerprinted`.
    pub preservation_ratio: f64,
    /// Every fingerprint item with its loss-detection verdict, in the
    /// same order as the source fingerprint's priority sort.
    pub items: Vec<ClassifiedItem>,
}

impl LossReport {
    /// Iterate over only the `Lost` items, in priority order. Used by
    /// the injection builder.
    pub fn lost_items(&self) -> impl Iterator<Item = &ClassifiedItem> {
        self.items
            .iter()
            .filter(|i| i.classification == LossClassification::Lost)
    }
}
