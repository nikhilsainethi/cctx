//! Compaction loss detection + injection payload types.
//!
//! Day 21 defines just the data types so the state store can read/write
//! them. Day 23 adds `loss_detector.rs` and `injection_builder.rs` with
//! the algorithms. Schema matches the Compaction Guard architecture (§3.4).

use serde::{Deserialize, Serialize};

use crate::fingerprint::ItemCategory;

/// Result of comparing a pre-compaction fingerprint against the
/// generated compaction summary.
///
/// Items in the fingerprint are bucketed into three groups:
///
/// - **preserved** — most tokens survived (`overlap_score >= preserved_threshold`)
/// - **paraphrased** — partially preserved, may have lost precision
/// - **lost** — effectively dropped (`overlap_score < lost_threshold`)
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
    /// The actual lost items, ranked by descending priority — Day 23's
    /// injection builder consumes this list.
    pub lost_items: Vec<LostItem>,
}

/// A single fingerprint item that did NOT survive compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LostItem {
    /// Identifier from the source [`crate::fingerprint::FingerprintItem`].
    pub fingerprint_id: String,
    /// Category copied from the source item for human-readable reports.
    pub category: ItemCategory,
    /// Textual content lost during compaction.
    pub content: String,
    /// Token count of `content`.
    pub tokens: usize,
    /// Original priority score from the fingerprint.
    pub priority_score: f64,
    /// Token-overlap score against the compaction summary (`< lost_threshold`).
    pub overlap_score: f64,
}
