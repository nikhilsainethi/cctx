//! Information fingerprint — what unique, irreplaceable information exists
//! in a conversation transcript.
//!
//! The fingerprint is the core data structure for the Compaction Guard
//! feature. Day 21 establishes the types and persistence; Day 22 adds the
//! extractor and scorer that populate `items`. The schema matches the
//! Compaction Guard architecture document (§3.3) so later phases can fill
//! in the algorithms without changing the type surface.

use serde::{Deserialize, Serialize};

/// Snapshot of a session's information landscape, written to disk before
/// compaction so we can later detect what was lost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    /// Claude Code session id (hex). Used to correlate pre/post compaction artifacts.
    pub session_id: String,
    /// ISO-8601 UTC timestamp recording when the fingerprint was generated.
    pub created_at: String,
    /// Total token count of the source transcript when fingerprinted.
    pub total_tokens: usize,
    /// Number of items in `items`. Cached for cheap reporting.
    pub total_items: usize,
    /// All extracted items, sorted by descending `priority_score` after
    /// scoring runs. Day 21 may save an empty vector; Day 22 populates.
    pub items: Vec<FingerprintItem>,
}

/// One unit of information considered "fingerprintable" — typically a
/// constraint, decision, technical fact, debug insight, or progress marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintItem {
    /// Stable identifier for cross-referencing in loss reports / injection.
    pub id: String,
    /// What kind of information this item represents.
    pub category: ItemCategory,
    /// The actual textual content, condensed to the essential phrasing.
    pub content: String,
    /// Token count of `content`.
    pub tokens: usize,
    /// Number of times this information appears in the conversation.
    /// `1` ⇒ unique / irreplaceable, `5+` ⇒ redundant / safe to compress.
    pub occurrence_count: u32,
    /// Message indices in the source transcript where this information appears.
    pub source_positions: Vec<usize>,
    /// Composite priority `uniqueness × recency × position_risk`. Higher = more important.
    pub priority_score: f64,
    /// Component scores broken out for transparency in tooling.
    pub scores: ItemScores,
}

/// Component scores that combine into [`FingerprintItem::priority_score`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemScores {
    /// `1.0 / occurrence_count` — rarer items score higher.
    pub uniqueness: f64,
    /// Exponential decay from the most recent occurrence: `e^(-rate × age)`.
    pub recency: f64,
    /// How deep in the attention dead zone the primary occurrence sits.
    /// `0.0` = start/end (safe), `1.0` = deep middle (high risk).
    pub position_risk: f64,
}

/// Coarse classification of fingerprinted information for display + filtering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ItemCategory {
    /// User-stated requirement or hard constraint
    /// (e.g. `"budget is $50K"`, `"must use PostgreSQL"`, `"deadline is Friday"`).
    Constraint,
    /// Architectural or design decision with rationale
    /// (e.g. `"chose REST over GraphQL because of team familiarity"`).
    Decision,
    /// Specific technical fact: port, file path, URL, IP, version, identifier.
    TechnicalFact,
    /// Bug root cause or debugging insight.
    DebugInsight,
    /// Task status / progress marker.
    ProgressMarker,
    /// Uncategorized but considered unique enough to track.
    Other,
}
