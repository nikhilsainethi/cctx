//! Data types for the fingerprint engine.
//!
//! Schema matches the Compaction Guard architecture (§3.3.1) plus two
//! additions for Day 22:
//!
//! - `extraction_method` on every item (`"regex"`, `"keyword"`, `"rake"`)
//!   for transparency / debugging when items behave unexpectedly.
//! - `textrank_boost` on `ItemScores` to record the keyphrase score from
//!   the layer-3 RAKE pass, separate from the architecture's three core
//!   scores so we can swap the underlying algorithm later without
//!   breaking the schema.

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
    /// Number of items in `items` after dedup + scoring + filtering.
    pub total_items: usize,
    /// All extracted items, sorted by descending `priority_score`.
    pub items: Vec<FingerprintItem>,
}

/// One unit of information considered "fingerprintable" — typically a
/// constraint, decision, technical fact, debug insight, or progress marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintItem {
    /// Stable identifier derived from the content hash.
    /// Format: `fp_{hex_hash}`. Same content always produces the same id.
    pub id: String,
    /// What kind of information this item represents.
    pub category: ItemCategory,
    /// The actual textual content, condensed to the essential phrasing.
    pub content: String,
    /// Token count of `content` (BPE under cl100k_base).
    pub tokens: usize,
    /// Number of times this information appears in the conversation.
    /// `1` ⇒ unique / irreplaceable, `5+` ⇒ redundant / safe to compress.
    pub occurrence_count: u32,
    /// Chunk indices in the normalized [`crate::core::context::Context`]
    /// where this information appears. Multi-occurrence items list every
    /// position they were seen at.
    pub source_positions: Vec<usize>,
    /// Composite `uniqueness × recency × (0.5 + position_risk)` plus
    /// the RAKE textrank_boost. Higher = more important to preserve.
    pub priority_score: f64,
    /// Component scores broken out for transparency in tooling.
    pub scores: ItemScores,
    /// Which extraction layer produced (or last touched) this item.
    /// One of `"regex"`, `"keyword"`, `"rake"`, `"gliner"`.
    pub extraction_method: String,
    /// GLiNER confidence in `[0.0, 1.0]`. Populated for items extracted
    /// or boosted by the Tier-1 NER pass; `None` for pure Tier-0 items.
    /// Omitted from output JSON when `None` to keep Tier-0 fingerprints
    /// schema-clean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Component scores that combine into [`FingerprintItem::priority_score`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemScores {
    /// `1.0 / occurrence_count` — rarer items score higher.
    pub uniqueness: f64,
    /// Exponential decay from the most recent occurrence:
    /// `exp(-decay_rate × (total_messages − last_position))`.
    pub recency: f64,
    /// How deep in the attention dead zone the worst occurrence sits.
    /// `0.0` = at the edge (safe), `1.0` = dead-center middle (high risk).
    pub position_risk: f64,
    /// Normalized RAKE keyphrase score, clamped to `[0.0, 1.0]`. Items
    /// that didn't appear in the layer-3 ranking carry `0.0`.
    pub textrank_boost: f64,
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

// ── Engine configuration ──────────────────────────────────────────────────────

/// Tunables for the fingerprint engine.
///
/// Defaults match `.cctx/config.json` written by [`crate::state::store::init`]:
/// `min_priority_score = 0.1`, `max_items = 200`, `recency_decay_rate = 0.05`,
/// `tier = 0`.
#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    /// `0` = Tier-0 only (regex + keyword + RAKE; zero ML deps).
    /// `1` = Tier-0 + GLiNER zero-shot NER (requires `--features gliner`
    /// and a downloaded model). Higher tiers reserved for future use.
    pub tier: u8,
    /// Drop items whose final priority score is below this threshold.
    pub min_priority_score: f64,
    /// Keep only the top `max_items` after sorting by priority.
    pub max_items: usize,
    /// Decay factor for the recency score: larger ⇒ older items penalized faster.
    pub recency_decay_rate: f64,
    /// Layer-3 RAKE phrase ceiling — top-N most-ranked keyphrases are
    /// considered for catch-all extraction.
    pub rake_top_n: usize,
    /// GLiNER detection threshold — entities below this are dropped.
    /// Used only at Tier 1+. The brief specifies 0.4 as the default.
    pub gliner_threshold: f64,
    /// Multiplier applied to priority when both Tier 0 and GLiNER agree
    /// on an item with high GLiNER confidence. Used only at Tier 1+.
    pub gliner_agreement_boost: f64,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            tier: 0,
            min_priority_score: 0.1,
            max_items: 200,
            recency_decay_rate: 0.05,
            rake_top_n: 30,
            gliner_threshold: 0.4,
            gliner_agreement_boost: 1.5,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a stable, content-derived id of the form `fp_{16-hex}`.
///
/// Uses FNV-1a — small, fast, deterministic across processes and Rust
/// versions (unlike `DefaultHasher` whose seed is per-process random).
pub fn make_id(content: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in content.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fp_{:016x}", hash)
}
