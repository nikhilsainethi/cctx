//! Append-only history of compaction events, persisted to
//! `.cctx/compaction-log.json` as a JSON array.
//!
//! The log answers: "how well has compaction preserved my conversations
//! over time?" — useful both as a debugging aid and as the data source
//! for the future `cctx compaction-history` command (Day 25+).

use serde::{Deserialize, Serialize};

/// A single compaction event recorded in the log.
///
/// Created on PostCompact runs once we have both the fingerprint and the
/// loss-report numbers. Day 21 defines the schema; Day 23/24 populate
/// real values from the loss detector and injection builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEvent {
    /// Claude Code session id.
    pub session_id: String,
    /// ISO-8601 UTC timestamp when the event was recorded.
    pub timestamp: String,
    /// `"auto"` (auto-compact at ~95%) or `"manual"` (`/compact`).
    pub trigger: String,
    /// Total fingerprinted items at pre-compaction time.
    pub total_items: usize,
    /// Of those, how many were classified PRESERVED in the summary.
    pub preserved: usize,
    /// Classified PARAPHRASED.
    pub paraphrased: usize,
    /// Classified LOST — the items that drove re-injection.
    pub lost: usize,
    /// Token count of the recovery payload that was queued for injection.
    pub injection_tokens: usize,
}

/// Read the entire compaction log from `.cctx/compaction-log.json`.
///
/// # Errors
///
/// Returns `Err` only when the file exists but is unreadable or contains
/// invalid JSON. A missing file is treated as "no history yet" and yields
/// an empty vector — callers don't need to special-case fresh projects.
pub fn read_history(log_path: &std::path::Path) -> anyhow::Result<Vec<CompactionEvent>> {
    use anyhow::Context as _;
    if !log_path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(log_path)
        .with_context(|| format!("Cannot read compaction log {}", log_path.display()))?;
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str::<Vec<CompactionEvent>>(&raw)
        .with_context(|| format!("Invalid JSON in compaction log {}", log_path.display()))
}
