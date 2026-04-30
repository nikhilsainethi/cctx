//! Priority scoring for fingerprint items.
//!
//! Three-component formula adapted from the Compaction Guard architecture
//! §3.3.1, with the position term shifted by 0.5 so items at the safe
//! ends of the window still score nonzero:
//!
//! ```text
//! priority = uniqueness × recency × (0.5 + position_risk) + textrank_boost
//! ```
//!
//! - **uniqueness** = `1.0 / occurrence_count` — a fact mentioned once is
//!   irreplaceable; a fact mentioned five times is fine to lose.
//! - **recency** = `exp(-decay × (total_messages − last_position))` —
//!   recent mentions get full credit; old mentions decay exponentially.
//! - **position_risk** = max over `source_positions` of distance into
//!   the dead zone, in `[0, 1]` (0 = at edge / safe, 1 = dead-center).
//! - **textrank_boost** is added in addition (not multiplied) so the
//!   RAKE signal contributes additively to a baseline rather than being
//!   gated on the multiplicative product going non-zero.

use super::types::{FingerprintConfig, FingerprintItem};

/// Score every item in place: fills `priority_score` and the component
/// scores in `ItemScores` (uniqueness, recency, position_risk).
///
/// `total_messages` is the number of chunks in the source [`crate::core::context::Context`].
/// Used to compute recency decay (newer = closer to `total_messages`)
/// and position risk (distance from the window edges).
pub fn score(items: &mut [FingerprintItem], total_messages: usize, config: &FingerprintConfig) {
    let total_messages = total_messages.max(1);
    let last_index = total_messages.saturating_sub(1) as f64;

    for item in items.iter_mut() {
        let occ = item.occurrence_count.max(1) as f64;
        item.scores.uniqueness = 1.0 / occ;

        // Most-recent position drives the recency score.
        let last_position = item.source_positions.iter().copied().max().unwrap_or(0) as f64;
        let age = (last_index - last_position).max(0.0);
        item.scores.recency = (-config.recency_decay_rate * age).exp();

        // Position risk = max distance into the dead zone across the
        // item's positions. 0 at edges, 1 dead-center.
        item.scores.position_risk = item
            .source_positions
            .iter()
            .map(|&p| position_risk(p, total_messages))
            .fold(0.0_f64, f64::max);

        // Composite + additive RAKE bonus.
        item.priority_score =
            item.scores.uniqueness * item.scores.recency * (0.5 + item.scores.position_risk)
                + item.scores.textrank_boost;
    }
}

/// Map a chunk position to a continuous dead-zone risk in `[0, 1]`.
///
/// `position / total` gives a normalized index. The minimum of that and
/// `1 - normalized` is the distance to the nearest edge (in `[0, 0.5]`).
/// Doubling yields the final `[0, 1]` score.
pub fn position_risk(position: usize, total_messages: usize) -> f64 {
    if total_messages <= 1 {
        return 0.0;
    }
    let normalized = position as f64 / (total_messages - 1) as f64;
    let distance_from_edge = normalized.min(1.0 - normalized);
    (distance_from_edge * 2.0).clamp(0.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::types::{make_id, FingerprintItem, ItemCategory, ItemScores};
    use super::*;

    fn make(occ: u32, positions: Vec<usize>) -> FingerprintItem {
        FingerprintItem {
            id: make_id(&format!("item_{}_{:?}", occ, positions)),
            category: ItemCategory::Other,
            content: "x".into(),
            tokens: 0,
            occurrence_count: occ,
            source_positions: positions,
            priority_score: 0.0,
            scores: ItemScores {
                uniqueness: 0.0,
                recency: 0.0,
                position_risk: 0.0,
                textrank_boost: 0.0,
            },
            extraction_method: "test".into(),
        }
    }

    #[test]
    fn unique_item_outscores_repeated_at_same_position() {
        let cfg = FingerprintConfig::default();
        let mut items = vec![make(1, vec![10]), make(5, vec![10])];
        score(&mut items, 30, &cfg);
        assert!(items[0].priority_score > items[1].priority_score);
    }

    #[test]
    fn recent_item_outscores_older_at_same_uniqueness() {
        let cfg = FingerprintConfig::default();
        let mut items = vec![make(1, vec![29]), make(1, vec![2])];
        // Both unique; recency favors index 29 (close to total=30) over 2.
        score(&mut items, 30, &cfg);
        assert!(items[0].priority_score > items[1].priority_score);
    }

    #[test]
    fn dead_zone_item_outscores_edge_at_same_uniqueness_and_recency() {
        // Same recency requires identical last_position relative to total.
        // Use two same-position items but vary position_risk by adjusting
        // total: position 5 in a 31-message conversation is mid-window
        // (dead zone), position 0 in a 31-message conversation is at the edge.
        let cfg = FingerprintConfig {
            recency_decay_rate: 0.0, // disable recency to isolate position_risk
            ..FingerprintConfig::default()
        };
        let mut a = make(1, vec![15]); // dead-center in 31-message window
        let mut b = make(1, vec![0]); // start of 31-message window
        let mut items = vec![a.clone(), b.clone()];
        score(&mut items, 31, &cfg);
        a = items[0].clone();
        b = items[1].clone();
        assert!(
            a.priority_score > b.priority_score,
            "dead-zone item ({}) should outscore edge ({})",
            a.priority_score,
            b.priority_score
        );
    }

    #[test]
    fn position_risk_function_extremes() {
        // Edges should be safe (risk ~0); middle should be high (risk ~1).
        assert!(position_risk(0, 100) < 0.05);
        assert!(position_risk(99, 100) < 0.05);
        let mid = position_risk(50, 100);
        assert!(mid > 0.95, "middle of 100 should be near 1.0, got {}", mid);
    }

    #[test]
    fn textrank_boost_adds_to_baseline() {
        let cfg = FingerprintConfig::default();
        let mut a = make(1, vec![15]);
        let mut b = make(1, vec![15]);
        b.scores.textrank_boost = 0.4;
        let mut items = vec![a.clone(), b.clone()];
        score(&mut items, 30, &cfg);
        a = items[0].clone();
        b = items[1].clone();
        assert!(b.priority_score > a.priority_score);
    }
}
