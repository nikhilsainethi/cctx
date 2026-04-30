//! Cross-item Jaccard merge for the fingerprint engine.
//!
//! The extractor emits one item per `(chunk, sentence)` pair, so the
//! same fact mentioned in three messages produces three near-duplicate
//! items. This module collapses them into a single item with
//! `occurrence_count = 3` and `source_positions = [c1, c2, c3]` so the
//! scorer's uniqueness term reflects reality.
//!
//! Algorithm: pairwise Jaccard over word sets, threshold 0.7. We
//! repeat the sweep until no merges happen — typical fingerprints
//! converge in 1–3 passes.

use std::collections::HashSet;

use super::types::{FingerprintItem, ItemCategory};

/// Similarity threshold above which two items are considered duplicates
/// and merged. 0.7 is permissive enough to catch paraphrases of the
/// same constraint but tight enough to keep distinct ports / paths apart.
const MERGE_THRESHOLD: f64 = 0.7;

/// Merge near-duplicate items in place. Returns the deduplicated list.
///
/// Merge semantics:
///
/// - **Content**: the longer item's content wins (more detail = more value).
/// - **occurrence_count**: summed across merged items.
/// - **source_positions**: union of the two lists, deduplicated.
/// - **category**: the longer item's category (avoids "Other" overriding
///   a more specific Constraint/Decision tag when the RAKE-found
///   sentence is a near-paraphrase of a keyword-found one).
/// - **extraction_method**: the surviving (longer) item's method.
/// - **textrank_boost**: max of the two — preserves any RAKE signal even
///   when the keyword-layer entry happens to be longer.
pub fn merge_duplicates(items: Vec<FingerprintItem>) -> Vec<FingerprintItem> {
    let mut working = items;

    // Iterate until a full pass produces no merges. Each pass is O(n²).
    loop {
        let (next, merged_any) = single_pass(working);
        working = next;
        if !merged_any {
            break;
        }
    }

    working
}

fn single_pass(items: Vec<FingerprintItem>) -> (Vec<FingerprintItem>, bool) {
    let n = items.len();
    if n < 2 {
        return (items, false);
    }

    // Pre-tokenize every item's word set once per pass to avoid
    // re-tokenizing inside the O(n²) loop.
    let word_sets: Vec<HashSet<String>> = items.iter().map(|i| word_set(&i.content)).collect();

    // dropped[i] tracks items absorbed into a survivor.
    let mut dropped = vec![false; n];
    let mut merged_any = false;

    // Mutable items so we can mutate the survivor in place.
    let mut items = items;

    for i in 0..n {
        if dropped[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dropped[j] {
                continue;
            }
            let sim = jaccard(&word_sets[i], &word_sets[j]);
            if sim < MERGE_THRESHOLD {
                continue;
            }

            // Pick the survivor — longer content stays.
            let (survivor_idx, absorbed_idx) = if items[i].content.len() >= items[j].content.len() {
                (i, j)
            } else {
                (j, i)
            };
            absorb(&mut items, survivor_idx, absorbed_idx);
            dropped[absorbed_idx] = true;
            merged_any = true;

            // The absorbed item is gone — don't compare it against later items.
            if absorbed_idx == i {
                break;
            }
        }
    }

    let kept: Vec<FingerprintItem> = items
        .into_iter()
        .zip(dropped)
        .filter_map(|(item, dropped)| if dropped { None } else { Some(item) })
        .collect();
    (kept, merged_any)
}

/// Move data from `src` into `dst` per the merge semantics in [`merge_duplicates`].
fn absorb(items: &mut [FingerprintItem], dst: usize, src: usize) {
    // Sum occurrence_count.
    let extra = items[src].occurrence_count;
    items[dst].occurrence_count = items[dst].occurrence_count.saturating_add(extra);

    // Union of source_positions, dedup-preserving.
    let mut positions: HashSet<usize> = items[dst].source_positions.iter().copied().collect();
    positions.extend(items[src].source_positions.iter().copied());
    let mut merged: Vec<usize> = positions.into_iter().collect();
    merged.sort_unstable();
    items[dst].source_positions = merged;

    // Max of textrank_boost (preserve any RAKE signal).
    let boost_src = items[src].scores.textrank_boost;
    if boost_src > items[dst].scores.textrank_boost {
        items[dst].scores.textrank_boost = boost_src;
    }

    // Category preference: keep the more specific tag if the survivor's
    // is `Other` but the absorbed item carries a real category.
    if items[dst].category == ItemCategory::Other && items[src].category != ItemCategory::Other {
        items[dst].category = items[src].category.clone();
    }
}

/// Stop-words filtered out before Jaccard. Mirrors the list used in
/// [`crate::analyzer::duplication`] so similarity scoring is consistent
/// across the codebase.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "it", "its", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "shouldn", "may", "might", "must", "can", "cannot",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "my", "your", "his",
    "her", "our", "their", "me", "him", "us", "them", "what", "which", "who", "whom", "if", "not",
    "no", "so", "as", "from", "about", "into", "through", "also", "just", "more", "very", "too",
    "than", "still", "again", "once",
];

/// Lower-cased word set with stop-words and punctuation stripped, used
/// as the basis for Jaccard similarity. Mirrors
/// [`crate::analyzer::duplication`]'s `extract_words` so similarity at
/// the fingerprint layer is consistent with the duplicate-detection
/// metric used elsewhere.
fn word_set(text: &str) -> HashSet<String> {
    let stops: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                // Strip leading / trailing punctuation but keep `$` so
                // "$50K" stays distinct from generic "fifty thousand".
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '$')
                .to_string()
        })
        .filter(|w| w.len() > 1 && !stops.contains(w.as_str()))
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::types::{make_id, FingerprintItem, ItemCategory, ItemScores};
    use super::*;

    fn item(content: &str, chunk_idx: usize, method: &str) -> FingerprintItem {
        FingerprintItem {
            id: make_id(content),
            category: ItemCategory::TechnicalFact,
            content: content.into(),
            tokens: 0,
            occurrence_count: 1,
            source_positions: vec![chunk_idx],
            priority_score: 0.0,
            scores: ItemScores {
                uniqueness: 1.0,
                recency: 1.0,
                position_risk: 0.0,
                textrank_boost: 0.0,
            },
            extraction_method: method.into(),
        }
    }

    #[test]
    fn near_duplicates_merge_into_one_item() {
        let items = vec![
            item("Auth service runs on port 8443 inside the mesh", 1, "regex"),
            item("Auth service is on port 8443 inside the mesh", 5, "regex"),
            item("Auth service runs on port 8443 in the mesh", 12, "regex"),
        ];
        let merged = merge_duplicates(items);
        assert_eq!(merged.len(), 1, "all three should collapse");
        let m = &merged[0];
        assert_eq!(m.occurrence_count, 3);
        assert_eq!(m.source_positions, vec![1, 5, 12]);
    }

    #[test]
    fn distinct_facts_are_not_merged() {
        let items = vec![
            item("Database is PostgreSQL with ACID compliance", 0, "keyword"),
            item("Auth uses JWT signed with RS256", 1, "keyword"),
            item("Deployed via Helm to GKE clusters", 2, "keyword"),
        ];
        let merged = merge_duplicates(items);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn category_upgrade_when_other_meets_specific() {
        let other = FingerprintItem {
            category: ItemCategory::Other,
            ..item("budget should not exceed fifty thousand dollars", 5, "rake")
        };
        let constraint = FingerprintItem {
            category: ItemCategory::Constraint,
            ..item(
                "budget shouldn't exceed fifty thousand dollars",
                5,
                "keyword",
            )
        };
        let merged = merge_duplicates(vec![other, constraint]);
        assert_eq!(merged.len(), 1);
        // Survivor's category should be Constraint regardless of which was
        // longer — the absorb() rule promotes specific over Other.
        assert_eq!(merged[0].category, ItemCategory::Constraint);
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let merged = merge_duplicates(vec![]);
        assert!(merged.is_empty());
    }

    #[test]
    fn single_item_passes_through_unchanged() {
        let it = item("only fact in the world", 0, "regex");
        let merged = merge_duplicates(vec![it.clone()]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].content, it.content);
    }
}
