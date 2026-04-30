//! End-to-end tests of the Tier-0 fingerprint engine against the
//! curated transcript fixtures. The unit tests inside the
//! `src/fingerprint/*` modules cover narrow behaviours; these tests
//! prove the pipeline catches the seeded facts on realistic input
//! and that the priority formula respects uniqueness over recurrence.

use std::path::Path;

use cctx::fingerprint::{fingerprint, Fingerprint, FingerprintConfig, ItemCategory};
use cctx::transcript::{normalize, parse_transcript};

const SAMPLE_FIXTURE: &str = "tests/fixtures/sample_transcript.jsonl";
const ACCURACY_FIXTURE: &str = "tests/fixtures/fingerprint_accuracy.jsonl";

fn fingerprint_fixture(path: &str) -> Fingerprint {
    let entries = parse_transcript(Path::new(path)).expect("fixture parses");
    let context = normalize(entries).expect("fixture normalizes");
    fingerprint(
        &context,
        &FingerprintConfig::default(),
        "test_session",
        "2026-04-30T00:00:00Z",
    )
}

// ── Sample transcript fixture (Day 21) ────────────────────────────────────────

#[test]
fn sample_transcript_extracts_seeded_facts() {
    let fp = fingerprint_fixture(SAMPLE_FIXTURE);
    assert!(!fp.items.is_empty(), "fingerprint should not be empty");

    // The five seeded facts from Day 21's fixture should each appear in
    // at least one extracted item.
    for fact in ["$50K", "PostgreSQL", "8443", "/etc/myapp/prod.yml", "HPA"] {
        assert!(
            fp.items.iter().any(|i| i.content.contains(fact)),
            "expected `{}` to appear in some fingerprint item",
            fact
        );
    }
}

#[test]
fn sample_transcript_categorizes_constraint_and_technical_fact() {
    let fp = fingerprint_fixture(SAMPLE_FIXTURE);

    // Budget statement should land as a Constraint via the keyword layer.
    let budget = fp
        .items
        .iter()
        .find(|i| i.content.contains("$50K"))
        .expect("budget item present");
    assert!(
        matches!(
            budget.category,
            ItemCategory::Constraint | ItemCategory::TechnicalFact
        ),
        "budget should categorise as Constraint or TechnicalFact, was {:?}",
        budget.category
    );

    // Port 8443 should land as a TechnicalFact via the regex layer.
    let port = fp
        .items
        .iter()
        .find(|i| i.content.contains("8443"))
        .expect("port item present");
    assert_eq!(
        port.category,
        ItemCategory::TechnicalFact,
        "port should be TechnicalFact"
    );
}

#[test]
fn sample_transcript_emits_extraction_method_metadata() {
    let fp = fingerprint_fixture(SAMPLE_FIXTURE);
    // Every item's extraction_method should be one of the three known layers.
    for item in &fp.items {
        assert!(
            matches!(
                item.extraction_method.as_str(),
                "regex" | "keyword" | "rake"
            ),
            "unexpected extraction_method `{}` on item `{}`",
            item.extraction_method,
            item.content
        );
    }
}

// ── Accuracy fixture (Day 22 spec) ────────────────────────────────────────────

/// Find the *repeated* CORS DebugInsight item — the one that's mentioned
/// in four separate chunks. Other CORS-mentioning items (e.g. an
/// assistant's one-time explanation captured by RAKE) are unrelated to
/// the priority comparison being tested.
fn find_repeated_cors(fp: &Fingerprint) -> Option<&cctx::fingerprint::FingerprintItem> {
    fp.items.iter().find(|i| {
        matches!(i.category, ItemCategory::DebugInsight)
            && i.content.to_lowercase().contains("cors")
    })
}

#[test]
fn accuracy_fixture_unique_constraint_outranks_repeated_debug() {
    // The accuracy fixture mentions the same CORS bug four times and
    // the $50K budget once. Per the priority formula, uniqueness
    // dominates: the budget Constraint must land above the merged
    // 4-occurrence CORS DebugInsight even though the CORS reports are
    // (on average) more recent.
    let fp = fingerprint_fixture(ACCURACY_FIXTURE);

    let budget = fp
        .items
        .iter()
        .find(|i| matches!(i.category, ItemCategory::Constraint) && i.content.contains("$50K"))
        .expect("budget Constraint should be extracted");

    let cors = find_repeated_cors(&fp).expect("repeated CORS DebugInsight should be extracted");

    assert!(
        budget.priority_score > cors.priority_score,
        "uniqueness should win: budget ({}) > merged CORS ({})",
        budget.priority_score,
        cors.priority_score
    );
}

#[test]
fn accuracy_fixture_repeated_cors_collapses_via_dedup() {
    // The "The issue was a missing CORS header." sentence appears
    // verbatim in four separate chunks. After Jaccard merge they should
    // collapse to one DebugInsight with occurrence_count == 4 and four
    // distinct source_positions.
    let fp = fingerprint_fixture(ACCURACY_FIXTURE);
    let cors = find_repeated_cors(&fp).expect("repeated CORS item should exist");
    assert_eq!(
        cors.occurrence_count, 4,
        "CORS DebugInsight should merge to count=4, got {}",
        cors.occurrence_count
    );
    assert_eq!(
        cors.source_positions.len(),
        4,
        "CORS DebugInsight should reference 4 source positions"
    );
}

#[test]
fn accuracy_fixture_extracts_postgresql_decision() {
    let fp = fingerprint_fixture(ACCURACY_FIXTURE);
    let postgres = fp
        .items
        .iter()
        .find(|i| i.content.to_lowercase().contains("postgresql"));
    assert!(
        postgres.is_some(),
        "PostgreSQL decision should be extracted"
    );
}

#[test]
fn accuracy_fixture_extracts_port_8443() {
    let fp = fingerprint_fixture(ACCURACY_FIXTURE);
    let port = fp.items.iter().find(|i| i.content.contains("8443"));
    assert!(port.is_some(), "port 8443 should be extracted");
    if let Some(p) = port {
        assert_eq!(
            p.extraction_method, "regex",
            "port number should come from the regex layer"
        );
    }
}

#[test]
fn fingerprint_is_serializable_to_json() {
    let fp = fingerprint_fixture(SAMPLE_FIXTURE);
    let json = serde_json::to_string_pretty(&fp).expect("fingerprint serializes");
    assert!(json.contains("\"items\""));
    assert!(json.contains("\"priority_score\""));

    // Round-trip — schema must parse back into the same struct.
    let parsed: Fingerprint = serde_json::from_str(&json).expect("round-trip parse");
    assert_eq!(parsed.total_items, fp.total_items);
}
