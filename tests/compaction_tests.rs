//! End-to-end tests for the loss-detector + injection-builder pipeline.
//!
//! Runs against the curated transcript fixture from Day 21. Proves the
//! full path from a transcript JSONL → fingerprint → loss report → injection
//! payload works without surprise schema mismatches between phases.

use std::path::Path;

use cctx::compaction::injection_builder::build_injection;
use cctx::compaction::loss_detector::{
    classify_item, compute_idf, detect_loss, weighted_overlap, DEFAULT_LOST_THRESHOLD,
    DEFAULT_PRESERVED_THRESHOLD,
};
use cctx::compaction::{LossClassification, LossReport};
use cctx::fingerprint::{fingerprint, FingerprintConfig};
use cctx::transcript::{normalize, parse_transcript};

const SAMPLE_FIXTURE: &str = "tests/fixtures/sample_transcript.jsonl";

fn fingerprint_sample() -> (cctx::core::context::Context, cctx::fingerprint::Fingerprint) {
    let entries = parse_transcript(Path::new(SAMPLE_FIXTURE)).unwrap();
    let context = normalize(entries).unwrap();
    let fp = fingerprint(
        &context,
        &FingerprintConfig::default(),
        "test_session",
        "2026-04-30T00:00:00Z",
    );
    (context, fp)
}

// ── Loss-detector integration ────────────────────────────────────────────────

#[test]
fn full_pipeline_classifies_against_summary_keeping_some_facts() {
    let (context, fp) = fingerprint_sample();
    // A summary that keeps PostgreSQL + budget mentions but drops port + prod.yml.
    let summary = "We agreed to use PostgreSQL with a $50K budget cap. \
                   Several TODOs remain in the codebase around caching and migrations.";
    let report = detect_loss(
        &fp.items,
        summary,
        &context,
        "test_session",
        "auto",
        context.total_tokens,
        50,
    );

    assert_eq!(report.session_id, "test_session");
    assert_eq!(report.compaction_trigger, "auto");
    assert_eq!(report.total_fingerprinted, fp.total_items);
    assert_eq!(report.items.len(), fp.total_items);
    assert_eq!(
        report.preserved_count + report.paraphrased_count + report.lost_count,
        report.total_fingerprinted
    );

    // The port 8443 fact mentioned only in entry 15 should land Lost.
    let port_classification = report
        .items
        .iter()
        .find(|c| c.fingerprint_item.content.contains("8443"))
        .map(|c| c.classification);
    assert!(
        matches!(port_classification, Some(LossClassification::Lost)),
        "port 8443 should be classified as Lost when summary omits it"
    );
}

#[test]
fn empty_summary_marks_everything_lost() {
    let (context, fp) = fingerprint_sample();
    let report = detect_loss(
        &fp.items,
        "",
        &context,
        "test_session",
        "auto",
        context.total_tokens,
        0,
    );
    assert_eq!(report.lost_count, report.total_fingerprinted);
    assert_eq!(report.preserved_count, 0);
}

#[test]
fn idf_reweights_overlap_correctly_on_real_fixture() {
    let (context, _) = fingerprint_sample();
    let idf = compute_idf(&context);

    // Words that appear once in the fixture should have noticeably
    // higher IDF than the boilerplate "the" / "is".
    let rare = idf.get("8443").copied().unwrap_or(0.0);
    let common = idf.get("the").copied().unwrap_or(0.0);
    assert!(
        rare > common,
        "rare token (8443: {}) must outweigh common token (the: {})",
        rare,
        common
    );
}

// ── Critical comparison: weighted vs naive ───────────────────────────────────
//
// This test mirrors the brief's "PostgreSQL/MongoDB/ACID" example.
// It's the entire reason we IDF-weight: a summary that keeps the
// surface-form noun (PostgreSQL) but drops the rationale tokens
// should be classified Lost, not Paraphrased.

#[test]
fn weighted_overlap_beats_naive_on_dropped_rationale_real_world() {
    use cctx::core::context::{AttentionZone, Chunk, Context};
    let mk = |s: &str| Chunk {
        index: 0,
        role: "user".into(),
        content: s.into(),
        token_count: 0,
        relevance_score: 0.5,
        attention_zone: AttentionZone::Strong,
    };
    // A 7-message conversation: "postgresql" mentioned in 6/7 chunks
    // (very low IDF), the rationale tokens mentioned exactly once (high IDF).
    let ctx = Context::new(vec![
        mk("we use postgresql for the api tier"),
        mk("postgresql holds session state"),
        mk("postgresql runs in the same vpc"),
        mk("postgresql has nightly backups"),
        mk("postgresql is the primary store"),
        mk("we considered mongodb but acid wins for compliance reasons"),
        mk("the team is comfortable with sql so postgresql it is"),
    ]);
    let idf = compute_idf(&ctx);

    // Build a fingerprint item by hand (matching the brief's example).
    let item_content = "Chose PostgreSQL over MongoDB for ACID compliance";
    let summary = "The project uses PostgreSQL as the database.";

    let weighted = weighted_overlap(item_content, summary, &idf);

    // Sanity: weighted should drop us below the Lost threshold (0.3),
    // proving the IDF re-weighting actually changes the verdict.
    assert!(
        weighted < DEFAULT_LOST_THRESHOLD,
        "weighted overlap ({}) must be < lost threshold ({}) — IDF should down-weight 'postgresql' enough that missing 'acid'/'mongodb'/'compliance' classifies as Lost",
        weighted,
        DEFAULT_LOST_THRESHOLD
    );

    // And the classify wrapper should agree.
    let item_real = make_test_item(item_content);
    let classified = classify_item(
        &item_real,
        summary,
        &idf,
        DEFAULT_PRESERVED_THRESHOLD,
        DEFAULT_LOST_THRESHOLD,
    );
    assert_eq!(classified.classification, LossClassification::Lost);
}

fn make_test_item(content: &str) -> cctx::fingerprint::FingerprintItem {
    use cctx::fingerprint::{make_id, ItemCategory, ItemScores};
    cctx::fingerprint::FingerprintItem {
        id: make_id(content),
        category: ItemCategory::Decision,
        content: content.into(),
        tokens: 0,
        occurrence_count: 1,
        source_positions: vec![5],
        priority_score: 0.8,
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

// ── Injection-builder integration ────────────────────────────────────────────

#[test]
fn injection_recovers_lost_items_within_budget() {
    let (context, fp) = fingerprint_sample();
    let summary = "we agreed on postgresql"; // intentionally minimal — most facts lost
    let report = detect_loss(
        &fp.items,
        summary,
        &context,
        "test_session",
        "auto",
        context.total_tokens,
        20,
    );

    let payload = build_injection(&report.items, 4096);

    // We expect SOMETHING to be recovered (port 8443, prod.yml, etc.).
    assert!(
        !payload.is_empty(),
        "minimal summary should leave items lost"
    );
    assert!(payload.payload.contains("[cctx] Context items recovered"));
    assert!(payload.payload.contains("Treat as authoritative context"));

    // Token count must respect the budget.
    assert!(
        payload.token_count <= 4096 + 200,
        "payload tokens ({}) should be near budget",
        payload.token_count
    );
}

#[test]
fn injection_is_empty_when_summary_preserves_everything() {
    // Construct a fingerprint where every item appears in the summary
    // verbatim — should produce zero lost items and an empty payload.
    use cctx::core::context::{AttentionZone, Chunk, Context};
    let ctx = Context::new(vec![Chunk {
        index: 0,
        role: "user".into(),
        content: "the budget is fifty thousand dollars and the auth port is 8443 in production"
            .into(),
        token_count: 0,
        relevance_score: 0.5,
        attention_zone: AttentionZone::Strong,
    }]);
    let summary = "the budget is fifty thousand dollars and the auth port is 8443 in production";
    let fp = fingerprint(
        &ctx,
        &FingerprintConfig::default(),
        "test_session",
        "2026-04-30T00:00:00Z",
    );
    if fp.total_items == 0 {
        // Nothing to test — the fingerprint engine found no items.
        return;
    }
    let report = detect_loss(&fp.items, summary, &ctx, "s", "auto", 100, 100);
    let payload = build_injection(&report.items, 4096);
    assert!(payload.is_empty());
}

// ── Round-trip the LossReport schema ─────────────────────────────────────────

#[test]
fn loss_report_serialization_round_trip() {
    let (context, fp) = fingerprint_sample();
    let summary = "we use postgresql";
    let report = detect_loss(&fp.items, summary, &context, "sess", "auto", 1000, 100);

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    let parsed: LossReport = serde_json::from_str(&json).expect("round-trip parse");
    assert_eq!(parsed.session_id, report.session_id);
    assert_eq!(parsed.items.len(), report.items.len());
    assert_eq!(parsed.lost_count, report.lost_count);
}
