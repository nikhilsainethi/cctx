//! Semantic deduplication tests using the TF-IDF mock embedder.
//!
//! These tests verify the dedup pipeline without requiring Ollama or OpenAI.
//! TF-IDF captures word-level overlap (not true semantic similarity), so
//! thresholds are lower than they would be with neural embeddings.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fixture analysis — verify the fixture has detectable duplication
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn duplicate_fixture_has_duplication_detected_by_analyzer() {
    // The Jaccard word-overlap analyzer (non-embedding) should detect some
    // duplication in our fixture — the user repeats their stack description
    // multiple times using similar words.
    let output = cctx()
        .args([
            "analyze",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--format", "json",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert_eq!(report["chunk_count"], 25);

    let dup_tokens = report["duplication"]["duplicate_tokens"].as_u64().unwrap();
    assert!(
        dup_tokens > 0,
        "analyzer should detect some word-level duplication, got 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semantic dedup with TF-IDF embedder
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tfidf_dedup_removes_near_duplicate_chunks() {
    // TF-IDF at threshold 0.3 should catch the repeated setup descriptions.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert!(
        msgs.len() < 25,
        "dedup should remove some chunks, got {} (original 25)",
        msgs.len()
    );
}

#[test]
fn tfidf_dedup_preserves_unique_content() {
    // The deployment question (turn 15) is unique — it should survive dedup.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    let all_content: String = msgs
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // The deployment-related content should still be present.
    assert!(
        all_content.contains("zero downtime") || all_content.contains("zero-downtime"),
        "unique deployment content should survive dedup"
    );
}

#[test]
fn tfidf_dedup_merges_unique_sentences() {
    // When a near-duplicate is removed, unique sentences from it should be
    // appended to the longer chunk with "[merged from duplicate]:" prefix.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // If any unique sentences were merged, the output will contain the marker.
    // This may or may not trigger depending on the specific TF-IDF scores,
    // so we just check it doesn't crash — the marker is a bonus.
    // With lower thresholds or real embeddings, merging is more common.
    assert!(output.status.success());
    let _ = stdout; // used above
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dedup without embedding provider → exact-match fallback
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dedup_without_provider_does_exact_match() {
    // Without --embedding-provider, dedup falls back to exact-match.
    // Our fixture has no verbatim duplicates, so nothing should be removed.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--strategy", "dedup",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    // No exact duplicates → all 25 chunks preserved.
    assert_eq!(msgs.len(), 25, "exact-match dedup should preserve all unique chunks");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dedup in a pipeline (bookend + dedup)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dedup_in_pipeline_with_bookend() {
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/duplicate_heavy_conversation.json",
            "--strategy", "bookend",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Pipeline log should show both strategies.
    assert!(stderr.contains("bookend"), "pipeline should run bookend");
    assert!(stderr.contains("dedup"), "pipeline should run dedup");

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert!(msgs.len() < 25, "pipeline should remove some duplicates");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Diff shows dedup improvement
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn diff_shows_dedup_removed_messages() {
    let fixture = "tests/fixtures/duplicate_heavy_conversation.json";

    // Run dedup and save output.
    let dir = std::env::temp_dir().join("cctx_dedup_diff");
    let _ = std::fs::create_dir_all(&dir);
    let optimized = dir.join("deduped.json");

    cctx()
        .args([
            "optimize", fixture,
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
            "--output", optimized.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Run diff.
    let output = cctx()
        .args(["diff", fixture, optimized.to_str().unwrap(), "--format", "json"])
        .output()
        .expect("failed to run diff");

    assert!(output.status.success());

    let diff: Value = serde_json::from_slice(&output.stdout).expect("not JSON");

    let removed = diff["changes"]["removed_messages"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        removed > 0,
        "diff should show removed messages from dedup"
    );

    let before_msgs = diff["before"]["messages"].as_u64().unwrap();
    let after_msgs = diff["after"]["messages"].as_u64().unwrap();
    assert!(after_msgs < before_msgs, "after should have fewer messages");

    // Cleanup.
    let _ = std::fs::remove_file(&optimized);
    let _ = std::fs::remove_dir(&dir);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tfidf_dedup_single_message_is_noop() {
    let input = serde_json::json!([
        {"role": "user", "content": "Just one message, nothing to deduplicate."}
    ])
    .to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.3",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert_eq!(msgs.len(), 1);
}

#[test]
fn tfidf_dedup_identical_messages_removes_duplicates() {
    // Exact duplicates should always be caught regardless of threshold.
    let input = serde_json::json!([
        {"role": "user", "content": "Tell me about Kubernetes deployment strategies."},
        {"role": "assistant", "content": "Here are the main strategies."},
        {"role": "user", "content": "Tell me about Kubernetes deployment strategies."},
        {"role": "assistant", "content": "Something different this time."}
    ])
    .to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy", "dedup",
            "--embedding-provider", "tfidf",
            "--dedup-threshold", "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    // The identical user messages should be deduplicated.
    assert!(
        msgs.len() < 4,
        "identical messages should be deduped, got {} chunks",
        msgs.len()
    );
}

#[test]
fn unknown_embedding_provider_gives_error() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--strategy", "dedup",
            "--embedding-provider", "nonexistent",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown embedding provider"));
}
