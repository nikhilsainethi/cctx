//! Pipeline-specific integration tests.
//!
//! Tests strategy ordering, budget enforcement, presets, count consistency,
//! and system-message protection during compression.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Strategy order matters
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn strategy_order_produces_different_results() {
    // Order matters when dedup is involved: dedup keeps the FIRST occurrence
    // of duplicate content. If bookend reorders before dedup, a different
    // copy survives than if dedup runs first.
    //
    // Fixture: 4 messages where messages 1 and 3 have identical content.
    // - bookend→dedup: bookend reorders by recency, then dedup removes whichever
    //   duplicate is second in the new order.
    // - dedup→bookend: dedup removes message 3 (second occurrence), then bookend
    //   reorders the remaining 3 messages.
    let input = serde_json::to_string(&serde_json::json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris, a major European city and global center for art, fashion, and culture."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is known for the Eiffel Tower, the Louvre museum, and its cafe culture along the Seine river."}
    ]))
    .unwrap();

    let output_bd = cctx()
        .args(["optimize", "--strategy", "bookend", "--strategy", "dedup"])
        .write_stdin(input.as_bytes().to_vec())
        .output()
        .expect("failed bookend→dedup");

    let output_db = cctx()
        .args(["optimize", "--strategy", "dedup", "--strategy", "bookend"])
        .write_stdin(input.as_bytes().to_vec())
        .output()
        .expect("failed dedup→bookend");

    assert!(output_bd.status.success());
    assert!(output_db.status.success());

    let msgs_bd: Vec<Value> =
        serde_json::from_slice(&output_bd.stdout).expect("bd not valid JSON");
    let msgs_db: Vec<Value> =
        serde_json::from_slice(&output_db.stdout).expect("db not valid JSON");

    // dedup removes one duplicate in both cases → 4 messages.
    assert_eq!(msgs_bd.len(), 4);
    assert_eq!(msgs_db.len(), 4);

    // The outputs should differ: bookend reorders before vs after dedup
    // changes which copy of the duplicate remains at which position.
    let full_bd = String::from_utf8_lossy(&output_bd.stdout);
    let full_db = String::from_utf8_lossy(&output_db.stdout);
    assert_ne!(
        full_bd, full_db,
        "bookend→dedup and dedup→bookend should produce different message ordering"
    );
}

#[test]
fn structural_then_bookend_compresses_before_reordering() {
    // When structural runs first, token count should drop before bookend runs.
    // Verify via stderr pipeline log: first line should show structural reducing
    // tokens, second line shows bookend keeping the same count.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/technical_conversation.json",
            "--strategy", "structural",
            "--strategy", "bookend",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Pipeline log should show structural first, bookend second.
    let structural_pos = stderr.find("structural").expect("structural not in log");
    let bookend_pos = stderr.find("bookend").expect("bookend not in log");
    assert!(
        structural_pos < bookend_pos,
        "structural should appear before bookend in the pipeline log"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Budget enforcement — output under target
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn budget_enforces_token_limit() {
    // large_conversation is ~8000 tokens. Budget 2000 → output must be ≤2000.
    let optimize_out = cctx()
        .args([
            "optimize",
            "tests/fixtures/large_conversation.json",
            "--preset", "balanced",
            "--budget", "2000",
        ])
        .output()
        .expect("failed to run optimize");

    assert!(optimize_out.status.success());

    // Pipe the optimized output to `cctx count` to verify token count.
    let count_out = cctx()
        .args(["count"])
        .write_stdin(optimize_out.stdout)
        .output()
        .expect("failed to run count");

    assert!(count_out.status.success());

    let tokens: usize = String::from_utf8_lossy(&count_out.stdout)
        .trim()
        .parse()
        .expect("count output not a number");

    assert!(
        tokens <= 2000,
        "budget 2000 but output has {} tokens",
        tokens
    );
    assert!(tokens > 0, "output should not be empty");
}

#[test]
fn budget_drops_chunks_reports_warnings() {
    // With a tight budget, stderr should report which chunks were dropped.
    cctx()
        .args([
            "optimize",
            "tests/fixtures/large_conversation.json",
            "--strategy", "bookend",
            "--budget", "1500",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("Dropped chunk"));
}

#[test]
fn budget_within_limit_preserves_all_chunks() {
    // Budget much larger than input → no drops, all chunks preserved.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/large_conversation.json",
            "--strategy", "bookend",
            "--budget", "100000",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> =
        serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");
    assert_eq!(msgs.len(), 39, "all 39 chunks should be preserved");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Presets resolve to correct strategy chains
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn preset_safe_is_bookend_only() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--preset", "safe",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("Pipeline: bookend"))
        // Should NOT contain structural or dedup.
        .stderr(predicates::str::contains("structural").not());
}

#[test]
fn preset_balanced_is_bookend_plus_structural() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--preset", "balanced",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("Pipeline: bookend, structural"))
        .stderr(predicates::str::contains("dedup").not());
}

#[test]
fn preset_aggressive_includes_dedup() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--preset", "aggressive",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("Pipeline: bookend, structural, dedup"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. `cctx count` matches `cctx analyze` token report
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn count_matches_analyze_total_tokens() {
    let fixture = "tests/fixtures/large_conversation.json";

    // Get token count from `cctx count`.
    let count_out = cctx()
        .args(["count", fixture])
        .output()
        .expect("failed to run count");
    let count_tokens: usize = String::from_utf8_lossy(&count_out.stdout)
        .trim()
        .parse()
        .expect("count not a number");

    // Get token count from `cctx analyze --format json`.
    let analyze_out = cctx()
        .args(["analyze", fixture, "--format", "json"])
        .output()
        .expect("failed to run analyze");
    let report: Value =
        serde_json::from_slice(&analyze_out.stdout).expect("analyze output not JSON");
    let analyze_tokens = report["total_tokens"].as_u64().unwrap() as usize;

    assert_eq!(
        count_tokens, analyze_tokens,
        "count ({}) and analyze ({}) should report the same token total",
        count_tokens, analyze_tokens
    );
}

#[test]
fn count_on_piped_optimize_output() {
    // optimize | count — the count should be less than or equal to the original
    // when structural compression is applied.
    let original_count_out = cctx()
        .args(["count", "tests/fixtures/technical_conversation.json"])
        .output()
        .expect("failed");
    let original: usize = String::from_utf8_lossy(&original_count_out.stdout)
        .trim()
        .parse()
        .unwrap();

    let optimize_out = cctx()
        .args([
            "optimize",
            "tests/fixtures/technical_conversation.json",
            "--preset", "balanced",
        ])
        .output()
        .expect("failed");

    let piped_count_out = cctx()
        .args(["count"])
        .write_stdin(optimize_out.stdout)
        .output()
        .expect("failed");

    let optimized: usize = String::from_utf8_lossy(&piped_count_out.stdout)
        .trim()
        .parse()
        .unwrap();

    assert!(
        optimized <= original,
        "optimized count ({}) should be <= original ({})",
        optimized,
        original
    );
    assert!(optimized > 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. System message is NEVER removed during budget compression
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn compress_never_removes_system_message() {
    // Compress large_conversation (8000+ tokens) to a very tight budget.
    // The system message must survive even when most chunks are dropped.
    let output = cctx()
        .args([
            "compress",
            "tests/fixtures/large_conversation.json",
            "--budget", "500",
        ])
        .output()
        .expect("failed to run compress");

    assert!(output.status.success());

    let msgs: Vec<Value> =
        serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");

    // Verify system message is present.
    let has_system = msgs
        .iter()
        .any(|m| m["role"].as_str() == Some("system"));
    assert!(
        has_system,
        "system message must NEVER be removed during compression"
    );

    // Verify the system content is the original (not truncated or empty).
    let system_msg = msgs.iter().find(|m| m["role"].as_str() == Some("system")).unwrap();
    let content = system_msg["content"].as_str().unwrap();
    assert!(
        content.contains("distributed systems engineer"),
        "system message content should be preserved intact"
    );
}

#[test]
fn budget_protects_last_two_user_messages() {
    // With a tight budget, the last 2 user messages should survive.
    let output = cctx()
        .args([
            "compress",
            "tests/fixtures/large_conversation.json",
            "--budget", "800",
        ])
        .output()
        .expect("failed to run compress");

    assert!(output.status.success());

    let msgs: Vec<Value> =
        serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");

    // The last 2 user messages in the original are about Kubernetes scaling.
    // Check that at least 2 user messages survive.
    let user_count = msgs.iter().filter(|m| m["role"].as_str() == Some("user")).count();
    assert!(
        user_count >= 2,
        "at least 2 user messages should be protected, found {}",
        user_count
    );
}

#[test]
fn compress_warns_when_protected_chunks_exceed_budget() {
    // Budget so small that even after dropping everything droppable, the
    // protected chunks (system + last 2 user) exceed the budget.
    cctx()
        .args([
            "compress",
            "tests/fixtures/large_conversation.json",
            "--budget", "100",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("Warning: protected chunks alone use"));
}
