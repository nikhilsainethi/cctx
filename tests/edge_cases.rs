//! Edge case tests — verify cctx never panics and always gives clean errors.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Empty input
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn empty_json_array_prints_no_context_and_exits_0() {
    // [] is a valid JSON array with zero messages.
    cctx()
        .args(["analyze"])
        .write_stdin("[]")
        .assert()
        .success() // exit 0, NOT exit 1
        .stderr(predicate::str::contains("No context to analyze"));
}

#[test]
fn empty_json_array_optimize_exits_0() {
    cctx()
        .args(["optimize", "--strategy", "bookend"])
        .write_stdin("[]")
        .assert()
        .success()
        .stderr(predicate::str::contains("No context to optimize"));
}

#[test]
fn empty_json_array_count_prints_zero() {
    cctx()
        .args(["count"])
        .write_stdin("[]")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("0"));
}

#[test]
fn empty_json_array_compress_exits_0() {
    cctx()
        .args(["compress", "--budget", "1000"])
        .write_stdin("[]")
        .assert()
        .success()
        .stderr(predicate::str::contains("No context to compress"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Single message
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn single_system_message_analyzes_cleanly() {
    let input = serde_json::json!([
        {"role": "system", "content": "You are a helpful assistant."}
    ])
    .to_string();

    let output = cctx()
        .args(["analyze", "--format", "json"])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");

    assert_eq!(report["chunk_count"], 1);
    assert!(report["total_tokens"].as_u64().unwrap() > 0);
    // Single message → nothing in dead zone → high health.
    assert!(report["health_score"].as_u64().unwrap() >= 80);
}

#[test]
fn single_message_bookend_is_noop() {
    let input = serde_json::json!([
        {"role": "system", "content": "You are a helpful assistant."}
    ])
    .to_string();

    let output = cctx()
        .args(["optimize", "--strategy", "bookend"])
        .write_stdin(input.clone())
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        msgs[0]["content"].as_str().unwrap(),
        "You are a helpful assistant."
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Massive single message (50,000+ tokens)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn massive_message_does_not_panic() {
    // ~50,000 tokens: "The quick brown fox jumps over the lazy dog. " × 5000
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    let huge = sentence.repeat(5000);
    let input = serde_json::json!([{"role": "user", "content": huge}]).to_string();

    // Count should work and return a large number.
    let output = cctx()
        .args(["count"])
        .write_stdin(input.clone())
        .output()
        .expect("failed to run count");

    assert!(output.status.success());
    let count: usize = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("not a number");
    assert!(count > 40_000, "expected >40K tokens, got {}", count);
}

#[test]
fn massive_message_structural_compression() {
    // Structural should handle a single huge message without panic.
    // No code/JSON/markdown to compress, so content passes through unchanged.
    let text = "This is a paragraph about database optimization. ".repeat(2000);
    let input = serde_json::json!([{"role": "user", "content": text}]).to_string();

    let output = cctx()
        .args(["optimize", "--strategy", "structural"])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");
    assert_eq!(msgs.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Invalid JSON → treated as raw text
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn invalid_json_auto_detects_as_raw() {
    // Not JSON at all → auto-detect falls back to raw text format.
    let input = "This is not JSON, it's just a paragraph of text.\nWith multiple lines.";

    let output = cctx()
        .args(["analyze", "--format", "json"])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");

    // Raw text → 1 chunk.
    assert_eq!(report["chunk_count"], 1);
    assert!(report["total_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn malformed_json_does_not_crash() {
    // Truncated JSON — parse fails, falls back to raw.
    let input = r#"[{"role": "user", "content": "hello"#; // missing closing

    let output = cctx()
        .args(["analyze", "--format", "json"])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).expect("stdout not valid JSON");
    assert_eq!(report["chunk_count"], 1); // treated as raw text
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Binary / non-UTF-8 file
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn binary_file_gives_clean_error() {
    let dir = std::env::temp_dir().join("cctx_edge_binary");
    let _ = std::fs::create_dir_all(&dir);
    let binfile = dir.join("binary.bin");

    // Write bytes that are not valid UTF-8.
    std::fs::write(&binfile, b"\x80\x81\x82\xff\xfe\x00\xc0\xc1").unwrap();

    cctx()
        .args(["analyze", binfile.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("non-UTF-8"));

    // Cleanup.
    let _ = std::fs::remove_file(&binfile);
    let _ = std::fs::remove_dir(&dir);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. --budget 0
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn budget_zero_on_optimize_gives_error() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--strategy",
            "bookend",
            "--budget",
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Budget must be a positive number"));
}

#[test]
fn budget_zero_on_compress_gives_error() {
    cctx()
        .args([
            "compress",
            "tests/fixtures/sample_conversation.json",
            "--budget",
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Budget must be a positive number"));
}

#[test]
fn budget_negative_is_rejected_by_clap() {
    // --budget takes usize, so "-5" is rejected by clap as an unexpected
    // argument (the leading dash makes clap think it's a flag, not a value).
    cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--budget",
            "-5",
        ])
        .assert()
        .failure();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Missing file
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn missing_file_gives_clean_error() {
    cctx()
        .args(["analyze", "nonexistent/path/to/file.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("File not found"));
}

#[test]
fn missing_file_on_optimize() {
    cctx()
        .args(["optimize", "does_not_exist.json", "--strategy", "bookend"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("File not found"));
}

#[test]
fn missing_file_on_diff() {
    cctx()
        .args(["diff", "missing_a.json", "missing_b.json"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("File not found").or(predicate::str::contains("Cannot read")),
        );
}
