//! Integration tests for the cctx CLI.
//!
//! These tests run the compiled binary as a subprocess using `assert_cmd`.
//! Each test exercises a real command and checks exit code, stdout, and stderr.
//!
//! # How assert_cmd works
//!
//! `Command::cargo_bin("cctx")` builds and returns a handle to the compiled
//! binary. You chain `.arg()` calls to set CLI arguments, `.write_stdin()` to
//! pipe data, and `.assert()` to run and check the result.
//!
//! `predicates` provides composable checks like `contains("text")`,
//! `is_empty()`, and `is_match(regex)` that you pass to `.stdout()` / `.stderr()`.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: get a Command pointing at the cctx binary
// ═══════════════════════════════════════════════════════════════════════════════

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Format auto-detection
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn analyze_openai_format_autodetect() {
    // The sample_conversation fixture is OpenAI format [{role, content}].
    // analyze should auto-detect it and produce a valid JSON report.
    let output = cctx()
        .args(["analyze", "tests/fixtures/sample_conversation.json", "--format", "json"])
        .output()
        .expect("failed to run");

    assert!(output.status.success(), "exit code was not 0");

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    // OpenAI fixture has 20 messages.
    assert_eq!(report["chunk_count"], 20);
    assert!(report["total_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn analyze_anthropic_format_autodetect() {
    // Content arrays with {type: "text"} blocks → should auto-detect as Anthropic.
    let output = cctx()
        .args(["analyze", "tests/fixtures/anthropic_conversation.json", "--format", "json"])
        .output()
        .expect("failed to run");

    assert!(output.status.success(), "exit code was not 0");

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    // Anthropic fixture has 6 messages (3 user, 3 assistant).
    assert_eq!(report["chunk_count"], 6);
    assert!(report["total_tokens"].as_u64().unwrap() > 100);
}

#[test]
fn analyze_rag_chunks_autodetect() {
    // Objects with "content" + "score" but no "role" → RAG chunks.
    let output = cctx()
        .args(["analyze", "tests/fixtures/rag_chunks.json", "--format", "json"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    assert_eq!(report["chunk_count"], 10);
    assert!(report["total_tokens"].as_u64().unwrap() > 500);
}

#[test]
fn analyze_raw_text_with_explicit_format() {
    // Plain text file requires --input-format raw (or auto-detect since it's not JSON).
    let output = cctx()
        .args(["analyze", "tests/fixtures/raw_document.txt", "--format", "json"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    // Raw text → 1 chunk.
    assert_eq!(report["chunk_count"], 1);
    // ~2000 words ≈ 1500+ tokens.
    assert!(report["total_tokens"].as_u64().unwrap() > 1000);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bookend reordering: high-score chunks at beginning/end
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bookend_places_high_score_chunks_at_edges() {
    // RAG chunks have explicit scores. After bookend reordering, the chunks
    // with the highest scores (0.95, 0.92, 0.91) should end up at the
    // beginning and end positions, NOT in the middle.
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/rag_chunks.json",
            "--strategy", "bookend",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    // stdout is the optimized JSON array of messages.
    let messages: Vec<Value> = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON array");

    assert_eq!(messages.len(), 10);

    // The first and last messages should contain high-relevance content.
    // Chunk with score 0.95 has "pg_stat_activity" in it.
    // Chunk with score 0.92 has "Connection pooling in PostgreSQL".
    let first_content = messages[0]["content"].as_str().unwrap();
    let last_content = messages[9]["content"].as_str().unwrap();

    // The two highest-score chunks should occupy the first and last positions.
    let top_phrases = ["pg_stat_activity", "Connection pooling in PostgreSQL"];
    let edges = format!("{} {}", first_content, last_content);
    for phrase in &top_phrases {
        assert!(
            edges.contains(phrase),
            "Expected '{}' at an edge position, but edges were:\nFIRST: {:.80}...\nLAST:  {:.80}...",
            phrase,
            first_content,
            last_content
        );
    }

    // The lowest-score chunk (0.30, Kubernetes liveness probes) should NOT
    // be at either edge — it should be in the middle.
    let k8s_content = "Kubernetes liveness probes";
    assert!(
        !first_content.contains(k8s_content) && !last_content.contains(k8s_content),
        "Low-score Kubernetes chunk should be in the middle, not at an edge"
    );
}

#[test]
fn bookend_summary_on_stderr() {
    // The summary (reordered N chunks, health score) goes to stderr,
    // leaving stdout as clean JSON.
    cctx()
        .args([
            "optimize",
            "tests/fixtures/rag_chunks.json",
            "--strategy", "bookend",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Reordered 10 chunks"))
        .stderr(predicate::str::contains("Health score:"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stdin piping
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn stdin_pipe_analyze_openai() {
    // Pipe OpenAI JSON through stdin → auto-detect → analyze.
    let fixture = std::fs::read("tests/fixtures/sample_conversation.json")
        .expect("fixture not found");

    let output = cctx()
        .args(["analyze", "--format", "json"])
        .write_stdin(fixture)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    assert_eq!(report["chunk_count"], 20);
}

#[test]
fn stdin_pipe_analyze_raw() {
    // Pipe raw text through stdin.
    cctx()
        .args(["analyze", "--input-format", "raw", "--format", "json"])
        .write_stdin("Hello, this is a raw text input for testing purposes.")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"chunk_count\": 1"));
}

#[test]
fn stdin_pipe_optimize_produces_valid_json() {
    // Pipe RAG chunks → optimize → verify stdout is parseable JSON.
    let fixture = std::fs::read("tests/fixtures/rag_chunks.json")
        .expect("fixture not found");

    let output = cctx()
        .args(["optimize", "--strategy", "bookend"])
        .write_stdin(fixture)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let messages: Vec<Value> = serde_json::from_slice(&output.stdout)
        .expect("optimize stdout is not valid JSON");

    assert_eq!(messages.len(), 10);
    // Every message should have "role" and "content" fields (normalized to OpenAI output).
    for msg in &messages {
        assert!(msg.get("role").is_some(), "missing 'role' field in output");
        assert!(msg.get("content").is_some(), "missing 'content' field in output");
    }
}

#[test]
fn stdin_dash_is_equivalent_to_no_file() {
    // `cctx analyze -` with piped stdin should work like `cctx analyze` with stdin.
    let fixture = std::fs::read("tests/fixtures/rag_chunks.json")
        .expect("fixture not found");

    let output = cctx()
        .args(["analyze", "-", "--format", "json"])
        .write_stdin(fixture)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    assert_eq!(report["chunk_count"], 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stdout / stderr separation (pipe-friendliness)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn optimize_stdout_is_pure_json_stderr_has_summary() {
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/sample_conversation.json",
            "--strategy", "bookend",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    // stdout must be valid JSON (no log messages mixed in).
    let _: Vec<Value> = serde_json::from_slice(&output.stdout)
        .expect("stdout must be pure JSON — no log lines allowed");

    // stderr must have the human-readable summary.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Reordered"),
        "stderr should contain summary, got: {}",
        stderr
    );
}

#[test]
fn analyze_json_format_stdout_is_valid_json() {
    let output = cctx()
        .args([
            "analyze",
            "tests/fixtures/anthropic_conversation.json",
            "--format", "json",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("analyze --format json stdout must be valid JSON");

    assert!(report.get("health_score").is_some());
    assert!(report.get("dead_zone").is_some());
    assert!(report.get("duplication").is_some());
    assert!(report.get("budget").is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// --input-format override
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn input_format_override_forces_raw_on_json_file() {
    // Force --input-format raw on a JSON file → treated as 1 raw text chunk.
    let output = cctx()
        .args([
            "analyze",
            "tests/fixtures/sample_conversation.json",
            "--input-format", "raw",
            "--format", "json",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is not valid JSON");

    // Forced raw → 1 chunk (the entire JSON file as text), not 20 messages.
    assert_eq!(report["chunk_count"], 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Output to file via --output
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn output_flag_writes_file_and_stderr_confirms() {
    let dir = std::env::temp_dir().join("cctx_test_output");
    let _ = std::fs::create_dir_all(&dir);
    let outfile = dir.join("optimized.json");

    // Remove leftover from previous runs.
    let _ = std::fs::remove_file(&outfile);

    cctx()
        .args([
            "optimize",
            "tests/fixtures/rag_chunks.json",
            "--strategy", "bookend",
            "--output", outfile.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Wrote optimized context to"));

    // File should exist and contain valid JSON.
    let content = std::fs::read_to_string(&outfile).expect("output file not created");
    let messages: Vec<Value> =
        serde_json::from_str(&content).expect("output file is not valid JSON");
    assert_eq!(messages.len(), 10);

    // Cleanup.
    let _ = std::fs::remove_file(&outfile);
    let _ = std::fs::remove_dir(&dir);
}
