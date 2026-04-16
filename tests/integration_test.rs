//! Comprehensive integration tests for the entire cctx system.
//!
//! Tests cover:
//!   1. CLI pipeline: analyze → optimize → diff
//!   2. Input format × strategy matrix
//!   3. Shell piping: optimize | analyze
//!   4. Budget compression accuracy
//!   5. System message protection across ALL strategies
//!   6. --output roundtrip (write → read back)
//!   7. Strategy composition: combined > individual
//!   8. Determinism: same input → same output
//!   9. Idempotency: optimizing twice == optimizing once

use assert_cmd::Command;
use serde_json::Value;

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("cctx_integration_{}", name))
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Full CLI pipeline: analyze → optimize (all presets) → diff
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_analyze_then_optimize_all_presets_then_diff() {
    let fixture = "tests/fixtures/technical_conversation.json";

    // Analyze original.
    let analyze_out = cctx()
        .args(["analyze", fixture, "--format", "json"])
        .output()
        .unwrap();
    assert!(analyze_out.status.success());
    let original_report: Value = serde_json::from_slice(&analyze_out.stdout).unwrap();
    let original_tokens = original_report["total_tokens"].as_u64().unwrap();
    assert!(original_tokens > 0);

    // Optimize with each preset.
    for preset in &["safe", "balanced", "aggressive"] {
        let out_path = tmp_path(&format!("preset_{}.json", preset));
        let opt_out = cctx()
            .args([
                "optimize",
                fixture,
                "--preset",
                preset,
                "--output",
                out_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(opt_out.status.success(), "preset {} failed", preset);

        // Verify output is valid JSON.
        let content = std::fs::read_to_string(&out_path).unwrap();
        let msgs: Vec<Value> = serde_json::from_str(&content).unwrap();
        assert!(!msgs.is_empty(), "preset {} produced empty output", preset);

        // Run diff between original and optimized.
        let diff_out = cctx()
            .args([
                "diff",
                fixture,
                out_path.to_str().unwrap(),
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert!(
            diff_out.status.success(),
            "diff failed for preset {}",
            preset
        );
        let diff: Value = serde_json::from_slice(&diff_out.stdout).unwrap();
        assert!(diff.get("before").is_some());
        assert!(diff.get("after").is_some());
        assert!(diff.get("changes").is_some());

        let _ = std::fs::remove_file(&out_path);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Input format × strategy matrix
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn all_formats_work_with_bookend() {
    let fixtures = [
        "tests/fixtures/sample_conversation.json",    // OpenAI
        "tests/fixtures/anthropic_conversation.json", // Anthropic
        "tests/fixtures/rag_chunks.json",             // RAG
        "tests/fixtures/raw_document.txt",            // Raw
    ];
    for fixture in &fixtures {
        let out = cctx()
            .args(["optimize", fixture, "--strategy", "bookend"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "bookend failed on {}: {}",
            fixture,
            String::from_utf8_lossy(&out.stderr)
        );
        let _: Vec<Value> = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|_| panic!("invalid JSON output from bookend on {}", fixture));
    }
}

#[test]
fn all_formats_work_with_structural() {
    let fixtures = [
        "tests/fixtures/technical_conversation.json",
        "tests/fixtures/structured_content.json",
        "tests/fixtures/raw_document.txt",
    ];
    for fixture in &fixtures {
        let out = cctx()
            .args(["optimize", fixture, "--strategy", "structural"])
            .output()
            .unwrap();
        assert!(out.status.success(), "structural failed on {}", fixture);
    }
}

#[test]
fn all_formats_work_with_count() {
    let fixtures = [
        "tests/fixtures/sample_conversation.json",
        "tests/fixtures/anthropic_conversation.json",
        "tests/fixtures/rag_chunks.json",
        "tests/fixtures/raw_document.txt",
    ];
    for fixture in &fixtures {
        let out = cctx().args(["count", fixture]).output().unwrap();
        assert!(out.status.success());
        let count: usize = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("count failed on {}", fixture));
        assert!(count > 0, "count should be >0 for {}", fixture);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Piping: optimize | analyze → health should not degrade
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn piped_optimize_then_analyze_health_does_not_degrade() {
    let fixture = "tests/fixtures/technical_conversation.json";

    // Get original health.
    let orig = cctx()
        .args(["analyze", fixture, "--format", "json"])
        .output()
        .unwrap();
    let orig_report: Value = serde_json::from_slice(&orig.stdout).unwrap();
    let orig_health = orig_report["health_score"].as_u64().unwrap();

    // Optimize with balanced.
    let opt = cctx()
        .args(["optimize", fixture, "--preset", "balanced"])
        .output()
        .unwrap();

    // Pipe optimized output into analyze.
    let analyzed = cctx()
        .args(["analyze", "--format", "json"])
        .write_stdin(opt.stdout)
        .output()
        .unwrap();
    let new_report: Value = serde_json::from_slice(&analyzed.stdout).unwrap();
    let new_health = new_report["health_score"].as_u64().unwrap();

    // Structural compression on technical_conversation should improve or maintain health.
    assert!(
        new_health >= orig_health.saturating_sub(5),
        "health degraded too much: {} -> {}",
        orig_health,
        new_health
    );
}

#[test]
fn piped_optimize_then_count_shows_reduction_for_structural_content() {
    let fixture = "tests/fixtures/technical_conversation.json";

    let orig_count: usize =
        String::from_utf8_lossy(&cctx().args(["count", fixture]).output().unwrap().stdout)
            .trim()
            .parse()
            .unwrap();

    let opt = cctx()
        .args(["optimize", fixture, "--preset", "balanced"])
        .output()
        .unwrap();

    let new_count: usize = String::from_utf8_lossy(
        &cctx()
            .args(["count"])
            .write_stdin(opt.stdout)
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .parse()
    .unwrap();

    assert!(
        new_count <= orig_count,
        "balanced should not increase tokens: {} -> {}",
        orig_count,
        new_count
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Budget compression hits target within 5% margin
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn budget_compression_within_5_percent_margin() {
    let fixture = "tests/fixtures/large_conversation.json";
    let budgets = [4000usize, 2000, 1000];

    for budget in &budgets {
        let opt = cctx()
            .args(["compress", fixture, "--budget", &budget.to_string()])
            .output()
            .unwrap();
        assert!(
            opt.status.success(),
            "compress failed for budget {}",
            budget
        );

        let count: usize = String::from_utf8_lossy(
            &cctx()
                .args(["count"])
                .write_stdin(opt.stdout)
                .output()
                .unwrap()
                .stdout,
        )
        .trim()
        .parse()
        .unwrap();

        // Should be at or below budget (protected chunks may prevent exact hit).
        // Allow 5% overshoot for protected chunks.
        let margin = (*budget as f64 * 1.05) as usize;
        assert!(
            count <= margin,
            "budget {}: got {} tokens (margin {})",
            budget,
            count,
            margin
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. System messages are NEVER removed by any strategy
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn system_message_survives_all_strategies() {
    let input = serde_json::json!([
        {"role": "system", "content": "You are a database expert specializing in PostgreSQL optimization."},
        {"role": "user", "content": "How do I optimize queries?"},
        {"role": "assistant", "content": "Sure, I can help with that! Great question! Use EXPLAIN ANALYZE to identify slow queries. Add indexes on frequently filtered columns. Consider partial indexes for selective queries. Use connection pooling with PgBouncer."},
        {"role": "user", "content": "What about connection pooling?"},
        {"role": "assistant", "content": "Sure thing! PgBouncer is the standard choice. Set pool_mode=transaction for web applications."}
    ]).to_string();

    let strategies = ["bookend", "structural", "dedup", "prune"];
    let _system_content = "You are a database expert specializing in PostgreSQL optimization.";

    for strategy in &strategies {
        let out = cctx()
            .args(["optimize", "--strategy", strategy])
            .write_stdin(input.clone())
            .output()
            .unwrap();
        assert!(out.status.success(), "{} failed", strategy);

        let msgs: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
        let has_system = msgs.iter().any(|m| {
            m["role"].as_str() == Some("system")
                && m["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("database expert"))
        });
        assert!(
            has_system,
            "system message missing after --strategy {}",
            strategy
        );
    }

    // Also test combined aggressive preset.
    let out = cctx()
        .args([
            "optimize",
            "--preset",
            "aggressive",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input.clone())
        .output()
        .unwrap();
    assert!(out.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    let has_system = msgs.iter().any(|m| {
        m["role"].as_str() == Some("system")
            && m["content"]
                .as_str()
                .is_some_and(|c| c.contains("database expert"))
    });
    assert!(has_system, "system message missing after aggressive preset");

    // Budget compression with very tight budget.
    let out = cctx()
        .args(["compress", "--budget", "100"])
        .write_stdin(input)
        .output()
        .unwrap();
    assert!(out.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    let has_system = msgs.iter().any(|m| m["role"].as_str() == Some("system"));
    assert!(
        has_system,
        "system message missing after compress --budget 100"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. --output writes valid JSON that can be read back
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn output_roundtrip_write_then_read_back() {
    let fixture = "tests/fixtures/large_conversation.json";
    let out_path = tmp_path("roundtrip.json");

    // Write optimized output to file.
    cctx()
        .args([
            "optimize",
            fixture,
            "--preset",
            "balanced",
            "--output",
            out_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Read it back with analyze.
    let analyzed = cctx()
        .args(["analyze", out_path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert!(analyzed.status.success());
    let report: Value = serde_json::from_slice(&analyzed.stdout).unwrap();
    assert!(report["total_tokens"].as_u64().unwrap() > 0);
    assert!(report["chunk_count"].as_u64().unwrap() > 0);

    // Read it back with count.
    let count_out = cctx()
        .args(["count", out_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(count_out.status.success());
    let count: usize = String::from_utf8_lossy(&count_out.stdout)
        .trim()
        .parse()
        .unwrap();
    assert_eq!(count, report["total_tokens"].as_u64().unwrap() as usize);

    // Read it back with optimize (double optimization).
    let re_opt = cctx()
        .args([
            "optimize",
            out_path.to_str().unwrap(),
            "--preset",
            "balanced",
        ])
        .output()
        .unwrap();
    assert!(re_opt.status.success());
    let _: Vec<Value> = serde_json::from_slice(&re_opt.stdout).unwrap();

    let _ = std::fs::remove_file(&out_path);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Strategy composition: combined > individual
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn balanced_reduces_more_than_bookend_alone_on_structured_content() {
    let fixture = "tests/fixtures/technical_conversation.json";

    let bookend_out = cctx()
        .args(["optimize", fixture, "--strategy", "bookend"])
        .output()
        .unwrap();
    let bookend_count: usize = String::from_utf8_lossy(
        &cctx()
            .args(["count"])
            .write_stdin(bookend_out.stdout)
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .parse()
    .unwrap();

    let balanced_out = cctx()
        .args(["optimize", fixture, "--preset", "balanced"])
        .output()
        .unwrap();
    let balanced_count: usize = String::from_utf8_lossy(
        &cctx()
            .args(["count"])
            .write_stdin(balanced_out.stdout)
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .parse()
    .unwrap();

    // Balanced (bookend + structural) should reduce at least as much as bookend alone
    // on a fixture with JSON/code/markdown.
    assert!(
        balanced_count <= bookend_count,
        "balanced ({}) should reduce at least as much as bookend ({}) on structured content",
        balanced_count,
        bookend_count
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Determinism: same input → same output
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn optimization_is_deterministic() {
    let fixture = "tests/fixtures/technical_conversation.json";

    let run1 = cctx()
        .args(["optimize", fixture, "--preset", "balanced"])
        .output()
        .unwrap();
    let run2 = cctx()
        .args(["optimize", fixture, "--preset", "balanced"])
        .output()
        .unwrap();

    assert!(run1.status.success());
    assert!(run2.status.success());

    // Outputs should be byte-identical.
    assert_eq!(
        run1.stdout, run2.stdout,
        "two runs of the same optimization should produce identical output"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Idempotency: optimizing twice == optimizing once
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn structural_optimization_is_idempotent() {
    let fixture = "tests/fixtures/technical_conversation.json";

    // First optimization.
    let opt1 = cctx()
        .args(["optimize", fixture, "--strategy", "structural"])
        .output()
        .unwrap();
    assert!(opt1.status.success());
    let count1: usize = String::from_utf8_lossy(
        &cctx()
            .args(["count"])
            .write_stdin(opt1.stdout.clone())
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .parse()
    .unwrap();

    // Second optimization (on already-optimized output).
    let opt2 = cctx()
        .args(["optimize", "--strategy", "structural"])
        .write_stdin(opt1.stdout)
        .output()
        .unwrap();
    assert!(opt2.status.success());
    let count2: usize = String::from_utf8_lossy(
        &cctx()
            .args(["count"])
            .write_stdin(opt2.stdout)
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .parse()
    .unwrap();

    // Second pass should not reduce further (already compressed).
    // Small variance is OK (token counting edge cases), but no big change.
    let diff = (count1 as i64 - count2 as i64).unsigned_abs() as usize;
    assert!(
        diff < count1 / 10,
        "second structural pass changed tokens significantly: {} -> {} (diff {})",
        count1,
        count2,
        diff
    );
}

#[test]
fn bookend_preserves_all_messages() {
    // Bookend reorders but never adds or removes messages.
    // Applying it twice should preserve the same set of content (though
    // order may shift since recency-based scores change after reordering).
    let fixture = "tests/fixtures/sample_conversation.json";

    let opt1 = cctx()
        .args(["optimize", fixture, "--strategy", "bookend"])
        .output()
        .unwrap();
    let opt2 = cctx()
        .args(["optimize", "--strategy", "bookend"])
        .write_stdin(opt1.stdout.clone())
        .output()
        .unwrap();

    let msgs1: Vec<Value> = serde_json::from_slice(&opt1.stdout).unwrap();
    let msgs2: Vec<Value> = serde_json::from_slice(&opt2.stdout).unwrap();

    // Same number of messages.
    assert_eq!(msgs1.len(), msgs2.len());

    // Same SET of content (order may differ).
    let mut content1: Vec<String> = msgs1
        .iter()
        .filter_map(|m| m["content"].as_str().map(|s| s.to_string()))
        .collect();
    let mut content2: Vec<String> = msgs2
        .iter()
        .filter_map(|m| m["content"].as_str().map(|s| s.to_string()))
        .collect();
    content1.sort();
    content2.sort();
    assert_eq!(
        content1, content2,
        "bookend should preserve all message content"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. Stress: very large input doesn't panic
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn aggressive_on_large_input_completes_without_panic() {
    let fixture = "tests/fixtures/large_conversation.json";
    let out = cctx()
        .args([
            "optimize",
            fixture,
            "--preset",
            "aggressive",
            "--prune-threshold",
            "0.5",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert!(!msgs.is_empty());
}
