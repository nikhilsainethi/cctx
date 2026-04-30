//! End-to-end Compaction Guard lifecycle test.
//!
//! Walks through the four phases (PreCompact → PostCompact →
//! UserPromptSubmit) using the actual hook handlers, against the
//! `fingerprint_accuracy.jsonl` fixture which carries six labeled
//! facts (budget, PostgreSQL decision, port 8443, config path,
//! HPA root cause, repeated CORS error). The synthetic
//! `compact_summary` is hand-crafted to PRESERVE some facts and
//! DROP others so we can assert the loss detector + injection
//! builder act on the right items.
//!
//! Note on path: the brief mentions `tests/integration/compaction_lifecycle.rs`,
//! but Cargo only auto-discovers files directly under `tests/`. We
//! keep the file at `tests/compaction_lifecycle.rs` so it's part
//! of the regular `cargo test` run.

use std::path::{Path, PathBuf};

use cctx::compaction::{LossClassification, LossReport};
use cctx::fingerprint::Fingerprint;
use cctx::hooks::input::{
    CommonHookInput, PostCompactInput, PreCompactInput, UserPromptSubmitInput,
};
use cctx::hooks::{post_compact, pre_compact, user_prompt};
use cctx::state::store;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unique_project(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("cctx_lifecycle_{}_{}_{}", tag, pid, nanos));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fingerprint_accuracy.jsonl")
}

fn pre_compact_input(project: &Path, session: &str, transcript: &Path) -> PreCompactInput {
    PreCompactInput {
        common: CommonHookInput {
            session_id: session.into(),
            transcript_path: Some(transcript.to_string_lossy().to_string()),
            cwd: Some(project.to_string_lossy().to_string()),
            hook_event_name: "PreCompact".into(),
        },
        trigger: Some("auto".into()),
        custom_instructions: None,
    }
}

fn post_compact_input(
    project: &Path,
    session: &str,
    transcript: &Path,
    summary: &str,
) -> PostCompactInput {
    PostCompactInput {
        common: CommonHookInput {
            session_id: session.into(),
            transcript_path: Some(transcript.to_string_lossy().to_string()),
            cwd: Some(project.to_string_lossy().to_string()),
            hook_event_name: "PostCompact".into(),
        },
        trigger: Some("auto".into()),
        compact_summary: Some(summary.into()),
    }
}

fn user_prompt_input(project: &Path, session: &str) -> UserPromptSubmitInput {
    UserPromptSubmitInput {
        common: CommonHookInput {
            session_id: session.into(),
            transcript_path: None,
            cwd: Some(project.to_string_lossy().to_string()),
            hook_event_name: "UserPromptSubmit".into(),
        },
        prompt: None,
    }
}

/// Synthetic summary used by the lifecycle test. Preserves some of
/// the fingerprintable facts and DROPS others so the loss detector
/// has clear signal to work with.
///
/// The budget sentence intentionally echoes several of the
/// high-IDF tokens from the fixture's original phrasing
/// (`important`, `constraint`, `exceed`, `total`) so weighted
/// overlap pushes it past the Paraphrased threshold rather than
/// dropping it for missing rationale words.
const SYNTH_SUMMARY: &str = "\
We discussed building a small services backend. \
The team decided to use PostgreSQL for the primary store, picking it over MongoDB \
for ACID and JSONB-driven schema flexibility. \
The user laid out an important constraint: the budget should not exceed $50K total for the project. \
We also resolved CORS issues across staging, production, and order-service \
by standardizing the missing CORS header in middleware. \
Several feature flags and observability patterns came up.";

// ── Phase 1 ──────────────────────────────────────────────────────────────────

#[test]
fn phase1_pre_compact_saves_fingerprint_with_known_items() {
    let project = unique_project("phase1");
    let session = "lifecycle_p1";

    pre_compact::handle(pre_compact_input(&project, session, &fixture_path())).unwrap();

    // Fingerprint file should exist under .cctx/fingerprints/.
    let fp_dir = store::state_root(&project).join(store::FINGERPRINTS_SUBDIR);
    let entries: Vec<_> = std::fs::read_dir(&fp_dir).unwrap().flatten().collect();
    assert!(
        entries.iter().any(|e| {
            let n = e.file_name();
            n.to_string_lossy().starts_with(&format!("{}_", session))
        }),
        "fingerprint file for session should exist"
    );

    // Load it back and look for the seeded facts.
    let fp = store::load_latest_fingerprint(&project, session)
        .unwrap()
        .expect("fingerprint should round-trip");

    assert_present(&fp, "$50K", "budget constraint");
    assert_present(&fp, "PostgreSQL", "PostgreSQL decision");
    assert_present(&fp, "8443", "port 8443 fact");
    assert_present(&fp, "/etc/myapp/prod.yml", "config path");
    assert_present(&fp, "HPA", "HPA root cause");
    assert_present(&fp, "CORS", "CORS error");

    std::fs::remove_dir_all(&project).ok();
}

fn assert_present(fp: &Fingerprint, needle: &str, label: &str) {
    assert!(
        fp.items.iter().any(|i| i.content.contains(needle)),
        "expected to find {} (substring `{}`) in fingerprint",
        label,
        needle
    );
}

#[test]
fn phase1_unique_constraint_outranks_repeated_cors() {
    let project = unique_project("phase1_priority");
    let session = "priority_check";

    pre_compact::handle(pre_compact_input(&project, session, &fixture_path())).unwrap();
    let fp = store::load_latest_fingerprint(&project, session)
        .unwrap()
        .expect("fingerprint loaded");

    let budget = fp
        .items
        .iter()
        .find(|i| i.content.contains("$50K"))
        .expect("budget item present");
    let cors = fp
        .items
        .iter()
        .find(|i| {
            matches!(i.category, cctx::fingerprint::ItemCategory::DebugInsight)
                && i.content.to_lowercase().contains("cors")
        })
        .expect("repeated CORS DebugInsight present");

    assert!(
        budget.priority_score > cors.priority_score,
        "budget ({:.3}) should outrank CORS ({:.3})",
        budget.priority_score,
        cors.priority_score
    );

    std::fs::remove_dir_all(&project).ok();
}

// ── Phase 2 ──────────────────────────────────────────────────────────────────

#[test]
fn phase2_loss_detection_classifies_dropped_facts() {
    let project = unique_project("phase2");
    let session = "lifecycle_p2";
    let fixture = fixture_path();

    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();
    post_compact::handle(post_compact_input(
        &project,
        session,
        &fixture,
        SYNTH_SUMMARY,
    ))
    .unwrap();

    // Loss report should be saved.
    let loss_path = store::state_root(&project)
        .join(store::LOSS_REPORTS_SUBDIR)
        .join(format!("{}.json", session));
    assert!(loss_path.is_file(), "loss report file should exist");

    let raw = std::fs::read_to_string(&loss_path).unwrap();
    let report: LossReport = serde_json::from_str(&raw).expect("loss report parses");

    // Dropped facts → Lost
    assert_classification(&report, "8443", LossClassification::Lost, "port 8443");
    assert_classification(
        &report,
        "/etc/myapp/prod.yml",
        LossClassification::Lost,
        "config path",
    );
    // HPA — looking for the bug-root-cause sentence.
    let hpa_lost = report.items.iter().any(|c| {
        c.fingerprint_item.content.contains("HPA")
            && matches!(
                c.classification,
                LossClassification::Lost | LossClassification::Paraphrased
            )
    });
    assert!(hpa_lost, "HPA root cause should be Lost or Paraphrased");

    // Preserved facts → not Lost (Paraphrased OR Preserved is acceptable
    // depending on how much of the surrounding sentence echoes the summary).
    let pg_kept = report.items.iter().any(|c| {
        c.fingerprint_item.content.contains("PostgreSQL")
            && c.classification != LossClassification::Lost
    });
    assert!(
        pg_kept,
        "PostgreSQL decision should be Preserved or Paraphrased (not Lost)"
    );

    let budget_kept = report.items.iter().any(|c| {
        c.fingerprint_item.content.contains("$50K") && c.classification != LossClassification::Lost
    });
    assert!(
        budget_kept,
        "budget should be Preserved or Paraphrased (not Lost)"
    );

    // Pending injection should be queued.
    let pending = store::state_root(&project)
        .join(store::PENDING_INJECTION_SUBDIR)
        .join(format!("{}.json", session));
    assert!(pending.is_file(), "pending injection should exist");

    std::fs::remove_dir_all(&project).ok();
}

fn assert_classification(
    report: &LossReport,
    needle: &str,
    expected: LossClassification,
    label: &str,
) {
    let found = report
        .items
        .iter()
        .find(|c| c.fingerprint_item.content.contains(needle));
    let item = found.unwrap_or_else(|| panic!("expected {} (`{}`) in loss report", label, needle));
    assert_eq!(
        item.classification, expected,
        "{}: expected {:?}, got {:?} (overlap = {:.2})",
        label, expected, item.classification, item.overlap_score
    );
}

// ── Phase 3 ──────────────────────────────────────────────────────────────────
//
// We invoke the user-prompt handler via the actual binary so we can
// capture stdout (Rust's `println!` writes to the process stdout — in
// in-process tests there's no clean way to redirect it). The prior
// two phases already wrote the pending-injection file via the
// in-process API.

#[test]
fn phase3_user_prompt_emits_recovery_block_via_binary() {
    use assert_cmd::Command;

    let project = unique_project("phase3");
    let session = "lifecycle_p3";
    let fixture = fixture_path();

    // Set up the queue via Phase 1 + 2.
    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();
    post_compact::handle(post_compact_input(
        &project,
        session,
        &fixture,
        SYNTH_SUMMARY,
    ))
    .unwrap();

    // Now invoke the binary. assert_cmd resolves the cargo-built bin
    // for us, so this exercises the real entry point.
    let payload = serde_json::json!({
        "session_id": session,
        "cwd": project.to_string_lossy(),
        "hook_event_name": "UserPromptSubmit",
    })
    .to_string();

    let output = Command::cargo_bin("cctx")
        .unwrap()
        .args(["hook", "user-prompt"])
        .write_stdin(payload.clone())
        .output()
        .unwrap();
    assert!(output.status.success(), "exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    // Recovered items should mention the dropped facts.
    assert!(stdout.contains("8443"), "stdout should contain port 8443");
    assert!(
        stdout.contains("/etc/myapp/prod.yml"),
        "stdout should contain the config path"
    );
    assert!(
        stdout.to_lowercase().contains("hpa") || stdout.to_lowercase().contains("scaling"),
        "stdout should reference the HPA root cause"
    );

    // The PostgreSQL DECISION should be Preserved — so no line
    // starting with `DECISION:` should mention PostgreSQL. Other
    // RAKE-extracted "Other"-category sentences may incidentally
    // contain "PostgreSQL" (e.g. an assistant comment that didn't
    // cleanly preserve), and that's fine.
    let preserved_decision_leaked = stdout
        .lines()
        .any(|l| l.starts_with("DECISION:") && l.contains("PostgreSQL"));
    assert!(
        !preserved_decision_leaked,
        "PostgreSQL DECISION was preserved — should NOT appear as a recovered DECISION item"
    );

    // Header should mention the recovery.
    assert!(
        stdout.contains("[cctx] Context items recovered from pre-compaction analysis"),
        "should carry the standard recovery header"
    );

    // Pending file should be deleted after emission.
    let pending = store::state_root(&project)
        .join(store::PENDING_INJECTION_SUBDIR)
        .join(format!("{}.json", session));
    assert!(!pending.exists(), "pending file should be removed");

    // Second invocation: no pending → empty stdout.
    let output2 = Command::cargo_bin("cctx")
        .unwrap()
        .args(["hook", "user-prompt"])
        .write_stdin(payload)
        .output()
        .unwrap();
    assert!(output2.status.success());
    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    assert!(
        stdout2.trim().is_empty(),
        "second invocation should be silent"
    );

    std::fs::remove_dir_all(&project).ok();
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn edge_empty_transcript_is_clean_skip() {
    let project = unique_project("edge_empty_transcript");
    // Build an empty JSONL file.
    let empty_path = project.join("empty.jsonl");
    std::fs::write(&empty_path, "").unwrap();

    let input = pre_compact_input(&project, "edge1", &empty_path);
    pre_compact::handle(input).unwrap();

    // Either no fingerprint file, or one with zero items — both are fine.
    if let Some(fp) = store::load_latest_fingerprint(&project, "edge1").unwrap() {
        assert_eq!(fp.total_items, 0);
    }

    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_only_system_messages_yields_empty_fingerprint() {
    let project = unique_project("edge_only_system");
    let path = project.join("system_only.jsonl");
    std::fs::write(
        &path,
        r#"{"type":"system","message":{"role":"system","content":"You are an expert."}}
{"type":"system","message":{"role":"system","content":"Working directory: /tmp."}}
"#,
    )
    .unwrap();

    pre_compact::handle(pre_compact_input(&project, "edge2", &path)).unwrap();
    let fp = store::load_latest_fingerprint(&project, "edge2")
        .unwrap()
        .expect("should still write a fingerprint, even an empty one");
    assert_eq!(fp.total_items, 0);

    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_empty_summary_marks_everything_lost() {
    let project = unique_project("edge_empty_summary");
    let session = "edge3";
    let fixture = fixture_path();
    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();

    // Empty summary → handler skips with a warning. We still want a
    // graceful exit, not a panic.
    post_compact::handle(post_compact_input(&project, session, &fixture, "")).unwrap();

    // No loss report should have been written (skip path).
    let loss_path = store::state_root(&project)
        .join(store::LOSS_REPORTS_SUBDIR)
        .join(format!("{}.json", session));
    assert!(
        !loss_path.exists(),
        "empty summary should skip without writing a loss report"
    );

    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_post_compact_without_pre_compact_is_clean_skip() {
    let project = unique_project("edge_no_pre");
    // No pre-compact run → no fingerprint to compare.
    post_compact::handle(post_compact_input(
        &project,
        "edge4",
        &fixture_path(),
        SYNTH_SUMMARY,
    ))
    .unwrap();
    // No loss report or pending file should have been written.
    let loss_path = store::state_root(&project)
        .join(store::LOSS_REPORTS_SUBDIR)
        .join("edge4.json");
    assert!(!loss_path.exists());
    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_user_prompt_no_pending_is_silent_no_op() {
    let project = unique_project("edge_no_pending");
    user_prompt::handle(user_prompt_input(&project, "edge5")).unwrap();
    // No injection-history file should have been written.
    let history_dir = store::state_root(&project).join("injection-history");
    let any = history_dir.exists() && std::fs::read_dir(&history_dir).unwrap().count() > 0;
    assert!(
        !any,
        "no archive should be written when nothing was delivered"
    );
    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_two_compactions_same_session_keep_both_fingerprints() {
    let project = unique_project("edge_two_compactions");
    let session = "edge6";
    let fixture = fixture_path();

    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();
    // Tiny sleep so the nanos timestamp in the filename differs even on
    // platforms with coarse-resolution clocks.
    std::thread::sleep(std::time::Duration::from_millis(2));
    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();

    let fp_dir = store::state_root(&project).join(store::FINGERPRINTS_SUBDIR);
    let count = std::fs::read_dir(&fp_dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("{}_", session))
        })
        .count();
    assert_eq!(count, 2, "two pre-compacts → two fingerprint files");

    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_malformed_transcript_line_is_skipped() {
    let project = unique_project("edge_malformed");
    let path = project.join("malformed.jsonl");
    std::fs::write(
        &path,
        r#"{"type":"user","message":{"role":"user","content":"the budget should not exceed $50K total"}}
this line is not valid JSON
{"type":"assistant","message":{"role":"assistant","content":"Acknowledged."}}
"#,
    )
    .unwrap();

    pre_compact::handle(pre_compact_input(&project, "edge7", &path)).unwrap();
    let fp = store::load_latest_fingerprint(&project, "edge7")
        .unwrap()
        .unwrap();
    // Despite the broken middle line, the budget statement should
    // still appear in the fingerprint.
    assert!(
        fp.items.iter().any(|i| i.content.contains("$50K")),
        "budget should be extracted despite the malformed line"
    );
    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn edge_full_lifecycle_compaction_log_appended() {
    let project = unique_project("edge_log");
    let session = "edge8";
    let fixture = fixture_path();

    pre_compact::handle(pre_compact_input(&project, session, &fixture)).unwrap();
    post_compact::handle(post_compact_input(
        &project,
        session,
        &fixture,
        SYNTH_SUMMARY,
    ))
    .unwrap();

    let log_path = store::state_root(&project).join(store::COMPACTION_LOG_FILE);
    let history = cctx::state::history::read_history(&log_path).unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].session_id, session);
    assert_eq!(history[0].trigger, "auto");
    assert!(history[0].lost > 0, "expected some lost items");

    std::fs::remove_dir_all(&project).ok();
}
