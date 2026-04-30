//! Phase 2 — diff the fingerprint against the compaction summary.
//!
//! By the time this fires, Claude Code has produced the compaction
//! summary text and our PreCompact handler should have already
//! written a fingerprint to `.cctx/fingerprints/`. We:
//!
//! 1. Load the latest fingerprint for the session.
//! 2. Run IDF-weighted loss detection against the summary.
//! 3. Save the loss report.
//! 4. Build + queue a recovery payload for the next user-prompt hook.
//! 5. Append a [`crate::state::history::CompactionEvent`] to the log.
//!
//! Like PreCompact, errors here are non-blocking — we log and exit 0
//! rather than disrupt Claude Code's lifecycle.

use anyhow::{Context as _, Result};

use crate::compaction::injection_builder::{build_injection, DEFAULT_INJECTION_BUDGET};
use crate::compaction::loss_detector::detect_loss;
use crate::core::tokenizer::Tokenizer;
use crate::state::history::CompactionEvent;
use crate::state::store;
use crate::transcript::{normalize, parse_transcript};

use super::input::{resolve_project_dir, resolve_transcript_path, PostCompactInput};

/// Run the PostCompact hook handler.
pub fn handle(input: PostCompactInput) -> Result<()> {
    let project_dir = resolve_project_dir(&input.common)?;
    store::init(&project_dir).context("Failed to initialize .cctx/")?;

    let session_id = if input.common.session_id.is_empty() {
        eprintln!("[cctx] PostCompact: missing session_id, skipping");
        return Ok(());
    } else {
        input.common.session_id.clone()
    };

    let summary = match input.compact_summary.as_deref() {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => {
            eprintln!("[cctx] PostCompact: empty compact_summary, skipping");
            return Ok(());
        }
    };

    // Load the most recent fingerprint for the session.
    let fp = match store::load_latest_fingerprint(&project_dir, &session_id)? {
        Some(f) => f,
        None => {
            eprintln!(
                "[cctx] PostCompact: no fingerprint for session `{}` — was PreCompact run?",
                session_id
            );
            return Ok(());
        }
    };

    // Re-parse the transcript so we can compute IDF over the
    // pre-compaction text (the fingerprint itself doesn't store the
    // raw chunks — its items are post-extraction). Best-effort:
    // missing transcript means we use a degenerate IDF where every
    // token weighs 1.0 (default in `weighted_overlap`). That's fine
    // — accuracy degrades but the system keeps working.
    let context = match resolve_transcript_path(&input.common) {
        Some(p) if p.is_file() => match parse_transcript(&p) {
            Ok(entries) => normalize(entries)
                .unwrap_or_else(|_| crate::core::context::Context::new(Vec::new())),
            Err(_) => crate::core::context::Context::new(Vec::new()),
        },
        _ => crate::core::context::Context::new(Vec::new()),
    };

    // Approx token counts for the compression-ratio field. The
    // tokenizer init can fail in pathological environments — fall
    // back to char/4 to keep the report populated.
    let tokenizer = Tokenizer::new().ok();
    let pre_tokens = fp.total_tokens; // captured at PreCompact time
    let post_tokens = match &tokenizer {
        Some(t) => t.count(&summary),
        None => summary.chars().count().div_ceil(4),
    };

    let trigger = input.trigger.as_deref().unwrap_or("auto");
    let report = detect_loss(
        &fp.items,
        &summary,
        &context,
        &session_id,
        trigger,
        pre_tokens,
        post_tokens,
    );

    store::save_loss_report(&project_dir, &session_id, &report)
        .context("Failed to save loss report")?;

    // Build the recovery payload and queue it for the user-prompt hook.
    let payload = build_injection(&report.items, DEFAULT_INJECTION_BUDGET);
    if !payload.is_empty() {
        store::save_pending_injection(&project_dir, &session_id, &payload.payload)
            .context("Failed to save pending injection")?;
    }

    // Append to the compaction log so `cctx compaction-history` has a row.
    let event = CompactionEvent {
        session_id: session_id.clone(),
        timestamp: store::now_iso8601(),
        trigger: trigger.to_string(),
        total_items: report.total_fingerprinted,
        preserved: report.preserved_count,
        paraphrased: report.paraphrased_count,
        lost: report.lost_count,
        injection_tokens: payload.token_count,
    };
    store::append_compaction_log(&project_dir, &event)
        .context("Failed to append compaction log")?;

    // Informational stdout — PostCompact has no decision control.
    println!(
        "[cctx] Compaction analysis: {} fingerprinted → {} preserved, {} paraphrased, {} lost. \
         Injection payload ready ({} items, {} tokens).",
        report.total_fingerprinted,
        report.preserved_count,
        report.paraphrased_count,
        report.lost_count,
        payload.item_count,
        payload.token_count,
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::input::CommonHookInput;
    use std::path::PathBuf;

    fn unique_project(tag: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("cctx_post_compact_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_transcript.jsonl")
    }

    fn run_pre_compact(project: &std::path::Path, session: &str) {
        let input = crate::hooks::input::PreCompactInput {
            common: CommonHookInput {
                session_id: session.into(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PreCompact".into(),
            },
            trigger: Some("auto".into()),
            custom_instructions: None,
        };
        crate::hooks::pre_compact::handle(input).unwrap();
    }

    #[test]
    fn missing_summary_is_a_clean_skip() {
        let project = unique_project("no_summary");
        let input = PostCompactInput {
            common: CommonHookInput {
                session_id: "sess1".into(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PostCompact".into(),
            },
            trigger: None,
            compact_summary: None,
        };
        handle(input).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn no_pre_fingerprint_is_a_clean_skip() {
        let project = unique_project("no_fingerprint");
        let input = PostCompactInput {
            common: CommonHookInput {
                session_id: "sess_solo".into(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PostCompact".into(),
            },
            trigger: Some("auto".into()),
            compact_summary: Some("we discussed PostgreSQL".into()),
        };
        handle(input).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn full_lifecycle_writes_loss_report_and_pending_injection() {
        let project = unique_project("full_post");
        let session = "lifecycle_post";

        // Phase 1 (precompact) writes a fingerprint.
        run_pre_compact(&project, session);

        // Phase 2 (postcompact) should produce loss-report + pending injection.
        let input = PostCompactInput {
            common: CommonHookInput {
                session_id: session.into(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PostCompact".into(),
            },
            trigger: Some("auto".into()),
            // Minimal summary — almost everything from the fingerprint
            // should be marked Lost (port 8443, prod.yml, …).
            compact_summary: Some("we discussed building a REST API; team uses PostgreSQL".into()),
        };
        handle(input).unwrap();

        let loss_report_path = store::state_root(&project)
            .join("loss-reports")
            .join(format!("{}.json", session));
        assert!(loss_report_path.is_file(), "loss report should exist");

        let pending_path = store::state_root(&project)
            .join("pending-injection")
            .join(format!("{}.json", session));
        assert!(
            pending_path.is_file(),
            "pending injection should be queued for the next user prompt"
        );

        // Compaction-log should have a row.
        let log_path = store::state_root(&project).join(crate::state::store::COMPACTION_LOG_FILE);
        assert!(log_path.is_file());
        let history = crate::state::history::read_history(&log_path).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].session_id, session);

        std::fs::remove_dir_all(&project).ok();
    }
}
