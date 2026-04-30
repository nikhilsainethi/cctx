//! Phase 3 — deliver the queued recovery payload (if any).
//!
//! This hook fires on **every** user prompt. Performance is the
//! whole story: the 99% case (no pending injection) must be a
//! single `metadata` syscall and an `exit 0`. We don't parse the
//! transcript, don't run any analysis, don't even initialize the
//! tokenizer — just check whether
//! `.cctx/pending-injection/<session>.json` exists.
//!
//! When the payload IS present (the first prompt after a compaction):
//!
//! 1. Read the file.
//! 2. Emit the contents to stdout (Claude Code treats this as
//!    injected context).
//! 3. Archive a copy to `.cctx/injection-history/`.
//! 4. Delete the pending file.

use anyhow::Result;

use crate::state::store;

use super::input::{resolve_project_dir, UserPromptSubmitInput};

/// Run the UserPromptSubmit hook handler.
///
/// Empty session id, missing project dir, missing pending file —
/// every "nothing to do" path exits silently and quickly. Any errors
/// are logged to stderr but the process still exits 0; we never want
/// to block a user prompt because of a cctx hiccup.
pub fn handle(input: UserPromptSubmitInput) -> Result<()> {
    if input.common.session_id.is_empty() {
        // Fast-path: no session id, no possible match — done.
        return Ok(());
    }

    let project_dir = match resolve_project_dir(&input.common) {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    // Cheap existence check — single stat, no JSON parse, no tokenizer.
    let pending_path = pending_injection_path(&project_dir, &input.common.session_id);
    if !pending_path.exists() {
        return Ok(());
    }

    // We have a payload waiting. Switch into the slow path.
    let payload = match store::take_pending_injection(&project_dir, &input.common.session_id) {
        Ok(Some(p)) => p,
        Ok(None) => {
            // Race with concurrent invocation — file vanished between
            // exists() and read. Treat as no-op.
            return Ok(());
        }
        Err(e) => {
            eprintln!(
                "[cctx] UserPromptSubmit: failed to read pending injection: {:#}",
                e
            );
            return Ok(());
        }
    };

    // Print the recovered context to stdout — Claude Code consumes it.
    print!("{}", payload);
    if !payload.ends_with('\n') {
        println!();
    }

    // Best-effort archive — failure here doesn't affect the
    // already-delivered injection.
    if let Err(e) = store::save_injection_history(&project_dir, &input.common.session_id, &payload)
    {
        eprintln!(
            "[cctx] UserPromptSubmit: could not archive injection: {:#}",
            e
        );
    }

    Ok(())
}

/// Resolve the path that the fast-path checks. Pulled out so the
/// tests can poke the same file the production code does.
fn pending_injection_path(project_dir: &std::path::Path, session_id: &str) -> std::path::PathBuf {
    store::state_root(project_dir)
        .join(store::PENDING_INJECTION_SUBDIR)
        .join(format!("{}.json", sanitize(session_id)))
}

/// Mirror of [`store::sanitize`] (which is private). Keeps the
/// fast-path filename-construction self-contained without relying on
/// the slow path's internal helper. Filenames stay strictly within
/// `[a-zA-Z0-9_-]` so user-supplied session ids can't escape the dir.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

/// `Result` alias kept for symmetry — we surface success regardless,
/// but the type signature lets `?` work cleanly inside `handle`.
#[allow(dead_code)]
type _UnusedAlias = Result<()>;

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
        let path = std::env::temp_dir().join(format!("cctx_user_prompt_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn make_input(session: &str, project: &std::path::Path) -> UserPromptSubmitInput {
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

    #[test]
    fn no_pending_injection_is_silent_no_op() {
        let project = unique_project("no_pending");
        // Don't even init() — exercise the absolute fastest path.
        let input = make_input("session_x", &project);
        handle(input).unwrap();
        // No injection-history file should appear either.
        let history_dir = store::state_root(&project).join("injection-history");
        assert!(!history_dir.exists() || std::fs::read_dir(&history_dir).unwrap().count() == 0);
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn pending_injection_is_delivered_and_archived() {
        let project = unique_project("delivers");
        store::init(&project).unwrap();
        let session = "session_d";
        store::save_pending_injection(&project, session, "[cctx] recovered text\nbody\n").unwrap();

        let input = make_input(session, &project);
        handle(input).unwrap();

        // Pending file must be gone.
        let pending = pending_injection_path(&project, session);
        assert!(
            !pending.exists(),
            "take_pending_injection should delete file"
        );

        // Injection history should have one archive file.
        let history_dir = store::state_root(&project).join("injection-history");
        let archived: Vec<_> = std::fs::read_dir(&history_dir).unwrap().flatten().collect();
        assert!(
            archived.iter().any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with(&format!("{}_", session)) && name.ends_with(".json")
            }),
            "archived copy of the injection should be saved"
        );

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn empty_session_id_is_fast_skip() {
        let project = unique_project("empty_session");
        let input = UserPromptSubmitInput {
            common: CommonHookInput {
                session_id: String::new(),
                transcript_path: None,
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "UserPromptSubmit".into(),
            },
            prompt: None,
        };
        handle(input).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }
}
