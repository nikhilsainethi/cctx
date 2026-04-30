//! Phase 1 — fingerprint the transcript before compaction runs.
//!
//! This handler is the entire reason we can later detect "this fact
//! was lost": without a snapshot of what existed before compaction,
//! we have nothing to diff against. We try to be generous with
//! failure modes — a missing transcript, no `.cctx/` directory, an
//! unparseable JSONL line — none of those should block compaction.
//! The handler logs a one-line warning and exits 0.
//!
//! Exit-code policy (per architecture §5.1):
//!
//! - 0 — success or non-blocking warning
//! - 1 — non-blocking error (stderr explains)
//! - **never 2** — exit 2 blocks compaction in Claude Code, which can
//!   cascade into a session error if the context is already at the
//!   limit. We swallow our own errors to keep the session alive.

use anyhow::{Context as _, Result};

use crate::core::tokenizer::Tokenizer;
use crate::fingerprint::{fingerprint, Fingerprint, FingerprintConfig};
use crate::state::store;
use crate::transcript::{normalize, parse_transcript};

use super::input::{resolve_project_dir, resolve_transcript_path, PreCompactInput};

/// Maximum tokens we let the stdout hint use. The hint goes INTO the
/// context being summarized, so a chatty hint costs us tokens we'd
/// rather spend on actual content.
const HINT_TOKEN_BUDGET: usize = 200;

/// Top-N items mentioned by name in the stdout hint. Three is enough
/// to be useful; more bloats the hint past the budget.
const TOP_AT_RISK_COUNT: usize = 3;

/// Run the PreCompact hook handler.
///
/// On any error: logs to stderr and returns the error so the binary
/// can exit with code 1. Per the architecture this is non-blocking —
/// compaction proceeds even if cctx errors out, just without our
/// fingerprint snapshot.
pub fn handle(input: PreCompactInput) -> Result<()> {
    let project_dir = resolve_project_dir(&input.common)?;

    // Auto-init `.cctx/` so the user doesn't need to run `cctx init`
    // before installing hooks. Idempotent — preserves existing config.
    store::init(&project_dir).context("Failed to initialize .cctx/ state directory")?;

    let session_id = if input.common.session_id.is_empty() {
        eprintln!("[cctx] PreCompact: missing session_id, skipping");
        return Ok(());
    } else {
        input.common.session_id.clone()
    };

    let transcript_path = match resolve_transcript_path(&input.common) {
        Some(p) => p,
        None => {
            eprintln!(
                "[cctx] PreCompact: no transcript_path on input or CLAUDE_TRANSCRIPT_PATH env, skipping"
            );
            return Ok(());
        }
    };

    if !transcript_path.is_file() {
        eprintln!(
            "[cctx] PreCompact: transcript {} not found, skipping",
            transcript_path.display()
        );
        return Ok(());
    }

    // Parse + normalize the transcript.
    let entries = parse_transcript(&transcript_path)
        .with_context(|| format!("Cannot parse transcript {}", transcript_path.display()))?;
    let context = normalize(entries).context("Failed to normalize transcript")?;

    // Pull tier from .cctx/config.json (Day 21 wrote a default with tier=0).
    let tier = read_tier(&project_dir).unwrap_or(0);

    let config = FingerprintConfig {
        tier,
        ..FingerprintConfig::default()
    };

    let created_at = store::now_iso8601();
    let fp = fingerprint(&context, &config, &session_id, &created_at);

    // Persist for the PostCompact handler to load.
    let saved_path = store::save_fingerprint(&project_dir, &session_id, &fp)
        .context("Failed to save fingerprint")?;
    eprintln!(
        "[cctx] PreCompact: fingerprint saved to {}",
        saved_path.display()
    );

    // The stdout hint goes INTO the context being summarized — keep it short.
    print_hint(&fp);

    Ok(())
}

/// Read `[fingerprint_extraction].fingerprint_tier` (top-level) from
/// `.cctx/config.json`. Returns `None` on any read/parse error so the
/// caller can default to Tier 0.
fn read_tier(project_dir: &std::path::Path) -> Option<u8> {
    let path = store::state_root(project_dir).join(store::CONFIG_FILE);
    let raw = std::fs::read_to_string(&path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("fingerprint_tier")
        .and_then(|v| v.as_u64())
        .and_then(|n| u8::try_from(n).ok())
}

/// Print the stdout hint. Truncates if it would exceed the token budget.
fn print_hint(fp: &Fingerprint) {
    let high_priority = fp.items.iter().filter(|i| i.priority_score >= 0.5).count();

    let top: Vec<String> = fp
        .items
        .iter()
        .take(TOP_AT_RISK_COUNT)
        .map(|item| {
            // Short preview — single sentence, capped at ~60 chars.
            let preview: String = item.content.chars().take(60).collect();
            let positions = if item.source_positions.is_empty() {
                String::new()
            } else {
                format!(" (msg {})", item.source_positions[0])
            };
            format!("{}{}", preview, positions)
        })
        .collect();

    let mut hint = format!(
        "[cctx] Fingerprinted {} items ({} high-priority) across {} messages.",
        fp.total_items,
        high_priority,
        fp.items
            .iter()
            .flat_map(|i| i.source_positions.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    );
    if !top.is_empty() {
        hint.push_str("\nTop items at risk: ");
        hint.push_str(&top.join("; "));
    }
    hint.push_str("\nFull fingerprint saved to .cctx/fingerprints/");

    // Cap to budget. A real overflow is rare given the format above,
    // but defensive truncation keeps us from blowing the context.
    if let Ok(tokenizer) = Tokenizer::new() {
        if tokenizer.count(&hint) > HINT_TOKEN_BUDGET {
            // Fall back to the headline only.
            hint = format!(
                "[cctx] Fingerprinted {} items ({} high-priority). Saved to .cctx/.",
                fp.total_items, high_priority
            );
        }
    }

    println!("{}", hint);
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
        let path = std::env::temp_dir().join(format!("cctx_pre_compact_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn fixture_path() -> std::path::PathBuf {
        // Resolve relative to the crate root (CARGO_MANIFEST_DIR).
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sample_transcript.jsonl")
    }

    #[test]
    fn missing_session_id_is_a_clean_skip() {
        let project = unique_project("no_session");
        let input = PreCompactInput {
            common: CommonHookInput {
                session_id: String::new(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PreCompact".into(),
            },
            trigger: None,
            custom_instructions: None,
        };
        // Should NOT error — PreCompact must never crash compaction.
        handle(input).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn missing_transcript_is_a_clean_skip() {
        let project = unique_project("no_transcript");
        let input = PreCompactInput {
            common: CommonHookInput {
                session_id: "test_sess".into(),
                transcript_path: None,
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PreCompact".into(),
            },
            trigger: None,
            custom_instructions: None,
        };
        handle(input).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn full_lifecycle_saves_fingerprint() {
        let project = unique_project("full_lifecycle");
        let input = PreCompactInput {
            common: CommonHookInput {
                session_id: "lifecycle_sess".into(),
                transcript_path: Some(fixture_path().to_string_lossy().to_string()),
                cwd: Some(project.to_string_lossy().to_string()),
                hook_event_name: "PreCompact".into(),
            },
            trigger: Some("auto".into()),
            custom_instructions: None,
        };
        handle(input).unwrap();

        // Verify a fingerprint landed under .cctx/fingerprints/.
        let fp_dir = store::state_root(&project).join("fingerprints");
        let entries: Vec<_> = std::fs::read_dir(&fp_dir).unwrap().flatten().collect();
        assert!(
            entries.iter().any(|e| e
                .file_name()
                .to_string_lossy()
                .starts_with("lifecycle_sess_")),
            "fingerprint file for the session should exist"
        );

        std::fs::remove_dir_all(&project).ok();
    }
}
