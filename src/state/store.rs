//! `.cctx/` directory manager — initialization, fingerprint persistence,
//! loss-report writes, and one-shot pending-injection handoff.
//!
//! Every function takes the project directory explicitly so the module
//! is testable against tmp dirs without `set_current_dir` gymnastics. The
//! production CLI passes `std::env::current_dir()`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::compaction::LossReport;
use crate::fingerprint::Fingerprint;
use crate::state::history::CompactionEvent;

// ── Layout constants ──────────────────────────────────────────────────────────

/// Root of the cctx state tree, relative to the project directory.
pub const STATE_DIR: &str = ".cctx";
/// Per-session pre-compaction fingerprints, timestamped.
pub const FINGERPRINTS_SUBDIR: &str = "fingerprints";
/// Per-session post-compaction loss analysis.
pub const LOSS_REPORTS_SUBDIR: &str = "loss-reports";
/// One-shot files awaiting the next SessionStart hook.
pub const PENDING_INJECTION_SUBDIR: &str = "pending-injection";
/// Archive of injections that have been delivered.
pub const INJECTION_HISTORY_SUBDIR: &str = "injection-history";
/// Append-only event log.
pub const COMPACTION_LOG_FILE: &str = "compaction-log.json";
/// State configuration file.
pub const CONFIG_FILE: &str = "config.json";

/// Default contents written to `.cctx/config.json` when the directory is
/// initialized. Hand-tuned to match the Compaction Guard architecture
/// (and the Day 21 brief). Pretty-printed for human edits.
pub const DEFAULT_CONFIG_JSON: &str = r#"{
  "version": "0.2.0",
  "injection_budget_tokens": 4096,
  "fingerprint_tier": 0,
  "fingerprint_extraction": {
    "min_priority_score": 0.1,
    "max_items": 200
  },
  "loss_detection": {
    "preserved_threshold": 0.7,
    "lost_threshold": 0.3
  },
  "recency_decay_rate": 0.05,
  "enabled": true
}
"#;

// ── Path helpers ──────────────────────────────────────────────────────────────

/// Path to the `.cctx/` root for `project_dir`.
pub fn state_root(project_dir: &Path) -> PathBuf {
    project_dir.join(STATE_DIR)
}

fn fingerprints_dir(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(FINGERPRINTS_SUBDIR)
}
fn loss_reports_dir(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(LOSS_REPORTS_SUBDIR)
}
fn pending_injection_dir(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(PENDING_INJECTION_SUBDIR)
}
// Day 24's session-start hook archives delivered injections into this dir.
// Day 21 only reserves the path during init().
#[allow(dead_code)]
fn injection_history_dir(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(INJECTION_HISTORY_SUBDIR)
}
fn compaction_log_path(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(COMPACTION_LOG_FILE)
}
fn config_path(project_dir: &Path) -> PathBuf {
    state_root(project_dir).join(CONFIG_FILE)
}

// ── Init ──────────────────────────────────────────────────────────────────────

/// Create the `.cctx/` tree under `project_dir`, idempotently.
///
/// Re-running `init` is safe: existing directories are left alone, and
/// an existing `config.json` is preserved (so user edits survive). Only
/// the missing pieces are written.
///
/// # Errors
///
/// Returns `Err` if directory creation fails or the default config can't
/// be written. Permission denied or read-only filesystems surface here.
pub fn init(project_dir: &Path) -> Result<()> {
    let root = state_root(project_dir);
    fs::create_dir_all(&root)
        .with_context(|| format!("Cannot create state directory {}", root.display()))?;

    for sub in [
        FINGERPRINTS_SUBDIR,
        LOSS_REPORTS_SUBDIR,
        PENDING_INJECTION_SUBDIR,
        INJECTION_HISTORY_SUBDIR,
    ] {
        let path = root.join(sub);
        fs::create_dir_all(&path)
            .with_context(|| format!("Cannot create state subdir {}", path.display()))?;
    }

    let cfg = config_path(project_dir);
    if !cfg.exists() {
        fs::write(&cfg, DEFAULT_CONFIG_JSON)
            .with_context(|| format!("Cannot write {}", cfg.display()))?;
    }

    Ok(())
}

// ── Fingerprints ──────────────────────────────────────────────────────────────

/// Write `data` to a freshly-named file under `.cctx/fingerprints/`.
///
/// Filenames take the form `{session_id}_{nanos}.json` so multiple
/// compactions per session produce distinct files (the latest wins for
/// [`load_latest_fingerprint`]).
///
/// # Errors
///
/// Returns `Err` if the destination directory doesn't exist (`init`
/// hasn't run), if serialization fails, or if the file write fails.
pub fn save_fingerprint(
    project_dir: &Path,
    session_id: &str,
    data: &Fingerprint,
) -> Result<PathBuf> {
    let dir = fingerprints_dir(project_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Cannot create fingerprints dir {}", dir.display()))?;

    let nanos = nanos_now();
    let filename = format!("{}_{}.json", sanitize(session_id), nanos);
    let path = dir.join(filename);

    let json =
        serde_json::to_string_pretty(data).context("Failed to serialize Fingerprint to JSON")?;
    fs::write(&path, json)
        .with_context(|| format!("Cannot write fingerprint to {}", path.display()))?;

    Ok(path)
}

/// Load the most recently written fingerprint for `session_id`.
///
/// Returns `Ok(None)` when no fingerprint files exist for the session —
/// callers like the SessionStart handler treat that as "nothing to recover."
///
/// # Errors
///
/// Returns `Err` only when files exist but can't be read or parsed. A
/// missing fingerprints directory is also treated as "no fingerprints"
/// (Ok(None)).
pub fn load_latest_fingerprint(
    project_dir: &Path,
    session_id: &str,
) -> Result<Option<Fingerprint>> {
    let dir = fingerprints_dir(project_dir);
    if !dir.exists() {
        return Ok(None);
    }

    let prefix = format!("{}_", sanitize(session_id));
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("Cannot read {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with(&prefix) && name.ends_with(".json") {
            candidates.push(entry.path());
        }
    }

    if candidates.is_empty() {
        return Ok(None);
    }
    // Filenames embed nanos so a lexicographic sort = chronological sort.
    candidates.sort();
    let newest = candidates.last().unwrap();

    let raw = fs::read_to_string(newest)
        .with_context(|| format!("Cannot read fingerprint {}", newest.display()))?;
    let fp: Fingerprint = serde_json::from_str(&raw)
        .with_context(|| format!("Invalid fingerprint JSON in {}", newest.display()))?;
    Ok(Some(fp))
}

// ── Loss reports ──────────────────────────────────────────────────────────────

/// Persist a [`LossReport`] under `.cctx/loss-reports/{session_id}.json`.
///
/// Subsequent compactions of the same session overwrite the file — the
/// loss report represents the *most recent* compaction's outcome.
///
/// # Errors
///
/// Returns `Err` if the directory can't be created or the file write fails.
pub fn save_loss_report(project_dir: &Path, session_id: &str, report: &LossReport) -> Result<()> {
    let dir = loss_reports_dir(project_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Cannot create loss-reports dir {}", dir.display()))?;

    let path = dir.join(format!("{}.json", sanitize(session_id)));
    let json =
        serde_json::to_string_pretty(report).context("Failed to serialize LossReport to JSON")?;
    fs::write(&path, json)
        .with_context(|| format!("Cannot write loss report to {}", path.display()))?;
    Ok(())
}

// ── Pending injection ─────────────────────────────────────────────────────────

/// Write a one-shot recovery payload for the next SessionStart hook to
/// pick up. `payload` is the already-formatted text that will go to stdout.
///
/// # Errors
///
/// Returns `Err` if the directory can't be created or the file write fails.
pub fn save_pending_injection(project_dir: &Path, session_id: &str, payload: &str) -> Result<()> {
    let dir = pending_injection_dir(project_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Cannot create pending-injection dir {}", dir.display()))?;

    let path = dir.join(format!("{}.json", sanitize(session_id)));
    fs::write(&path, payload)
        .with_context(|| format!("Cannot write pending injection {}", path.display()))?;
    Ok(())
}

/// Read AND remove the pending injection for `session_id`.
///
/// Returns `Ok(None)` when no payload is queued. The remove step happens
/// AFTER the read succeeds so a transient I/O error doesn't lose the
/// payload — the next call will retry. If the remove itself fails, we
/// log to stderr and still return the payload (the caller should not be
/// blocked from acting on a recovered payload by a stale file).
///
/// # Errors
///
/// Returns `Err` if a payload exists but the read fails.
pub fn take_pending_injection(project_dir: &Path, session_id: &str) -> Result<Option<String>> {
    let dir = pending_injection_dir(project_dir);
    let path = dir.join(format!("{}.json", sanitize(session_id)));
    if !path.exists() {
        return Ok(None);
    }
    let payload = fs::read_to_string(&path)
        .with_context(|| format!("Cannot read pending injection {}", path.display()))?;

    if let Err(e) = fs::remove_file(&path) {
        // Log but don't fail — better to deliver the payload than block on cleanup.
        eprintln!(
            "[cctx] warn: could not remove pending injection {}: {}",
            path.display(),
            e
        );
    }

    Ok(Some(payload))
}

// ── Compaction log ────────────────────────────────────────────────────────────

/// Append `event` to `.cctx/compaction-log.json`.
///
/// The log is stored as a JSON array (read-modify-write). At the scales
/// this feature targets — handful of compactions per project — the
/// rewrite cost is negligible. If volume grows, swap to JSONL.
///
/// # Errors
///
/// Returns `Err` if the existing file is unreadable / contains invalid
/// JSON, or if the rewrite fails.
pub fn append_compaction_log(project_dir: &Path, event: &CompactionEvent) -> Result<()> {
    let path = compaction_log_path(project_dir);
    fs::create_dir_all(state_root(project_dir))
        .with_context(|| format!("Cannot ensure {}", state_root(project_dir).display()))?;

    let mut events: Vec<CompactionEvent> = if path.exists() {
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("Cannot read compaction log {}", path.display()))?;
        if raw.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&raw)
                .with_context(|| format!("Invalid JSON in {}", path.display()))?
        }
    } else {
        Vec::new()
    };
    events.push(event.clone());

    let json =
        serde_json::to_string_pretty(&events).context("Failed to serialize compaction log")?;
    fs::write(&path, json)
        .with_context(|| format!("Cannot write compaction log {}", path.display()))?;
    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Wall-clock nanoseconds since the UNIX epoch, used in fingerprint filenames
/// so multiple saves per session sort chronologically.
fn nanos_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Replace path-hostile characters in a session id so user input can't
/// escape the state directory or produce illegal filenames on Windows.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| match c {
            // Allow the boring identifier set; replace anything else with `_`.
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::{Fingerprint, FingerprintItem, ItemCategory, ItemScores};

    /// Allocate a unique tmp project directory per test so parallel runs
    /// don't collide. Returns the path; explicit cleanup at the end of
    /// each test removes it.
    fn unique_project(tag: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cctx_state_test_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn dummy_fingerprint(session: &str) -> Fingerprint {
        Fingerprint {
            session_id: session.to_string(),
            created_at: "2026-04-30T00:00:00Z".to_string(),
            total_tokens: 100,
            total_items: 1,
            items: vec![FingerprintItem {
                id: "item1".into(),
                category: ItemCategory::Constraint,
                content: "budget is $50K".into(),
                tokens: 4,
                occurrence_count: 1,
                source_positions: vec![5],
                priority_score: 0.9,
                scores: ItemScores {
                    uniqueness: 1.0,
                    recency: 0.9,
                    position_risk: 1.0,
                },
            }],
        }
    }

    #[test]
    fn init_creates_all_directories_and_config() {
        let project = unique_project("init");
        init(&project).unwrap();

        let root = state_root(&project);
        assert!(root.is_dir());
        for sub in [
            FINGERPRINTS_SUBDIR,
            LOSS_REPORTS_SUBDIR,
            PENDING_INJECTION_SUBDIR,
            INJECTION_HISTORY_SUBDIR,
        ] {
            assert!(root.join(sub).is_dir(), "missing subdir {}", sub);
        }
        assert!(config_path(&project).is_file(), "config.json not written");

        // Idempotency: second call must not error and must not overwrite.
        let cfg_path = config_path(&project);
        let edited = "{\"custom\":true}";
        std::fs::write(&cfg_path, edited).unwrap();
        init(&project).unwrap();
        let after = std::fs::read_to_string(&cfg_path).unwrap();
        assert_eq!(after, edited, "init must preserve user-edited config");

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn save_and_load_fingerprint_roundtrip() {
        let project = unique_project("fp_roundtrip");
        init(&project).unwrap();

        let original = dummy_fingerprint("session_alpha");
        let path = save_fingerprint(&project, "session_alpha", &original).unwrap();
        assert!(path.is_file());

        let loaded = load_latest_fingerprint(&project, "session_alpha")
            .unwrap()
            .expect("fingerprint should round-trip");
        assert_eq!(loaded.session_id, original.session_id);
        assert_eq!(loaded.total_items, original.total_items);
        assert_eq!(loaded.items[0].content, "budget is $50K");

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn load_returns_none_when_no_fingerprint_for_session() {
        let project = unique_project("fp_none");
        init(&project).unwrap();
        let loaded = load_latest_fingerprint(&project, "nonexistent").unwrap();
        assert!(loaded.is_none());
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn second_save_does_not_overwrite_first() {
        let project = unique_project("fp_no_overwrite");
        init(&project).unwrap();

        let mut fp_old = dummy_fingerprint("session_beta");
        fp_old.total_items = 1;
        let p1 = save_fingerprint(&project, "session_beta", &fp_old).unwrap();

        // Sleep a couple ms so nanosecond timestamps are guaranteed distinct
        // even on filesystems / timers with coarser-than-nano resolution.
        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut fp_new = dummy_fingerprint("session_beta");
        fp_new.total_items = 42;
        let p2 = save_fingerprint(&project, "session_beta", &fp_new).unwrap();

        assert_ne!(p1, p2, "second save must produce a distinct file");
        assert!(p1.is_file(), "first fingerprint file must still exist");
        assert!(p2.is_file(), "second fingerprint file must exist");

        // load_latest must return the newer one.
        let latest = load_latest_fingerprint(&project, "session_beta")
            .unwrap()
            .unwrap();
        assert_eq!(latest.total_items, 42);

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn pending_injection_take_reads_then_deletes() {
        let project = unique_project("inj");
        init(&project).unwrap();

        save_pending_injection(&project, "session_x", "recovery payload").unwrap();
        let path = pending_injection_dir(&project).join("session_x.json");
        assert!(path.exists(), "save should produce a file");

        let payload = take_pending_injection(&project, "session_x")
            .unwrap()
            .unwrap();
        assert_eq!(payload, "recovery payload");
        assert!(!path.exists(), "take should delete the file");

        // Second take returns None — file is gone.
        let again = take_pending_injection(&project, "session_x").unwrap();
        assert!(again.is_none());

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn loss_report_save_overwrites() {
        let project = unique_project("loss");
        init(&project).unwrap();

        let report = crate::compaction::LossReport {
            session_id: "session_y".into(),
            compaction_trigger: "auto".into(),
            pre_compaction_tokens: 1000,
            post_compaction_tokens: 200,
            compression_ratio: 0.2,
            total_fingerprinted: 10,
            preserved_count: 7,
            paraphrased_count: 1,
            lost_count: 2,
            preservation_ratio: 0.7,
            lost_items: vec![],
        };
        save_loss_report(&project, "session_y", &report).unwrap();

        let path = loss_reports_dir(&project).join("session_y.json");
        assert!(path.is_file());

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn append_compaction_log_round_trip() {
        let project = unique_project("log");
        init(&project).unwrap();

        let event_a = CompactionEvent {
            session_id: "s1".into(),
            timestamp: "2026-04-30T00:00:00Z".into(),
            trigger: "auto".into(),
            total_items: 10,
            preserved: 7,
            paraphrased: 1,
            lost: 2,
            injection_tokens: 512,
        };
        let event_b = CompactionEvent {
            session_id: "s2".into(),
            timestamp: "2026-04-30T01:00:00Z".into(),
            trigger: "manual".into(),
            total_items: 20,
            preserved: 18,
            paraphrased: 1,
            lost: 1,
            injection_tokens: 100,
        };

        append_compaction_log(&project, &event_a).unwrap();
        append_compaction_log(&project, &event_b).unwrap();

        let history = crate::state::history::read_history(&compaction_log_path(&project)).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].session_id, "s1");
        assert_eq!(history[1].session_id, "s2");

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn sanitize_strips_path_traversal_chars() {
        assert_eq!(sanitize("../escape"), "___escape");
        assert_eq!(sanitize("normal-id_123"), "normal-id_123");
        assert_eq!(sanitize("session id with spaces"), "session_id_with_spaces");
    }
}
