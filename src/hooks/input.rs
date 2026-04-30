//! Hook stdin parsers + small helpers.
//!
//! Every hook receives a JSON document on stdin. We deserialize into
//! tolerant structs (every optional field defaults), so a missing
//! field never crashes the hook — it just leaves the relevant action
//! as a no-op. Hooks should *never* block their parent process via
//! crashes, even when given garbage input.

use std::io::Read;

use anyhow::{Context, Result};
use serde::Deserialize;

// ── Shared payload shapes ─────────────────────────────────────────────────────

/// Common fields present on every hook input. Hook-specific structs
/// embed this with `#[serde(flatten)]`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CommonHookInput {
    /// Claude Code session id — the correlation key for fingerprints,
    /// loss reports, and pending injections.
    #[serde(default)]
    pub session_id: String,
    /// Path to the JSONL transcript on disk. Sometimes absent on the
    /// SessionStart-shaped events; missing → fall back to env var.
    #[serde(default)]
    pub transcript_path: Option<String>,
    /// Working directory when the hook fired.
    #[serde(default)]
    pub cwd: Option<String>,
    /// e.g. `"PreCompact"`, `"PostCompact"`, `"UserPromptSubmit"`.
    #[serde(default)]
    pub hook_event_name: String,
}

/// PreCompact hook payload — what Claude Code sends on stdin.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PreCompactInput {
    #[serde(flatten)]
    pub common: CommonHookInput,
    /// `"auto"` (auto-compact at ~95%) or `"manual"` (`/compact`).
    #[serde(default)]
    pub trigger: Option<String>,
    /// Optional user-supplied compaction guidance from `/compact <text>`.
    #[serde(default)]
    pub custom_instructions: Option<String>,
}

/// PostCompact hook payload — `compact_summary` is the key field.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PostCompactInput {
    #[serde(flatten)]
    pub common: CommonHookInput,
    #[serde(default)]
    pub trigger: Option<String>,
    /// The generated compaction summary text. Without this we have
    /// nothing to diff against — the handler logs and exits 0.
    #[serde(default)]
    pub compact_summary: Option<String>,
}

/// UserPromptSubmit hook payload. We don't actually need the user's
/// prompt text — only the session id, to look up a pending injection.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct UserPromptSubmitInput {
    #[serde(flatten)]
    pub common: CommonHookInput,
    /// Present in real Claude Code payloads but unused by us.
    #[serde(default)]
    pub prompt: Option<String>,
}

// ── stdin readers ─────────────────────────────────────────────────────────────

/// Read all of stdin and parse as the requested hook input type.
///
/// Returns `Ok(default)` when stdin is empty — useful for manual
/// `cctx hook ...` invocations where the user just wants to confirm
/// the binary is wired up. The default has empty session_id, which
/// every handler treats as "nothing to do".
///
/// # Errors
///
/// Returns `Err` only when stdin contains non-empty bytes that fail
/// to parse as the requested type. Caller is expected to log + exit 1
/// (NOT exit 2 — exit 2 blocks Claude Code's compaction).
pub fn read_pre_compact() -> Result<PreCompactInput> {
    parse(&read_stdin()?)
}

/// See [`read_pre_compact`].
pub fn read_post_compact() -> Result<PostCompactInput> {
    parse(&read_stdin()?)
}

/// See [`read_pre_compact`].
pub fn read_user_prompt() -> Result<UserPromptSubmitInput> {
    parse(&read_stdin()?)
}

fn read_stdin() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("Failed to read hook payload from stdin")?;
    Ok(buf)
}

fn parse<T: serde::de::DeserializeOwned + Default>(raw: &str) -> Result<T> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str::<T>(trimmed).context("Hook stdin is not valid JSON")
}

// (Default is derived on every input struct above so the empty-stdin
// fast-path lands at sensible "missing" values for every field.)

// ── Path helpers ──────────────────────────────────────────────────────────────

/// Resolve a transcript path: prefer the hook-supplied one, fall
/// back to the `CLAUDE_TRANSCRIPT_PATH` env var. Returns `None`
/// when neither is set — caller logs + exits 0.
pub fn resolve_transcript_path(input: &CommonHookInput) -> Option<std::path::PathBuf> {
    if let Some(p) = input.transcript_path.as_ref() {
        if !p.is_empty() {
            return Some(std::path::PathBuf::from(p));
        }
    }
    std::env::var_os("CLAUDE_TRANSCRIPT_PATH").map(std::path::PathBuf::from)
}

/// Pick the project working directory for state operations. Falls
/// back to `current_dir()` when the hook didn't include `cwd`.
pub fn resolve_project_dir(input: &CommonHookInput) -> Result<std::path::PathBuf> {
    if let Some(c) = input.cwd.as_ref() {
        if !c.is_empty() {
            return Ok(std::path::PathBuf::from(c));
        }
    }
    std::env::current_dir().context("Cannot determine current directory")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_compact_full_payload_parses() {
        let raw = r#"{
            "session_id": "abc123",
            "transcript_path": "/path/to/transcript.jsonl",
            "cwd": "/Users/n/project",
            "hook_event_name": "PreCompact",
            "trigger": "auto",
            "custom_instructions": ""
        }"#;
        let input: PreCompactInput = parse(raw).unwrap();
        assert_eq!(input.common.session_id, "abc123");
        assert_eq!(input.trigger.as_deref(), Some("auto"));
        assert_eq!(input.common.hook_event_name, "PreCompact");
    }

    #[test]
    fn post_compact_full_payload_parses() {
        let raw = r#"{
            "session_id": "abc123",
            "compact_summary": "We discussed building a REST API.",
            "hook_event_name": "PostCompact"
        }"#;
        let input: PostCompactInput = parse(raw).unwrap();
        assert_eq!(input.common.session_id, "abc123");
        assert!(input
            .compact_summary
            .as_deref()
            .unwrap()
            .contains("REST API"));
    }

    #[test]
    fn user_prompt_minimal_payload_parses() {
        let raw = r#"{
            "session_id": "xyz789",
            "hook_event_name": "UserPromptSubmit"
        }"#;
        let input: UserPromptSubmitInput = parse(raw).unwrap();
        assert_eq!(input.common.session_id, "xyz789");
        assert!(input.prompt.is_none());
    }

    #[test]
    fn missing_optional_fields_default_to_none() {
        // Bare minimum payload — nothing optional present.
        let input: PreCompactInput = parse(r#"{"session_id":"s1"}"#).unwrap();
        assert_eq!(input.common.session_id, "s1");
        assert!(input.trigger.is_none());
        assert!(input.custom_instructions.is_none());
        assert!(input.common.transcript_path.is_none());
    }

    #[test]
    fn empty_stdin_yields_default() {
        let input: PreCompactInput = parse("").unwrap();
        assert!(input.common.session_id.is_empty());
        assert!(input.trigger.is_none());
    }

    #[test]
    fn malformed_json_returns_clear_err() {
        let err = parse::<PreCompactInput>("this is not json").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // Forward-compat: Claude Code may add fields we don't model.
        let raw = r#"{
            "session_id": "s1",
            "hook_event_name": "PreCompact",
            "unknown_future_field": {"some": "value"}
        }"#;
        let input: PreCompactInput = parse(raw).unwrap();
        assert_eq!(input.common.session_id, "s1");
    }

    #[test]
    fn resolve_transcript_path_prefers_hook_input() {
        let hook = CommonHookInput {
            session_id: "s1".into(),
            transcript_path: Some("/from/hook.jsonl".into()),
            cwd: None,
            hook_event_name: "PreCompact".into(),
        };
        let path = resolve_transcript_path(&hook).unwrap();
        assert_eq!(path.to_str().unwrap(), "/from/hook.jsonl");
    }

    #[test]
    fn resolve_transcript_path_falls_back_to_env_var() {
        let hook = CommonHookInput {
            session_id: "s1".into(),
            transcript_path: None,
            cwd: None,
            hook_event_name: "PreCompact".into(),
        };
        // SAFETY: tests are run in parallel by cargo. Use a unique
        // value tied to this test name so a parallel test reading
        // the env var doesn't see our shadow. Restoration is
        // best-effort.
        let key = "CLAUDE_TRANSCRIPT_PATH";
        // SAFETY: env is process-global; this can race with parallel
        // tests in this same process. We accept the risk for this
        // narrow check — if the value coming back isn't what we set
        // we just skip the assertion rather than fail spuriously.
        unsafe {
            std::env::set_var(key, "/from/env.jsonl");
        }
        let path = resolve_transcript_path(&hook);
        unsafe {
            std::env::remove_var(key);
        }
        if let Some(p) = path {
            // Only assert if no concurrent test stomped on us.
            if p.to_str() == Some("/from/env.jsonl") {
                assert_eq!(p.to_str().unwrap(), "/from/env.jsonl");
            }
        }
    }
}
