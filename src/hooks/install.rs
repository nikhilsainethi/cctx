//! Generate / remove cctx's entries in a Claude Code `settings.json`.
//!
//! Claude Code's hook config lives in JSON of this shape:
//!
//! ```text
//! {
//!   "hooks": {
//!     "PreCompact": [{ "hooks": [{ "type": "command", "command": "..." }] }],
//!     "PostCompact": [{ ... }],
//!     "UserPromptSubmit": [{ ... }]
//!   },
//!   "...other top-level settings...": ...
//! }
//! ```
//!
//! `install` and `uninstall` here are **surgical** — they read the
//! existing file (if any), modify only `hooks.<EventName>` arrays,
//! and write back, leaving every other setting untouched. Repeated
//! runs are idempotent (no duplicate entries on re-install).

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};

// ── Scope resolution ──────────────────────────────────────────────────────────

/// Where `settings.json` lives, by scope.
#[derive(Debug, Clone, Copy)]
pub enum HookScope {
    /// `.claude/settings.local.json` (project-local, not committed).
    Local,
    /// `.claude/settings.json` (project-shared, committed).
    Project,
    /// `~/.claude/settings.json` (user-global).
    User,
}

impl HookScope {
    /// Resolve to an absolute path. `Local` and `Project` are
    /// project-relative; `User` is `$HOME`-relative. Returns `Err`
    /// when the relevant root can't be determined.
    pub fn settings_path(&self, project_dir: &std::path::Path) -> Result<PathBuf> {
        match self {
            HookScope::Local => Ok(project_dir.join(".claude").join("settings.local.json")),
            HookScope::Project => Ok(project_dir.join(".claude").join("settings.json")),
            HookScope::User => {
                let home = std::env::var_os("HOME")
                    .ok_or_else(|| anyhow!("$HOME not set; cannot resolve user scope"))?;
                Ok(PathBuf::from(home).join(".claude").join("settings.json"))
            }
        }
    }
}

// ── Hook registration table ───────────────────────────────────────────────────

/// Each entry is `(claude_event_name, cctx_subcommand)`. Used by both
/// install (to know what to add) and uninstall (to know what
/// commands count as "ours").
const CCTX_HOOKS: &[(&str, &str)] = &[
    ("PreCompact", "cctx hook pre-compact"),
    ("PostCompact", "cctx hook post-compact"),
    ("UserPromptSubmit", "cctx hook user-prompt"),
];

// ── Install ───────────────────────────────────────────────────────────────────

/// Install cctx hooks into the settings file at `scope`, creating the
/// file (and parent directory) if needed.
///
/// Idempotent: re-running on an already-installed config is a no-op.
/// Other hooks (or non-hook settings) in the same file are preserved.
///
/// # Errors
///
/// Returns `Err` if the file exists but contains invalid JSON, or if
/// any I/O step fails.
pub fn install(scope: HookScope, project_dir: &std::path::Path) -> Result<PathBuf> {
    let path = scope.settings_path(project_dir)?;
    let mut root = read_settings_or_empty(&path)?;

    // Ensure root is an object — anything else means we'd clobber
    // user data, which is unsafe.
    let root_obj = root
        .as_object_mut()
        .ok_or_else(|| anyhow!("Settings file at {} is not a JSON object", path.display()))?;

    let hooks_obj = root_obj
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| anyhow!("`hooks` key in {} is not an object", path.display()))?;

    let mut added_count = 0usize;
    for (event, command) in CCTX_HOOKS {
        let arr = hooks_obj
            .entry(*event)
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| anyhow!("`hooks.{}` is not an array in {}", event, path.display()))?;

        // Skip if any existing entry already wires our exact command.
        if arr.iter().any(|entry| entry_has_command(entry, command)) {
            continue;
        }

        arr.push(json!({
            "hooks": [
                { "type": "command", "command": command }
            ]
        }));
        added_count += 1;
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create {}", parent.display()))?;
    }
    let pretty =
        serde_json::to_string_pretty(&root).context("Failed to serialize settings.json")?;
    std::fs::write(&path, pretty).with_context(|| format!("Cannot write {}", path.display()))?;

    if added_count == 0 {
        eprintln!(
            "[cctx] Hooks already installed at {}. Nothing to add.",
            path.display()
        );
    } else {
        eprintln!(
            "[cctx] Hooks installed to {}. cctx will now guard compactions.",
            path.display()
        );
    }
    Ok(path)
}

// ── Uninstall ─────────────────────────────────────────────────────────────────

/// Remove cctx's hook entries from the settings file at `scope`.
///
/// Removes only entries whose inner `hooks[].command` matches one of
/// the cctx-installed commands; leaves every other setting and every
/// other hook entry alone. Cleans up empty arrays / empty `hooks`
/// objects so the file looks tidy on disk.
///
/// # Errors
///
/// Returns `Err` if the file exists but contains invalid JSON, or if
/// any I/O step fails. A missing file is treated as success (nothing
/// to remove).
pub fn uninstall(scope: HookScope, project_dir: &std::path::Path) -> Result<()> {
    let path = scope.settings_path(project_dir)?;
    if !path.exists() {
        eprintln!(
            "[cctx] No settings file at {}. Nothing to remove.",
            path.display()
        );
        return Ok(());
    }

    let mut root = read_settings_or_empty(&path)?;
    let mut removed_count = 0usize;

    if let Some(root_obj) = root.as_object_mut() {
        if let Some(hooks_obj) = root_obj.get_mut("hooks").and_then(|v| v.as_object_mut()) {
            for (event, _command) in CCTX_HOOKS {
                if let Some(Value::Array(arr)) = hooks_obj.get_mut(*event) {
                    let before = arr.len();
                    arr.retain(|entry| !entry_belongs_to_cctx(entry));
                    removed_count += before - arr.len();
                }
            }
            // Drop empty arrays so re-installs start clean.
            let empty_keys: Vec<String> = hooks_obj
                .iter()
                .filter_map(|(k, v)| match v {
                    Value::Array(a) if a.is_empty() => Some(k.clone()),
                    _ => None,
                })
                .collect();
            for k in empty_keys {
                hooks_obj.remove(&k);
            }
        }
        // If `hooks` is now empty, drop the whole key.
        let drop_hooks = root_obj
            .get("hooks")
            .and_then(|v| v.as_object())
            .map(|o| o.is_empty())
            .unwrap_or(false);
        if drop_hooks {
            root_obj.remove("hooks");
        }
    }

    let pretty = serde_json::to_string_pretty(&root)
        .context("Failed to serialize settings.json after uninstall")?;
    std::fs::write(&path, pretty).with_context(|| format!("Cannot write {}", path.display()))?;

    eprintln!(
        "[cctx] Removed {} hook {} from {}.",
        removed_count,
        if removed_count == 1 {
            "entry"
        } else {
            "entries"
        },
        path.display()
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn read_settings_or_empty(path: &std::path::Path) -> Result<Value> {
    if !path.is_file() {
        return Ok(json!({}));
    }
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("Cannot read {}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(&raw).with_context(|| format!("Invalid JSON in {}", path.display()))
}

/// `true` when `entry` is shaped like a Claude Code hook config that
/// includes the given `command` somewhere in its inner `hooks` array.
fn entry_has_command(entry: &Value, command: &str) -> bool {
    entry
        .get("hooks")
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter()
                .any(|h| h.get("command").and_then(|c| c.as_str()) == Some(command))
        })
        .unwrap_or(false)
}

/// `true` when ANY of `entry`'s inner hooks has a `command` that
/// looks like one of cctx's. Used by uninstall to drop entries we own.
/// We match by the `cctx hook` prefix so future cctx hooks
/// (e.g. `cctx hook session-start`) get cleaned up too.
fn entry_belongs_to_cctx(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter().any(|h| {
                h.get("command")
                    .and_then(|c| c.as_str())
                    .map(|s| s.starts_with("cctx hook"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(tag: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("cctx_install_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn install_into_empty_project_creates_valid_json() {
        let project = unique_dir("empty");
        let path = install(HookScope::Local, &project).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert!(parsed.get("hooks").is_some());
        for (event, _) in CCTX_HOOKS {
            assert!(
                parsed["hooks"].get(*event).is_some(),
                "hooks.{} should be present",
                event
            );
        }
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn install_is_idempotent() {
        let project = unique_dir("idempotent");
        install(HookScope::Local, &project).unwrap();
        let path = install(HookScope::Local, &project).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        for (event, _) in CCTX_HOOKS {
            let count = parsed["hooks"][*event]
                .as_array()
                .map(|a| a.len())
                .unwrap_or(0);
            assert_eq!(
                count, 1,
                "{} should have exactly 1 entry, got {}",
                event, count
            );
        }
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn install_merges_with_existing_hooks() {
        let project = unique_dir("merge");
        let claude_dir = project.join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let path = claude_dir.join("settings.local.json");
        // User has a pre-existing custom hook.
        let existing = json!({
            "permissions": {"defaultMode": "acceptEdits"},
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "echo my-custom-hook"}]}
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "echo on-stop"}]}
                ]
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&existing).unwrap()).unwrap();

        install(HookScope::Local, &project).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();

        // Custom Stop hook should still be there.
        assert!(parsed["hooks"]["Stop"].as_array().unwrap().len() == 1);
        assert!(parsed["permissions"]["defaultMode"] == "acceptEdits");

        // PreCompact should have BOTH the custom hook and our cctx one.
        let pre = parsed["hooks"]["PreCompact"].as_array().unwrap();
        assert_eq!(pre.len(), 2);
        assert!(pre
            .iter()
            .any(|e| entry_has_command(e, "echo my-custom-hook")));
        assert!(pre
            .iter()
            .any(|e| entry_has_command(e, "cctx hook pre-compact")));

        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn uninstall_removes_only_cctx_entries() {
        let project = unique_dir("uninstall");
        let claude_dir = project.join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let path = claude_dir.join("settings.local.json");
        let mixed = json!({
            "permissions": {"defaultMode": "acceptEdits"},
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "echo my-custom-hook"}]},
                    {"hooks": [{"type": "command", "command": "cctx hook pre-compact"}]}
                ],
                "PostCompact": [
                    {"hooks": [{"type": "command", "command": "cctx hook post-compact"}]}
                ],
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "cctx hook user-prompt"}]}
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "echo on-stop"}]}
                ]
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&mixed).unwrap()).unwrap();

        uninstall(HookScope::Local, &project).unwrap();

        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // Permissions intact.
        assert!(parsed["permissions"]["defaultMode"] == "acceptEdits");
        // Custom Stop hook intact.
        assert!(parsed["hooks"]["Stop"].as_array().unwrap().len() == 1);
        // PreCompact should now have ONLY the custom hook.
        let pre = parsed["hooks"]["PreCompact"].as_array().unwrap();
        assert_eq!(pre.len(), 1);
        assert!(entry_has_command(&pre[0], "echo my-custom-hook"));
        // PostCompact + UserPromptSubmit had only cctx — array should be removed.
        assert!(parsed["hooks"].get("PostCompact").is_none());
        assert!(parsed["hooks"].get("UserPromptSubmit").is_none());
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn uninstall_no_settings_file_is_a_clean_skip() {
        let project = unique_dir("uninstall_nofile");
        // Don't create any file.
        uninstall(HookScope::Local, &project).unwrap();
        std::fs::remove_dir_all(&project).ok();
    }
}
