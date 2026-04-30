//! Tolerant JSONL parser for Claude Code transcripts.
//!
//! "Tolerant" in two senses:
//!
//! 1. Per-line errors don't abort the whole file. Each malformed line is
//!    logged to stderr and skipped; the parse keeps going. Real
//!    transcripts can pick up garbled content, half-flushed writes, or
//!    schema drift, and we'd rather extract 199/200 entries than reject
//!    them all.
//! 2. The schema types use `#[serde(default)]` and a catch-all `Unknown`
//!    content block, so unexpected fields or block kinds don't fail.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};

use super::schema::TranscriptEntry;

/// Parse a Claude Code JSONL transcript file into a vector of entries.
///
/// One line in the file = one [`TranscriptEntry`]. Empty/whitespace-only
/// lines are skipped silently. Lines that fail to parse as a JSON object
/// matching [`TranscriptEntry`] are skipped with a one-line warning to
/// stderr; the rest of the file continues to parse.
///
/// # Errors
///
/// Returns `Err` only if the file cannot be opened (missing, permission
/// denied, etc.). Per-line parse errors are non-fatal by design.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use cctx::transcript::parse_transcript;
///
/// let entries = parse_transcript(Path::new("session.jsonl")).unwrap();
/// println!("parsed {} entries", entries.len());
/// ```
pub fn parse_transcript(path: &Path) -> Result<Vec<TranscriptEntry>> {
    let file = File::open(path)
        .with_context(|| format!("Cannot open transcript at {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut entries: Vec<TranscriptEntry> = Vec::new();
    let mut skipped = 0usize;

    // BufRead::lines() yields Result<String> per line — a read error here
    // is rare (truncated UTF-8, IO failure mid-file). Treat each as a
    // skippable bad line so one corrupt byte doesn't kill the whole parse.
    for (line_no, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[cctx] transcript {} line {}: read error, skipping: {}",
                    path.display(),
                    line_no + 1,
                    e
                );
                skipped += 1;
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<TranscriptEntry>(trimmed) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                // Truncate the offending line in the warning so we don't
                // dump multi-kilobyte tool outputs to stderr.
                let preview: String = trimmed.chars().take(80).collect();
                let suffix = if trimmed.len() > preview.len() {
                    "…"
                } else {
                    ""
                };
                eprintln!(
                    "[cctx] transcript {} line {}: parse error, skipping: {} | {}{}",
                    path.display(),
                    line_no + 1,
                    e,
                    preview,
                    suffix
                );
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        eprintln!(
            "[cctx] parsed {} entries from {} ({} skipped)",
            entries.len(),
            path.display(),
            skipped
        );
    }

    Ok(entries)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::schema::TranscriptContent;
    use std::io::Write;

    /// Helper: write `contents` to a unique temp file and return its path.
    fn tmp_jsonl(name: &str, contents: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cctx_parser_{}_{}.jsonl", name, nanos));
        let mut f = File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parses_valid_jsonl() {
        let raw = r#"{"type":"user","message":{"role":"user","content":"hi"}}
{"type":"assistant","message":{"role":"assistant","content":"hello"}}
"#;
        let path = tmp_jsonl("valid", raw);
        let entries = parse_transcript(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].entry_type, "user");
        assert_eq!(entries[1].entry_type, "assistant");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn skips_malformed_line_continues_parsing() {
        let raw = "\
{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}
this is not json
{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"third\"}}
";
        let path = tmp_jsonl("malformed", raw);
        let entries = parse_transcript(&path).unwrap();
        // The middle line is dropped; the outer two survive.
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].entry_type, "user");
        assert_eq!(entries[1].entry_type, "assistant");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_file_returns_empty_vec() {
        let path = tmp_jsonl("empty", "");
        let entries = parse_transcript(&path).unwrap();
        assert!(entries.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn whitespace_only_lines_are_ignored() {
        let raw = "\
{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}


\t
";
        let path = tmp_jsonl("whitespace", raw);
        let entries = parse_transcript(&path).unwrap();
        assert_eq!(entries.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parses_tool_use_and_tool_result_blocks() {
        let raw = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"file1\nfile2"}]}}
"#;
        let path = tmp_jsonl("tool_blocks", raw);
        let entries = parse_transcript(&path).unwrap();
        assert_eq!(entries.len(), 2);
        // Both entries should have message content as Blocks variant.
        match &entries[0].message.as_ref().unwrap().content {
            TranscriptContent::Blocks(_) => {}
            _ => panic!("expected blocks for assistant entry"),
        }
        match &entries[1].message.as_ref().unwrap().content {
            TranscriptContent::Blocks(_) => {}
            _ => panic!("expected blocks for user/tool_result entry"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_returns_err() {
        let path = std::env::temp_dir().join("cctx_parser_does_not_exist_xyz.jsonl");
        let err = parse_transcript(&path).unwrap_err();
        assert!(err.to_string().contains("Cannot open transcript"));
    }

    #[test]
    fn unknown_block_type_does_not_break_parse() {
        // An "image" block isn't in our explicit variants — should fall
        // through to the Unknown catch-all rather than failing the line.
        let raw = r#"{"type":"user","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"abc"}},{"type":"text","text":"see image"}]}}
"#;
        let path = tmp_jsonl("unknown_block", raw);
        let entries = parse_transcript(&path).unwrap();
        assert_eq!(entries.len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
