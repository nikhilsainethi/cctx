//! Convert parsed transcript entries into cctx's existing [`Context`] form.
//!
//! Mapping rules (Compaction Guard architecture §3.2.3):
//!
//! - User text                      → `Chunk { role: "user", … }`
//! - Assistant text blocks          → `Chunk { role: "assistant", … }`
//! - Tool-use + tool-result pair    → single `Chunk { role: "tool_interaction", … }`
//!   formatted as `[Tool: NAME] Input: … Output: …`
//! - System messages                → `Chunk { role: "system", … }` pinned at index 0
//! - Ordering follows the transcript's original order (system entries
//!   bubble to the front but otherwise the sequence is preserved).
//!
//! Tool outputs longer than [`TOOL_OUTPUT_TOKEN_LIMIT`] tokens are truncated
//! with a `[output truncated, N tokens]` marker so a single multi-megabyte
//! file dump can't dominate the analysis.

use std::collections::HashMap;

use anyhow::{Context as _, Result};

use crate::analyzer::health::assign_attention_zones;
use crate::core::context::{AttentionZone, Chunk, Context};
use crate::core::tokenizer::Tokenizer;

use super::schema::{ContentBlock, TranscriptContent, TranscriptEntry};

/// Tool outputs above this many tokens are truncated by the normalizer.
/// Keeps a single huge file-read from blowing out the dead-zone analysis
/// while preserving the fact that the tool ran and what it returned a slice of.
pub const TOOL_OUTPUT_TOKEN_LIMIT: usize = 500;

/// Normalize a parsed transcript into a [`Context`].
///
/// Pairs `tool_use` blocks with their matching `tool_result` blocks (by
/// `tool_use_id`) so a tool call shows up as one chunk in the timeline
/// rather than two. Any orphan tool_use without a result still becomes a
/// chunk (`Output: (no result)`); orphan tool_results are dropped with a
/// warning since they have no calling context.
///
/// # Errors
///
/// Returns `Err` if the BPE tokenizer fails to initialize. Per-entry
/// errors are non-fatal — malformed individual entries are skipped.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use cctx::transcript::{parse_transcript, normalize};
///
/// let entries = parse_transcript(Path::new("session.jsonl")).unwrap();
/// let ctx = normalize(entries).unwrap();
/// println!("{} chunks, {} tokens", ctx.chunk_count(), ctx.total_tokens);
/// ```
pub fn normalize(entries: Vec<TranscriptEntry>) -> Result<Context> {
    let tokenizer =
        Tokenizer::new().context("Failed to initialize tokenizer for transcript normalize")?;

    // ── Pass 1: collect all tool_result content keyed by tool_use_id ──────────
    //
    // We build a map up-front so when we encounter a tool_use block we can
    // attach its matching result without scanning the rest of the
    // transcript every time. Drops orphan results silently — they're rare
    // and have no context to attach them to.
    let tool_results = collect_tool_results(&entries);

    // ── Pass 2: walk entries in order, emit chunks ────────────────────────────
    let mut system_chunks: Vec<Chunk> = Vec::new();
    let mut other_chunks: Vec<Chunk> = Vec::new();

    for entry in &entries {
        let Some(message) = entry.message.as_ref() else {
            continue;
        };

        // Branch by role + content shape.
        match (message.role.as_str(), &message.content) {
            // ── System messages: pinned to position 0 ─────────────────────
            ("system", TranscriptContent::Text(text)) => {
                system_chunks.push(make_chunk("system", text.clone(), &tokenizer));
            }
            ("system", TranscriptContent::Blocks(blocks)) => {
                let text = collect_block_text(blocks);
                if !text.is_empty() {
                    system_chunks.push(make_chunk("system", text, &tokenizer));
                }
            }

            // ── User / assistant plain text ───────────────────────────────
            (role, TranscriptContent::Text(text)) if !text.is_empty() => {
                let role = canonical_role(role);
                other_chunks.push(make_chunk(role, text.clone(), &tokenizer));
            }

            // ── User / assistant block-form content ───────────────────────
            (role, TranscriptContent::Blocks(blocks)) => {
                emit_blocks(role, blocks, &tool_results, &mut other_chunks, &tokenizer);
            }

            // Empty text is skipped silently.
            _ => {}
        }
    }

    // ── Stitch: system messages first, others in order ────────────────────────
    let mut chunks: Vec<Chunk> = Vec::with_capacity(system_chunks.len() + other_chunks.len());
    chunks.append(&mut system_chunks);
    chunks.append(&mut other_chunks);

    // Re-index so each chunk's `index` matches its final position.
    let total = chunks.len();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
        // Default relevance: system pinned high; others assigned by recency.
        chunk.relevance_score = default_relevance(&chunk.role, i, total);
    }

    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);
    Ok(Context::new(chunks))
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Walk every entry once and build a `tool_use_id → output text` map so
/// the second pass can attach results to their matching tool_use blocks
/// without doing a full re-scan per call.
fn collect_tool_results(entries: &[TranscriptEntry]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for entry in entries {
        let Some(message) = entry.message.as_ref() else {
            continue;
        };
        let TranscriptContent::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
            } = block
            {
                // Last write wins if the same id appears twice — extremely
                // rare and either resolution is fine.
                if let Some(text) = content.as_ref() {
                    map.insert(tool_use_id.clone(), text.clone());
                }
            }
        }
    }
    map
}

/// Concatenate text from every text block, joined with double newlines.
/// Tool blocks inside a system message are ignored — system content is
/// always plain text in practice.
fn collect_block_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Emit chunks for one assistant or user message expressed as content blocks.
///
/// Text blocks become standalone chunks. Tool-use blocks pair with their
/// matching tool_result and become a single `tool_interaction` chunk.
/// Tool-result blocks that appear here are skipped — they were already
/// matched via [`collect_tool_results`] and shouldn't double-emit.
fn emit_blocks(
    role: &str,
    blocks: &[ContentBlock],
    tool_results: &HashMap<String, String>,
    out: &mut Vec<Chunk>,
    tokenizer: &Tokenizer,
) {
    for block in blocks {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                let role = canonical_role(role);
                out.push(make_chunk(role, text.clone(), tokenizer));
            }
            ContentBlock::ToolUse { id, name, input } => {
                let output = tool_results
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| "(no result)".to_string());
                let merged = format_tool_interaction(name, input, &output, tokenizer);
                out.push(make_chunk("tool_interaction", merged, tokenizer));
            }
            // Tool-result blocks are consumed by their matching tool_use;
            // emitting them again would double-count. Same for unknown.
            ContentBlock::ToolResult { .. } | ContentBlock::Unknown => {}
            ContentBlock::Text { .. } => {} // empty text
        }
    }
}

/// Produce the merged content string for a tool_use + tool_result pair.
///
/// Format: `[Tool: NAME] Input: COMPACT_JSON Output: TEXT`. When `TEXT`
/// exceeds [`TOOL_OUTPUT_TOKEN_LIMIT`] tokens, it's truncated to roughly
/// that many tokens with a `[output truncated, N tokens]` suffix
/// recording the original count.
fn format_tool_interaction(
    name: &str,
    input: &serde_json::Value,
    output: &str,
    tokenizer: &Tokenizer,
) -> String {
    // Compact JSON (no pretty-print) keeps the input dense; tool inputs
    // are usually small so this rarely matters, but it's a safer default.
    let input_str = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
    let input_str = if input_str.len() > 300 {
        // Hard char-limit cap on input only — token budget for the chunk
        // is dominated by output, so a tight char cap on input is fine.
        let truncated: String = input_str.chars().take(297).collect();
        format!("{}…", truncated)
    } else {
        input_str
    };

    let output_str = truncate_tool_output(output, tokenizer);

    format!(
        "[Tool: {}] Input: {} Output: {}",
        name, input_str, output_str
    )
}

/// Truncate `output` to ~`TOOL_OUTPUT_TOKEN_LIMIT` tokens if it's larger.
/// Returns the original string when within budget.
fn truncate_tool_output(output: &str, tokenizer: &Tokenizer) -> String {
    let total_tokens = tokenizer.count(output);
    if total_tokens <= TOOL_OUTPUT_TOKEN_LIMIT {
        return output.to_string();
    }

    // Approximate truncation: find a byte cut-off that lands close to
    // the token limit. Linear scan over chars is fine — typical large
    // outputs are still under a megabyte.
    let mut buf = String::new();
    for ch in output.chars() {
        buf.push(ch);
        // Recount only every 64 chars to keep this O(n / 64) rather
        // than O(n²) — tiktoken encode is cheap but not free.
        if buf.len().is_multiple_of(64) && tokenizer.count(&buf) >= TOOL_OUTPUT_TOKEN_LIMIT {
            break;
        }
    }

    format!(
        "{} … [output truncated, {} tokens]",
        buf.trim_end(),
        total_tokens
    )
}

/// Canonicalize role strings: anything Claude emits maps to one of
/// `system`/`user`/`assistant` for downstream code. Role names from
/// outside that triad are passed through unchanged so future variants
/// (e.g. `tool`) still flow.
fn canonical_role(role: &str) -> &str {
    match role {
        "system" | "user" | "assistant" => role,
        other => other,
    }
}

/// Build a [`Chunk`] with token count populated. `index` is set to a
/// placeholder; the caller re-indexes after stitching system + others.
fn make_chunk(role: &str, content: String, tokenizer: &Tokenizer) -> Chunk {
    let tokens = tokenizer.count(&content);
    Chunk {
        index: 0,
        role: role.to_string(),
        content,
        token_count: tokens,
        relevance_score: 0.5,
        attention_zone: AttentionZone::Strong,
    }
}

/// Default relevance scoring used when the transcript doesn't carry
/// explicit scores. System messages are always max-relevance; others
/// decay smoothly from oldest (0.1) to newest (0.9).
fn default_relevance(role: &str, index: usize, total: usize) -> f64 {
    if role == "system" {
        return 1.0;
    }
    if total <= 1 {
        return 0.5;
    }
    0.1 + (index as f64 / (total - 1) as f64) * 0.8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::schema::{ContentBlock, TranscriptContent, TranscriptMessage};

    fn user_text(text: &str) -> TranscriptEntry {
        TranscriptEntry {
            entry_type: "user".into(),
            message: Some(TranscriptMessage {
                role: "user".into(),
                content: TranscriptContent::Text(text.into()),
            }),
            timestamp: None,
            session_id: None,
        }
    }

    fn assistant_text(text: &str) -> TranscriptEntry {
        TranscriptEntry {
            entry_type: "assistant".into(),
            message: Some(TranscriptMessage {
                role: "assistant".into(),
                content: TranscriptContent::Text(text.into()),
            }),
            timestamp: None,
            session_id: None,
        }
    }

    fn system_text(text: &str) -> TranscriptEntry {
        TranscriptEntry {
            entry_type: "system".into(),
            message: Some(TranscriptMessage {
                role: "system".into(),
                content: TranscriptContent::Text(text.into()),
            }),
            timestamp: None,
            session_id: None,
        }
    }

    fn assistant_tool_use(id: &str, name: &str, input: serde_json::Value) -> TranscriptEntry {
        TranscriptEntry {
            entry_type: "assistant".into(),
            message: Some(TranscriptMessage {
                role: "assistant".into(),
                content: TranscriptContent::Blocks(vec![ContentBlock::ToolUse {
                    id: id.into(),
                    name: name.into(),
                    input,
                }]),
            }),
            timestamp: None,
            session_id: None,
        }
    }

    fn user_tool_result(id: &str, content: &str) -> TranscriptEntry {
        TranscriptEntry {
            entry_type: "user".into(),
            message: Some(TranscriptMessage {
                role: "user".into(),
                content: TranscriptContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: id.into(),
                    content: Some(content.into()),
                }]),
            }),
            timestamp: None,
            session_id: None,
        }
    }

    #[test]
    fn user_messages_become_user_chunks() {
        let entries = vec![
            user_text("first user message"),
            assistant_text("first reply"),
            user_text("second user message"),
        ];
        let ctx = normalize(entries).unwrap();
        assert_eq!(ctx.chunk_count(), 3);
        assert_eq!(ctx.chunks[0].role, "user");
        assert_eq!(ctx.chunks[1].role, "assistant");
        assert_eq!(ctx.chunks[2].role, "user");
        // Indices are reassigned to match position in the final context.
        assert_eq!(ctx.chunks[0].index, 0);
        assert_eq!(ctx.chunks[2].index, 2);
    }

    #[test]
    fn tool_use_and_result_merge_into_single_chunk() {
        let entries = vec![
            user_text("please list files"),
            assistant_tool_use("toolu_1", "Bash", serde_json::json!({"command": "ls"})),
            user_tool_result("toolu_1", "file_a.txt\nfile_b.txt"),
            assistant_text("here are the files."),
        ];
        let ctx = normalize(entries).unwrap();
        // Expected chunks: user, tool_interaction, assistant.
        // The user-tool-result entry does NOT produce its own chunk.
        assert_eq!(ctx.chunk_count(), 3);
        assert_eq!(ctx.chunks[0].role, "user");
        assert_eq!(ctx.chunks[1].role, "tool_interaction");
        assert_eq!(ctx.chunks[2].role, "assistant");
        assert!(ctx.chunks[1].content.starts_with("[Tool: Bash]"));
        assert!(ctx.chunks[1].content.contains("ls"));
        assert!(ctx.chunks[1].content.contains("file_a.txt"));
    }

    #[test]
    fn system_messages_pin_to_position_zero() {
        let entries = vec![
            user_text("hello"),
            system_text("you are a helpful assistant"),
            user_text("anything?"),
        ];
        let ctx = normalize(entries).unwrap();
        assert_eq!(ctx.chunks[0].role, "system");
        assert_eq!(ctx.chunks[0].index, 0);
        // System content survives intact even though it appeared later in the source.
        assert!(ctx.chunks[0].content.contains("helpful assistant"));
    }

    #[test]
    fn large_tool_output_is_truncated() {
        // Build an output well over the 500-token limit.
        let big = "alpha beta gamma delta epsilon zeta ".repeat(800);
        let entries = vec![
            assistant_tool_use("toolu_x", "Read", serde_json::json!({"path": "/big"})),
            user_tool_result("toolu_x", &big),
        ];
        let ctx = normalize(entries).unwrap();
        let merged = &ctx.chunks[0].content;
        assert!(merged.contains("[output truncated"));
        // Truncated output should be far smaller than the original.
        assert!(merged.len() < big.len());
    }

    #[test]
    fn orphan_tool_use_emits_no_result_marker() {
        let entries = vec![assistant_tool_use(
            "toolu_orphan",
            "Bash",
            serde_json::json!({"command": "true"}),
        )];
        let ctx = normalize(entries).unwrap();
        assert_eq!(ctx.chunk_count(), 1);
        assert!(ctx.chunks[0].content.contains("(no result)"));
    }

    #[test]
    fn empty_transcript_yields_empty_context() {
        let ctx = normalize(vec![]).unwrap();
        assert_eq!(ctx.chunk_count(), 0);
        assert_eq!(ctx.total_tokens, 0);
    }

    #[test]
    fn ordering_is_preserved_for_non_system_entries() {
        let entries = vec![
            user_text("turn 1"),
            assistant_text("turn 2"),
            user_text("turn 3"),
            assistant_text("turn 4"),
        ];
        let ctx = normalize(entries).unwrap();
        assert_eq!(ctx.chunks[0].content, "turn 1");
        assert_eq!(ctx.chunks[1].content, "turn 2");
        assert_eq!(ctx.chunks[2].content, "turn 3");
        assert_eq!(ctx.chunks[3].content, "turn 4");
    }
}
