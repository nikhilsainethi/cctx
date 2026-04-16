//! Input format detection and parsing.
//!
//! cctx accepts four input formats:
//!   - **OpenAI** — `[{role, content}]` (content is a string)
//!   - **Anthropic** — `[{role, content}]` (content can be a string OR an array
//!     of `{type: "text", text: "..."}` blocks)
//!   - **RAG chunks** — `[{content, score?, metadata?}]` (no role field)
//!   - **Raw text** — anything that isn't valid JSON
//!
//! `detect_format` inspects the raw input and picks the right parser.
//! `parse_input` runs detection (or uses an explicit override) then normalizes
//! everything into `Vec<Message>`.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::core::context::Message;

// ── Format enum ───────────────────────────────────────────────────────────────

/// Which parser to use on a piece of input.
///
/// Pass this to [`parse_input`] as `Some(InputFormat::…)` to skip
/// auto-detection, or `None` to let [`detect_format`] pick.
#[derive(Debug, Clone, PartialEq)]
pub enum InputFormat {
    /// OpenAI chat format: `[{"role": ..., "content": "..."}]`.
    OpenAi,
    /// Anthropic messages — like OpenAI but `content` may be an array of
    /// typed blocks (`{"type": "text", "text": "..."}`).
    Anthropic,
    /// RAG retrieval output: `[{"content": ..., "score": ..., "metadata": ...}]`
    /// with no `role` field. `score` becomes `relevance_score`.
    RagChunks,
    /// Arbitrary text — wraps the whole input in a single `document` message.
    Raw,
}

// ── Detection ─────────────────────────────────────────────────────────────────

/// Inspect raw text and decide which format it is.
///
/// Detection rules (checked in order):
///   1. Parse as JSON array of objects
///   2. If objects have "role" + "content" as array with "type" → Anthropic
///   3. If objects have "role" → OpenAI
///   4. If objects have "content" (no "role") → RAG chunks
///   5. Anything else → Raw text
pub fn detect_format(raw: &str) -> InputFormat {
    // serde_json::from_str::<Vec<Value>> attempts to parse as a JSON array.
    // Each element is a serde_json::Value — a dynamic JSON type.
    let arr: Vec<Value> = match serde_json::from_str(raw.trim()) {
        Ok(arr) => arr,
        Err(_) => return InputFormat::Raw,
    };

    if arr.is_empty() {
        // Valid JSON array with zero elements → treat as empty OpenAI format.
        // This ensures [] produces 0 messages (not a raw chunk containing "[]").
        return InputFormat::OpenAi;
    }

    let first = &arr[0];

    // ── Conversation formats (have "role") ────────────────────────────────
    if first.get("role").is_some() {
        // Check if ANY message has content as an array of typed blocks.
        // .any() short-circuits: stops as soon as one match is found.
        let has_block_content = arr.iter().any(|msg| {
            msg.get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|blocks| blocks.iter().any(|b| b.get("type").is_some()))
        });
        if has_block_content {
            return InputFormat::Anthropic;
        }
        return InputFormat::OpenAi;
    }

    // ── RAG chunks (have "content" but no "role") ─────────────────────────
    if first.get("content").is_some() {
        return InputFormat::RagChunks;
    }

    InputFormat::Raw
}

// ── Public parsing API ────────────────────────────────────────────────────────

/// Parse raw input into a `Vec<Message>`, auto-detecting the format if needed.
///
/// When `format` is `None`, calls [`detect_format`] first. When `Some(f)`, uses
/// `f` directly — this is the path used by the `--input-format` CLI flag.
///
/// # Errors
///
/// Returns `Err` if the input is the wrong shape for the chosen format
/// (malformed JSON, missing required fields, etc.). The [`InputFormat::Raw`]
/// path is infallible.
///
/// # Examples
///
/// ```
/// use cctx::formats::{parse_input, InputFormat};
///
/// let raw = r#"[{"role": "user", "content": "hi"}]"#;
/// let msgs = parse_input(raw, None).unwrap();
/// assert_eq!(msgs.len(), 1);
/// assert_eq!(msgs[0].role, "user");
///
/// let msgs = parse_input("hello", Some(InputFormat::Raw)).unwrap();
/// assert_eq!(msgs.len(), 1);
/// assert_eq!(msgs[0].role, "document");
/// ```
pub fn parse_input(raw: &str, format: Option<InputFormat>) -> Result<Vec<Message>> {
    let format = format.unwrap_or_else(|| detect_format(raw));

    match format {
        InputFormat::OpenAi => parse_openai(raw),
        InputFormat::Anthropic => parse_anthropic(raw),
        InputFormat::RagChunks => parse_rag_chunks(raw),
        InputFormat::Raw => Ok(parse_raw(raw)),
    }
}

// ── OpenAI parser ─────────────────────────────────────────────────────────────
// Simplest: JSON array of {role: string, content: string}.

fn parse_openai(raw: &str) -> Result<Vec<Message>> {
    serde_json::from_str(raw.trim())
        .context("Invalid OpenAI format — expected [{\"role\": ..., \"content\": ...}, ...]")
}

// ── Anthropic parser ──────────────────────────────────────────────────────────
//
// Anthropic messages look like OpenAI but `content` can be either:
//   - a plain string: {"role": "user", "content": "Hello"}
//   - an array of blocks: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
//
// We normalize both to a plain string by joining all text blocks.

/// Intermediate type for deserialization — not exposed outside this module.
#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

/// `#[serde(untagged)]` tries each variant in order until one succeeds.
/// First attempt: parse as String. If that fails: parse as Vec<Block>.
/// This is how you model "string | array" in serde without a tag field.
#[derive(Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Deserialize)]
struct AnthropicBlock {
    /// The block type: "text", "image", "tool_use", etc.
    /// We only extract text from "text" blocks; everything else is skipped.
    #[serde(default)]
    r#type: String,
    /// The text payload (only present on "text" blocks).
    text: Option<String>,
}

fn parse_anthropic(raw: &str) -> Result<Vec<Message>> {
    let messages: Vec<AnthropicMessage> =
        serde_json::from_str(raw.trim()).context("Invalid Anthropic format")?;

    Ok(messages
        .into_iter()
        .map(|msg| {
            let content = match msg.content {
                AnthropicContent::Text(s) => s,
                AnthropicContent::Blocks(blocks) => {
                    // Collect text from "text" blocks, skip images/tool_use.
                    // .filter_map() = filter + map in one: returns only Some values.
                    blocks
                        .iter()
                        .filter(|b| b.r#type == "text")
                        .filter_map(|b| b.text.as_deref())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
            };
            Message {
                role: msg.role,
                content,
                relevance_score: None,
            }
        })
        .collect())
}

// ── RAG chunks parser ─────────────────────────────────────────────────────────
//
// RAG retrieval output: [{content: "...", score: 0.92, metadata: {...}}, ...]
// No "role" field. We assign role "document" and map score → relevance_score.

#[derive(Deserialize)]
struct RagChunk {
    content: String,
    /// Retrieval relevance score (0.0–1.0). Used by bookend for placement.
    score: Option<f64>,
    // `metadata` may be present in the JSON but we don't need it internally.
    // serde ignores unknown fields by default, so no need to declare it.
}

fn parse_rag_chunks(raw: &str) -> Result<Vec<Message>> {
    let chunks: Vec<RagChunk> = serde_json::from_str(raw.trim()).context(
        "Invalid RAG chunks format — expected [{\"content\": ..., \"score\": ...}, ...]",
    )?;

    Ok(chunks
        .into_iter()
        .map(|c| Message {
            role: "document".to_string(),
            content: c.content,
            relevance_score: c.score,
        })
        .collect())
}

// ── Raw text parser ───────────────────────────────────────────────────────────
// Entire input becomes a single chunk.

fn parse_raw(raw: &str) -> Vec<Message> {
    vec![Message {
        role: "document".to_string(),
        content: raw.to_string(),
        relevance_score: None,
    }]
}
