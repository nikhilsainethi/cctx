//! Claude Code transcript JSONL schema — loosely typed for forward compatibility.
//!
//! Real transcripts mix several entry shapes: user/assistant/system messages,
//! plus events like cache control, session metadata, and others we don't
//! need today. We use `#[serde(default)]` on every optional field and a
//! catch-all `Unknown` content block so unfamiliar shapes parse instead of
//! rejecting the whole line.

use serde::{Deserialize, Serialize};

/// One JSONL line. The wire field name is `"type"` (a Rust keyword), so
/// we rename to `entry_type` and let serde do the bridging.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptEntry {
    /// Entry kind (e.g. `"user"`, `"assistant"`, `"system"`, `"summary"`).
    /// Stored as a free-form string rather than an enum so unknown values
    /// don't break parsing of the whole transcript.
    #[serde(rename = "type")]
    pub entry_type: String,
    /// Inner message payload — present on chat entries, absent on
    /// administrative entries like compaction-summary markers.
    #[serde(default)]
    pub message: Option<TranscriptMessage>,
    /// Wall-clock timestamp from Claude Code, if present.
    #[serde(default)]
    pub timestamp: Option<String>,
    /// Session identifier, if present on the entry.
    #[serde(default)]
    pub session_id: Option<String>,
}

/// The chat-message payload nested inside [`TranscriptEntry::message`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptMessage {
    /// `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Either a plain string (most common for short user messages) or an
    /// array of typed content blocks (Anthropic's structured form).
    pub content: TranscriptContent,
}

/// Anthropic's `content` is polymorphic: string OR array of blocks. We
/// model both with `#[serde(untagged)]` so serde tries each variant in
/// declaration order and picks the first that matches.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TranscriptContent {
    /// Single plain-text payload, no formatting.
    Text(String),
    /// Ordered sequence of typed blocks (text, tool_use, tool_result, …).
    Blocks(Vec<ContentBlock>),
}

/// One element of a structured content array. Discriminated by the `type`
/// field on the wire; the `Unknown` variant catches anything not modeled.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text block.
    #[serde(rename = "text")]
    Text {
        /// The text payload.
        text: String,
    },
    /// Assistant invocation of a tool.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Tool-use identifier; matched against `tool_use_id` on the result block.
        id: String,
        /// Name of the tool the assistant called (e.g. `"Bash"`, `"Read"`).
        name: String,
        /// Tool input arguments — kept as raw JSON because the schema
        /// varies per tool.
        input: serde_json::Value,
    },
    /// Result returned by a tool, sent back as part of a "user" message.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// Identifier from the matching `ToolUse` block.
        tool_use_id: String,
        /// Tool output. Optional because aborted/error tool calls may
        /// surface a result block without textual content.
        #[serde(default)]
        content: Option<String>,
    },
    /// Anything else Claude Code may emit (e.g. `"image"`, future block
    /// types). Catching them here keeps parsing tolerant to schema drift.
    #[serde(other)]
    Unknown,
}
