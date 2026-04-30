//! Claude Code JSONL transcript ingestion.
//!
//! Claude Code stores conversation history as JSONL files (one JSON object
//! per line) — *not* the OpenAI chat array format that the rest of cctx
//! handles via [`crate::formats`]. This module bridges the two:
//!
//! 1. [`schema`] — strongly-typed but tolerant transcript types.
//! 2. [`parser`] — line-by-line JSONL → `Vec<TranscriptEntry>`. Skips
//!    malformed lines with a stderr warning rather than aborting.
//! 3. [`normalizer`] — `Vec<TranscriptEntry>` → [`crate::core::context::Context`],
//!    so the existing analyzer / strategies can operate on transcripts
//!    unchanged.

pub mod normalizer;
pub mod parser;
pub mod schema;

pub use normalizer::normalize;
pub use parser::parse_transcript;
pub use schema::{ContentBlock, TranscriptContent, TranscriptEntry, TranscriptMessage};
