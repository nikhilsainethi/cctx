//! Claude Code hook handlers.
//!
//! cctx integrates with Claude Code's lifecycle via three hooks:
//!
//! 1. **PreCompact** — fires just before context compaction. We
//!    fingerprint the transcript and save the snapshot to disk.
//! 2. **PostCompact** — fires after the compaction summary is
//!    generated. We diff the fingerprint against the summary and
//!    queue a recovery payload.
//! 3. **UserPromptSubmit** — fires on every user prompt. The first
//!    one after a compaction picks up the queued payload and emits
//!    it to stdout (Claude treats SessionStart-style stdout as
//!    injected context).
//!
//! **Why UserPromptSubmit and not SessionStart?**
//!
//! Empirical testing showed Claude Code fires hooks in this order
//! during mid-session compaction:
//!
//! ```text
//! 1. PreCompact
//! 2. SessionStart (source = "compact")  ← BEFORE PostCompact
//! 3. PostCompact
//! 4. UserPromptSubmit (on next user message)
//! ```
//!
//! SessionStart runs before the loss report exists, so it can't
//! inject anything useful. UserPromptSubmit is the first reliable
//! emission point after PostCompact has finished, so that's where
//! we deliver the payload. We optimize this hook hard — the fast-
//! path (99% of prompts, no pending injection) is just one
//! `metadata` call.

pub mod input;
pub mod install;
pub mod post_compact;
pub mod pre_compact;
pub mod user_prompt;
