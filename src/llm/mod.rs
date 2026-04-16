//! LLM provider abstraction for text generation (summarization, etc.).
//!
//! # How optional async dependencies work in cctx
//!
//! cctx has three tiers of network functionality:
//!
//!   1. **Core CLI** (`cargo build`) — zero network deps. No reqwest, no tokio.
//!      analyze, optimize, compress, count, diff all work offline. Fast compile.
//!
//!   2. **Embeddings/LLM** (`--features embeddings` or `--features llm`) — adds
//!      `reqwest` (with `blocking` feature). HTTP calls use `reqwest::blocking::Client`,
//!      which internally spins up a minimal tokio runtime per-call. The user's code
//!      stays synchronous — no `async fn main`, no `#[tokio::main]`.
//!
//!   3. **Proxy** (`--features proxy`) — adds `reqwest` + `tokio` + `axum` + tower.
//!      Full async runtime for the HTTP server. This is the only feature that
//!      requires tokio in the binary.
//!
//! The key design: `reqwest` is a single optional dep shared by all three features.
//! Cargo unifies the feature flags — if you enable both `embeddings` and `proxy`,
//! you get one copy of reqwest with all features (`blocking` + `stream` + `json`).
//!
//! The LLM trait is sync (blocking) for the same reason as the embedding trait:
//! it runs inside `spawn_blocking` in the proxy, and on the main thread in the CLI.
//! No async-trait complexity needed.

use anyhow::Result;

// Providers are only compiled with --features llm.
#[cfg(feature = "llm")]
pub mod ollama;
#[cfg(feature = "llm")]
pub mod openai;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// A provider that generates text completions from an LLM.
///
/// `Send + Sync` because the provider is stored in `Arc` and shared
/// across threads (blocking thread pool in the proxy, main thread in CLI).
pub trait LlmProvider: Send + Sync {
    /// Generate a completion given a system prompt and user prompt.
    /// Returns the generated text.
    fn complete(&self, system: &str, prompt: &str) -> Result<String>;
}
