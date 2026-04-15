//! OpenAI-compatible proxy server.
//!
//! Sits between your application and the LLM API, optimizing context
//! transparently. Your app changes one line (the base URL) and every
//! API call flows through cctx.
//!
//! # Architecture
//!
//! ```text
//! Your App ──POST /v1/chat/completions──> cctx proxy ──forward──> LLM API
//!          <─────── response ────────────            <── response ──
//! ```
//!
//! This module is only compiled when the `proxy` feature is enabled:
//!   cargo build --features proxy

pub mod config;
pub mod handler;
pub mod metrics;
pub mod server;
pub mod upstream;
