// lib.rs is the root of the library crate. main.rs imports from here via `use cctx::...`.
// `pub mod` makes a module public — without `pub`, code in other crates can't see it.
pub mod analyzer;
pub mod compaction;
pub mod config;
pub mod core;
pub mod embeddings;
pub mod fingerprint;
pub mod formats;
pub mod llm;
pub mod pipeline;
pub mod state;
pub mod strategies;
pub mod transcript;
pub mod watch;

// The proxy module is only compiled when the "proxy" feature flag is enabled.
// #[cfg(feature = "...")] is Rust's conditional compilation — the compiler
// skips this entire module (and its dependencies) when the feature is off.
// This keeps the default binary small and fast to compile.
#[cfg(feature = "proxy")]
pub mod proxy;
