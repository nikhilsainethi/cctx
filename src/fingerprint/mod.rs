//! Information fingerprint — what unique, irreplaceable information
//! exists in a conversation transcript.
//!
//! Day 21 stubbed the data types so the state store could read/write
//! [`Fingerprint`] without the algorithms in place. Day 22 fills in the
//! Tier-0 engine: zero ML dependencies, three layered passes (regex,
//! keyword triggers, RAKE catch-all) plus Jaccard-based cross-item
//! dedup and a uniqueness × recency × position-risk priority score.
//!
//! Submodules:
//!
//! - [`types`] — `Fingerprint`, `FingerprintItem`, `ItemCategory`,
//!   `ItemScores`, `FingerprintConfig`, plus the `make_id` content hasher.
//! - [`extractor`] — three-layer extraction over a [`crate::core::context::Context`].
//! - [`dedup`] — Jaccard-merge near-duplicate items.
//! - [`scorer`] — priority scoring on the merged items.
//! - [`engine`] — orchestrator (`fingerprint(ctx, config, …)`).
//!
//! The top-level [`fingerprint()`] re-export is the entry point most
//! callers want.

pub mod dedup;
pub mod engine;
pub mod extractor;
pub mod gliner_iface;
pub mod scorer;
pub mod types;

#[cfg(feature = "gliner")]
pub mod gliner;

pub use engine::fingerprint;
pub use gliner_iface::{label_to_category, GlinerEngine, GlinerEntity, ENTITY_LABELS};
pub use types::{
    make_id, Fingerprint, FingerprintConfig, FingerprintItem, ItemCategory, ItemScores,
};
