// lib.rs is the root of the library crate. main.rs imports from here via `use cctx::...`.
// `pub mod` makes a module public — without `pub`, code in other crates can't see it.
pub mod core;
pub mod analyzer;
pub mod formats;
pub mod pipeline;
pub mod strategies;
