//! Strategy implementations — one module per transformation.
//!
//! Each strategy exposes one or more `apply*` functions that take a
//! [`crate::core::context::Context`] and return new chunks. The trait
//! objects in [`crate::pipeline`] wrap these plain functions so they can be
//! composed in a pipeline.

pub mod bookend;
pub mod dedup;
pub mod prune;
pub mod structural;
pub mod summarize;
