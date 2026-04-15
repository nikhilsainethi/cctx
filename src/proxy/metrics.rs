//! In-memory metrics counters, safe for concurrent access.
//!
//! # Atomics
//!
//! `AtomicU64` is a 64-bit integer that can be read/written from multiple
//! threads without a mutex. The hardware provides atomic load/store/add
//! instructions, so there's no locking overhead. This is how high-performance
//! servers count requests without contention.
//!
//! `Ordering::Relaxed` is the cheapest memory ordering — it guarantees the
//! atomic operation itself is correct but doesn't enforce ordering relative
//! to other memory operations. For simple counters this is sufficient.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// Live counters shared across all handler tasks via `Arc<AppState>`.
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub tokens_original_total: AtomicU64,
    pub tokens_optimized_total: AtomicU64,
}

impl Metrics {
    /// Take a consistent snapshot for the /cctx/metrics endpoint.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            tokens_original_total: self.tokens_original_total.load(Ordering::Relaxed),
            tokens_optimized_total: self.tokens_optimized_total.load(Ordering::Relaxed),
            tokens_saved_total: self
                .tokens_original_total
                .load(Ordering::Relaxed)
                .saturating_sub(self.tokens_optimized_total.load(Ordering::Relaxed)),
        }
    }
}

/// Serializable snapshot of the current metrics. Returned as JSON.
#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub tokens_original_total: u64,
    pub tokens_optimized_total: u64,
    pub tokens_saved_total: u64,
}
