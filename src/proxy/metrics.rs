//! In-memory metrics counters, safe for concurrent access.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// Live counters shared across all handler tasks via `Arc<AppState>`.
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub streaming_requests: AtomicU64,
    pub tokens_original_total: AtomicU64,
    pub tokens_optimized_total: AtomicU64,
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        let original = self.tokens_original_total.load(Ordering::Relaxed);
        let optimized = self.tokens_optimized_total.load(Ordering::Relaxed);
        let saved = original.saturating_sub(optimized);
        let avg_ratio = if original > 0 {
            optimized as f64 / original as f64
        } else {
            1.0
        };

        MetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            streaming_requests: self.streaming_requests.load(Ordering::Relaxed),
            tokens_original_total: original,
            tokens_optimized_total: optimized,
            tokens_saved_total: saved,
            avg_compression_ratio: (avg_ratio * 1000.0).round() / 1000.0, // 3 decimal places
        }
    }
}

/// Serializable snapshot of the current metrics.
#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub streaming_requests: u64,
    pub tokens_original_total: u64,
    pub tokens_optimized_total: u64,
    pub tokens_saved_total: u64,
    /// Average ratio: optimized / original. 1.0 = no compression, 0.7 = 30% saved.
    pub avg_compression_ratio: f64,
}
