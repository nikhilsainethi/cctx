//! Comprehensive proxy metrics — thread-safe counters and per-model stats.
//!
//! Atomic counters (AtomicU64) for high-frequency counters that don't need
//! grouping. Mutex<HashMap> for per-model and per-strategy breakdowns where
//! we need key-value tracking with occasional writes.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

// ── Model pricing (USD per 1M input tokens) ──────────────────────────────────

pub fn model_price_per_million(model: &str) -> f64 {
    match model {
        "gpt-4o" => 2.50,
        "gpt-4o-mini" => 0.15,
        "gpt-4.1" => 2.00,
        "gpt-4.1-mini" => 0.40,
        "gpt-4.1-nano" => 0.10,
        "claude-sonnet-4-6" | "claude-sonnet-4-5-20250514" => 3.00,
        "claude-haiku-4-5" | "claude-haiku-4-5-20251001" => 0.80,
        "claude-opus-4-6" => 15.00,
        _ => 1.00, // conservative default for unknown models
    }
}

// ── Live metrics ──────────────────────────────────────────────────────────────

pub struct Metrics {
    started_at: Instant,

    // Request counters (atomic — lock-free, one add per request).
    pub requests_total: AtomicU64,
    pub requests_optimized: AtomicU64,
    pub requests_passthrough: AtomicU64,
    pub requests_failed: AtomicU64,
    pub requests_streaming: AtomicU64,

    // Token counters.
    pub tokens_input_original: AtomicU64,
    pub tokens_input_optimized: AtomicU64,

    // Optimization latency (our processing time, not upstream latency).
    pub optimize_latency_total_us: AtomicU64, // microseconds total
    pub optimize_latency_count: AtomicU64,    // number of measurements

    // Per-strategy and per-model breakdowns need a map → use Mutex.
    pub strategies_applied: Mutex<HashMap<String, u64>>,
    pub per_model: Mutex<HashMap<String, ModelStats>>,

    // Last request info for the dashboard.
    pub last_request: Mutex<Option<LastRequest>>,
}

#[derive(Clone, Serialize)]
pub struct LastRequest {
    pub model: String,
    pub original_tokens: u64,
    pub optimized_tokens: u64,
    pub latency_ms: u64,
}

#[derive(Default, Clone)]
pub struct ModelStats {
    pub requests: u64,
    pub tokens_saved: u64,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            started_at: Instant::now(),
            requests_total: AtomicU64::new(0),
            requests_optimized: AtomicU64::new(0),
            requests_passthrough: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_streaming: AtomicU64::new(0),
            tokens_input_original: AtomicU64::new(0),
            tokens_input_optimized: AtomicU64::new(0),
            optimize_latency_total_us: AtomicU64::new(0),
            optimize_latency_count: AtomicU64::new(0),
            strategies_applied: Mutex::new(HashMap::new()),
            per_model: Mutex::new(HashMap::new()),
            last_request: Mutex::new(None),
        }
    }
}

impl Metrics {
    /// Record a completed request with optimization results.
    pub fn record_request(
        &self,
        model: &str,
        strategies: &[String],
        original_tokens: u64,
        optimized_tokens: u64,
        was_optimized: bool,
        was_fallback: bool,
        optimize_latency_ms: u64,
    ) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);

        if was_fallback {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        } else if was_optimized {
            self.requests_optimized.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_passthrough.fetch_add(1, Ordering::Relaxed);
        }

        if original_tokens > 0 {
            self.tokens_input_original
                .fetch_add(original_tokens, Ordering::Relaxed);
            self.tokens_input_optimized
                .fetch_add(optimized_tokens, Ordering::Relaxed);

            // Per-model stats.
            if let Ok(mut map) = self.per_model.lock() {
                let entry = map.entry(model.to_string()).or_default();
                entry.requests += 1;
                entry.tokens_saved += original_tokens.saturating_sub(optimized_tokens);
            }
        }

        // Per-strategy counts.
        if was_optimized {
            if let Ok(mut map) = self.strategies_applied.lock() {
                for name in strategies {
                    *map.entry(name.clone()).or_default() += 1;
                }
            }
        }

        // Optimization latency.
        if optimize_latency_ms > 0 {
            self.optimize_latency_total_us
                .fetch_add(optimize_latency_ms * 1000, Ordering::Relaxed);
            self.optimize_latency_count.fetch_add(1, Ordering::Relaxed);
        }

        // Last request (for dashboard).
        if let Ok(mut last) = self.last_request.lock() {
            *last = Some(LastRequest {
                model: model.to_string(),
                original_tokens,
                optimized_tokens,
                latency_ms: optimize_latency_ms,
            });
        }
    }

    pub fn record_streaming(&self) {
        self.requests_streaming.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a consistent snapshot for the /cctx/metrics endpoint.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let original = self.tokens_input_original.load(Ordering::Relaxed);
        let optimized = self.tokens_input_optimized.load(Ordering::Relaxed);
        let saved = original.saturating_sub(optimized);
        let ratio = if original > 0 {
            (optimized as f64 / original as f64 * 1000.0).round() / 1000.0
        } else {
            1.0
        };

        // Per-model cost savings.
        let model_map = self.per_model.lock().map(|m| m.clone()).unwrap_or_default();
        let mut by_model: HashMap<String, ModelCostSnapshot> = HashMap::new();
        let mut total_cost_saved = 0.0;

        for (model, stats) in &model_map {
            let price = model_price_per_million(model);
            let cost = stats.tokens_saved as f64 * price / 1_000_000.0;
            total_cost_saved += cost;
            by_model.insert(
                model.clone(),
                ModelCostSnapshot {
                    requests: stats.requests,
                    tokens_saved: stats.tokens_saved,
                    saved_usd: (cost * 100.0).round() / 100.0,
                },
            );
        }

        let strategies = self
            .strategies_applied
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();

        let lat_count = self.optimize_latency_count.load(Ordering::Relaxed);
        let lat_total_us = self.optimize_latency_total_us.load(Ordering::Relaxed);
        let avg_latency_ms = if lat_count > 0 {
            (lat_total_us as f64 / lat_count as f64 / 1000.0 * 10.0).round() / 10.0
        } else {
            0.0
        };

        let last_req = self.last_request.lock().ok().and_then(|l| l.clone());

        MetricsSnapshot {
            uptime_seconds: self.started_at.elapsed().as_secs(),
            avg_optimize_latency_ms: avg_latency_ms,
            last_request: last_req,
            requests: RequestsSnapshot {
                total: self.requests_total.load(Ordering::Relaxed),
                optimized: self.requests_optimized.load(Ordering::Relaxed),
                passthrough: self.requests_passthrough.load(Ordering::Relaxed),
                failed: self.requests_failed.load(Ordering::Relaxed),
                streaming: self.requests_streaming.load(Ordering::Relaxed),
            },
            tokens: TokensSnapshot {
                input_original: original,
                input_optimized: optimized,
                saved,
                compression_ratio: ratio,
            },
            cost: CostSnapshot {
                estimated_saved_usd: (total_cost_saved * 100.0).round() / 100.0,
                by_model,
            },
            strategies,
        }
    }

    /// Reset all counters (for /cctx/metrics/reset).
    pub fn reset(&self) {
        self.requests_total.store(0, Ordering::Relaxed);
        self.requests_optimized.store(0, Ordering::Relaxed);
        self.requests_passthrough.store(0, Ordering::Relaxed);
        self.requests_failed.store(0, Ordering::Relaxed);
        self.requests_streaming.store(0, Ordering::Relaxed);
        self.tokens_input_original.store(0, Ordering::Relaxed);
        self.tokens_input_optimized.store(0, Ordering::Relaxed);
        self.optimize_latency_total_us.store(0, Ordering::Relaxed);
        self.optimize_latency_count.store(0, Ordering::Relaxed);
        if let Ok(mut m) = self.strategies_applied.lock() {
            m.clear();
        }
        if let Ok(mut m) = self.per_model.lock() {
            m.clear();
        }
        if let Ok(mut l) = self.last_request.lock() {
            *l = None;
        }
        // Note: started_at is NOT reset — uptime always reflects true uptime.
    }
}

// ── Serializable snapshot ─────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub uptime_seconds: u64,
    pub avg_optimize_latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_request: Option<LastRequest>,
    pub requests: RequestsSnapshot,
    pub tokens: TokensSnapshot,
    pub cost: CostSnapshot,
    pub strategies: HashMap<String, u64>,
}

#[derive(Serialize)]
pub struct RequestsSnapshot {
    pub total: u64,
    pub optimized: u64,
    pub passthrough: u64,
    pub failed: u64,
    pub streaming: u64,
}

#[derive(Serialize)]
pub struct TokensSnapshot {
    pub input_original: u64,
    pub input_optimized: u64,
    pub saved: u64,
    pub compression_ratio: f64,
}

#[derive(Serialize)]
pub struct CostSnapshot {
    pub estimated_saved_usd: f64,
    pub by_model: HashMap<String, ModelCostSnapshot>,
}

#[derive(Serialize)]
pub struct ModelCostSnapshot {
    pub requests: u64,
    pub tokens_saved: u64,
    pub saved_usd: f64,
}
