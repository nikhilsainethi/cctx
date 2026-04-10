// `crate::` means "from the root of *this* library crate" (lib.rs).
// This is how modules inside the same crate import each other.
use crate::core::context::{AttentionZone, Chunk, Context};

// ── Health report ─────────────────────────────────────────────────────────────

pub struct HealthReport {
    pub total_tokens: usize,
    pub chunk_count: usize,
    pub dead_zone_count: usize,
    pub dead_zone_tokens: usize,
    /// Fraction of tokens sitting in the attention dead zone (0.0–1.0).
    pub dead_zone_ratio: f64,
    /// Overall score 0–100. Higher is better. Penalizes dead-zone content.
    pub health_score: u32,
}

impl HealthReport {
    pub fn print(&self, filename: &str) {
        println!("cctx Context Health Report");
        println!("==========================");
        println!("File:         {filename}");
        println!(
            "Chunks:       {} messages  |  Total tokens: {}",
            self.chunk_count, self.total_tokens
        );
        println!();
        println!(
            "Dead zone:    {}/{} chunks  ({} tokens, {:.1}% of total)",
            self.dead_zone_count,
            self.chunk_count,
            self.dead_zone_tokens,
            self.dead_zone_ratio * 100.0,
        );

        // Format the health score with a status label.
        let status = match self.health_score {
            80..=100 => "GOOD",
            60..=79 => "NEEDS WORK",
            _ => "POOR",
        };
        println!("Health score: {}/100 — {status}", self.health_score);

        if self.health_score < 80 {
            println!();
            println!("Tip: run `cctx optimize --strategy bookend` to reorder chunks out of the dead zone.");
        }
    }
}

// ── Attention zone assignment ─────────────────────────────────────────────────

/// Walk the chunk list and label each chunk with its AttentionZone based on
/// where it falls in the total token stream.
///
/// U-curve model (Liu et al., TACL 2024):
///   STRONG   = first 25% or last 25% of tokens
///   DEAD ZONE = middle 50%
///
/// `chunks: &mut Vec<Chunk>` — a mutable borrow of the vector.
/// We modify chunks in place; the caller still owns the Vec after this returns.
pub fn assign_attention_zones(chunks: &mut Vec<Chunk>, total_tokens: usize) {
    let mut cumulative = 0usize;

    for chunk in chunks.iter_mut() {
        // Normalized start position: 0.0 = very beginning, 1.0 = very end.
        let pos = if total_tokens == 0 {
            0.0f64
        } else {
            cumulative as f64 / total_tokens as f64
        };

        chunk.attention_zone = if pos < 0.25 || pos >= 0.75 {
            AttentionZone::Strong
        } else {
            AttentionZone::DeadZone
        };

        cumulative += chunk.token_count;
    }
}

// ── Analysis ──────────────────────────────────────────────────────────────────

/// Analyze a context and return a health report.
/// `context: &Context` — immutable borrow; we read but don't modify.
pub fn analyze(context: &Context) -> HealthReport {
    let dead_chunks: Vec<&Chunk> = context.dead_zone_chunks();
    let dead_zone_count = dead_chunks.len();
    // Iterators are lazy in Rust — nothing runs until you call a consumer like .sum().
    let dead_zone_tokens: usize = dead_chunks.iter().map(|c| c.token_count).sum();

    let dead_zone_ratio = if context.total_tokens == 0 {
        0.0
    } else {
        dead_zone_tokens as f64 / context.total_tokens as f64
    };

    // Simple health score: penalise dead-zone content linearly.
    // Week 2 will add duplication, budget-utilisation, and freshness sub-scores.
    let health_score = ((1.0 - dead_zone_ratio) * 100.0).round() as u32;

    HealthReport {
        total_tokens: context.total_tokens,
        chunk_count: context.chunk_count(),
        dead_zone_count,
        dead_zone_tokens,
        dead_zone_ratio,
        health_score,
    }
}

/// Print a per-chunk position table — called by the `analyze` command.
pub fn print_chunk_table(context: &Context) {
    println!();
    println!(
        "  {:<4} {:<12} {:>8}  {:<12} {}",
        "idx", "role", "tokens", "zone", "relevance"
    );
    println!("  {}", "-".repeat(54));

    for chunk in &context.chunks {
        println!(
            "  [{:<2}] {:<12} {:>6} t  [{}] score={:.2}",
            chunk.index,
            chunk.role,
            chunk.token_count,
            chunk.attention_zone,
            chunk.relevance_score,
        );
    }
    println!();
}
