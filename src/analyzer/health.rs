use owo_colors::OwoColorize;
use serde::Serialize;

use crate::analyzer::duplication::{self, DuplicatePair};
use crate::core::context::{AttentionZone, Chunk, Context};

// ── Model budget lookup ───────────────────────────────────────────────────────

/// Map a model name to its context window size in tokens.
pub fn model_budget(model: &str) -> usize {
    // match on &str works by comparing string content (not pointer identity).
    match model {
        "gpt-4o" => 128_000,
        "gpt-4" => 128_000,
        "gpt-3.5-turbo" => 16_385,
        "claude-sonnet" => 200_000,
        "claude-opus" => 200_000,
        "claude-haiku" => 200_000,
        _ => 128_000, // safe default
    }
}

// ── Health metric structs ─────────────────────────────────────────────────────
//
// Each metric is a separate struct so the JSON output has clean nesting.
// #[derive(Serialize)] lets serde_json turn these into JSON automatically.

#[derive(Serialize)]
pub struct DeadZoneMetric {
    pub score: u32,
    pub chunk_count: usize,
    pub tokens: usize,
    pub ratio: f64,
}

#[derive(Serialize)]
pub struct DuplicationMetric {
    pub score: u32,
    pub duplicate_tokens: usize,
    pub ratio: f64,
    pub pairs: Vec<DuplicatePair>,
}

#[derive(Serialize)]
pub struct BudgetMetric {
    pub score: u32,
    pub utilization: f64,
    pub used: usize,
    pub budget: usize,
    pub model: String,
}

#[derive(Serialize)]
pub struct HealthReport {
    pub total_tokens: usize,
    pub chunk_count: usize,
    pub health_score: u32,
    pub dead_zone: DeadZoneMetric,
    pub duplication: DuplicationMetric,
    pub budget: BudgetMetric,
    pub recommendations: Vec<String>,
}

// ── Attention zone assignment ─────────────────────────────────────────────────

/// Label each chunk's attention zone based on its position in the token stream.
///
/// U-curve model (Liu et al., TACL 2024):
///   STRONG    = first 25% or last 25% of tokens
///   DEAD ZONE = middle 50%
pub fn assign_attention_zones(chunks: &mut [Chunk], total_tokens: usize) {
    let mut cumulative = 0usize;
    for chunk in chunks.iter_mut() {
        let pos = if total_tokens == 0 {
            0.0
        } else {
            cumulative as f64 / total_tokens as f64
        };
        chunk.attention_zone = if !(0.25..0.75).contains(&pos) {
            AttentionZone::Strong
        } else {
            AttentionZone::DeadZone
        };
        cumulative += chunk.token_count;
    }
}

// ── Analysis ──────────────────────────────────────────────────────────────────

/// Run all health checks and return a full report.
///
/// `model` is used to look up the context window budget (e.g. "claude-sonnet" → 200k).
pub fn analyze(context: &Context, model: &str) -> HealthReport {
    let budget = model_budget(model);

    // ── Dead zone metric ──────────────────────────────────────────────────
    let dead_chunks: Vec<&Chunk> = context.dead_zone_chunks();
    let dz_tokens: usize = dead_chunks.iter().map(|c| c.token_count).sum();
    let dz_ratio = if context.total_tokens == 0 {
        0.0
    } else {
        dz_tokens as f64 / context.total_tokens as f64
    };
    let dz_score = ((1.0 - dz_ratio) * 100.0).round() as u32;

    // ── Duplication metric ────────────────────────────────────────────────
    // Threshold 0.15: catches topical overlap without flagging unrelated messages.
    let (dup_pairs, dup_tokens) = duplication::detect_duplicates(&context.chunks, 0.15);
    let dup_ratio = if context.total_tokens == 0 {
        0.0
    } else {
        (dup_tokens as f64 / context.total_tokens as f64).min(1.0)
    };
    let dup_score = ((1.0 - dup_ratio) * 100.0).round().min(100.0) as u32;

    // ── Budget utilization metric ─────────────────────────────────────────
    // Simple linear score: more headroom = higher score.
    // Over budget (>100%) → score 0.
    let utilization = if budget == 0 {
        0.0
    } else {
        context.total_tokens as f64 / budget as f64
    };
    let budget_score = if utilization > 1.0 {
        0u32
    } else {
        ((1.0 - utilization) * 100.0).round() as u32
    };

    // ── Composite health score ────────────────────────────────────────────
    //
    // Reference doc formula uses 5 metrics with weights summing to 1.0.
    // We implement 3 of 5; renormalize the weights so they still sum to 1.0.
    //
    //   dead_zone:   0.30 / 0.75 ≈ 0.40
    //   duplication:  0.25 / 0.75 ≈ 0.33
    //   budget:       0.20 / 0.75 ≈ 0.27
    const W_TOTAL: f64 = 0.30 + 0.25 + 0.20;
    let health_score = ((dz_score as f64 * (0.30 / W_TOTAL))
        + (dup_score as f64 * (0.25 / W_TOTAL))
        + (budget_score as f64 * (0.20 / W_TOTAL)))
        .round() as u32;

    // ── Recommendations ───────────────────────────────────────────────────
    let mut recs = Vec::new();
    if dz_score < 80 {
        recs.push(
            "Run `cctx optimize --strategy bookend` to fix dead zone placement".to_string(),
        );
    }
    if dup_score < 90 && dup_tokens > 0 {
        recs.push(format!(
            "Run `cctx optimize --strategy dedup` to remove ~{} duplicate tokens",
            format_number(dup_tokens)
        ));
    }
    if budget_score < 20 {
        recs.push(format!(
            "Context is near capacity — consider `cctx compress --budget={}k`",
            budget / 2000
        ));
    }

    HealthReport {
        total_tokens: context.total_tokens,
        chunk_count: context.chunk_count(),
        health_score,
        dead_zone: DeadZoneMetric {
            score: dz_score,
            chunk_count: dead_chunks.len(),
            tokens: dz_tokens,
            ratio: dz_ratio,
        },
        duplication: DuplicationMetric {
            score: dup_score,
            duplicate_tokens: dup_tokens,
            ratio: dup_ratio,
            pairs: dup_pairs,
        },
        budget: BudgetMetric {
            score: budget_score,
            utilization,
            used: context.total_tokens,
            budget,
            model: model.to_string(),
        },
        recommendations: recs,
    }
}

// ── Terminal output ───────────────────────────────────────────────────────────
//
// Box-drawing characters form borders: ╭╮╰╯ corners, ─ horizontal, │ vertical, ├┤ divider.
// These are regular Unicode chars — Rust strings are UTF-8, so they Just Work.

/// Inner width of the report box (content between the │ borders).
const BOX_W: usize = 56;

impl HealthReport {
    /// Render the full health report with box-drawing borders and colors.
    pub fn print_terminal(&self) {
        // ── Header ────────────────────────────────────────────────────
        println!("╭{}╮", "─".repeat(BOX_W));
        box_line_centered("cctx Context Health Report");
        println!("├{}┤", "─".repeat(BOX_W));

        // ── Overall score ─────────────────────────────────────────────
        let label = status_label(self.health_score);
        let icon = status_icon(self.health_score);
        box_line(&format!(
            "  Overall Health:  {}  {}  {}",
            color_score(self.health_score),
            icon,
            label
        ));
        println!("├{}┤", "─".repeat(BOX_W));

        // ── Token summary ─────────────────────────────────────────────
        box_line(&format!(
            "  Total Tokens:     {:>8}",
            format_number(self.total_tokens)
        ));
        box_line(&format!(
            "  Model Budget:     {:>8} ({})",
            format_number(self.budget.budget),
            self.budget.model
        ));
        box_line(&format!(
            "  Utilization:      {:>7.1}%",
            self.budget.utilization * 100.0
        ));
        println!("├{}┤", "─".repeat(BOX_W));

        // ── Dead zone detail ──────────────────────────────────────────
        let dz = &self.dead_zone;
        box_line(&format!(
            "  {} Dead Zone:      {}",
            status_icon(dz.score),
            color_score(dz.score)
        ));
        box_line(&format!(
            "    -> {} chunks in the middle 50% of context",
            dz.chunk_count
        ));
        box_line(&format!(
            "    -> {} tokens ({:.1}%) sitting in dead zone",
            format_number(dz.tokens),
            dz.ratio * 100.0
        ));
        box_empty();

        // ── Duplication detail ────────────────────────────────────────
        let dup = &self.duplication;
        box_line(&format!(
            "  {} Duplication:    {}",
            status_icon(dup.score),
            color_score(dup.score)
        ));
        if dup.duplicate_tokens > 0 {
            box_line(&format!(
                "    -> ~{} tokens ({:.1}%) are near-duplicates",
                format_number(dup.duplicate_tokens),
                dup.ratio * 100.0
            ));
            // Show top 3 most-similar pairs.
            for pair in dup.pairs.iter().take(3) {
                box_line(&format!(
                    "    -> Turns {} and {} overlap {:.0}%",
                    pair.chunk_a,
                    pair.chunk_b,
                    pair.similarity * 100.0
                ));
            }
        } else {
            box_line("    -> No significant duplication detected");
        }
        box_empty();

        // ── Budget detail ─────────────────────────────────────────────
        let bg = &self.budget;
        box_line(&format!(
            "  {} Budget:         {}",
            status_icon(bg.score),
            color_score(bg.score)
        ));
        box_line(&format!(
            "    -> Using {:.1}% of {} token budget",
            bg.utilization * 100.0,
            format_number(bg.budget)
        ));

        // ── Close box ─────────────────────────────────────────────────
        println!("╰{}╯", "─".repeat(BOX_W));

        // ── Recommendations (outside the box) ─────────────────────────
        if !self.recommendations.is_empty() {
            println!();
            println!("Recommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }
    }

    /// Emit the report as pretty-printed JSON.
    pub fn print_json(&self) {
        // serde_json::to_string_pretty can't fail on our types (no maps with
        // non-string keys), so unwrap is safe here.
        println!("{}", serde_json::to_string_pretty(self).unwrap());
    }
}

// ── Chunk table (printed above the box) ───────────────────────────────────────

/// Print a per-chunk position table with colored zone labels.
pub fn print_chunk_table(context: &Context) {
    println!();
    println!(
        "  {:<4} {:<12} {:>8}  {:<12} relevance",
        "idx", "role", "tokens", "zone"
    );
    // "─" is a Unicode box-drawing dash — nicer than ASCII "-".
    println!("  {}", "─".repeat(56));

    for chunk in &context.chunks {
        let zone_colored = match chunk.attention_zone {
            AttentionZone::Strong => format!("{}", "STRONG   ".green()),
            AttentionZone::DeadZone => format!("{}", "DEAD ZONE".red()),
        };
        println!(
            "  [{:<2}] {:<12} {:>6} t  [{}] score={:.2}",
            chunk.index, chunk.role, chunk.token_count, zone_colored, chunk.relevance_score,
        );
    }
    println!();
}

// ── Box-drawing helpers ───────────────────────────────────────────────────────

/// Print a left-aligned content line padded to fit inside the box.
///
/// The tricky part: `content` may contain ANSI escape codes (from owo-colors)
/// that are invisible but take up bytes. We strip them to measure the *visible*
/// length, then pad with spaces so the right │ border lines up.
fn box_line(content: &str) {
    let vis = visible_len(content);
    let pad = BOX_W.saturating_sub(vis);
    println!("│{}{}│", content, " ".repeat(pad));
}

fn box_line_centered(text: &str) {
    let total_pad = BOX_W.saturating_sub(text.len());
    let left = total_pad / 2;
    let right = total_pad - left;
    println!("│{}{}{}│", " ".repeat(left), text.bold(), " ".repeat(right));
}

fn box_empty() {
    println!("│{}│", " ".repeat(BOX_W));
}

// ── Formatting utilities ──────────────────────────────────────────────────────

/// Compute visible character count, ignoring ANSI SGR escape sequences.
///
/// ANSI SGR codes look like: ESC [ <digits;digits...> m
/// owo-colors produces these for colors/bold. They're invisible in the terminal
/// but inflate .len() and .chars().count(). We strip them for padding math.
fn visible_len(s: &str) -> usize {
    let mut count = 0usize;
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
            // Skip all chars inside the escape sequence.
        } else {
            count += 1;
        }
    }
    count
}

/// Format a score with color: green ≥ 80, yellow ≥ 60, red otherwise.
fn color_score(score: u32) -> String {
    let text = format!("{}/100", score);
    if score >= 80 {
        format!("{}", text.green().bold())
    } else if score >= 60 {
        format!("{}", text.yellow().bold())
    } else {
        format!("{}", text.red().bold())
    }
}

/// Status icon for a metric score.
fn status_icon(score: u32) -> &'static str {
    if score >= 80 {
        "✓"
    } else if score >= 60 {
        "⚠"
    } else {
        "✗"
    }
}

/// Human-readable status label.
fn status_label(score: u32) -> &'static str {
    if score >= 80 {
        "GOOD"
    } else if score >= 60 {
        "NEEDS WORK"
    } else {
        "POOR"
    }
}

/// Insert commas into a number: 128000 → "128,000".
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, &ch) in chars.iter().enumerate() {
        // Insert comma before every group of 3 digits (not at position 0).
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(ch);
    }
    result
}
