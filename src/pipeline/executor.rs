//! Pipeline executor — runs strategies in sequence with per-step logging
//! and optional token budget enforcement.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;

use crate::core::context::{Chunk, Context};
use crate::pipeline::{PipelineConfig, Strategy};

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// An ordered list of strategies applied in sequence.
/// Output of each strategy feeds as input to the next.
pub struct Pipeline {
    strategies: Vec<Box<dyn Strategy>>,
    config: PipelineConfig,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Pipeline {
            strategies: Vec::new(),
            config,
        }
    }

    pub fn add(&mut self, strategy: Box<dyn Strategy>) {
        self.strategies.push(strategy);
    }

    /// Run all strategies in order. Logs per-step metrics to stderr.
    /// Returns the final context after all strategies have been applied.
    pub fn run(&self, context: Context) -> Result<Context> {
        let names: Vec<&str> = self.strategies.iter().map(|s| s.name()).collect();
        eprintln!("Pipeline: {}", names.join(", "));

        let mut current = context;

        for strategy in &self.strategies {
            let before = current.total_tokens;
            let start = Instant::now();

            let chunks = strategy.apply(&current, &self.config)?;
            current = Context::new(chunks);

            let ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  {:<12} :: {} -> {} tokens  ({:.1}ms)",
                strategy.name(),
                before,
                current.total_tokens,
                ms
            );
        }

        Ok(current)
    }

    /// Run the pipeline, then enforce a token budget by truncating if needed.
    /// Returns the final context and any warnings about dropped chunks.
    pub fn run_with_budget(
        &self,
        context: Context,
        budget: usize,
    ) -> Result<(Context, Vec<String>)> {
        let mut current = self.run(context)?;

        if current.total_tokens <= budget {
            return Ok((current, vec![]));
        }

        eprintln!(
            "Budget enforcement: {} tokens -> {} budget",
            current.total_tokens, budget
        );

        let (truncated, warnings) = truncate_to_budget(&current.chunks, budget);
        current = Context::new(truncated);

        for w in &warnings {
            eprintln!("  {}", w);
        }

        Ok((current, warnings))
    }
}

// ── Budget enforcement ────────────────────────────────────────────────────────
//
// When the optimized context still exceeds the token budget, we drop the
// oldest unprotected chunks until we're under budget.
//
// Protected chunks (never dropped):
//   - System messages (define the LLM's role/instructions)
//   - Last 2 user messages (represent the current intent)

pub fn truncate_to_budget(chunks: &[Chunk], budget: usize) -> (Vec<Chunk>, Vec<String>) {
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    if total <= budget {
        return (chunks.to_vec(), vec![]);
    }

    // ── Identify protected positions ──────────────────────────────────────
    let user_positions: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.role == "user")
        .map(|(i, _)| i)
        .collect();
    // .rev().take(2) gives the last 2 user message positions.
    let protected_users: HashSet<usize> = user_positions.iter().rev().take(2).copied().collect();

    let protected: HashSet<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(i, c)| c.role == "system" || protected_users.contains(i))
        .map(|(i, _)| i)
        .collect();

    // ── Drop oldest unprotected until under budget ────────────────────────
    let mut drop_set: HashSet<usize> = HashSet::new();
    let mut running = total;
    let mut warnings: Vec<String> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if running <= budget {
            break;
        }
        if protected.contains(&i) {
            continue;
        }
        drop_set.insert(i);
        running -= chunk.token_count;
        warnings.push(format!(
            "Dropped chunk {} ({}, {} tokens)",
            chunk.index, chunk.role, chunk.token_count
        ));
    }

    let result: Vec<Chunk> = chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop_set.contains(i))
        .map(|(_, c)| c.clone())
        .collect();

    if running > budget {
        warnings.push(format!(
            "Warning: protected chunks alone use {} tokens (budget {})",
            running, budget
        ));
    }

    (result, warnings)
}
