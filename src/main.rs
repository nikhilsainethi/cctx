use std::collections::HashSet;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use cctx::analyzer::health::{analyze, assign_attention_zones, print_chunk_table};
use cctx::core::context::{AttentionZone, Chunk, Context as AppContext, Message};
use cctx::core::tokenizer::Tokenizer;
use cctx::strategies::bookend;

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "cctx",
    about = "Context Compiler for LLMs — analyze and optimize your LLM context window",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Terminal,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a context file and report health metrics (tokens, dead zones, duplication, score).
    Analyze {
        /// Path to a JSON file: array of {"role": "...", "content": "..."} objects.
        file: PathBuf,

        /// Target model for budget calculation.
        /// Supported: gpt-4o (128k), claude-sonnet (200k), claude-opus (200k), gpt-3.5-turbo (16k).
        #[arg(long, default_value = "default")]
        model: String,

        /// Output format: terminal (pretty box report) or json (machine-readable).
        #[arg(long, value_enum, default_value_t = OutputFormat::Terminal)]
        format: OutputFormat,
    },

    /// Optimize a context file by applying a strategy and output the result as JSON.
    Optimize {
        /// Path to a JSON file (same format as `analyze`).
        file: PathBuf,

        /// Strategy to apply. Currently supported: bookend
        #[arg(long, default_value = "bookend")]
        strategy: String,

        /// Score chunk relevance against this query using TF-IDF.
        /// Without this, uses a heuristic: system messages and last 3 user
        /// messages get highest priority.
        #[arg(long)]
        query: Option<String>,

        /// Write output to a file instead of stdout.
        /// When omitted, optimized JSON goes to stdout (pipe-friendly).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Analyze {
            file,
            model,
            format,
        } => cmd_analyze(file, model, format),
        Commands::Optimize {
            file,
            strategy,
            query,
            output,
        } => cmd_optimize(file, strategy, query, output),
    }
}

// ── Commands ──────────────────────────────────────────────────────────────────

fn cmd_analyze(file: PathBuf, model: String, format: OutputFormat) -> Result<()> {
    let context = build_context(&file)?;
    let report = analyze(&context, &model);

    match format {
        OutputFormat::Terminal => {
            print_chunk_table(&context);
            report.print_terminal();
        }
        OutputFormat::Json => {
            report.print_json();
        }
    }

    Ok(())
}

fn cmd_optimize(
    file: PathBuf,
    strategy: String,
    query: Option<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    if strategy != "bookend" {
        anyhow::bail!(
            "Unknown strategy '{}'. Currently supported: bookend",
            strategy
        );
    }

    let context = build_context(&file)?;
    let original_score = analyze(&context, "default").health_score;

    // Snapshot: which chunk indices sit in the dead zone BEFORE optimization.
    // HashSet<usize> is a set of unique integers — O(1) contains() checks.
    let before_dead: HashSet<usize> = context
        .chunks
        .iter()
        .filter(|c| c.attention_zone == AttentionZone::DeadZone)
        .map(|c| c.index)
        .collect();
    let dead_count_before = before_dead.len();

    // Apply bookend with optional query.
    // .as_deref() converts Option<String> → Option<&str> by borrowing the
    // inner String as a &str slice. This avoids moving ownership.
    let mut reordered = bookend::apply(&context, query.as_deref());

    // Re-assign attention zones based on the new ordering.
    let total_tokens = reordered.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut reordered, total_tokens);

    // Count chunks rescued: were in dead zone before, now in strong zone.
    let moved_from_dead = reordered
        .iter()
        .filter(|c| before_dead.contains(&c.index) && c.attention_zone == AttentionZone::Strong)
        .count();

    let new_context = AppContext::new(reordered);
    let new_score = analyze(&new_context, "default").health_score;

    // ── Serialize output ──────────────────────────────────────────────────
    let messages: Vec<Message> = new_context.chunks.iter().map(|c| c.to_message()).collect();
    let json = serde_json::to_string_pretty(&messages).context("Failed to serialize output")?;

    // Write to file or stdout. File path goes to --output; without it, stdout.
    match &output {
        Some(path) => {
            std::fs::write(path, &json)
                .with_context(|| format!("Cannot write to '{}'", path.display()))?;
            eprintln!("Wrote optimized context to {}", path.display());
        }
        None => {
            println!("{json}");
        }
    }

    // ── Summary to stderr ─────────────────────────────────────────────────
    let n = new_context.chunk_count();
    eprintln!();

    if let Some(q) = &query {
        eprintln!("Query: \"{}\"", q);
    }

    eprintln!(
        "Reordered {} chunks. Moved {} high-relevance chunks from dead zone to safe positions.",
        n, moved_from_dead
    );

    if moved_from_dead > 0 {
        // Estimate based on Liu et al.: 10–30% recall improvement for rescued content.
        // Scale by what fraction of the dead zone we actually rescued.
        let rescue_ratio = moved_from_dead as f64 / dead_count_before.max(1) as f64;
        let estimate = if rescue_ratio > 0.5 {
            "~20-30%"
        } else {
            "~10-20%"
        };
        eprintln!("Estimated recall improvement: {}", estimate);
    }

    eprintln!("Health score: {} -> {}", original_score, new_score);

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_context(file: &PathBuf) -> Result<AppContext> {
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read file '{}'", file.display()))?;

    let messages: Vec<Message> = serde_json::from_str(&raw)
        .context("Invalid JSON — expected an array of {\"role\": ..., \"content\": ...} objects")?;

    let tokenizer = Tokenizer::new().context("Failed to initialize BPE tokenizer")?;
    let n = messages.len();

    let mut chunks: Vec<Chunk> = messages
        .into_iter()
        .enumerate()
        .map(|(i, msg)| {
            let relevance = msg
                .relevance_score
                .map(|s| s.clamp(0.0, 1.0))
                .unwrap_or_else(|| default_relevance(&msg.role, i, n));

            Chunk {
                index: i,
                role: msg.role,
                content: msg.content.clone(),
                token_count: tokenizer.count(&msg.content),
                relevance_score: relevance,
                attention_zone: AttentionZone::Strong, // placeholder
            }
        })
        .collect();

    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);

    Ok(AppContext::new(chunks))
}

/// Initial relevance score assigned during context building.
/// The bookend strategy overrides these with its own scoring, but other
/// strategies (future) may use these defaults.
fn default_relevance(role: &str, index: usize, total: usize) -> f64 {
    if role == "system" {
        return 1.0;
    }
    if total <= 1 {
        return 0.5;
    }
    0.1 + (index as f64 / (total - 1) as f64) * 0.8
}
