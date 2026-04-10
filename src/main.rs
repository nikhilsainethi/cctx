// The binary crate (main.rs) imports the library crate by its package name.
// `use cctx::...` is how main.rs reaches into lib.rs and its child modules.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use cctx::analyzer::health::{analyze, assign_attention_zones, print_chunk_table};
use cctx::core::context::{AttentionZone, Chunk, Context as AppContext, Message};
use cctx::core::tokenizer::Tokenizer;
use cctx::strategies::bookend;

// ── CLI definition ────────────────────────────────────────────────────────────
//
// clap's derive API: annotate a struct with #[derive(Parser)] and clap generates
// all the argument parsing, help text, and error messages automatically.

#[derive(Parser)]
#[command(
    name = "cctx",
    about = "Context Compiler for LLMs — analyze and optimize your LLM context window",
    version
)]
struct Cli {
    // #[command(subcommand)] tells clap that this field holds one of the Commands variants.
    #[command(subcommand)]
    command: Commands,
}

// Each variant of this enum becomes a subcommand.
// The doc comment on each variant becomes the help text shown in `cctx --help`.
#[derive(Subcommand)]
enum Commands {
    /// Analyze a context file and report health metrics (tokens, dead zones, score).
    Analyze {
        /// Path to a JSON file: array of {"role": "...", "content": "..."} objects.
        file: PathBuf,
    },

    /// Optimize a context file by applying a strategy and output the result as JSON.
    Optimize {
        /// Path to a JSON file (same format as `analyze`).
        file: PathBuf,

        /// Strategy to apply. Currently supported: bookend
        #[arg(long, default_value = "bookend")]
        strategy: String,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

// Returning Result<()> from main lets us use ? throughout.
// If we return Err, Rust prints the error message and exits with code 1.
fn main() -> Result<()> {
    let cli = Cli::parse();

    // match on an enum is exhaustive — the compiler ensures every variant is handled.
    match cli.command {
        Commands::Analyze { file } => cmd_analyze(file),
        Commands::Optimize { file, strategy } => cmd_optimize(file, strategy),
    }
}

// ── Commands ──────────────────────────────────────────────────────────────────

fn cmd_analyze(file: PathBuf) -> Result<()> {
    let filename = file.display().to_string();
    let context = build_context(&file)?;

    print_chunk_table(&context);

    let report = analyze(&context);
    report.print(&filename);

    Ok(())
}

fn cmd_optimize(file: PathBuf, strategy: String) -> Result<()> {
    if strategy != "bookend" {
        anyhow::bail!("Unknown strategy '{}'. Currently supported: bookend", strategy);
    }

    let context = build_context(&file)?;
    let original_tokens = context.total_tokens;
    let original_score = analyze(&context).health_score;

    // Apply the bookend strategy — returns a new Vec<Chunk> in reordered order.
    let mut reordered_chunks = bookend::apply(&context);

    // Re-assign attention zones: after reordering, each chunk occupies a new
    // position in the token stream, so its zone label must be recomputed.
    let total_tokens = reordered_chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut reordered_chunks, total_tokens);

    let new_context = AppContext::new(reordered_chunks);
    let new_score = analyze(&new_context).health_score;

    // Emit the optimized messages as JSON to stdout (pipe-friendly).
    let messages: Vec<Message> = new_context.chunks.iter().map(|c| c.to_message()).collect();
    // serde_json::to_string_pretty formats JSON with indentation.
    let json = serde_json::to_string_pretty(&messages).context("Failed to serialize output")?;
    println!("{json}");

    // Print stats to stderr so stdout stays clean JSON.
    eprintln!();
    eprintln!("--- bookend applied ---");
    eprintln!("Tokens:       {} (unchanged — pure reorder)", original_tokens);
    eprintln!("Health score: {} → {}", original_score, new_score);
    eprintln!("Chunk order:  {}", chunk_order_summary(&new_context));

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Read a file, parse the JSON, count tokens, assign attention zones, and
/// return a fully-populated Context ready for analysis or optimization.
fn build_context(file: &PathBuf) -> Result<AppContext> {
    // std::fs::read_to_string returns Result<String>.
    // .with_context() attaches a descriptive message if it returns Err.
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read file '{}'", file.display()))?;

    // serde_json::from_str parses JSON into our Message type.
    // The turbofish ::<Vec<Message>> is the type hint — tells serde what to produce.
    let messages: Vec<Message> = serde_json::from_str(&raw)
        .context("Invalid JSON — expected an array of {\"role\": ..., \"content\": ...} objects")?;

    let tokenizer = Tokenizer::new().context("Failed to initialize BPE tokenizer")?;
    let n = messages.len();

    // Build initial chunks. into_iter() consumes the Vec (moves each element).
    // enumerate() pairs each element with its index: (0, msg0), (1, msg1), ...
    let mut chunks: Vec<Chunk> = messages
        .into_iter()
        .enumerate()
        .map(|(i, msg)| {
            let relevance = msg
                .relevance_score
                // .map() transforms Some(v) → Some(f(v)), leaves None unchanged.
                .map(|s| s.clamp(0.0, 1.0))
                // .unwrap_or_else() provides a fallback when the Option is None.
                .unwrap_or_else(|| default_relevance(&msg.role, i, n));

            Chunk {
                index: i,
                role: msg.role,
                content: msg.content.clone(),
                token_count: tokenizer.count(&msg.content),
                relevance_score: relevance,
                attention_zone: AttentionZone::Strong, // placeholder — set below
            }
        })
        .collect();

    // Total tokens needed before we can assign zones (zones depend on position/total).
    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);

    Ok(AppContext::new(chunks))
}

/// Heuristic relevance score when the user doesn't provide one.
/// System prompts are always critical. For user/assistant turns, recency = relevance.
fn default_relevance(role: &str, index: usize, total: usize) -> f64 {
    if role == "system" {
        return 1.0;
    }
    if total <= 1 {
        return 0.5;
    }
    // Map index 0..n-1 linearly to 0.1..0.9 — later messages score higher.
    0.1 + (index as f64 / (total - 1) as f64) * 0.8
}

/// Build a compact string showing new chunk order, e.g. "[2] [4] [0] [3] [1]".
fn chunk_order_summary(context: &AppContext) -> String {
    context
        .chunks
        .iter()
        .map(|c| format!("[{}]", c.index))
        .collect::<Vec<_>>()
        .join(" → ")
}
