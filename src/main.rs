use std::collections::HashSet;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use cctx::analyzer::health::{analyze, assign_attention_zones, print_chunk_table};
use cctx::core::context::{AttentionZone, Chunk, Context as AppContext, Message};
use cctx::core::tokenizer::Tokenizer;
use cctx::formats::{self, InputFormat};
use cctx::strategies::{bookend, structural};

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

/// Output format for the `analyze` report.
#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Terminal,
    Json,
}

/// Input format override. Default "auto" inspects the content to decide.
///
/// ValueEnum auto-generates CLI parsing: the user types `--input-format openai`
/// and clap produces `InputFormatArg::Openai`.
#[derive(Clone, ValueEnum)]
enum InputFormatArg {
    Auto,
    Openai,
    Anthropic,
    Chunks,
    Raw,
}

impl InputFormatArg {
    /// Convert to the library's InputFormat. `Auto` → `None` (let detection decide).
    fn to_lib(&self) -> Option<InputFormat> {
        match self {
            Self::Auto => None,
            Self::Openai => Some(InputFormat::OpenAi),
            Self::Anthropic => Some(InputFormat::Anthropic),
            Self::Chunks => Some(InputFormat::RagChunks),
            Self::Raw => Some(InputFormat::Raw),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze context health (tokens, dead zones, duplication, score).
    Analyze {
        /// Input file. Omit or use "-" to read from stdin.
        file: Option<PathBuf>,

        /// Input format. Auto-detects by default.
        /// openai: [{role, content}]  anthropic: [{role, content:[{type,text}]}]
        /// chunks: [{content, score?}]  raw: plain text
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,

        /// Target model for budget calculation.
        #[arg(long, default_value = "default")]
        model: String,

        /// Output format: terminal (pretty report) or json (machine-readable).
        #[arg(long, value_enum, default_value_t = OutputFormat::Terminal)]
        format: OutputFormat,
    },

    /// Optimize context and emit result as JSON to stdout (or --output file).
    Optimize {
        /// Input file. Omit or use "-" to read from stdin.
        file: Option<PathBuf>,

        /// Input format. Auto-detects by default.
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,

        /// Strategy to apply. Supported: bookend, structural
        #[arg(long, default_value = "bookend")]
        strategy: String,

        /// Score chunk relevance against this query (TF-IDF for bookend,
        /// section relevance for structural markdown collapse).
        #[arg(long)]
        query: Option<String>,

        /// Write output to a file instead of stdout.
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
            input_format,
            model,
            format,
        } => {
            let raw = read_input(&file)?;
            let context = build_context(&raw, input_format.to_lib())?;
            cmd_analyze(&context, &model, format)
        }
        Commands::Optimize {
            file,
            input_format,
            strategy,
            query,
            output,
        } => {
            let raw = read_input(&file)?;
            let context = build_context(&raw, input_format.to_lib())?;
            cmd_optimize(&context, &strategy, query.as_deref(), &output)
        }
    }
}

// ── Input reading ─────────────────────────────────────────────────────────────

/// Read from a file path, or from stdin when the path is omitted / is "-".
///
/// `IsTerminal` (std since Rust 1.70) checks whether stdin is connected to a
/// keyboard (interactive) vs a pipe. If it's a terminal we bail instead of
/// blocking forever waiting for the user to type EOF.
fn read_input(file: &Option<PathBuf>) -> Result<String> {
    match file {
        Some(path) if path.to_str() != Some("-") => std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read '{}'", path.display())),
        _ => {
            // No file or "-" → read from stdin.
            if io::stdin().is_terminal() {
                anyhow::bail!(
                    "No input file. Provide a path or pipe data to stdin:\n  \
                     cctx analyze input.json\n  \
                     cat input.json | cctx analyze\n  \
                     echo 'hello world' | cctx analyze --input-format raw"
                );
            }
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .context("Failed to read from stdin")?;
            Ok(buf)
        }
    }
}

// ── Commands ──────────────────────────────────────────────────────────────────

fn cmd_analyze(context: &AppContext, model: &str, format: OutputFormat) -> Result<()> {
    let report = analyze(context, model);
    match format {
        OutputFormat::Terminal => {
            print_chunk_table(context);
            report.print_terminal();
        }
        OutputFormat::Json => report.print_json(),
    }
    Ok(())
}

fn cmd_optimize(
    context: &AppContext,
    strategy: &str,
    query: Option<&str>,
    output: &Option<PathBuf>,
) -> Result<()> {
    match strategy {
        "bookend" => run_bookend(context, query, output),
        "structural" => run_structural(context, query, output),
        other => anyhow::bail!(
            "Unknown strategy '{}'. Supported: bookend, structural",
            other
        ),
    }
}

// ── Strategy runners ──────────────────────────────────────────────────────────

fn run_bookend(
    context: &AppContext,
    query: Option<&str>,
    output: &Option<PathBuf>,
) -> Result<()> {
    let original_score = analyze(context, "default").health_score;

    let before_dead: HashSet<usize> = context
        .chunks
        .iter()
        .filter(|c| c.attention_zone == AttentionZone::DeadZone)
        .map(|c| c.index)
        .collect();
    let dead_count_before = before_dead.len();

    let mut reordered = bookend::apply(context, query);
    let total_tokens = reordered.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut reordered, total_tokens);

    let moved_from_dead = reordered
        .iter()
        .filter(|c| before_dead.contains(&c.index) && c.attention_zone == AttentionZone::Strong)
        .count();

    let new_context = AppContext::new(reordered);
    let new_score = analyze(&new_context, "default").health_score;

    emit_json(&new_context, output)?;

    let n = new_context.chunk_count();
    eprintln!();
    if let Some(q) = query {
        eprintln!("Query: \"{}\"", q);
    }
    eprintln!(
        "Reordered {} chunks. Moved {} high-relevance chunks from dead zone to safe positions.",
        n, moved_from_dead
    );
    if moved_from_dead > 0 {
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

fn run_structural(
    context: &AppContext,
    query: Option<&str>,
    output: &Option<PathBuf>,
) -> Result<()> {
    let tokenizer = Tokenizer::new().context("Failed to initialize tokenizer")?;
    let before_tokens = context.total_tokens;

    let compressed = structural::apply(context, query, &tokenizer);
    let new_context = AppContext::new(compressed);
    let after_tokens = new_context.total_tokens;
    let saved = before_tokens.saturating_sub(after_tokens);
    let pct = if before_tokens > 0 {
        (saved as f64 / before_tokens as f64) * 100.0
    } else {
        0.0
    };

    emit_json(&new_context, output)?;

    eprintln!();
    if let Some(q) = query {
        eprintln!("Query: \"{}\"", q);
    }
    eprintln!(
        "Compressed {} chunks. Tokens: {} -> {} (saved {}, {:.1}% reduction)",
        new_context.chunk_count(),
        before_tokens,
        after_tokens,
        saved,
        pct
    );
    Ok(())
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Data goes to stdout (pipe-friendly), human messages go to stderr.
fn emit_json(context: &AppContext, output: &Option<PathBuf>) -> Result<()> {
    let messages: Vec<Message> = context.chunks.iter().map(|c| c.to_message()).collect();
    let json = serde_json::to_string_pretty(&messages).context("Failed to serialize output")?;
    match output {
        Some(path) => {
            std::fs::write(path, &json)
                .with_context(|| format!("Cannot write to '{}'", path.display()))?;
            eprintln!("Wrote optimized context to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

// ── Context building ──────────────────────────────────────────────────────────

/// Parse raw input into messages (using format detection), count tokens,
/// assign attention zones, and return a ready-to-use Context.
fn build_context(raw: &str, input_format: Option<InputFormat>) -> Result<AppContext> {
    let messages =
        formats::parse_input(raw, input_format).context("Failed to parse input")?;

    if messages.is_empty() {
        anyhow::bail!("Input is empty — no messages or chunks found");
    }

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
                attention_zone: AttentionZone::Strong,
            }
        })
        .collect();

    let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total_tokens);

    Ok(AppContext::new(chunks))
}

fn default_relevance(role: &str, index: usize, total: usize) -> f64 {
    if role == "system" {
        return 1.0;
    }
    if total <= 1 {
        return 0.5;
    }
    0.1 + (index as f64 / (total - 1) as f64) * 0.8
}
