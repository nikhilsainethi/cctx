use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use cctx::analyzer::health::{analyze, assign_attention_zones, print_chunk_table};
use cctx::core::context::{AttentionZone, Chunk, Context as AppContext, Message};
use cctx::core::tokenizer::Tokenizer;
use cctx::formats::{self, InputFormat};
use cctx::pipeline::executor::Pipeline;
use cctx::pipeline::{make_strategy, preset_strategies, PipelineConfig};

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

#[derive(Clone, ValueEnum)]
enum InputFormatArg {
    Auto,
    Openai,
    Anthropic,
    Chunks,
    Raw,
}

impl InputFormatArg {
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
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,
        #[arg(long, default_value = "default")]
        model: String,
        /// Output format: terminal (pretty report) or json (machine-readable).
        #[arg(long, value_enum, default_value_t = OutputFormat::Terminal)]
        format: OutputFormat,
    },

    /// Optimize context by applying a pipeline of strategies.
    ///
    /// Strategies run in the order specified. Use --strategy multiple times
    /// to chain them, or --preset for a predefined pipeline.
    Optimize {
        /// Input file. Omit or use "-" to read from stdin.
        file: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,
        /// Strategy to apply. Repeat for chaining: --strategy bookend --strategy structural.
        /// Supported: bookend, structural, dedup
        #[arg(long)]
        strategy: Vec<String>,
        /// Preset pipeline: safe (bookend), balanced (bookend+structural),
        /// aggressive (bookend+structural+dedup).
        #[arg(long)]
        preset: Option<String>,
        /// Score relevance against this query.
        #[arg(long)]
        query: Option<String>,
        /// Token budget. If the result exceeds this, oldest non-critical chunks are dropped.
        #[arg(long)]
        budget: Option<usize>,
        /// Write output to a file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Compress context to fit a hard token budget.
    ///
    /// Applies structural compression, then truncates oldest messages
    /// until the budget is met. System messages and the last 2 user
    /// messages are never dropped.
    Compress {
        file: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,
        /// Target token budget (required).
        #[arg(long)]
        budget: usize,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Count tokens in the input. Pipe-friendly — prints just the number.
    ///
    /// Useful as a pipeline component:
    ///   cctx optimize input.json --preset balanced | cctx count
    Count {
        file: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,
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
            let ctx = build_context(&raw, input_format.to_lib())?;
            cmd_analyze(&ctx, &model, format)
        }
        Commands::Optimize {
            file,
            input_format,
            strategy,
            preset,
            query,
            budget,
            output,
        } => {
            let raw = read_input(&file)?;
            let ctx = build_context(&raw, input_format.to_lib())?;
            cmd_optimize(ctx, strategy, preset, query, budget, &output)
        }
        Commands::Compress {
            file,
            input_format,
            budget,
            query,
            output,
        } => {
            let raw = read_input(&file)?;
            let ctx = build_context(&raw, input_format.to_lib())?;
            cmd_compress(ctx, budget, query, &output)
        }
        Commands::Count {
            file,
            input_format,
        } => {
            let raw = read_input(&file)?;
            let ctx = build_context(&raw, input_format.to_lib())?;
            // Just print the token count — nothing else. Pipe-friendly.
            println!("{}", ctx.total_tokens);
            Ok(())
        }
    }
}

// ── Input reading ─────────────────────────────────────────────────────────────

fn read_input(file: &Option<PathBuf>) -> Result<String> {
    match file {
        Some(path) if path.to_str() != Some("-") => std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read '{}'", path.display())),
        _ => {
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
    context: AppContext,
    strategy_args: Vec<String>,
    preset: Option<String>,
    query: Option<String>,
    budget: Option<usize>,
    output: &Option<PathBuf>,
) -> Result<()> {
    let before_tokens = context.total_tokens;

    // ── Resolve strategy list ─────────────────────────────────────────────
    let names: Vec<String> = if let Some(p) = &preset {
        preset_strategies(p)?
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    } else if strategy_args.is_empty() {
        vec!["bookend".to_string()]
    } else {
        strategy_args
    };

    // ── Build and run pipeline ────────────────────────────────────────────
    let config = PipelineConfig {
        query,
        tokenizer: Tokenizer::new().context("Failed to initialize tokenizer")?,
    };
    let mut pipeline = Pipeline::new(config);
    for name in &names {
        pipeline.add(make_strategy(name)?);
    }

    let (final_ctx, warnings) = if let Some(b) = budget {
        let (ctx, w) = pipeline.run_with_budget(context, b)?;
        // Re-assign attention zones on the final result.
        finalize_context(ctx, w)
    } else {
        let ctx = pipeline.run(context)?;
        finalize_context(ctx, vec![])
    };

    emit_json(&final_ctx, output)?;
    print_summary(before_tokens, &final_ctx, &warnings);

    Ok(())
}

fn cmd_compress(
    context: AppContext,
    budget: usize,
    query: Option<String>,
    output: &Option<PathBuf>,
) -> Result<()> {
    let before_tokens = context.total_tokens;

    // Compress always runs structural first, then enforces budget.
    let config = PipelineConfig {
        query,
        tokenizer: Tokenizer::new().context("Failed to initialize tokenizer")?,
    };
    let mut pipeline = Pipeline::new(config);
    pipeline.add(make_strategy("structural")?);

    let (final_ctx, warnings) = {
        let (ctx, w) = pipeline.run_with_budget(context, budget)?;
        finalize_context(ctx, w)
    };

    emit_json(&final_ctx, output)?;
    print_summary(before_tokens, &final_ctx, &warnings);

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Re-assign attention zones after the pipeline finishes and bundle warnings.
fn finalize_context(context: AppContext, warnings: Vec<String>) -> (AppContext, Vec<String>) {
    let mut chunks = context.chunks;
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total);
    (AppContext::new(chunks), warnings)
}

fn print_summary(before: usize, context: &AppContext, warnings: &[String]) {
    let after = context.total_tokens;
    let saved = before.saturating_sub(after);

    eprintln!();
    if saved > 0 && before > 0 {
        let pct = (saved as f64 / before as f64) * 100.0;
        eprintln!(
            "Tokens: {} -> {} (saved {}, {:.1}% reduction)",
            before, after, saved, pct
        );
    } else {
        eprintln!("Tokens: {} -> {}", before, after);
    }

    if !warnings.is_empty() {
        eprintln!();
        for w in warnings {
            eprintln!("  {}", w);
        }
    }
}

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

fn build_context(raw: &str, input_format: Option<InputFormat>) -> Result<AppContext> {
    let messages = formats::parse_input(raw, input_format).context("Failed to parse input")?;

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
