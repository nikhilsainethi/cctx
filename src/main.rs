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
        /// Embedding provider for semantic dedup: ollama or openai.
        /// Requires: cargo build --features embeddings
        #[arg(long)]
        embedding_provider: Option<String>,
        /// Cosine similarity threshold for semantic dedup (0.0-1.0).
        #[arg(long, default_value_t = 0.85)]
        dedup_threshold: f64,
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

    /// Run as an OpenAI-compatible proxy server.
    ///
    /// Intercepts /v1/chat/completions requests, optimizes context, and
    /// forwards to the upstream API. Change your base URL and every API
    /// call flows through cctx automatically.
    ///
    /// Requires: cargo build --features proxy
    #[cfg(feature = "proxy")]
    Proxy {
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        /// Upstream LLM API base URL.
        #[arg(long)]
        upstream: String,
        /// Optimization strategies to apply. Repeat for chaining.
        /// Without this, the proxy is a passthrough (no optimization).
        #[arg(long)]
        strategy: Vec<String>,
        /// Token budget. After strategies run, if tokens still exceed this,
        /// oldest non-critical messages are dropped.
        #[arg(long)]
        budget: Option<usize>,
        /// Dry-run: optimize and log savings, but forward the original
        /// unmodified request. Safe for testing in production.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// Upstream request timeout in seconds.
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Embedding provider for semantic dedup: ollama or openai.
        #[arg(long)]
        embedding_provider: Option<String>,
        /// Cosine similarity threshold for semantic dedup.
        #[arg(long, default_value_t = 0.85)]
        dedup_threshold: f64,
    },

    /// Compare two context files side-by-side (before vs after).
    ///
    /// Shows token changes, message differences, dead zone improvements,
    /// and health score delta. Useful for evaluating optimization impact.
    Diff {
        /// Original (before) file.
        before: PathBuf,
        /// Optimized (after) file.
        after: PathBuf,
        #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
        input_format: InputFormatArg,
        /// Output format: terminal (table) or json.
        #[arg(long, value_enum, default_value_t = OutputFormat::Terminal)]
        format: OutputFormat,
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
            embedding_provider,
            dedup_threshold,
        } => {
            let raw = read_input(&file)?;
            let ctx = build_context(&raw, input_format.to_lib())?;
            let provider = make_embedding_provider(embedding_provider.as_deref())?;
            cmd_optimize(ctx, strategy, preset, query, budget, &output, provider, dedup_threshold)
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
            if ctx.chunk_count() == 0 {
                println!("0");
            } else {
                println!("{}", ctx.total_tokens);
            }
            Ok(())
        }
        // ── Proxy command ──────────────────────────────────────────────────
        //
        // #[cfg(feature = "proxy")] means this arm only exists when compiled
        // with --features proxy. Without it, `cctx proxy` is "unrecognized".
        //
        // We create a tokio Runtime manually rather than marking main() as
        // #[tokio::main] because we don't want to require tokio for CLI-only
        // users. The runtime is only constructed when someone actually runs
        // the proxy command.
        //
        // # Sync vs Async (the key difference from everything else in cctx)
        //
        // All our CLI commands so far are synchronous: each function call
        // blocks the current thread until it completes. This is fine for a
        // CLI that does one thing and exits.
        //
        // The proxy is different: it handles many concurrent HTTP requests.
        // With sync code, you'd need one OS thread per concurrent request
        // (expensive — each thread uses ~8MB of stack). With async, a single
        // thread can handle thousands of requests by switching between them
        // at `.await` points (when waiting for network I/O).
        //
        // tokio is the async runtime that makes this work. It provides:
        //   - A multi-threaded task scheduler (like a green-thread executor)
        //   - An I/O driver that wakes tasks when sockets are ready
        //   - Timers, channels, and sync primitives for async code
        //
        // `runtime.block_on(future)` bridges sync → async: it runs the
        // future on the tokio runtime and blocks the current thread until
        // the future completes (which for a server means "until killed").
        #[cfg(feature = "proxy")]
        Commands::Proxy {
            listen,
            upstream,
            strategy,
            budget,
            dry_run,
            timeout,
            embedding_provider,
            dedup_threshold,
        } => {
            for name in &strategy {
                cctx::pipeline::make_strategy(name)?;
            }
            if let Some(b) = budget {
                if b == 0 {
                    anyhow::bail!("Budget must be a positive number");
                }
            }
            let rt = tokio::runtime::Runtime::new()
                .context("Failed to create tokio async runtime")?;
            rt.block_on(cctx::proxy::server::run(
                cctx::proxy::config::ProxyConfig {
                    listen_addr: listen,
                    upstream_url: upstream,
                    strategy_names: strategy,
                    budget,
                    dry_run,
                    timeout_secs: timeout,
                    embedding_provider,
                    dedup_threshold,
                },
            ))
        }

        Commands::Diff {
            before,
            after,
            input_format,
            format,
        } => {
            let fmt = input_format.to_lib();
            let raw_b = read_file(&before)?;
            let raw_a = read_file(&after)?;
            let ctx_b = build_context(&raw_b, fmt.clone())?;
            let ctx_a = build_context(&raw_a, fmt)?;
            cmd_diff(&ctx_b, &ctx_a, format)
        }
    }
}

// ── Input reading ─────────────────────────────────────────────────────────────

fn read_input(file: &Option<PathBuf>) -> Result<String> {
    match file {
        Some(path) if path.to_str() != Some("-") => read_file(path),
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
    if context.chunk_count() == 0 {
        eprintln!("No context to analyze");
        return Ok(());
    }
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
    embedding_provider: Option<std::sync::Arc<dyn cctx::embeddings::EmbeddingProvider>>,
    dedup_threshold: f64,
) -> Result<()> {
    if context.chunk_count() == 0 {
        eprintln!("No context to optimize");
        return Ok(());
    }
    if let Some(b) = budget {
        if b == 0 {
            anyhow::bail!("Budget must be a positive number");
        }
    }

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
        embedding_provider,
        dedup_threshold,
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
    if context.chunk_count() == 0 {
        eprintln!("No context to compress");
        return Ok(());
    }
    if budget == 0 {
        anyhow::bail!("Budget must be a positive number");
    }

    let before_tokens = context.total_tokens;

    // Compress always runs structural first, then enforces budget.
    let config = PipelineConfig {
        query,
        tokenizer: Tokenizer::new().context("Failed to initialize tokenizer")?,
        embedding_provider: None,
        dedup_threshold: 0.85,
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

// ── Diff command ──────────────────────────────────────────────────────────

fn cmd_diff(before: &AppContext, after: &AppContext, format: OutputFormat) -> Result<()> {
    let report_b = analyze(before, "default");
    let report_a = analyze(after, "default");

    // ── Detect per-message changes ────────────────────────────────────────
    //
    // The optimized file has no persistent IDs — messages get new positions
    // when re-parsed. We match by content: exact match = moved (or same),
    // prefix match = compressed, no match = removed.
    use std::collections::HashSet;

    // (before_pos, after_pos, role)
    let mut moved: Vec<(usize, usize, String)> = Vec::new();
    // (before_pos, before_tokens, after_tokens, role)
    let mut compressed: Vec<(usize, usize, usize, String)> = Vec::new();
    // (before_pos, role, tokens)
    let mut removed: Vec<(usize, String, usize)> = Vec::new();

    let mut after_matched: HashSet<usize> = HashSet::new();
    let mut before_matched: HashSet<usize> = HashSet::new();

    // Pass 1: exact content matches → same position or moved.
    for (b_pos, b_chunk) in before.chunks.iter().enumerate() {
        if let Some((a_pos, _)) = after
            .chunks
            .iter()
            .enumerate()
            .find(|(a_pos, a)| !after_matched.contains(a_pos) && a.content == b_chunk.content)
        {
            after_matched.insert(a_pos);
            before_matched.insert(b_pos);
            if a_pos != b_pos {
                moved.push((b_pos, a_pos, b_chunk.role.clone()));
            }
        }
    }

    // Pass 2: unmatched → try role+prefix match (compressed) or mark removed.
    for (b_pos, b_chunk) in before.chunks.iter().enumerate() {
        if before_matched.contains(&b_pos) {
            continue;
        }
        let prefix_len = 80.min(b_chunk.content.len());
        let b_prefix = &b_chunk.content[..prefix_len];

        let role_match = after.chunks.iter().enumerate().find(|(a_pos, a)| {
            !after_matched.contains(a_pos)
                && a.role == b_chunk.role
                && (a.content.starts_with(b_prefix)
                    || b_chunk
                        .content
                        .starts_with(&a.content[..80.min(a.content.len())]))
        });

        if let Some((a_pos, a_chunk)) = role_match {
            after_matched.insert(a_pos);
            before_matched.insert(b_pos);
            compressed.push((
                b_pos,
                b_chunk.token_count,
                a_chunk.token_count,
                b_chunk.role.clone(),
            ));
        } else {
            removed.push((b_pos, b_chunk.role.clone(), b_chunk.token_count));
        }
    }

    match format {
        OutputFormat::Terminal => print_diff_terminal(
            &report_b, &report_a, &moved, &compressed, &removed, before, after,
        ),
        OutputFormat::Json => print_diff_json(
            &report_b, &report_a, &moved, &compressed, &removed,
        ),
    }

    Ok(())
}

fn print_diff_terminal(
    before: &cctx::analyzer::health::HealthReport,
    after: &cctx::analyzer::health::HealthReport,
    moved: &[(usize, usize, String)],
    compressed: &[(usize, usize, usize, String)],
    removed: &[(usize, String, usize)],
    ctx_b: &AppContext,
    ctx_a: &AppContext,
) {
    use owo_colors::OwoColorize;

    // ── Table rendering ───────────────────────────────────────────────────
    // Column widths: label=21, before=10, after=10, change=10
    let w0 = 21usize;
    let w1 = 10usize;
    let w2 = 10usize;
    let w3 = 10usize;

    let sep_h = format!(
        "├{:─<w0$}┼{:─<w1$}┼{:─<w2$}┼{:─<w3$}┤",
        "", "", "", ""
    );
    let top = format!(
        "┌{:─<w0$}┬{:─<w1$}┬{:─<w2$}┬{:─<w3$}┐",
        "", "", "", ""
    );
    let bot = format!(
        "└{:─<w0$}┴{:─<w1$}┴{:─<w2$}┴{:─<w3$}┘",
        "", "", "", ""
    );

    let row = |label: &str, b: &str, a: &str, ch: &str| {
        format!(
            "│ {:<w$}│ {:>w1$} │ {:>w2$} │ {:>w3$} │",
            label,
            b,
            a,
            ch,
            w = w0 - 1,
            w1 = w1 - 2,
            w2 = w2 - 2,
            w3 = w3 - 2,
        )
    };

    // ── Compute change strings ────────────────────────────────────────────
    let fmt_num = |n: usize| -> String {
        let s = n.to_string();
        let chars: Vec<char> = s.chars().collect();
        let mut r = String::with_capacity(s.len() + s.len() / 3);
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 && (chars.len() - i) % 3 == 0 {
                r.push(',');
            }
            r.push(ch);
        }
        r
    };

    let pct_change = |b: usize, a: usize| -> String {
        if b == 0 {
            return "—".to_string();
        }
        if a == b {
            return "—".to_string();
        }
        let pct = ((a as f64 - b as f64) / b as f64) * 100.0;
        if pct < 0.0 {
            format!("{:.1}%", pct)
        } else {
            format!("+{:.1}%", pct)
        }
    };

    let score_change = |b: u32, a: u32| -> String {
        if a == b {
            "—".to_string()
        } else if a > b {
            format!("+{}", a - b)
        } else {
            format!("-{}", b - a)
        }
    };

    // ── Print table ───────────────────────────────────────────────────────
    println!("{}", top);
    println!("{}", row("Metric", "Before", "After", "Change"));
    println!("{}", sep_h);
    println!(
        "{}",
        row(
            "Total tokens",
            &fmt_num(before.total_tokens),
            &fmt_num(after.total_tokens),
            &pct_change(before.total_tokens, after.total_tokens),
        )
    );
    println!(
        "{}",
        row(
            "Messages",
            &fmt_num(before.chunk_count),
            &fmt_num(after.chunk_count),
            &pct_change(before.chunk_count, after.chunk_count),
        )
    );
    println!(
        "{}",
        row(
            "Dead zone tokens",
            &fmt_num(before.dead_zone.tokens),
            &fmt_num(after.dead_zone.tokens),
            &pct_change(before.dead_zone.tokens, after.dead_zone.tokens),
        )
    );
    println!(
        "{}",
        row(
            "Duplicate tokens",
            &fmt_num(before.duplication.duplicate_tokens),
            &fmt_num(after.duplication.duplicate_tokens),
            &pct_change(
                before.duplication.duplicate_tokens,
                after.duplication.duplicate_tokens
            ),
        )
    );
    println!(
        "{}",
        row(
            "Health score",
            &before.health_score.to_string(),
            &after.health_score.to_string(),
            &score_change(before.health_score, after.health_score),
        )
    );
    println!("{}", bot);

    // ── Change details ────────────────────────────────────────────────────
    if !moved.is_empty() {
        println!();
        println!(
            "{}",
            format!("Moved: {} messages reordered", moved.len()).yellow()
        );
        for (b_pos, a_pos, role) in moved.iter().take(5) {
            println!(
                "  message {} ({}) :: position {}/{} -> {}/{}",
                b_pos, role, b_pos, ctx_b.chunk_count(), a_pos, ctx_a.chunk_count()
            );
        }
        if moved.len() > 5 {
            println!("  ... and {} more", moved.len() - 5);
        }
    }

    if !compressed.is_empty() {
        println!();
        println!(
            "{}",
            format!("Compressed: {} messages reduced", compressed.len()).cyan()
        );
        for &(b_pos, bt, at, ref role) in compressed.iter().take(5) {
            let saved = bt.saturating_sub(at);
            println!(
                "  message {} ({}) :: {} -> {} tokens (-{})",
                b_pos, role, bt, at, saved
            );
        }
        if compressed.len() > 5 {
            println!("  ... and {} more", compressed.len() - 5);
        }
    }

    if !removed.is_empty() {
        println!();
        println!(
            "{}",
            format!("Removed: {} messages dropped", removed.len()).red()
        );
        for (b_pos, role, tokens) in removed.iter().take(5) {
            println!("  message {} ({}, {} tokens)", b_pos, role, tokens);
        }
        if removed.len() > 5 {
            println!("  ... and {} more", removed.len() - 5);
        }
    }
}

fn print_diff_json(
    before: &cctx::analyzer::health::HealthReport,
    after: &cctx::analyzer::health::HealthReport,
    moved: &[(usize, usize, String)],
    compressed: &[(usize, usize, usize, String)],
    removed: &[(usize, String, usize)],
) {
    let diff = serde_json::json!({
        "before": {
            "total_tokens": before.total_tokens,
            "messages": before.chunk_count,
            "dead_zone_tokens": before.dead_zone.tokens,
            "duplicate_tokens": before.duplication.duplicate_tokens,
            "health_score": before.health_score,
        },
        "after": {
            "total_tokens": after.total_tokens,
            "messages": after.chunk_count,
            "dead_zone_tokens": after.dead_zone.tokens,
            "duplicate_tokens": after.duplication.duplicate_tokens,
            "health_score": after.health_score,
        },
        "changes": {
            "token_change_pct": if before.total_tokens > 0 {
                ((after.total_tokens as f64 - before.total_tokens as f64) / before.total_tokens as f64) * 100.0
            } else { 0.0 },
            "health_score_delta": after.health_score as i64 - before.health_score as i64,
            "moved_messages": moved.iter().map(|(b, a, role)| {
                serde_json::json!({"from": b, "to": a, "role": role})
            }).collect::<Vec<_>>(),
            "compressed_messages": compressed.iter().map(|&(pos, bt, at, ref role)| {
                serde_json::json!({"position": pos, "before_tokens": bt, "after_tokens": at, "role": role})
            }).collect::<Vec<_>>(),
            "removed_messages": removed.iter().map(|(pos, role, tokens)| {
                serde_json::json!({"position": pos, "role": role, "tokens": tokens})
            }).collect::<Vec<_>>(),
        }
    });
    println!("{}", serde_json::to_string_pretty(&diff).unwrap());
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

/// Read a file with clean error messages (shared by read_input and diff).
fn read_file(path: &PathBuf) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            anyhow::anyhow!("File not found: {}", path.display())
        } else if e.kind() == io::ErrorKind::PermissionDenied {
            anyhow::anyhow!("Permission denied: {}", path.display())
        } else {
            anyhow::anyhow!("Cannot read '{}': {}", path.display(), e)
        }
    })?;
    String::from_utf8(bytes).map_err(|_| {
        anyhow::anyhow!(
            "File contains non-UTF-8 data (binary file?): {}",
            path.display()
        )
    })
}

fn build_context(raw: &str, input_format: Option<InputFormat>) -> Result<AppContext> {
    let messages = formats::parse_input(raw, input_format).context("Failed to parse input")?;

    // Empty input (e.g. []) → return a valid but empty Context.
    // Each command checks for this and prints a friendly message.
    if messages.is_empty() {
        return Ok(AppContext::new(vec![]));
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

/// Create an embedding provider from the CLI flag.
/// Returns None if no provider specified (dedup falls back to exact-match).
fn make_embedding_provider(
    name: Option<&str>,
) -> Result<Option<std::sync::Arc<dyn cctx::embeddings::EmbeddingProvider>>> {
    match name {
        None => Ok(None),
        // TF-IDF is always available — no external API needed.
        Some("tfidf") => Ok(Some(std::sync::Arc::new(
            cctx::embeddings::tfidf::TfIdfEmbedder,
        ))),
        #[cfg(feature = "embeddings")]
        Some("ollama") => Ok(Some(std::sync::Arc::new(
            cctx::embeddings::ollama::OllamaEmbedder::default_local(),
        ))),
        #[cfg(feature = "embeddings")]
        Some("openai") => {
            let embedder = cctx::embeddings::openai::OpenAIEmbedder::from_env()?;
            Ok(Some(std::sync::Arc::new(embedder)))
        }
        #[cfg(not(feature = "embeddings"))]
        Some(name) if name == "ollama" || name == "openai" => {
            anyhow::bail!(
                "Provider '{}' requires --features embeddings.\n\
                 Rebuild with: cargo build --features embeddings\n\
                 Or use --embedding-provider tfidf (no external deps)",
                name
            );
        }
        Some(other) => {
            anyhow::bail!(
                "Unknown embedding provider '{}'. Supported: tfidf, ollama, openai",
                other
            );
        }
    }
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
