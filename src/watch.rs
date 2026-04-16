//! `cctx watch` — poll a file, re-analyze on change, with optional auto-optimize.
//!
//! Use cases:
//!   1. Monitor a growing conversation / agent context file for health
//!      degradation in real time.
//!   2. Auto-optimize: read an input file, write an optimized copy on each
//!      change. A poor-man's proxy for when swapping the base URL isn't an
//!      option — the app reads from the optimized file.
//!
//! Polling (not inotify/fsevents) is deliberate: mtime polling is portable,
//! trivial, and low-overhead at human-scale intervals.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use owo_colors::OwoColorize;

use crate::analyzer::health::{HealthReport, analyze, assign_attention_zones};
use crate::core::context::{AttentionZone, Chunk, Context as AppContext, Message};
use crate::core::tokenizer::Tokenizer;
use crate::embeddings::EmbeddingProvider;
use crate::formats;
use crate::llm::LlmProvider;
use crate::pipeline::executor::Pipeline;
use crate::pipeline::{PipelineConfig, make_strategy, preset_strategies};

// ── Public config ─────────────────────────────────────────────────────────────

pub struct WatchConfig {
    pub file: PathBuf,
    pub interval_secs: u64,
    pub alert_threshold: u32,
    pub auto_optimize: bool,
    pub output: Option<PathBuf>,
    pub strategies: Vec<String>,
    pub preset: Option<String>,
    pub query: Option<String>,
    pub budget: Option<usize>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub llm_provider: Option<Arc<dyn LlmProvider>>,
    pub dedup_threshold: f64,
    pub prune_threshold: f64,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run(cfg: WatchConfig) -> Result<()> {
    validate(&cfg)?;

    let interval = Duration::from_secs(cfg.interval_secs.max(1));
    let tty = std::io::stderr().is_terminal();
    // Build the tokenizer once — tiktoken init loads BPE tables and is
    // relatively expensive. Reused across every tick.
    let tokenizer = Tokenizer::new().context("Failed to initialize tokenizer")?;
    let panel_height = if cfg.auto_optimize { 10 } else { 8 };

    eprintln!(
        "cctx watching {} (every {}s, Ctrl+C to stop)",
        cfg.file.display(),
        cfg.interval_secs
    );
    eprintln!();

    // Initial tick: read once so the first render has data.
    let mut last_mtime = file_mtime(&cfg.file);
    let mut status = tick(&cfg, &tokenizer)
        .with_context(|| format!("Initial read of {} failed", cfg.file.display()))?;
    let mut last_change_check: u64 = 1;
    let mut checks: u64 = 1;
    let mut changes: u64 = 1;
    let mut first_render = true;

    let mut render_changed = true; // first render always counts as a "change" for logging

    loop {
        render(
            &status,
            &cfg,
            tty,
            first_render,
            render_changed,
            panel_height,
            checks,
            changes,
            last_change_check,
        );
        first_render = false;

        thread::sleep(interval);
        checks += 1;

        let mtime = file_mtime(&cfg.file);
        render_changed = match (last_mtime, mtime) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        };

        if render_changed {
            last_mtime = mtime;
            changes += 1;
            last_change_check = checks;

            match tick(&cfg, &tokenizer) {
                Ok(next) => status = next,
                Err(e) => {
                    // Don't crash — just report and keep watching. The user
                    // may be mid-edit writing invalid JSON; next tick recovers.
                    eprintln!("\n{} {:#}", "watch error:".red().bold(), e);
                    first_render = true; // cursor math is now off; restart render from scratch
                }
            }
        }
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate(cfg: &WatchConfig) -> Result<()> {
    if !cfg.file.is_file() {
        anyhow::bail!("Watch target not found: {}", cfg.file.display());
    }
    if cfg.auto_optimize && cfg.output.is_none() {
        anyhow::bail!("--auto-optimize requires --output <path>");
    }
    if !cfg.auto_optimize && cfg.output.is_some() {
        anyhow::bail!("--output requires --auto-optimize");
    }
    if let Some(out) = &cfg.output {
        let in_canon = std::fs::canonicalize(&cfg.file).ok();
        let out_canon = std::fs::canonicalize(out).ok();
        // Compare by canonical path; if both resolve, they must differ.
        if let (Some(a), Some(b)) = (in_canon, out_canon)
            && a == b
        {
            anyhow::bail!("--output must differ from the watched file (would create a write loop)");
        }
    }
    Ok(())
}

// ── Per-tick work ─────────────────────────────────────────────────────────────

struct TickStatus {
    report: HealthReport,
    optimized: Option<OptimizedInfo>,
}

struct OptimizedInfo {
    before_tokens: usize,
    after_tokens: usize,
    output: PathBuf,
}

fn tick(cfg: &WatchConfig, tokenizer: &Tokenizer) -> Result<TickStatus> {
    let raw = std::fs::read_to_string(&cfg.file)
        .with_context(|| format!("Cannot read {}", cfg.file.display()))?;
    let ctx = build_ctx(&raw, tokenizer)?;
    let report = analyze(&ctx, "default");

    let optimized = if cfg.auto_optimize {
        let before_tokens = ctx.total_tokens;
        let out_path = cfg
            .output
            .as_ref()
            .expect("validate() enforces this")
            .clone();
        let opt_ctx = run_pipeline(ctx, cfg)?;
        let after_tokens = opt_ctx.total_tokens;
        write_output(&opt_ctx, &out_path)?;
        Some(OptimizedInfo {
            before_tokens,
            after_tokens,
            output: out_path,
        })
    } else {
        None
    };

    Ok(TickStatus { report, optimized })
}

fn build_ctx(raw: &str, tokenizer: &Tokenizer) -> Result<AppContext> {
    let messages = formats::parse_input(raw, None).context("Failed to parse input")?;
    if messages.is_empty() {
        return Ok(AppContext::new(vec![]));
    }
    let n = messages.len();
    let mut chunks: Vec<Chunk> = messages
        .into_iter()
        .enumerate()
        .map(|(i, msg)| {
            let relevance = msg
                .relevance_score
                .map(|s| s.clamp(0.0, 1.0))
                .unwrap_or_else(|| {
                    if msg.role == "system" {
                        1.0
                    } else if n <= 1 {
                        0.5
                    } else {
                        0.1 + (i as f64 / (n - 1) as f64) * 0.8
                    }
                });
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
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total);
    Ok(AppContext::new(chunks))
}

fn run_pipeline(ctx: AppContext, cfg: &WatchConfig) -> Result<AppContext> {
    let names: Vec<String> = if let Some(p) = &cfg.preset {
        preset_strategies(p)?
            .into_iter()
            .map(String::from)
            .collect()
    } else if cfg.strategies.is_empty() {
        vec!["bookend".to_string()]
    } else {
        cfg.strategies.clone()
    };

    let pipeline_config = PipelineConfig {
        query: cfg.query.clone(),
        tokenizer: Tokenizer::new()?,
        embedding_provider: cfg.embedding_provider.clone(),
        dedup_threshold: cfg.dedup_threshold,
        prune_threshold: cfg.prune_threshold,
        llm_provider: cfg.llm_provider.clone(),
    };
    let mut pipeline = Pipeline::new(pipeline_config);
    for name in &names {
        pipeline.add(make_strategy(name)?);
    }

    let result = if let Some(b) = cfg.budget {
        let (c, _warnings) = pipeline.run_with_budget(ctx, b)?;
        c
    } else {
        pipeline.run(ctx)?
    };

    let mut chunks = result.chunks;
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assign_attention_zones(&mut chunks, total);
    Ok(AppContext::new(chunks))
}

fn write_output(ctx: &AppContext, path: &Path) -> Result<()> {
    let messages: Vec<Message> = ctx.chunks.iter().map(|c| c.to_message()).collect();
    let json =
        serde_json::to_string_pretty(&messages).context("Failed to serialize optimized context")?;
    std::fs::write(path, json).with_context(|| format!("Cannot write {}", path.display()))
}

// ── Rendering ─────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn render(
    status: &TickStatus,
    cfg: &WatchConfig,
    tty: bool,
    first: bool,
    changed_this_tick: bool,
    panel_height: usize,
    checks: u64,
    changes: u64,
    last_change_check: u64,
) {
    // Non-TTY branch: emit plain log lines so piping to a file produces
    // clean, grep-able output. Print only when something actually changed
    // (first render or file change) — avoid spamming one line per tick.
    if !tty {
        if changed_this_tick {
            log_line(status, checks);
        }
        return;
    }

    let w = 50;

    // On every redraw after the first, move up and overwrite the previous frame.
    if !first {
        eprint!("\x1b[{}A", panel_height);
    }
    let c = "\x1b[2K"; // clear-line before writing each row

    let since = (checks.saturating_sub(last_change_check)) * cfg.interval_secs.max(1);
    let since_label = if since == 0 {
        "just now".to_string()
    } else {
        format!("{}s ago", since)
    };

    let file_str = cfg.file.display().to_string();
    let title = format!("cctx watch — {}", truncate(&file_str, w - 17));
    let tick_line = format!(
        "Check #{:<5} Last change: {} ({} total)",
        checks, since_label, changes
    );
    let tokens_line = format!("Tokens:       {}", fmt_n(status.report.total_tokens));
    let dz_line = format!(
        "Dead zone:    {} ({:.1}%)",
        fmt_n(status.report.dead_zone.tokens),
        status.report.dead_zone.ratio * 100.0
    );
    let health_line = format!("Health:       {}/100", status.report.health_score);

    eprintln!("{c}╭{}╮", "─".repeat(w));
    eprintln!("{c}│  {:<width$}│", title, width = w - 2);
    eprintln!("{c}│  {:<width$}│", tick_line, width = w - 2);
    eprintln!("{c}│  {:<width$}│", tokens_line, width = w - 2);
    eprintln!("{c}│  {:<width$}│", dz_line, width = w - 2);
    eprintln!("{c}│  {:<width$}│", health_line, width = w - 2);

    if let Some(opt) = &status.optimized {
        let pct = reduction_pct(opt.before_tokens, opt.after_tokens);
        let opt_line = format!(
            "Optimized:    {} -> {} (-{:.1}%)",
            fmt_n(opt.before_tokens),
            fmt_n(opt.after_tokens),
            pct
        );
        let out_str = opt.output.display().to_string();
        let out_trunc = truncate(&out_str, w - 16);
        let out_line = format!("Output:       {}", out_trunc);
        eprintln!("{c}│  {:<width$}│", opt_line, width = w - 2);
        eprintln!("{c}│  {:<width$}│", out_line, width = w - 2);
    }

    eprintln!("{c}╰{}╯", "─".repeat(w));

    // Alert line is always present — red when below threshold, blank otherwise.
    // Keeping its height constant means the cursor-up math (panel_height)
    // always matches the number of lines we emitted.
    let score = status.report.health_score;
    if score < cfg.alert_threshold {
        eprintln!(
            "{c}  {} Health {} below alert threshold {}",
            "!".red().bold(),
            score.to_string().red().bold(),
            cfg.alert_threshold
        );
    } else {
        eprintln!("{c}");
    }
}

fn log_line(status: &TickStatus, tick: u64) {
    eprintln!(
        "[tick #{}] health={}/100 tokens={} dead_zone={}",
        tick,
        status.report.health_score,
        status.report.total_tokens,
        status.report.dead_zone.tokens,
    );
    if let Some(opt) = &status.optimized {
        let pct = reduction_pct(opt.before_tokens, opt.after_tokens);
        eprintln!(
            "[tick #{}] optimized {} -> {} (-{:.1}%) -> {}",
            tick,
            opt.before_tokens,
            opt.after_tokens,
            pct,
            opt.output.display(),
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

fn reduction_pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        (before as f64 - after as f64) / before as f64 * 100.0
    }
}

fn fmt_n(n: usize) -> String {
    let s = n.to_string();
    let chars: Vec<char> = s.chars().collect();
    let mut r = String::with_capacity(s.len() + s.len() / 3);
    for (i, &ch) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            r.push(',');
        }
        r.push(ch);
    }
    r
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_with_contents(name: &str, contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cctx_watch_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    #[test]
    fn validate_rejects_missing_file() {
        let cfg = WatchConfig {
            file: PathBuf::from("/nonexistent/cctx_watch_test.json"),
            interval_secs: 1,
            alert_threshold: 50,
            auto_optimize: false,
            output: None,
            strategies: vec![],
            preset: None,
            query: None,
            budget: None,
            embedding_provider: None,
            llm_provider: None,
            dedup_threshold: 0.85,
            prune_threshold: 0.3,
        };
        assert!(validate(&cfg).is_err());
    }

    #[test]
    fn validate_rejects_auto_optimize_without_output() {
        let input = tmp_with_contents("in1.json", "[]");
        let cfg = WatchConfig {
            file: input,
            interval_secs: 1,
            alert_threshold: 50,
            auto_optimize: true,
            output: None,
            strategies: vec!["bookend".into()],
            preset: None,
            query: None,
            budget: None,
            embedding_provider: None,
            llm_provider: None,
            dedup_threshold: 0.85,
            prune_threshold: 0.3,
        };
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("--output"));
    }

    #[test]
    fn validate_rejects_output_equals_input() {
        let input = tmp_with_contents("inout.json", "[]");
        let cfg = WatchConfig {
            file: input.clone(),
            interval_secs: 1,
            alert_threshold: 50,
            auto_optimize: true,
            output: Some(input),
            strategies: vec!["bookend".into()],
            preset: None,
            query: None,
            budget: None,
            embedding_provider: None,
            llm_provider: None,
            dedup_threshold: 0.85,
            prune_threshold: 0.3,
        };
        let err = validate(&cfg).unwrap_err();
        assert!(err.to_string().contains("must differ"));
    }

    #[test]
    fn reduction_pct_handles_zero() {
        assert_eq!(reduction_pct(0, 0), 0.0);
        assert_eq!(reduction_pct(100, 50), 50.0);
    }

    #[test]
    fn fmt_n_adds_commas() {
        assert_eq!(fmt_n(0), "0");
        assert_eq!(fmt_n(999), "999");
        assert_eq!(fmt_n(1_000), "1,000");
        assert_eq!(fmt_n(1_234_567), "1,234,567");
    }
}
