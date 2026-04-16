//! Criterion benchmarks — measures execution time with statistical rigor.
//!
//! Run: cargo bench
//! Report: target/criterion/report/index.html

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cctx::analyzer::health::{analyze, assign_attention_zones};
use cctx::core::context::{AttentionZone, Chunk, Context, Message};
use cctx::core::tokenizer::Tokenizer;
use cctx::formats;
use cctx::pipeline::executor::Pipeline;
use cctx::pipeline::{PipelineConfig, make_strategy};

fn build_context(raw: &str) -> Context {
    let messages = formats::parse_input(raw, None).unwrap();
    let tokenizer = Tokenizer::new().unwrap();
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
    Context::new(chunks)
}

fn load_fixture(name: &str) -> (String, Context) {
    let raw = std::fs::read_to_string(format!("tests/fixtures/{}", name)).unwrap();
    let ctx = build_context(&raw);
    (raw, ctx)
}

fn make_pipeline(strategies: &[&str]) -> Pipeline {
    let config = PipelineConfig {
        query: None,
        tokenizer: Tokenizer::new().unwrap(),
        embedding_provider: None,
        dedup_threshold: 0.85,
        prune_threshold: 0.3,
        llm_provider: None,
    };
    let mut pipeline = Pipeline::new(config);
    for name in strategies {
        pipeline.add(make_strategy(name).unwrap());
    }
    pipeline
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_analyze(c: &mut Criterion) {
    let fixtures = [
        "large_conversation.json",
        "technical_conversation.json",
        "bench_long_chat.json",
    ];

    let mut group = c.benchmark_group("analyze");
    for name in &fixtures {
        let (_, ctx) = load_fixture(name);
        group.bench_with_input(BenchmarkId::from_parameter(name), &ctx, |b, ctx| {
            b.iter(|| analyze(ctx, "default"));
        });
    }
    group.finish();
}

fn bench_strategies(c: &mut Criterion) {
    let (_, ctx) = load_fixture("technical_conversation.json");

    let mut group = c.benchmark_group("strategy");
    for strat in &["bookend", "structural", "dedup", "prune"] {
        let pipeline = make_pipeline(&[strat]);
        group.bench_with_input(BenchmarkId::from_parameter(strat), &ctx, |b, ctx| {
            b.iter(|| pipeline.run(ctx.clone()));
        });
    }
    group.finish();
}

fn bench_presets(c: &mut Criterion) {
    let (_, ctx) = load_fixture("large_conversation.json");

    let mut group = c.benchmark_group("preset");
    let presets = [
        ("safe", vec!["bookend"]),
        ("balanced", vec!["bookend", "structural"]),
        (
            "aggressive_no_llm",
            vec!["bookend", "structural", "dedup", "prune"],
        ),
    ];

    for (name, strats) in &presets {
        let pipeline = make_pipeline(&strats.iter().map(|s| s.as_ref()).collect::<Vec<&str>>());
        group.bench_with_input(BenchmarkId::from_parameter(name), &ctx, |b, ctx| {
            b.iter(|| pipeline.run(ctx.clone()));
        });
    }
    group.finish();
}

fn bench_context_building(c: &mut Criterion) {
    let fixtures = [
        "large_conversation.json",
        "bench_long_chat.json",
        "bench_agent_history.json",
    ];

    let mut group = c.benchmark_group("build_context");
    for name in &fixtures {
        let raw = std::fs::read_to_string(format!("tests/fixtures/{}", name)).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(name), &raw, |b, raw| {
            b.iter(|| build_context(raw));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_analyze,
    bench_strategies,
    bench_presets,
    bench_context_building
);
criterion_main!(benches);
