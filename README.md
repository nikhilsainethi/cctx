# cctx — Context Compiler for LLMs

> Your LLM is only as smart as the context you feed it.

[![CI](https://github.com/nikhilsainethi/cctx/actions/workflows/ci.yml/badge.svg)](https://github.com/nikhilsainethi/cctx/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fast, provider-agnostic Rust CLI and HTTP proxy that analyzes, optimizes, and compresses LLM context before it hits the API.

![cctx demo](demo.gif)

---

## The Problem

LLMs lose **~30% accuracy** on information placed in the middle of the context window — a structural property of transformer attention, not a bug ([Liu et al., *Lost in the Middle*, TACL 2024](https://arxiv.org/abs/2307.03172)). Industry data attributes **~65% of enterprise AI agent failures** to context drift during multi-step reasoning. Bigger context windows don't fix the attention curve — they just give you more space to get lost in. Every API call is billed by the token; most of those tokens earn nothing.

## What cctx Does

cctx analyzes context for dead zones, duplication, and bloat, then runs research-backed optimization strategies — attention-aware reordering, structural compression, semantic deduplication, importance pruning, and LLM-powered summarization. It ships as a single static Rust binary with no runtime dependencies, works as a **CLI** or a drop-in **OpenAI-compatible HTTP proxy**, and is provider-agnostic (OpenAI, Anthropic, Ollama, anything OpenAI-compatible).

## Results

**Up to 94% fewer tokens · <10ms pipeline overhead · zero provider lock-in.**

Measured on the included benchmark fixtures (release build, Apple Silicon):

| Fixture | Original | Safe | Balanced | Aggressive |
|---|---:|---:|---:|---:|
| `bench_long_chat` (dev chat, 50 turns) | 11,219 | 11,219 | 9,531 (**-15%**) | 1,880 (**-83%**) |
| `bench_rag_chunks` (15 RAG chunks) | 10,000 | 10,000 | 10,000 | 1,529 (**-85%**) |
| `bench_agent_history` (30-turn agent) | 16,986 | 16,986 | 16,986 | 1,045 (**-94%**) |
| `bench_codebase_context` (5 source files) | 10,064 | 10,064 | 3,137 (**-69%**) | 1,067 (**-89%**) |
| `large_conversation` (39-turn debug) | 8,011 | 8,011 | 8,011 | 4,154 (**-48%**) |
| `technical_conversation` (API debug) | 4,868 | 4,868 | 3,643 (**-25%**) | 3,643 (**-25%**) |

`safe` reorders only (no tokens removed). `balanced` compresses structured content. `aggressive` applies all five strategies. **Quality retention** (estimated pending LLM-as-judge runs): ~10/10 safe, ~9/10 balanced, ~7–8/10 aggressive — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §3.

## Quick Start

```bash
cargo install cctx

cctx analyze conversation.json                              # health report
cctx optimize conversation.json --preset balanced -o out.json
cctx diff conversation.json out.json                        # side-by-side
```

## Proxy Mode

The killer feature: change one environment variable, every API call gets optimized transparently.

```bash
cargo install cctx --features proxy

cctx proxy --listen 127.0.0.1:8080 \
  --upstream https://api.openai.com \
  --preset balanced

export OPENAI_BASE_URL=http://127.0.0.1:8080
# Your app is now optimized. No code changes.
```

Live metrics over HTTP:

```bash
$ curl -s http://127.0.0.1:8080/cctx/metrics | jq
{
  "requests": 1284,
  "tokens_in": 10_312_544,
  "tokens_out": 7_431_820,
  "tokens_saved": 2_880_724,
  "estimated_savings_usd": 7.20,
  "avg_latency_ms": 4.2,
  "by_model": { "gpt-4o": { "saved": 2_240_110 }, "gpt-4o-mini": { "saved": 640_614 } }
}
```

Features: SSE streaming passthrough · hard token budgets (`--budget`) · dry-run mode · live terminal dashboard · per-model cost tracking.

## Strategies

- **`bookend`** — reorders chunks to place critical content at the ends of the window, where LLM attention is strongest ([Liu et al., TACL 2024](https://arxiv.org/abs/2307.03172)).
- **`structural`** — compresses JSON, code, and markdown inline (prunes metadata, keeps function signatures, collapses irrelevant sections). Up to 50% reduction on structured content.
- **`dedup`** — removes near-duplicate chunks via cosine similarity; built-in TF-IDF embedder, optional Ollama or OpenAI backends.
- **`prune`** — drops low-signal filler by TF-IDF-style scoring ("Sure, I can help!", empty acknowledgments).
- **`summarize`** — LLM-powered hierarchical compression (recent turns verbatim → aging turns as bullets → archived turns merged into a paragraph). Fallback: drop archived turns without LLM.

Compose as `--strategy a --strategy b`, or use a preset: `--preset safe|balanced|aggressive`.

## Installation

```bash
cargo install cctx                            # core CLI, zero network deps
cargo install cctx --features proxy           # + HTTP proxy
cargo install cctx --features embeddings      # + Ollama/OpenAI embedders
cargo install cctx --features llm             # + Ollama/OpenAI summarizer
cargo install cctx --features "proxy,embeddings,llm"   # everything
```

Or clone and build:

```bash
git clone https://github.com/nikhilsainethi/cctx.git
cd cctx && cargo build --release --features "proxy,embeddings,llm"
# Binary at target/release/cctx
```

Homebrew formula (`brew install cctx`) and prebuilt release binaries: coming soon — see [Roadmap](#roadmap).

## Benchmarks

Full report: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Summary:

| Preset | Avg Reduction | Pipeline Time | Quality (est.) |
|---|---:|---:|---:|
| `safe` | 0% | 12 µs | 10 / 10 |
| `balanced` | 16% (0–69% per fixture) | 2.4 ms | ~9 / 10 |
| `aggressive` (no LLM) | 71% (25–94%) | 7.1 ms | ~7–8 / 10 |

Regenerate: `./scripts/run_benchmarks.sh && cargo bench`.

## Architecture

Full reference: [`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md).

```
            ┌────────────────┐
JSON / stdin │   formats::    │  auto-detect OpenAI / Anthropic / RAG / raw
 ──────────▶ │   parse_input  │
            └───────┬────────┘
                    ▼
            ┌────────────────┐
            │  core::Context │  chunks + tokens + attention zones
            └───────┬────────┘
                    ▼
            ┌────────────────────────────────────────┐
            │  pipeline::executor                    │
            │  [bookend] [structural] [dedup]        │  strategies run in sequence
            │  [prune]   [summarize]                 │
            └───────┬────────────────────────────────┘
                    ▼
            ┌────────────────┐
            │ optimized JSON │ ── CLI stdout · proxy → upstream LLM API
            └────────────────┘
```

Core is sync Rust. The proxy wraps the same pipeline in an axum server using `tokio::task::spawn_blocking` to bridge back to sync.

## Roadmap

- **More strategies** — entity-preserving summarization, query-aware chunk selection, conversation-turn merging.
- **More input formats** — LangChain messages, Pydantic schemas, tool-call traces with structured arguments.
- **WASM target** — run cctx in the browser for client-side context trimming in web apps.
- **Homebrew + prebuilt binaries** — `brew install cctx` and Linux/macOS/Windows release artifacts.
- **Quality benchmark CI** — automated LLM-as-judge runs against fixtures on every release.

## Contributing

Issues and pull requests welcome. The codebase is intentionally over-commented — it doubles as a Rust learning resource. Start with `docs/PROJECT_REFERENCE.md` for the architecture tour.

## License

MIT — see [LICENSE](LICENSE).
