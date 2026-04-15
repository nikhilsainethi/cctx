# cctx — Context Compiler for LLMs

> Your LLM is only as smart as the context you feed it.

[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/nikhilsainethi/cctx)](https://github.com/nikhilsainethi/cctx)

A Rust CLI that analyzes, optimizes, and compresses LLM context at the application layer. Works with any provider (OpenAI, Anthropic, Ollama, local models) and any framework. Zero dependencies beyond the binary.

## The Problem

LLMs advertise massive context windows but can't use them effectively. Stanford's *Lost in the Middle* paper (Liu et al., TACL 2024) showed a **30% accuracy drop** for information placed in the middle of the context window — a structural property of how transformers attend to position, not a bug. Zylos Research found that **65% of enterprise AI agent failures** in 2025 were caused by context drift or memory loss during multi-step reasoning. And it gets worse: beyond ~30K tokens, performance degrades regardless of the advertised window size.

Every developer building with LLMs faces this. There is no standalone, fast, provider-agnostic tool that fixes it.

## The Solution

**cctx** sits between your application and the LLM API. It analyzes your context for problems (dead zones, duplication, bloat), then applies research-backed optimization strategies — reordering chunks to match the attention curve, compressing structured content without losing meaning, and enforcing token budgets with smart truncation. It reads JSON, outputs JSON, and works in shell pipelines. No API keys, no network calls, no runtime dependencies.

## Quick Start

```bash
cargo install cctx

# Analyze context health
cctx analyze conversation.json

# Optimize with a strategy pipeline
cctx optimize conversation.json --strategy bookend --strategy structural

# Compress to a hard token budget
cctx compress conversation.json --budget 32000
```

## Demo

```
$ cctx analyze conversation.json
╭────────────────────────────────────────────────────────╮
│             cctx Context Health Report                 │
├────────────────────────────────────────────────────────┤
│  Overall Health:  82/100  ✓  GOOD                      │
├────────────────────────────────────────────────────────┤
│  Total Tokens:        4,868                            │
│  Dead zone:       1,862 tokens (38.2%) in dead zone    │
│  Duplication:       232 tokens (4.8%) near-duplicates  │
╰────────────────────────────────────────────────────────╯

$ cctx optimize conversation.json --preset balanced --output optimized.json
Pipeline: bookend, structural
  bookend      :: 4868 -> 4868 tokens  (0.0ms)
  structural   :: 4868 -> 3643 tokens  (1.6ms)
Tokens: 4868 -> 3643 (saved 1225, 25.2% reduction)

$ cctx diff conversation.json optimized.json
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Metric              │   Before │    After │   Change │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Total tokens        │    4,868 │    3,643 │   -25.2% │
│ Dead zone tokens    │    1,862 │    1,043 │   -44.0% │
│ Health score        │       82 │       86 │       +4 │
└─────────────────────┴──────────┴──────────┴──────────┘
```

Run the full interactive demo: `./scripts/demo.sh`

Record it: install [vhs](https://github.com/charmbracelet/vhs) then `vhs scripts/record-demo.tape`

## What It Does

| Command | Purpose |
|---------|---------|
| `cctx analyze <input>` | Health report: token count, dead zone detection, duplication scoring, composite health score (0-100). Supports `--format json` for machine-readable output. |
| `cctx optimize <input>` | Apply one or more strategies: `--strategy bookend --strategy structural`. Supports `--preset safe\|balanced\|aggressive`, `--budget`, `--query`, `--output`. |
| `cctx compress <input> --budget N` | Hit a hard token budget. Applies structural compression, then drops oldest non-critical messages. System prompts and recent user messages are never removed. |
| `cctx count <input>` | Print the token count. Pipe-friendly: `cctx optimize input.json \| cctx count` |
| `cctx diff <before> <after>` | Side-by-side comparison table: tokens, messages, dead zones, health score, with per-message change tracking (moved, compressed, removed). |

All commands read from files or stdin, auto-detect input format (OpenAI, Anthropic, RAG chunks, raw text), and follow the Unix convention: data to stdout, messages to stderr.

## Strategies

### Bookend Reordering

**Research basis:** Liu et al., "Lost in the Middle" (Stanford, TACL 2024)

LLMs attend most strongly to the beginning and end of context. Bookend sorts chunks by relevance and alternates placement: highest relevance at position 0, second-highest at position N-1, third at position 1, and so on inward. When `--query` is provided, relevance is scored via TF-IDF against the query. Without a query, system messages and the last 3 user messages get priority. **Cost: zero tokens added or removed.** Expected recall improvement: 10-30%.

### Structural Compression

**Principle:** Structured data (JSON, code, markdown) contains significant redundancy when used as LLM context.

Three sub-strategies: **JSON pruning** removes timestamps, UUIDs, metadata fields, and collapses objects deeper than 3 levels. **Code signature extraction** keeps function/class signatures and docstrings, replaces bodies with line counts (Python, JavaScript, TypeScript, Rust). **Markdown collapse** uses `--query` to identify relevant sections and collapses the rest to headers only. **Cost: typically 25-50% token reduction** on structured content.

### Semantic Deduplication

Detects near-duplicate content using embeddings (cosine similarity). Keeps the longer chunk; merges unique sentences from the shorter one. Three providers: `--embedding-provider tfidf` (built-in, no deps), `ollama` (local, requires `--features embeddings`), `openai` (API, requires `--features embeddings`).

### Importance Pruning

Scores each line by stop-word ratio, repetition, structural importance, filler phrase detection, and length. Removes low-scoring lines (conversational filler like "Sure, I can help!", "Great question!", short acknowledgments). System messages and last 2 user messages are never pruned.

### Presets

| Preset | Strategies | Use case |
|--------|-----------|----------|
| `--preset safe` | bookend | Reorder only. Never removes content. |
| `--preset balanced` | bookend + structural | Good default. Reorder + compress structured content. |
| `--preset aggressive` | bookend + structural + dedup + prune | Maximum reduction. Dedup + filler removal. |

## Input Formats

cctx auto-detects the input format, or you can override with `--input-format`:

```bash
# OpenAI messages: [{role, content}]
cctx analyze messages.json

# Anthropic messages: [{role, content: [{type: "text", text: "..."}]}]
cctx analyze anthropic.json

# RAG chunks: [{content, score?, metadata?}] — score drives bookend placement
cctx optimize chunks.json --strategy bookend

# Raw text
cat document.txt | cctx analyze --input-format raw

# Stdin pipe
curl -s api.example.com/context | cctx optimize --preset balanced | curl -X POST ...
```

## Proxy Mode

An OpenAI-compatible HTTP proxy that optimizes context transparently. Change one line (the base URL) and every API call flows through cctx automatically. No code changes to your application.

```bash
# Install with proxy support
cargo install cctx --features proxy

# Start the proxy
cctx proxy --listen 127.0.0.1:8080 \
  --upstream https://api.openai.com \
  --strategy bookend --strategy structural

# Point your app at the proxy
export OPENAI_BASE_URL=http://127.0.0.1:8080
# Every API call now gets optimized automatically.
```

Features:
- **Streaming support** — SSE responses forwarded chunk-by-chunk, zero buffering
- **Token budget** — `--budget 32000` enforces a hard limit per request
- **Dry-run mode** — `--dry-run` logs savings without modifying requests
- **Live dashboard** — `--dashboard` shows real-time stats on stderr
- **Metrics API** — `GET /cctx/metrics` returns JSON with requests, tokens saved, cost estimates, per-model breakdowns
- **Catch-all routing** — all OpenAI endpoints (embeddings, models, audio) pass through transparently
- **Model pricing** — tracks estimated cost savings per model (GPT-4o, Claude Sonnet, etc.)

## Benchmarks

Results from `./benches/run_benchmark.sh` on the included test fixtures (release build, Apple Silicon):

| Fixture | Tokens | Safe | Balanced | Reduction | Health | Time |
|---------|--------|------|----------|-----------|--------|------|
| sample_conversation (20 turns) | 1,945 | 1,945 | 1,945 | — | 83 → 79 | 71ms |
| technical_conversation (5 turns, JSON+code+markdown) | 4,868 | 4,868 | 3,643 | **-25.2%** | 82 → 86 | 72ms |
| structured_content (6 turns, JSON+Python+JS+markdown) | 1,714 | 1,714 | 965 | **-43.7%** | 75 → 78 | 69ms |
| large_conversation (39 turns) | 8,011 | 8,011 | 8,011 | — | 77 → 76 | 75ms |
| rag_chunks (10 retrieval chunks) | 988 | 988 | 988 | — | 80 → 80 | 73ms |
| anthropic_conversation (6 turns) | 757 | 757 | 757 | — | 70 → 76 | 71ms |

**Safe** (bookend only) never removes tokens — pure reordering. **Balanced** (bookend + structural) compresses structured content (JSON, code, markdown) while preserving plain conversation text. Fixtures without structured content show no reduction, which is correct — there's nothing to compress.

With `--budget`, larger reductions are possible: `cctx compress large_conversation.json --budget 2000` produces 1,642 tokens (79.5% reduction) by combining structural compression with smart truncation.

## Installation

### Core CLI (no network dependencies)

```bash
cargo install cctx
```

### With proxy server

```bash
cargo install cctx --features proxy
```

### With embedding providers (Ollama, OpenAI)

```bash
cargo install cctx --features embeddings
```

### Everything

```bash
cargo install cctx --features "proxy,embeddings"
```

### From git

```bash
git clone https://github.com/nikhilsainethi/cctx.git
cd cctx
cargo build --release                          # Core CLI
cargo build --release --features "proxy,embeddings"  # Full
# Binary at target/release/cctx
```

## Architecture

```
src/
├── main.rs                  # CLI entry point (clap)
├── lib.rs                   # Library root
├── core/
│   ├── context.rs           # Context, Chunk, AttentionZone data model
│   └── tokenizer.rs         # tiktoken-rs wrapper for token counting
├── analyzer/
│   ├── health.rs            # Health scoring (dead zone, duplication, budget)
│   └── duplication.rs       # Jaccard word-similarity duplicate detection
├── embeddings/
│   ├── mod.rs               # EmbeddingProvider trait, cosine similarity
│   ├── tfidf.rs             # TF-IDF mock embedder (no external deps)
│   ├── ollama.rs            # Ollama local embedding provider [feature: embeddings]
│   └── openai.rs            # OpenAI embedding provider [feature: embeddings]
├── formats/
│   └── mod.rs               # Input format auto-detection and parsing
├── strategies/
│   ├── bookend.rs           # Attention-aware reordering (TF-IDF or heuristic)
│   ├── structural.rs        # JSON pruning, code signatures, markdown collapse
│   ├── dedup.rs             # Exact-match + semantic deduplication
│   └── prune.rs             # Importance-aware sentence pruning
├── pipeline/
│   ├── mod.rs               # Strategy trait, presets, factory
│   └── executor.rs          # Sequential pipeline execution, budget enforcement
└── proxy/                   # [feature: proxy]
    ├── server.rs            # axum router, startup, dashboard task
    ├── handler.rs           # Request parsing, optimization, forwarding
    ├── upstream.rs          # HTTP client for upstream API
    ├── metrics.rs           # Atomic counters, per-model cost tracking
    └── dashboard.rs         # Live terminal stats display
```

## Contributing

Contributions welcome. The codebase is intentionally well-commented — it serves as a Rust learning resource alongside being a real tool.

Areas where help is needed:
- **Benchmarks** — formal evaluation on public datasets with LLM-as-judge quality scoring
- **Hierarchical summarization** — LLM-powered context compression (Tier 2 strategy)
- **Streaming optimization** — parse and count tokens from SSE response chunks
- **More languages** — code signature extraction for Java, Go, C#, PHP
- **Homebrew formula** — `brew install cctx`

## License

MIT
