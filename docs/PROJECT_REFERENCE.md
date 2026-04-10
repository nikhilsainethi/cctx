# cctx — Context Compiler for LLMs

## Project Reference Document

**Author:** Nikhil Sai Nethi
**Started:** April 2026
**Language:** Rust
**License:** MIT (recommended for maximum adoption)

---

## 1. Problem Statement

### The Core Problem

Large Language Models advertise massive context windows (200K–10M tokens), but **effective context utilization degrades significantly before those limits are reached**. Research consistently shows:

- **Lost in the Middle (Liu et al., Stanford, TACL 2024):** LLMs follow a U-shaped attention curve — strong recall at the beginning and end of context, but 30%+ accuracy drop for information placed in the middle third. This is a structural property of Rotary Position Embedding (RoPE), not a bug.

- **Context Rot (Chroma Research, 2025):** Performance degrades accelerates beyond ~30,000 tokens even in models with much larger windows. Irrelevant history crowds the context window, silently undermining reasoning quality.

- **Enterprise Impact (Zylos Research, 2026):** 65% of enterprise AI agent failures in 2025 were attributed to context drift or memory loss during multi-step reasoning — not raw context exhaustion.

- **Diminishing Returns (NoLiMa Benchmark):** For many popular LLMs, performance degrades significantly as context length increases, regardless of the advertised window size.

### Why Existing Solutions Are Insufficient

| Solution | Limitation |
|----------|-----------|
| Anthropic `compact` (Jan 2026) | Claude-only. Operates at KV-cache level. Black box. No user control. |
| LLMLingua (Microsoft Research) | Academic. Python-only. Requires model perplexity access. Slow. |
| KVzip (Seoul National University) | Requires model internals (KV-cache access). Not application-layer. |
| ACON (Oct 2025) | Framework for agent compression guidelines, not a standalone tool. |
| Factory.ai anchored summarization | Proprietary. Internal to Factory's coding agent. |
| Provence | Trains a separate pruner model. Not plug-and-play. |
| Framework-specific solutions | LangChain/LangGraph have basic truncation. Not reusable outside their ecosystem. |

### The Gap

**There is no standalone, fast, provider-agnostic CLI tool that optimizes LLM context at the application layer.**

Every developer building with LLMs — whether using Claude, GPT, Gemini, Ollama, or local models — faces the same problem. They deserve a tool that works everywhere.

---

## 2. Solution: cctx

### One-Line Pitch

> "Your LLM is only as smart as the context you feed it. cctx makes your context smarter."

### What cctx Does

cctx is a Rust CLI + local proxy that analyzes, optimizes, and compresses LLM context before it reaches the model. It operates at the application layer — above the API, below your application logic — making it compatible with any provider, any framework, any language.

### Design Principles

1. **Zero-dependency core.** The basic optimization strategies (reordering, structural compression, analysis) require no ML models, no API calls, nothing but the binary.
2. **Progressive enhancement.** Adding a tiny embedding model (80MB) unlocks semantic features. Adding LLM access (local or API) unlocks summarization. Each tier is optional.
3. **Pipe-friendly.** Works with stdin/stdout, JSON I/O, exit codes for CI/CD. Composable with any toolchain.
4. **Provider-agnostic.** Works with OpenAI, Anthropic, Google, Ollama, llama.cpp, vLLM — anything that accepts text.
5. **Transparent.** Shows exactly what it changed, why, and how much it saved. No black boxes.

---

## 3. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
│  (Python script, Agent framework, CLI tool, Web app, etc.)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        cctx Layer                            │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Analyzer │→ │ Strategy │→ │ Optimizer │→ │  Emitter   │  │
│  │          │  │ Selector │  │ Pipeline  │  │            │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────────┘  │
│       │                          │                │          │
│       ▼                          ▼                ▼          │
│  ┌─────────┐            ┌──────────────┐  ┌────────────┐   │
│  │ Context  │            │  Strategy    │  │   Report   │   │
│  │ Health   │            │  Registry    │  │  Generator │   │
│  │ Report   │            │              │  │            │   │
│  └─────────┘            └──────────────┘  └────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Provider API                        │
│           (OpenAI, Anthropic, Ollama, etc.)                  │
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

```
cctx/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI entry point (clap)
│   ├── lib.rs                   # Library root (for use as a crate)
│   │
│   ├── core/
│   │   ├── mod.rs
│   │   ├── context.rs           # Context data model (chunks, metadata, zones)
│   │   ├── tokenizer.rs         # Token counting (tiktoken-rs or custom BPE)
│   │   ├── attention_map.rs     # U-curve attention zone mapper
│   │   └── budget.rs            # Token budget allocation engine
│   │
│   ├── analyzer/
│   │   ├── mod.rs
│   │   ├── health.rs            # Context health scoring
│   │   ├── dead_zone.rs         # Dead zone detection (middle-third analysis)
│   │   ├── duplication.rs       # Exact/near-duplicate detection
│   │   └── structure.rs         # Document structure parser (MD, JSON, code)
│   │
│   ├── strategies/
│   │   ├── mod.rs
│   │   ├── bookend.rs           # Attention-aware reordering
│   │   ├── structural.rs        # Structure-aware compression
│   │   ├── dedup.rs             # Semantic deduplication (needs embeddings)
│   │   ├── summarize.rs         # Hierarchical summarization (needs LLM)
│   │   └── prune.rs             # Importance-aware token pruning
│   │
│   ├── pipeline/
│   │   ├── mod.rs
│   │   ├── config.rs            # Strategy pipeline configuration
│   │   └── executor.rs          # Composable strategy execution
│   │
│   ├── proxy/
│   │   ├── mod.rs
│   │   ├── server.rs            # OpenAI-compatible proxy server
│   │   ├── interceptor.rs       # Request/response interceptor
│   │   └── passthrough.rs       # Upstream API passthrough
│   │
│   ├── formats/
│   │   ├── mod.rs
│   │   ├── openai.rs            # OpenAI message format parser/emitter
│   │   ├── anthropic.rs         # Anthropic message format parser/emitter
│   │   ├── raw.rs               # Raw text input/output
│   │   └── chunks.rs            # RAG chunk format (JSON array)
│   │
│   └── report/
│       ├── mod.rs
│       ├── terminal.rs          # Pretty-printed terminal output
│       └── json.rs              # Machine-readable JSON reports
│
├── tests/
│   ├── fixtures/                # Real-world context samples for testing
│   │   ├── long_conversation.json
│   │   ├── rag_chunks.json
│   │   ├── codebase_context.json
│   │   └── agent_history.json
│   ├── test_bookend.rs
│   ├── test_structural.rs
│   ├── test_analyzer.rs
│   └── test_pipeline.rs
│
├── benches/                     # Criterion benchmarks
│   ├── compression_bench.rs
│   └── tokenizer_bench.rs
│
└── docs/
    ├── ARCHITECTURE.md
    ├── STRATEGIES.md
    └── BENCHMARKS.md
```

### Dependency Tiers

**Tier 0 — Zero Dependencies (core binary)**

Works offline, no models, no API keys. Pure algorithms.

- Token counting (BPE tokenizer, bundled)
- Attention zone mapping (U-curve model from research)
- Bookend reordering
- Structural compression (JSON pruning, code→signatures, markdown collapsing)
- Context health analysis and scoring
- Exact/near-duplicate detection (hash-based)
- Token budget management

**Tier 1 — Lightweight Embeddings (optional feature flag)**

Adds an 80MB embedding model (`all-MiniLM-L6-v2` via `rust-bert` or ONNX runtime). CPU-only, no GPU needed.

- Semantic deduplication (cosine similarity between chunks)
- Semantic clustering (group related content)
- Relevance scoring against a query

**Tier 2 — LLM-Powered (optional, user-configured)**

Uses a local model (Ollama) or API call for operations that need language understanding.

- Hierarchical summarization
- Importance-aware pruning with LLM scoring
- Context-aware compression (understanding what matters for the task)

---

## 4. Optimization Strategies — Research Basis

### Strategy 1: Bookend Reordering

**Research basis:** Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (Stanford, TACL 2024)

**Principle:** LLMs attend most strongly to the beginning and end of context. Place highest-relevance content at positions 1–2 and N-1–N. Push lower-relevance content to the middle.

**Algorithm:**
```
Input: chunks[] with relevance_scores[]
1. Sort chunks by relevance (descending)
2. Assign to positions using alternating placement:
   - Highest relevance → position 0 (beginning)
   - 2nd highest → position N-1 (end)
   - 3rd highest → position 1 (near beginning)
   - 4th highest → position N-2 (near end)
   - Continue alternating inward
3. Output: reordered chunks[]
```

**Cost:** Zero. No tokens added or removed. Pure reordering.

**Expected impact:** 10–30% improvement in recall for information previously in the dead zone (based on Liu et al. findings).

**Implementation priority:** Week 1 — this is the simplest, most provably effective strategy.

### Strategy 2: Structural Compression

**Research basis:** Application-layer insight. No single paper, but grounded in the observation that structured data (JSON, code, markdown) contains significant redundancy when used as LLM context.

**Principle:** Use document structure to compress intelligently without losing meaning.

**Sub-strategies:**

**a) JSON Pruning:**
```
Input:  {"user": {"name": "John", "id": 12345, "email": "john@example.com",
         "created_at": "2024-01-01", "last_login": "2024-03-15",
         "preferences": {"theme": "dark", "lang": "en"}}}

Output: {"user": {"name": "John", "id": 12345, "email": "john@example.com"}}
```
Remove fields not referenced in the prompt/query. Flatten unnecessary nesting.

**b) Code → Signatures:**
```
Input:  fn calculate_tax(income: f64, rate: f64) -> f64 {
            let base = income * rate;
            let adjusted = if income > 100000.0 {
                base * 1.1
            } else {
                base
            };
            adjusted.round()
        }

Output: fn calculate_tax(income: f64, rate: f64) -> f64 { /* calculates tax with bracket adjustment */ }
```
Preserve function signatures and brief descriptions. Drop implementation when only API understanding is needed.

**c) Markdown Hierarchy Collapse:**
```
Input:  # Chapter 1: Introduction
        ## 1.1 Background
        Long paragraph about background...
        ## 1.2 Motivation
        Long paragraph about motivation...
        # Chapter 2: Methods (this is what the user asked about)
        ## 2.1 Data Collection
        Detailed methodology...

Output: # Chapter 1: Introduction [collapsed: background and motivation, 2 sections]
        # Chapter 2: Methods
        ## 2.1 Data Collection
        Detailed methodology...
```
Collapse sections not relevant to the query. Preserve relevant sections verbatim.

**Cost:** Reduces tokens, sometimes dramatically (50–80% for large JSON payloads).

**Implementation priority:** Week 2.

### Strategy 3: Semantic Deduplication (Tier 1)

**Research basis:** Common observation in RAG systems and long conversations. Retrieval often returns overlapping chunks. Conversations repeat information across turns.

**Principle:** Embed chunks, compute pairwise cosine similarity, merge near-duplicates.

**Algorithm:**
```
Input: chunks[] with embeddings[]
1. Compute pairwise cosine similarity matrix
2. For each pair with similarity > threshold (default: 0.85):
   a. Keep the longer/more detailed chunk
   b. Append any unique information from the shorter chunk
   c. Mark the shorter chunk for removal
3. Output: deduplicated chunks[]
```

**Cost:** Requires embedding model. Reduces tokens by 15–30% in typical multi-turn conversations.

**Implementation priority:** Week 3 (requires embedding integration).

### Strategy 4: Hierarchical Summarization (Tier 2)

**Research basis:** Factory.ai anchored summarization, ACON framework (arXiv, Oct 2025), adaptive context compression (arXiv, Mar 2026).

**Principle:** Maintain a rolling summary of older context. Recent turns stay verbatim. Older turns are compressed into an incrementally-updated summary.

**Algorithm:**
```
Input: conversation_turns[], budget_tokens
1. Divide turns into zones:
   - RECENT (last N turns): Keep verbatim
   - AGING (turns N+1 to 2N): Summarize into bullet points
   - ARCHIVED (turns > 2N): Merge into rolling summary
2. Rolling summary update:
   a. Take existing summary + newly aging turns
   b. Produce updated summary (via local LLM or API)
   c. Anchor summary to the last archived turn ID
3. Assemble: [system_prompt] + [rolling_summary] + [aging_bullets] + [recent_verbatim]
4. Verify total tokens <= budget
```

**Cost:** Requires an LLM call (local or API). But a 1-cent compression call that saves 50 cents of context tokens is net positive.

**Implementation priority:** Week 3–4 (requires LLM integration).

### Strategy 5: Importance-Aware Token Pruning (Tier 1/2)

**Research basis:** LLMLingua (Microsoft), ATACompressor (SIGIR 2025).

**Principle:** Score individual tokens or sentences by information density. Remove low-information content (filler phrases, redundant explanations, verbose formatting) under a budget constraint.

**Heuristic scoring (Tier 0, no model needed):**
- Sentence length vs information density ratio
- Repetition detection (n-gram frequency)
- Stop-word ratio
- Structural importance (headings > body text > footnotes)

**Model-based scoring (Tier 1/2):**
- Use embedding model to score relevance against query
- Use LLM to rate importance (expensive but accurate)

**Implementation priority:** Week 4 (stretch goal for v0.1).

---

## 5. Context Health Scoring

cctx provides a "health score" for any context before it hits the model. This is analogous to a linting report.

### Health Metrics

| Metric | What It Measures | Scoring |
|--------|-----------------|---------|
| **Dead Zone Ratio** | % of high-relevance content in the middle third | 0–100 (100 = all critical content in safe zones) |
| **Duplication Score** | % of tokens that are near-duplicates | 0–100 (100 = no duplication) |
| **Budget Utilization** | How well the token budget is used | 0–100 (100 = optimal fill, no waste) |
| **Structure Score** | Whether document structure is preserved | 0–100 (100 = clean structure) |
| **Freshness Score** | Ratio of recent vs stale information | 0–100 (100 = all information is current/relevant) |

### Composite Health Score

```
health_score = weighted_average(
    dead_zone_ratio * 0.30,      # Most impactful
    duplication_score * 0.25,
    budget_utilization * 0.20,
    structure_score * 0.15,
    freshness_score * 0.10
)
```

### Health Report Example

```
$ cctx analyze conversation.json --model claude-sonnet

╭─────────────────────────────────────────────╮
│           cctx Context Health Report         │
├─────────────────────────────────────────────┤
│  Overall Health:  62/100  ⚠️  NEEDS WORK     │
├─────────────────────────────────────────────┤
│  Total Tokens:     47,832                    │
│  Model Budget:     200,000 (claude-sonnet)   │
│  Utilization:      23.9%                     │
├─────────────────────────────────────────────┤
│  ⚠ Dead Zone:      38/100                    │
│    → 3 high-relevance chunks in middle third │
│    → Chunk "api_docs.md" at position 4/9     │
│    → Chunk "error_log.txt" at position 5/9   │
│                                              │
│  ✗ Duplication:    45/100                    │
│    → 12,340 tokens (25.8%) are duplicates    │
│    → Turns 3 and 7 overlap 89%              │
│    → Turns 12 and 15 overlap 73%            │
│                                              │
│  ✓ Budget:         82/100                    │
│  ✓ Structure:      91/100                    │
│  ⚠ Freshness:      68/100                    │
│    → 8 turns older than 20 exchanges         │
╰─────────────────────────────────────────────╯

Recommendations:
  1. Run `cctx optimize --strategy=bookend` to fix dead zone placement
  2. Run `cctx optimize --strategy=dedup` to remove 12K duplicate tokens
  3. Consider `cctx compress --budget=32k` to reduce stale context
```

---

## 6. CLI Interface Design

### Commands

```bash
# Analyze context health (no modifications)
cctx analyze <input> [--model <model>] [--format json|terminal]

# Optimize context (apply strategies)
cctx optimize <input> [--strategy <name>] [--budget <tokens>] [--output <file>]

# Compress context to fit a budget
cctx compress <input> --budget <tokens> [--method <method>]

# Count tokens accurately
cctx count <input> [--model <model>]

# Run as OpenAI-compatible proxy
cctx proxy --listen <addr> --upstream <url> [--strategy <name>] [--budget <tokens>]

# Compare two contexts (before/after diff)
cctx diff <before> <after> [--semantic]
```

### Input Formats

```bash
# OpenAI messages format (JSON array of {role, content})
cctx analyze messages.json --format openai

# Anthropic messages format
cctx analyze messages.json --format anthropic

# Raw text
cat document.txt | cctx analyze --format raw

# RAG chunks (JSON array of {content, metadata, score})
cctx analyze chunks.json --format chunks

# Stdin pipe
curl -s api.example.com/context | cctx optimize --strategy=bookend | curl -X POST api.openai.com/...
```

### Strategy Composition

```bash
# Apply multiple strategies in sequence
cctx optimize input.json \
  --strategy=bookend \
  --strategy=structural \
  --strategy=dedup \
  --budget=32000

# Use a preset
cctx optimize input.json --preset=aggressive  # all strategies, tight budget
cctx optimize input.json --preset=safe        # reorder only, no content removal
cctx optimize input.json --preset=balanced    # reorder + dedup + structural
```

---

## 7. Proxy Mode Architecture

The killer feature for adoption. Zero code changes required.

### How It Works

```
Your App (unchanged)              cctx proxy                    LLM Provider
      │                              │                              │
      │ POST /v1/chat/completions    │                              │
      │ ─────────────────────────→   │                              │
      │                              │ 1. Parse messages             │
      │                              │ 2. Analyze context health     │
      │                              │ 3. Apply optimization pipeline│
      │                              │ 4. Log metrics                │
      │                              │                              │
      │                              │ POST /v1/chat/completions    │
      │                              │ ─────────────────────────→   │
      │                              │                              │
      │                              │ ←───────────────────────── │
      │                              │ (response)                   │
      │ ←───────────────────────── │                              │
      │ (response, unchanged)        │                              │
```

### Usage

```bash
# Start proxy
cctx proxy --listen 127.0.0.1:8080 --upstream https://api.openai.com --strategy=bookend,dedup

# Your existing code — change ONE line (base URL)
# Before: client = OpenAI(base_url="https://api.openai.com")
# After:  client = OpenAI(base_url="http://127.0.0.1:8080")

# Or use environment variable
export OPENAI_BASE_URL=http://127.0.0.1:8080
# Now ALL OpenAI SDK calls go through cctx automatically
```

### Proxy Metrics Endpoint

```bash
# cctx exposes metrics at /cctx/metrics
curl http://127.0.0.1:8080/cctx/metrics

{
  "requests_total": 1847,
  "tokens_saved_total": 2_340_000,
  "tokens_original_total": 8_120_000,
  "compression_ratio_avg": 0.71,
  "estimated_cost_saved_usd": 14.20,
  "health_score_avg": 78,
  "strategies_applied": {
    "bookend": 1847,
    "dedup": 1203,
    "structural": 892
  }
}
```

---

## 8. Threat & Safety Model

### What cctx Touches

cctx sits between the user's application and the LLM API. It sees and modifies the full context, including potentially sensitive data. This requires careful security design.

### Threat Categories

| Threat | Mitigation |
|--------|-----------|
| **Data exposure in proxy mode** | cctx NEVER logs message content by default. Metrics are aggregate only (token counts, not content). Opt-in verbose logging with explicit flag. |
| **Content modification corrupts meaning** | Every strategy preserves a "semantic integrity" score. Users can set a minimum threshold. If compression would drop below threshold, cctx warns and skips. |
| **Proxy as man-in-the-middle** | cctx runs locally only (127.0.0.1). No remote proxy mode. TLS passthrough for upstream connections. No API key storage — keys are passed through headers. |
| **Embedding model data leakage** | Tier 1 embeddings run locally. No data leaves the machine. No telemetry. |
| **Supply chain (dependency) attacks** | Minimal dependencies. Audit all crates. Use `cargo-audit` in CI. |
| **Adversarial context injection** | cctx does not interpret content semantically in Tier 0. It reorders and compresses structurally. Not vulnerable to prompt injection because it doesn't execute prompts. |

### Privacy Principles

1. **Local-first.** All processing happens on the user's machine.
2. **No telemetry.** Zero data collection. No analytics. No phone-home.
3. **No key storage.** API keys pass through headers; cctx never writes them to disk.
4. **Opt-in only.** Every feature that touches content (logging, semantic analysis) is opt-in.
5. **Auditable.** Open source. Every transformation is deterministic and reproducible.

---

## 9. Implementation Roadmap

### Week 1: Foundation + First Strategy

**Learning goals:** Rust basics — ownership, borrowing, structs, enums, error handling, cargo, crates.

**Deliverables:**
- [ ] Repository setup (Cargo workspace, CI with GitHub Actions, README skeleton)
- [ ] Core data model: `Context`, `Chunk`, `AttentionZone`, `HealthReport`
- [ ] Tokenizer integration (tiktoken-rs for BPE token counting)
- [ ] Attention zone mapper (U-curve model from Liu et al.)
- [ ] Strategy 1: Bookend reordering (pure algorithm, no dependencies)
- [ ] Context health analyzer (dead zone detection, basic scoring)
- [ ] CLI skeleton with `cctx analyze` and `cctx optimize --strategy=bookend`
- [ ] Unit tests with fixture data
- [ ] README: problem statement, installation, basic usage

**Rust learning plan for Week 1:**
- Day 1: Rust Book chapters 1–6 (basics, ownership, structs, enums)
- Day 1–2: Set up project, implement `Context` and `Chunk` structs
- Day 3: Rust Book chapters 7–10 (modules, error handling, generics, traits)
- Day 3–4: Implement tokenizer + attention zone mapper
- Day 5–6: Implement bookend reordering + analyzer
- Day 7: CLI with clap, integration tests

### Week 2: Structural Compression + CLI Polish

**Deliverables:**
- [ ] Strategy 2: Structural compression (JSON pruning, code→signatures, markdown collapse)
- [ ] Full CLI interface (`cctx analyze`, `cctx optimize`, `cctx compress`, `cctx count`)
- [ ] Pipe-friendly I/O (stdin/stdout, JSON output mode)
- [ ] Input format parsers (OpenAI messages, Anthropic messages, raw text, RAG chunks)
- [ ] Strategy composition (apply multiple strategies in sequence)
- [ ] Presets (safe, balanced, aggressive)
- [ ] Token budget management
- [ ] Comprehensive test suite with real-world fixtures
- [ ] README: full CLI documentation, strategy descriptions

### Week 3: Semantic Layer + Proxy

**Deliverables:**
- [ ] Tier 1: Lightweight embedding integration (ONNX runtime or rust-bert)
  - Feature flag: `--features embeddings`
  - Model: all-MiniLM-L6-v2 (80MB, CPU-only)
- [ ] Strategy 3: Semantic deduplication
- [ ] Proxy mode: OpenAI-compatible HTTP server (hyper/axum)
- [ ] Proxy: Request interception, context optimization, upstream passthrough
- [ ] Proxy: Metrics endpoint (/cctx/metrics)
- [ ] Proxy: Streaming support (SSE passthrough)
- [ ] `cctx diff` command (before/after comparison)

### Week 4: Benchmarks + Launch

**Deliverables:**
- [ ] Benchmark suite:
  - Cost savings: tokens before vs after across standard datasets
  - Quality retention: LLM-as-judge scoring on QA tasks with/without cctx
  - Latency: time overhead per request
  - Comparison vs raw context, vs Anthropic compact (where possible)
- [ ] Benchmark results in docs/BENCHMARKS.md with charts
- [ ] Demo GIF/video for README
- [ ] Launch blog post: problem → research → solution → benchmarks → install
- [ ] Strategy 4 (stretch): Hierarchical summarization with Ollama integration
- [ ] Cross-compilation: Linux, macOS, Windows binaries in GitHub Releases
- [ ] Post to: X (thread with charts), Reddit (r/LocalLLaMA, r/MachineLearning, r/rust, r/programming), HackerNews

---

## 10. Key Rust Crates (Dependency Plan)

### Core (always included)
| Crate | Purpose |
|-------|---------|
| `clap` | CLI argument parsing (derive API) |
| `serde` + `serde_json` | JSON serialization/deserialization |
| `tiktoken-rs` | BPE tokenizer (OpenAI-compatible token counting) |
| `anyhow` | Error handling |
| `thiserror` | Custom error types |
| `colored` / `owo-colors` | Terminal colors for health reports |

### Proxy mode (feature flag)
| Crate | Purpose |
|-------|---------|
| `axum` or `hyper` | HTTP server for proxy mode |
| `reqwest` | HTTP client for upstream API calls |
| `tokio` | Async runtime |
| `tower` | Middleware layers for the proxy |

### Embeddings (feature flag)
| Crate | Purpose |
|-------|---------|
| `ort` (ONNX Runtime) | Run embedding model locally |
| `ndarray` | Tensor operations for cosine similarity |

### Development
| Crate | Purpose |
|-------|---------|
| `criterion` | Benchmarking |
| `assert_cmd` + `predicates` | CLI integration testing |
| `tempfile` | Temporary files in tests |

---

## 11. Benchmarking Plan

### Datasets

1. **Long conversation (multi-turn chat):** 50+ turn conversation with repeated information
2. **RAG pipeline output:** 10–20 retrieved chunks with varying relevance
3. **Codebase context:** Large code file + documentation stuffed into context
4. **Agent history:** Multi-step agent with tool call results accumulating

### Metrics

| Metric | How Measured |
|--------|-------------|
| Token reduction | Before/after token count |
| Cost savings | Token reduction × model pricing |
| Quality retention | LLM-as-judge scoring (ask model to answer questions with original vs optimized context) |
| Latency overhead | Time for cctx processing per request |
| Dead zone improvement | Critical info recall before vs after reordering |

### Baseline Comparisons

- Raw context (no optimization)
- Simple truncation (drop oldest turns)
- Random sampling (keep N random chunks)
- cctx Tier 0 (bookend + structural)
- cctx Tier 1 (+ deduplication)
- cctx Tier 2 (+ summarization)

---

## 12. Launch & Marketing Strategy

### README Structure (this IS the marketing)

```
# cctx — Context Compiler for LLMs

> Your LLM is only as smart as the context you feed it.

[One-line install] [Demo GIF] [Badge: tokens saved]

## The Problem (3 sentences + research citation)
## The Solution (what cctx does, 1 paragraph)
## Quick Start (3 commands: install, analyze, optimize)
## Benchmarks (chart showing cost savings + quality retention)
## How It Works (architecture diagram)
## Strategies (brief description of each)
## Proxy Mode (the killer feature, 5-line setup)
## Installation (cargo install, brew, binaries)
## Contributing
## License (MIT)
```

### Launch Posts

**X/Twitter thread:**
```
1/ I built cctx — a Rust CLI that optimizes LLM context before it hits the API.

Research shows LLMs lose 30% accuracy on info placed in the middle of context. Most developers don't know this.

cctx fixes it. Here's how 🧵

2/ [Benchmark chart: tokens saved]
3/ [Demo GIF: cctx analyze → cctx optimize]
4/ [Proxy mode: zero code change setup]
5/ [Link to repo]
```

**Reddit posts:**
- r/LocalLLaMA: Focus on local model + offline angle
- r/MachineLearning: Focus on research basis (Liu et al.)
- r/rust: Focus on implementation decisions + performance
- r/programming: Focus on practical cost savings

---

## 13. Research References

1. Liu, N.F., Lin, K., Hewitt, J., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." Transactions of the ACL, 12.
2. ACON: Agent Context Optimization. arXiv:2510.00615 (Oct 2025).
3. KVzip: Compressing KV Cache for Efficient Long-Context LLM Inference. Seoul National University (Nov 2025).
4. Adaptive Context Compression Techniques for LLMs. arXiv:2603.29193 (Mar 2026).
5. Factory.ai. "Compressing Context." (Jul 2025).
6. Chroma Research. "Context Rot" terminology and findings (2025).
7. Zylos Research. "AI Agent Context Compression Strategies." (Feb 2026).
8. NoLiMa: Long-context evaluation benchmark showing performance degradation.
9. LLMLingua: Microsoft Research prompt compression.
10. ATACompressor: Task-aware compression. SIGIR 2025.

---

## 14. Success Metrics

### Month 1 (Launch)
- [ ] Working CLI with 2–3 strategies
- [ ] Proxy mode functional
- [ ] Benchmarks showing measurable improvement
- [ ] README that tells a compelling story
- [ ] First 100 GitHub stars

### Month 2–3 (Growth)
- [ ] Community contributions (strategies, format support)
- [ ] Integration guides (LangChain, LlamaIndex, Claude Code)
- [ ] Homebrew formula
- [ ] 500+ GitHub stars
- [ ] Featured in an AI newsletter

### Month 6 (Maturity)
- [ ] Used in production by at least one company
- [ ] All 5 strategies implemented and benchmarked
- [ ] Mentioned in a conference talk or blog post
- [ ] 1000+ GitHub stars