# I built a context compiler for LLMs — here's why your AI is dumber than it should be

## The problem nobody talks about

You paid for GPT-4o's 128K-token context window. You stuffed 50,000 tokens of documentation, chat history, and system prompts into a single request. The model gave you a polite, confident, *wrong* answer.

It didn't miss the details. It never saw them.

In 2024, Stanford researchers published ["Lost in the Middle"](https://arxiv.org/abs/2307.03172) — one of those quiet papers that should have changed how the industry thinks about context engineering, and mostly didn't. They placed the answer to a question at different positions in long contexts and measured recall. The result was a U-shaped curve: strong recall at the beginning of the window, strong recall at the end, and a **~30% accuracy drop** for information buried in the middle third.

This isn't a bug. It's a structural property of how transformer attention works — specifically, how rotary position encodings distribute attention mass. Bigger context windows don't fix it. They just give you more space to lose information in.

And it gets worse in production. A 2025 Zylos Research report attributes **65% of enterprise AI agent failures** to context drift and memory loss during multi-step reasoning — not raw context exhaustion. Agents that work fine on short traces fall apart as history accumulates, not because the window fills up, but because critical facts migrate into the attention dead zone.

The uncomfortable implication: the way most of us use LLMs — dumping conversation history, retrieval results, and tool outputs into one long prompt — is actively hostile to how the models actually read. Every extra token between the signal and the question is a penalty. Your context is making your AI stupid.

The fix isn't bigger windows. It's better packing.

## What I built

**[cctx](https://github.com/nikhilsainethi/cctx)** is a context compiler: a Rust CLI and HTTP proxy that analyzes your LLM context, flags what's wrong, and rewrites it before it hits the model. Provider-agnostic (OpenAI, Anthropic, Ollama, anything OpenAI-compatible), single static binary, no runtime dependencies, no API keys required for the core features.

Five strategies, each backed by research or a measurable property of real input:

- **`bookend`** — reorders chunks so critical content sits where the model actually attends
- **`structural`** — compresses JSON payloads, collapses code bodies to signatures, prunes irrelevant markdown
- **`dedup`** — removes near-duplicate chunks via cosine similarity (built-in TF-IDF approximator, or pluggable Ollama/OpenAI embedders)
- **`prune`** — drops low-signal filler (`"Sure, I can help!"` and its friends)
- **`summarize`** — LLM-powered hierarchical compression: recent turns verbatim, aging turns as bullets, old turns merged into a paragraph

Compose them on the command line, or run `cctx proxy` and point your app's `OPENAI_BASE_URL` at it. **Zero code changes.** Optimization happens on the wire.

## The results

Measured on six representative fixtures — a 50-turn dev chat, 15 RAG chunks on Kubernetes networking, a 30-turn agent trace with tool calls, five source files plus system prompt, a 39-turn infra debugging session, and a 5-turn API debug:

| Fixture | Original | Balanced | Aggressive |
|---|---:|---:|---:|
| 50-turn dev chat | 11,219 | 9,531 (**-15%**) | 1,880 (**-83%**) |
| 15 RAG chunks | 10,000 | 10,000 | 1,529 (**-85%**) |
| 30-turn agent trace | 16,986 | 16,986 | 1,045 (**-94%**) |
| 5 source files + prompt | 10,064 | 3,137 (**-69%**) | 1,067 (**-89%**) |
| 39-turn debug session | 8,011 | 8,011 | 4,154 (**-48%**) |
| 5-turn API debug | 4,868 | 3,643 (**-25%**) | 3,643 (**-25%**) |

Average reduction: **15.7%** (balanced), **70.7%** (aggressive). Up to **94% fewer tokens** on long agent traces.

`balanced` is the conservative default — it only touches structured content (JSON, code, markdown) and reorders the rest. On plain prose it's a no-op. On the kind of context that agents and code assistants actually produce — structured tool calls, serialized state, RAG chunks — it's 25–69% savings with near-zero risk of changing meaning.

`aggressive` adds deduplication, pruning, and summarization. It reliably hits 48–94% on long traces where most of the tokens are redundant history.

Here's what it looks like on a real debugging conversation:

```
$ cctx analyze conversation.json
╭────────────────────────────────────────────────────────╮
│             cctx Context Health Report                 │
├────────────────────────────────────────────────────────┤
│  Overall Health:  82/100  ✓  GOOD                      │
│  Total Tokens:        4,868                            │
│  Dead zone:       1,862 tokens (38.2%) in dead zone    │
│  Duplication:       232 tokens (4.8%) near-duplicates  │
╰────────────────────────────────────────────────────────╯

$ cctx optimize conversation.json --preset balanced -o out.json
Tokens: 4,868 -> 3,643 (-25.2%)

$ cctx analyze out.json
│  Overall Health:  86/100  ✓  GOOD
│  Dead zone:       1,043 tokens (28.6%)
```

Thirty-eight percent of the tokens in the original sat in the attention dead zone. A single structural pass dropped that to 29% *and* cut 25% of total tokens. Meaning-preserving — just rearranged and tightened.

Performance, measured with `criterion`: **2.4 ms** for the balanced pipeline on an 8K-token fixture, **7.1 ms** for aggressive-without-LLM. Tokenization dominates; the strategies themselves run in microseconds.

### Cost

Real OpenAI pricing (GPT-4o input: $2.50 / 1M tokens) and a typical team workload — 10,000 calls/day, 8,000 input tokens per call:

- Baseline spend: **$200/day · $6,000/month**
- `balanced` (15.7% avg): **~$940/month saved**
- `aggressive` (70.7% avg): **~$4,240/month saved**

Workload-dependent, obviously. If your traffic is 95% plain-prose chat with no structure, `balanced` won't help and you want `aggressive`. If it's code-assistant or agent traces — where `balanced` hits 25–69% — it pays for itself on day one.

## How it works

Each strategy exists because something measurable is wrong with typical LLM context.

**Bookend reordering** is the direct response to Liu et al. Given N chunks, sort by relevance and alternate placement: highest at position 0, second-highest at position N-1, third at position 1, inward. Relevance is TF-IDF against an optional `--query`, or a heuristic (system messages + recent user messages) without one. No tokens added or removed — just putting the important material where the model is paying attention. The Stanford results suggest a 10–30% recall improvement on buried facts.

**Structural compression** exploits the fact that structured data is mostly punctuation. A typical API response has 60% of its tokens in braces, quotes, and whitespace. Code blocks carry redundant indentation and rarely-useful implementation details. Three sub-passes: JSON pruning (strip timestamps, UUIDs, metadata; collapse objects more than 3 levels deep), code signature extraction (keep function and class signatures, replace bodies with `# 42 lines`), markdown collapse (keep sections matching the query, header-only for the rest). 25–50% reduction on structured content.

**Dedup** computes cosine similarity between chunks. Near-duplicates (e.g. overlapping RAG chunks) keep the longer version and merge any unique sentences from the shorter one. Three embedding providers: a zero-dependency TF-IDF approximator, local Ollama, or OpenAI.

**Pruning** scores sentences by stop-word ratio, repetition, structural importance, and filler-phrase detection. Drops the low-scoring ones. System messages and the last two user messages are always preserved.

**Summarization** is hierarchical: the last six turns stay verbatim, the next six get bulletized by an LLM, older turns get merged into a single paragraph. Without an LLM provider configured it falls back to dropping archived turns — still effective, just coarser.

### The pipeline

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

### Proxy mode

The part I'm most proud of: setup is one line.

```bash
cctx proxy --upstream https://api.openai.com --preset balanced

export OPENAI_BASE_URL=http://127.0.0.1:8080
```

Every API call now flows through cctx. Streaming responses pass through chunk-by-chunk with no buffering. Metrics at `GET /cctx/metrics` (requests, tokens saved, estimated cost savings, per-model breakdown). Optional live dashboard on stderr.

No code changes. No SDK. Just a base URL.

## Try it

```bash
cargo install cctx

cctx analyze conversation.json
cctx optimize conversation.json --preset balanced -o out.json
cctx diff conversation.json out.json
```

Repo: **[github.com/nikhilsainethi/cctx](https://github.com/nikhilsainethi/cctx)**. MIT licensed. The codebase is deliberately over-commented — I'm learning Rust on this project, and the comments are half documentation, half pedagogy.

Star it if it's useful. PRs very welcome — especially for additional input formats, more code-signature languages (Java, Go, C#), and a WASM build for browser-side trimming.

Your LLM is only as smart as the context you feed it. Feed it better.
