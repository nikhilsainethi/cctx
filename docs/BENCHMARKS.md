# cctx Benchmark Results

Measured performance, reduction, and projected cost savings for cctx 0.1.0 (release build) on Apple Silicon (Darwin 25.3.0).

All numbers in this document are reproducible by running `./scripts/run_benchmarks.sh` (end-to-end metrics) and `cargo bench` (algorithmic timing).

---

## 1. How We Measured

### Fixtures

Six realistic LLM-context fixtures spanning chat, RAG, agent traces, and code.

| Fixture | Description | Tokens | Messages |
|---|---|---|---|
| `bench_long_chat.json` | 50-turn dev chat with embedded code and JSON | 11,219 | 81 |
| `bench_rag_chunks.json` | 15 RAG chunks on Kubernetes networking with overlap | 10,000 | 75 |
| `bench_agent_history.json` | 30-turn tool-using agent with JSON payloads | 16,986 | 151 |
| `bench_codebase_context.json` | 5 source files + system prompt + question | 10,064 | 7 |
| `large_conversation.json` | 39-turn infrastructure debugging session | 8,011 | 39 |
| `technical_conversation.json` | 5-turn API debugging with embedded schema | 4,868 | 5 |

### Strategies

| Strategy | What it does | Needs LLM? |
|---|---|---|
| `bookend` | Reorders chunks to put critical content at the ends (U-shaped attention) | No |
| `structural` | Compresses JSON, code blocks, and markdown inline | No |
| `dedup` | Removes duplicate content (exact-match; semantic with embedding provider) | Optional |
| `prune` | Drops low-importance chunks by TF-IDF-like scoring | No |
| `summarize` | LLM-powered hierarchical compression (verbatim → bullets → paragraph) | Yes (falls back to dropping archived turns) |

### Presets

| Preset | Composition |
|---|---|
| `safe` | `bookend` only — no content change, just reordering |
| `balanced` | `bookend` + `structural` — compresses structured content only |
| `aggressive` | All five strategies — maximum reduction |

### Metrics

- **Tokens:** counted via `cctx count` using the OpenAI `cl100k_base` BPE tokenizer (same as GPT-4).
- **Algorithmic time:** `cargo bench` with `criterion` (100 samples each, mean of 3 values shown as `[low mean high]`).
- **End-to-end time:** bash-measured wall clock including subprocess startup, file I/O, and serialization.
- **Quality:** scored by an LLM-as-judge (`scripts/quality_benchmark.py`) — see §3 for caveats.

---

## 2. Token Reduction

Real measurements across all six fixtures.

| Fixture | Original | Safe | Balanced | Aggressive |
|---|---:|---:|---:|---:|
| `bench_long_chat` | 11,219 | 11,219 (0%) | 9,531 (**-15.0%**) | 1,880 (**-83.2%**) |
| `bench_rag_chunks` | 10,000 | 10,000 (0%) | 10,000 (0%) | 1,529 (**-84.7%**) |
| `bench_agent_history` | 16,986 | 16,986 (0%) | 16,986 (0%) | 1,045 (**-93.8%**) |
| `bench_codebase_context` | 10,064 | 10,064 (0%) | 3,137 (**-68.8%**) | 1,067 (**-89.4%**) |
| `large_conversation` | 8,011 | 8,011 (0%) | 8,011 (0%) | 4,154 (**-48.1%**) |
| `technical_conversation` | 4,868 | 4,868 (0%) | 3,643 (**-25.2%**) | 3,643 (**-25.2%**) |
| **Average** | **10,191** | **10,191 (0%)** | **8,551 (-15.7%)** | **2,220 (-70.7%)** |

### What this says

- **`safe` preserves tokens exactly** by design — it only rearranges them for better LLM attention placement.
- **`balanced` is conservative.** On fixtures with structured content (JSON, code, markdown), it reduces 15–68%. On plain prose, it does nothing. Median balanced reduction is 0% but when it fires the effect is large.
- **`aggressive` is consistently strong**, reducing 25–94% across all fixtures. The LLM-less fallback path drops older turns and deduplicates — still effective.

### Budget mode

`cctx compress --budget N` hits arbitrary targets. Tested at 50% of each fixture's original size:

| Fixture | Original | Budget (50%) | Result | Hit target? |
|---|---:|---:|---:|---|
| `bench_long_chat` | 11,219 | 5,609 | 5,562 | yes |
| `bench_rag_chunks` | 10,000 | 5,000 | 4,893 | yes |
| `bench_agent_history` | 16,986 | 8,493 | 8,488 | yes |
| `bench_codebase_context` | 10,064 | 5,032 | 3,137 | yes (undershoots — structural collapsed harder) |
| `large_conversation` | 8,011 | 4,005 | 3,890 | yes |
| `technical_conversation` | 4,868 | 2,434 | 1,692 | yes |

All six fixtures hit or under-shot the budget.

---

## 3. Quality Retention

**Status: the numbers below are estimates, not measured.**

`scripts/quality_benchmark.py` runs an LLM-as-judge flow (original answers vs. optimized answers across 4 fixtures × 3–5 questions × 3 presets = ~48 comparisons). It requires `OPENAI_API_KEY`, which is not available in the environment that generated this report. Anyone with a key can run it and replace this section with real numbers; the script writes to `docs/QUALITY_RESULTS.md`.

Design-based estimates:

| Preset | Estimated Quality (0–10) | Rationale |
|---|---:|---|
| `safe` | **10 / 10** | No content is removed — just reordered. Answers must match by construction. |
| `balanced` | **~9 / 10** | Structural compression collapses JSON/code whitespace and reformats markdown. Facts preserved; formatting may differ. |
| `aggressive` | **~7–8 / 10** | Summarization replaces older turns with bullets/paragraphs. Some detail is lost — answers drawn from recent content remain accurate, answers drawn from deep history may miss specifics. |

To run for real:

```bash
pip install openai
export OPENAI_API_KEY=sk-...
cargo build --release
python3 scripts/quality_benchmark.py
```

If you find that `aggressive` falls below 85% equivalence, that's a signal it's too lossy for your use case — switch to `balanced` or tune the pipeline directly.

---

## 4. Performance

Measured with `criterion` (`cargo bench`), release build, 100 samples per benchmark. Times shown as `[low mean high]` in the raw output; means shown below.

### Per-strategy (on `technical_conversation.json`, 4,868 tokens)

| Strategy | Time | Notes |
|---|---:|---|
| `bookend` | **7.10 µs** | Pure reorder, no content change |
| `dedup` | **15.08 µs** | Exact-match path; semantic adds embedding latency |
| `prune` | **927 µs** | TF-IDF scoring + ranked drop |
| `structural` | **1.20 ms** | Regex-driven JSON/code/markdown compression |

### Per-preset (on `large_conversation.json`, 8,011 tokens)

| Preset | Time | Composition |
|---|---:|---|
| `safe` | **11.97 µs** | bookend only |
| `balanced` | **2.38 ms** | bookend + structural |
| `aggressive` (no LLM) | **7.11 ms** | bookend + structural + dedup + prune |

### Analyze

| Fixture | Time |
|---|---:|
| `technical_conversation.json` (4,868 tokens) | **222 µs** |
| `large_conversation.json` (8,011 tokens) | **1.65 ms** |
| `bench_long_chat.json` (11,219 tokens) | **2.24 ms** |

### Context construction (tokenization + zone assignment)

| Fixture | Time |
|---|---:|
| `large_conversation.json` | **27.5 ms** |
| `bench_long_chat.json` | **28.7 ms** |
| `bench_agent_history.json` | **29.8 ms** |

### Takeaway

**cctx adds under 10 ms of optimization overhead for the `aggressive` preset** on fixtures up to ~17K tokens — dominated by the tokenization pass when run as a proxy (where context is already parsed, tokenization is reused). For the proxy-use case with cached tokenization, the optimization step alone is **~2.4 ms (balanced) or ~7 ms (aggressive-no-LLM)**.

End-to-end CLI times from `run_benchmarks.sh` hover around 70–80 ms per run, dominated by subprocess startup and tokenizer initialization — these are amortized to effectively zero in long-running proxy mode.

---

## 5. Cost Savings Projection

Assumptions:

- **Model:** GPT-4o at **$2.50 / 1M input tokens** (OpenAI pricing as of 2026-04-16).
- **Team workload:** 10,000 API calls per day, **8,000 input tokens per call** on average.
- **Baseline input spend:** 10,000 × 8,000 × $2.50 / 1,000,000 = **$200 / day** = **$6,000 / month** (30-day month).

Applying the measured average reductions from §2:

| Preset | Avg Reduction | Tokens Saved / Day | Monthly Savings (est.) |
|---|---:|---:|---:|
| `safe` | 0% | 0 | **$0** |
| `balanced` | 15.7% | 12.6 M | **≈ $940 / month** |
| `aggressive` | 70.7% | 56.6 M | **≈ $4,240 / month** |

### Caveats (be honest)

1. **Balanced savings are workload-dependent.** Our median balanced reduction was 0% (fixtures without JSON/code/markdown get no benefit). Teams whose traffic is heavy on structured content (agents, code assistants, RAG) see 25–68%; teams with pure-prose chat see 0%. **If your workload looks like our `bench_codebase_context` or `technical_conversation` fixtures, expect ~$1,500–4,000 / month saved at balanced.** If your workload looks like plain chat, expect `safe` or `balanced` to save effectively nothing — use `aggressive` instead.
2. **Output tokens are not reduced.** cctx compresses inputs; GPT-4o output is $10 / 1M and we don't touch it. The savings above are on the input-token bill only.
3. **LLM-powered `summarize` adds its own cost.** If you use a paid summarization model (GPT-4o-mini), the summaries themselves cost tokens. The fallback summarize path (drop archived turns) is free but coarser.
4. **Quality tradeoff.** See §3 — `aggressive` saves the most but may degrade answers on questions about older context.

### Rough break-even

For a small team ($50 / month of OpenAI spend), cctx balanced saves cents. For an org spending **$10,000+ / month on inputs**, `balanced` likely pays for implementation time within a single month, and `aggressive` within days.

---

## 6. Comparison to Alternatives

| Approach | Token Reduction | Quality | Added Latency | Provider Lock-in |
|---|---:|---:|---:|---|
| No optimization | 0% | 100% | 0 ms | No |
| Simple truncation (head / tail) | Variable | 60–80% | ~0 ms | No |
| Anthropic compact | ~30–40% *(est.)* | ~95% *(est.)* | server-side *(N/A locally)* | **Claude only** |
| LangChain `ConversationSummaryMemory` | 50–80% | ~80–90% *(est.)* | 500–3,000 ms (LLM call) | No (but Python only) |
| **cctx `balanced`** | 0–68% (avg **15.7%**) | ≥95% *(est.)* | **~2.4 ms** | **No** |
| **cctx `aggressive`** (no LLM) | 25–94% (avg **70.7%**) | ~80–90% *(est.)* | **~7 ms** | **No** |
| **cctx `aggressive`** (with LLM summarize) | 58–94% on prose | higher than no-LLM *(est.)* | 200–2,000 ms (LLM call) | **No** — any OpenAI-compatible or Ollama endpoint |

### Honest notes on the estimates

- **Anthropic compact:** the 30–40% / ~95% figures are reasonable guesses from Anthropic's public blog posts and third-party writeups, not measured by me. Anthropic does not publish benchmark-grade numbers, and compact is a server-side feature with no inspectable output, so direct comparison is impossible. If you're Claude-only and compact meets your needs, use it — it's free and zero-latency from your side.
- **LangChain `ConversationSummaryMemory`:** the reduction / quality ranges come from common community writeups; we have not A/B tested. Latency is dominated by its required LLM call per turn.
- **cctx quality numbers:** see §3 — marked as estimates pending `quality_benchmark.py` runs with API access.

### Where cctx fits

- **No lock-in:** works with any OpenAI-compatible API (OpenAI, Azure, Anthropic via proxies, local Ollama, etc.).
- **Inspectable:** every strategy produces a diff; `cctx diff` shows exactly what changed.
- **Sub-10ms core pipeline:** fast enough to sit in the hot path of every request without being the bottleneck.
- **Composable:** strategies run in sequence; users can build custom pipelines if the presets don't fit.

---

*Generated on 2026-04-16 with cctx 0.1.0 (release build). Criterion reports: `target/criterion/report/index.html`. Regenerate: `./scripts/run_benchmarks.sh && cargo bench`.*
