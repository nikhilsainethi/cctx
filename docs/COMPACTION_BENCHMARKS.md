# Compaction Guard — benchmark report

This report captures the measured behavior of the Tier-0 (zero-ML)
compaction guard on three axes: **fingerprint accuracy**, **runtime
performance**, and **loss-detector quality**. All numbers are
reproducible — every result below comes from a script committed to
this repo, run on a clean checkout against the fixtures listed.

> Hardware: Apple Silicon dev laptop (Darwin 25.3.0).
> Build: `cargo build --release --features "proxy,embeddings,llm"`.
> Date: 2026-04-16.

---

## 1. Fingerprint accuracy

**Script:** `scripts/accuracy_benchmark.sh`
**Fixture:** `tests/fixtures/fingerprint_accuracy.jsonl` — 35-message
curated conversation containing six load-bearing facts placed at
realistic depths (constraint, decision, two operational facts, two
debug-insight root causes).

| Metric        | Result        |
|---------------|---------------|
| Recall        | **6/6 (100%)** |
| Precision     | **10/15 (67%)** |
| Items kept    | 15            |

**Recall** asks: of the six facts a future Claude session would need
to recover, how many show up *somewhere* in the extracted fingerprint?
All six are present:

- ✓ Budget $50K constraint
- ✓ PostgreSQL primary-store decision
- ✓ Auth service on port 8443
- ✓ Config path `/etc/myapp/prod.yml`
- ✓ HPA scaling-delay root cause
- ✓ Missing CORS header root cause

**Precision** is a deliberately conservative measure — it counts an
item as "load-bearing" only if it textually echoes one of the six
known fact patterns. The five "filler" items are mostly RAKE-extracted
elaboration around the CORS discussion (the fix landed, the redirect
behavior, etc.) — useful context, not noise — but the script doesn't
get credit for them. So 67% is a floor, not a ceiling.

---

## 2. Runtime performance

**Script:** `scripts/perf_benchmark.sh`
**Generator:** `cctx bench generate --messages N --output FILE` — a
deterministic JSONL synthesizer that interleaves user/assistant turns,
sprinkles in tool_use cycles every 8 entries, and seeds the six known
facts at fixed positions so the fingerprinter has something real to
chew on. Five timed runs per size; the median is reported.

| Messages | Bytes      | Median   | Min     | Max     |
|---------:|-----------:|---------:|--------:|--------:|
|       50 |     9 071  |  58.2 ms | 57.1 ms | 66.6 ms |
|      200 |    36 221  |  62.4 ms | 61.7 ms | 64.9 ms |
|      500 |    90 507  |  69.8 ms | 68.8 ms | 72.0 ms |
|     1500 |   271 458  |  **93.4 ms** | 91.9 ms | 94.8 ms |

**Target:** 1500 messages (~200K tokens — Claude Code's compaction
trigger) in under 3 seconds. **Achieved:** 93.4 ms. That's roughly
**32× under target**, and the baseline ~58 ms cost visible at N=50 is
mostly process startup — actual fingerprint work scales sub-linearly
because RAKE and the regex layers are O(n) over tokens with low
constants.

The takeaway: there is no perceptible latency cost to running the
compaction guard at the moment a Claude Code session is about to
compact, even on the largest realistic transcripts.

---

## 3. Loss detector — weighted vs. naive overlap

**Test:** `tests/weighted_vs_naive_overlap.rs`

The loss detector classifies each fingerprint item against the
post-compaction summary into one of three buckets based on
**IDF-weighted token overlap**:

- **Preserved** — overlap ≥ 0.7 (the summary clearly retains it)
- **Paraphrased** — 0.3 ≤ overlap < 0.7 (signal kept, surface form changed)
- **Lost**       — overlap < 0.3 (rationale tokens absent — must be re-injected)

A naive set-based Jaccard overlap is the obvious baseline. It works
fine when the summary echoes every important word, but it has a
predictable failure mode: when a verbose original collapses into a
short summary that keeps the *common* nouns and drops the *rare*
specifics, naive overlap reports "Paraphrased" while the rationale is
gone — a false negative that lets the recovery system stay silent on
items it should be re-injecting.

The integration test runs **ten adversarial cases** through both
classifiers with a deterministic IDF map (common tokens = 0.1, rare
tokens = 2.0). Three cases are sanity baselines (identical input,
genuinely lost, common-word echo) where both should agree. The
remaining seven are the interesting ones:

| Case                                               | Naive verdict | Weighted verdict |
|----------------------------------------------------|---------------|------------------|
| PostgreSQL/MongoDB/ACID rationale dropped          | Paraphrased   | **Lost**         |
| Auth service port number dropped                   | Paraphrased   | **Lost**         |
| Budget cap value dropped                           | Paraphrased   | **Lost**         |
| Config file path dropped                           | Paraphrased   | **Lost**         |
| Bug root cause specifics dropped                   | Paraphrased   | **Lost**         |
| Constraint kept; details dropped                   | Paraphrased   | **Lost**         |
| Design decision: rationale trimmed                 | Paraphrased   | **Lost**         |

In every one of these, the summary kept the common framing words
("PostgreSQL", "service", "budget", "config", "deadline", "Rust") but
dropped the high-IDF rationale ("ACID", "8443", "50K", the file path,
"HPA", "March 15", "borrow checker"). Naive overlap sees the surface
echo and shrugs; weighted overlap correctly flags rationale loss and
queues those items for recovery injection.

The test asserts `weighted_wins >= 7` so this margin is wired into
CI — if a future change makes weighted overlap drift toward naive
behavior, the build fails.

---

## 4. End-to-end lifecycle

**Test:** `tests/compaction_lifecycle.rs` (12 tests, all passing)

A realistic three-phase walk-through:

1. **PreCompact** — feed the curated fixture through the fingerprint
   pipeline; assert the six known facts land in the saved fingerprint
   on disk and that uniqueness scoring puts the budget constraint
   above the (multiply-occurring) CORS chatter.
2. **PostCompact** — feed a synthetic post-compaction summary that
   deliberately drops port 8443, the config path, and the HPA root
   cause but preserves PostgreSQL, the budget, and the CORS
   resolution. Assert the loss detector classifies each item into the
   right bucket and writes a pending-injection payload for the lost
   ones only.
3. **UserPromptSubmit** — invoke the actual `cctx` binary and capture
   stdout. Assert the recovery block contains exactly the lost items
   and does *not* leak the preserved ones (specifically, no
   `DECISION:` line for PostgreSQL).

Plus eight edge-case tests covering empty transcripts, summary-only
inputs, no-fingerprint state, double compaction in one session,
malformed JSONL, and oversized tool output.

---

## Reproducing this report

```bash
# Build the release binary once.
cargo build --release --features "proxy,embeddings,llm"

# Accuracy.
./scripts/accuracy_benchmark.sh

# Performance.
./scripts/perf_benchmark.sh

# Loss detector quality.
cargo test --features "proxy,embeddings,llm" \
  --test weighted_vs_naive_overlap -- --nocapture

# Lifecycle integration.
cargo test --features "proxy,embeddings,llm" \
  --test compaction_lifecycle
```
