# Changelog

All notable changes to cctx are documented here. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-04-16

First public release.

### Added

- **Context health analysis** — `cctx analyze` reports dead-zone tokens, duplication metrics, budget utilization, and a composite 0–100 health score with actionable recommendations.
- **Bookend reordering strategy** — attention-aware placement based on Liu et al., *Lost in the Middle* (TACL 2024). Sorts by relevance (TF-IDF against `--query` or a recency heuristic) and interleaves at the window edges.
- **Structural compression** — inline compression of JSON payloads (prune timestamps/UUIDs/metadata, collapse deep nesting), code blocks (keep signatures, replace bodies), and markdown (query-aware section collapse).
- **Semantic deduplication** — cosine similarity over embeddings from Ollama, OpenAI, or a built-in TF-IDF approximator. Merges unique sentences from shorter duplicates.
- **Importance-aware token pruning** — heuristic sentence scoring (stop-word ratio, repetition, structural markers, filler-phrase detection) with system-message and recent-user-turn protection.
- **Hierarchical summarization** — three-tier LLM-powered compression: recent turns verbatim, aging turns bulletized, archived turns merged into a paragraph. Fallback path drops archived turns when no LLM is configured.
- **OpenAI-compatible proxy mode** — `cctx proxy` runs an HTTP server that optimizes `/v1/chat/completions` requests and forwards to any OpenAI-compatible upstream. SSE streaming passes through chunk-by-chunk. Catch-all route proxies unrelated endpoints unchanged.
- **Multi-format input** — auto-detects OpenAI chat, Anthropic messages (string or typed-block content), RAG-chunk arrays, and raw text; `--input-format` overrides.
- **Strategy composition** — `--strategy a --strategy b` chains strategies in order; presets (`safe`, `balanced`, `aggressive`) provide curated bundles.
- **Token budget enforcement** — `--budget N` on `optimize`/`proxy` and `cctx compress --budget N` drop oldest unprotected chunks until the budget is met; system messages and last two user turns are preserved.
- **Config file support** — `.cctx.toml` in the current directory or `~/.config/cctx/config.toml` with `[default]`, `[optimize]`, `[proxy]`, `[dedup]`, `[prune]`, `[summarize]` sections. `cctx init` writes a commented template. Precedence: CLI flags > project config > user config > built-in defaults.
- **Watch mode with auto-optimization** — `cctx watch <file>` polls the file and re-analyzes on change with an in-place TTY panel; `--auto-optimize --output <path>` writes an optimized copy on every change, creating a file-based pipeline alternative to the HTTP proxy.
- **Cost tracking and metrics dashboard** — proxy exposes `GET /cctx/metrics` with per-model cost-savings breakdowns and `GET /cctx/metrics/reset`; optional `--dashboard` flag renders live stats on stderr.
- **LLM-as-judge quality benchmark harness** (`scripts/quality_benchmark.py`) — evaluates whether optimized contexts still produce equivalent answers across fixtures × questions × presets.
- **Reproducible benchmark suite** (`scripts/run_benchmarks.sh`, `cargo bench`) — end-to-end metrics and `criterion`-measured algorithmic timings exported to `docs/BENCHMARKS.md`.

### Notes

- `balanced` preset is workload-dependent: averages 15.7% reduction across all fixtures but reaches 25–69% on structured content (code, JSON, markdown). `aggressive` averages 70.7% across all fixtures.
- Quality-retention numbers are currently design-based estimates; a full LLM-judge run is tracked in the v0.2 milestone.

[0.1.0]: https://github.com/nikhilsainethi/cctx/releases/tag/v0.1.0
