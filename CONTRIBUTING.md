# Contributing to cctx

Thanks for your interest in improving cctx. The codebase is deliberately over-commented — it doubles as a Rust learning artifact, so readability wins over cleverness.

## Build from source

Prerequisites: Rust 1.88 or newer (stable).

```bash
git clone https://github.com/nikhilsainethi/cctx.git
cd cctx

# Core CLI only (fastest build, no network dependencies)
cargo build --release

# Everything: proxy, embeddings, LLM providers
cargo build --release --features "proxy,embeddings,llm"
```

The release binary lands at `target/release/cctx`.

## Run tests

```bash
# Full suite (all 95+ tests)
cargo test --features "proxy,embeddings,llm"

# Core only
cargo test

# One module
cargo test --features embeddings embeddings::
```

CI also runs `cargo fmt --check` and `cargo clippy -- -D warnings` against both feature sets — please run them locally before sending a PR.

Benchmarks (measured, not regression-gated):

```bash
./scripts/run_benchmarks.sh        # end-to-end fixtures → docs/BENCHMARKS.md
cargo bench --bench benchmark_suite  # criterion timings
```

## Adding a new strategy

Strategies are pluggable via the [`Strategy`](src/pipeline/mod.rs) trait. To add one:

1. **Create the strategy module.** Add `src/strategies/my_strategy.rs` with a public `apply(context, ...)` function that takes a `&Context` and returns `Vec<Chunk>` (or `Result<Vec<Chunk>>` if it can fail).

2. **Register the module.** Add `pub mod my_strategy;` to `src/strategies/mod.rs`.

3. **Wrap it as a `Strategy` impl.** In `src/pipeline/mod.rs`, add a thin wrapper:

    ```rust
    pub struct MyStrategy;

    impl Strategy for MyStrategy {
        fn name(&self) -> &str { "my_strategy" }
        fn apply(&self, ctx: &Context, cfg: &PipelineConfig) -> Result<Vec<Chunk>> {
            Ok(my_strategy::apply(ctx, /* args from cfg */))
        }
    }
    ```

4. **Register in the factory.** Add the new name to `make_strategy` in `src/pipeline/mod.rs` so `--strategy my_strategy` works from the CLI.

5. **Decide preset membership.** If the new strategy should be part of a preset, update `preset_strategies` accordingly.

6. **Tests.** Add a `tests/my_strategy_tests.rs` (or extend `tests/pipeline_tests.rs`) with at least one happy-path and one edge-case test. Use the fixtures under `tests/fixtures/`.

7. **Docs.** Update the strategy table in `README.md`, and add a bullet to the relevant section of `docs/BENCHMARKS.md` when you have measurements.

## Code style

- **Let the formatter decide.** `cargo fmt` is law. Don't hand-format.
- **Clippy clean.** `cargo clippy --all-features -- -D warnings` must pass. Use `#[allow(clippy::X)]` sparingly and only with a one-line justification comment.
- **Comments explain "why," not "what."** If a comment paraphrases what the code does, delete it. If it captures a non-obvious constraint, invariant, or historical reason, keep it.
- **`///` for pub items.** Every public function, struct, enum, and trait gets a doc comment: brief description, `# Errors` section on fallible functions, `# Examples` on the main public API.
- **Error handling.** `anyhow::Result` in binaries and test code; keep the library surface clean. Never `unwrap()` on user input.
- **Feature gates.** New network-touching code goes behind `embeddings`, `llm`, or `proxy`. Core stays dependency-light.
- **No buzzwords.** We avoid "revolutionary," "paradigm," "leverage," and friends in user-facing text. Plain prose.

## Pull request process

1. **Open an issue first** for non-trivial changes (new strategies, CLI changes, architecture shifts) so we can align on scope before you write code.
2. **One concern per PR.** Rebase your work into focused commits; split unrelated changes.
3. **Include tests** that fail on `main` and pass with your change.
4. **Update docs** in the same PR: `README.md`, `docs/BENCHMARKS.md`, `CHANGELOG.md`.
5. **Run the full local check:**
    ```bash
    cargo fmt --check
    cargo clippy --features "proxy,embeddings,llm" -- -D warnings
    cargo test --features "proxy,embeddings,llm"
    ```
6. **CI must be green** before merge.

## License

By contributing you agree that your contributions will be licensed under the MIT License (see `LICENSE`).
