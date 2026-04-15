# cctx development commands
# Usage: make <target>

.PHONY: build test lint release proxy-test clean fmt check all

# ── Default ────────────────────────────────────────────────────────────────────

all: lint test build

# ── Build ──────────────────────────────────────────────────────────────────────

build:
	cargo build

build-release:
	cargo build --release

build-full:
	cargo build --features "proxy,embeddings"

build-release-full:
	cargo build --release --features "proxy,embeddings"

# ── Test ───────────────────────────────────────────────────────────────────────

test:
	cargo test

test-full:
	cargo test --features "proxy,embeddings"

proxy-test: build-full
	./tests/test_proxy.sh

test-all: test test-full proxy-test

# ── Lint ───────────────────────────────────────────────────────────────────────

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all --check

clippy:
	cargo clippy -- -D warnings
	cargo clippy --features "proxy,embeddings" -- -D warnings

lint: fmt-check clippy

# ── Release ────────────────────────────────────────────────────────────────────

release: lint test-all build-release build-release-full
	@echo "Release builds ready:"
	@ls -lh target/release/cctx 2>/dev/null || true
	@echo ""
	@echo "To publish: git tag v0.1.0 && git push --tags"

# ── Benchmark ──────────────────────────────────────────────────────────────────

bench: build-release
	./benches/run_benchmark.sh

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:
	cargo clean

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo "cctx development commands:"
	@echo ""
	@echo "  make build          Build core (no proxy)"
	@echo "  make build-full     Build with proxy + embeddings"
	@echo "  make test           Run core tests"
	@echo "  make test-full      Run all tests (proxy + embeddings)"
	@echo "  make proxy-test     Run proxy shell tests"
	@echo "  make test-all       Run everything"
	@echo "  make lint           Check formatting + clippy"
	@echo "  make fmt            Auto-format code"
	@echo "  make bench          Run benchmarks"
	@echo "  make release        Full release build with checks"
	@echo "  make clean          Remove build artifacts"
