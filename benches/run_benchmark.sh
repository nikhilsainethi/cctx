#!/usr/bin/env bash
#
# cctx benchmark runner
#
# Runs each preset (safe, balanced, aggressive) on every fixture and prints
# a summary table with token counts, health scores, reduction percentages,
# and wall-clock timing.
#
# Usage:
#   cargo build --release
#   ./benches/run_benchmark.sh
#
# Or with a debug build (slower but fine for correctness):
#   cargo build
#   CCTX=target/debug/cctx ./benches/run_benchmark.sh

set -euo pipefail

# ── Resolve the cctx binary ──────────────────────────────────────────────────

CCTX="${CCTX:-}"
if [[ -z "$CCTX" ]]; then
    if [[ -x target/release/cctx ]]; then
        CCTX=target/release/cctx
    elif [[ -x target/debug/cctx ]]; then
        CCTX=target/debug/cctx
    else
        echo "Error: no cctx binary found. Run 'cargo build --release' first." >&2
        exit 1
    fi
fi

echo "Using: $CCTX ($($CCTX --version))"
echo

# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURES=(
    "tests/fixtures/sample_conversation.json"
    "tests/fixtures/technical_conversation.json"
    "tests/fixtures/structured_content.json"
    "tests/fixtures/large_conversation.json"
    "tests/fixtures/rag_chunks.json"
    "tests/fixtures/anthropic_conversation.json"
)

# Short display names (strip path and extension).
fixture_name() {
    basename "$1" .json
}

# ── Temp directory ────────────────────────────────────────────────────────────

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# ── Comma-format a number ────────────────────────────────────────────────────

comma() {
    python3 -c "print(f'{$1:,}')"
}

# ── Measure wall-clock time in milliseconds ──────────────────────────────────
# Runs a command, captures its exit code, and prints elapsed ms to stdout.
# The command's own stdout/stderr go to /dev/null.

time_ms() {
    local start end
    start=$(python3 -c "import time; print(int(time.time()*1000))")
    "$@" >/dev/null 2>&1
    end=$(python3 -c "import time; print(int(time.time()*1000))")
    echo $(( end - start ))
}

# ── Collect data ──────────────────────────────────────────────────────────────

# Arrays to hold results (parallel indexed with FIXTURES).
declare -a NAMES ORIGINALS SAFES BALANCEDS AGGRESSIVES
declare -a HEALTH_BEFORES HEALTH_AFTERS
declare -a TIME_SAFES TIME_BALANCEDS TIME_AGGRESSIVES

for fixture in "${FIXTURES[@]}"; do
    name=$(fixture_name "$fixture")
    NAMES+=("$name")

    # Original token count.
    orig=$($CCTX count "$fixture")
    ORIGINALS+=("$orig")

    # Original health score.
    health_before=$($CCTX analyze "$fixture" --format json | python3 -c "import sys,json; print(json.load(sys.stdin)['health_score'])")
    HEALTH_BEFORES+=("$health_before")

    # ── Safe preset ───────────────────────────────────────────────────────
    safe_out="$TMPDIR/${name}_safe.json"
    ms=$(time_ms $CCTX optimize "$fixture" --preset safe --output "$safe_out")
    safe_tokens=$($CCTX count "$safe_out")
    SAFES+=("$safe_tokens")
    TIME_SAFES+=("$ms")

    # ── Balanced preset ───────────────────────────────────────────────────
    balanced_out="$TMPDIR/${name}_balanced.json"
    ms=$(time_ms $CCTX optimize "$fixture" --preset balanced --output "$balanced_out")
    balanced_tokens=$($CCTX count "$balanced_out")
    BALANCEDS+=("$balanced_tokens")
    TIME_BALANCEDS+=("$ms")

    # ── Aggressive preset ─────────────────────────────────────────────────
    aggressive_out="$TMPDIR/${name}_aggressive.json"
    ms=$(time_ms $CCTX optimize "$fixture" --preset aggressive --output "$aggressive_out")
    aggressive_tokens=$($CCTX count "$aggressive_out")
    AGGRESSIVES+=("$aggressive_tokens")
    TIME_AGGRESSIVES+=("$ms")

    # Health score after balanced optimization.
    health_after=$($CCTX analyze "$balanced_out" --format json | python3 -c "import sys,json; print(json.load(sys.stdin)['health_score'])")
    HEALTH_AFTERS+=("$health_after")
done

# ── Print summary table ──────────────────────────────────────────────────────

pct() {
    # Compute percentage change: (after - before) / before * 100.
    local before=$1 after=$2
    if [[ "$before" -eq 0 ]]; then
        echo "—"
        return
    fi
    python3 -c "print(f'{($after - $before) / $before * 100:+.1f}%')"
}

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                           cctx Benchmark Results"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo

# ── Token counts table ────────────────────────────────────────────────────────

printf "%-26s │ %8s │ %8s │ %8s │ %8s │ %8s\n" \
    "Fixture" "Original" "Safe" "Balanced" "Aggressv" "Reductn"
printf "%-26s─┼─%8s─┼─%8s─┼─%8s─┼─%8s─┼─%8s\n" \
    "──────────────────────────" "────────" "────────" "────────" "────────" "────────"

for i in "${!NAMES[@]}"; do
    reduction=$(pct "${ORIGINALS[$i]}" "${BALANCEDS[$i]}")
    printf "%-26s │ %8s │ %8s │ %8s │ %8s │ %8s\n" \
        "${NAMES[$i]}" \
        "$(comma "${ORIGINALS[$i]}")" \
        "$(comma "${SAFES[$i]}")" \
        "$(comma "${BALANCEDS[$i]}")" \
        "$(comma "${AGGRESSIVES[$i]}")" \
        "$reduction"
done

echo

# ── Health scores table ───────────────────────────────────────────────────────

printf "%-26s │ %8s │ %8s │ %8s\n" \
    "Fixture" "Before" "After" "Delta"
printf "%-26s─┼─%8s─┼─%8s─┼─%8s\n" \
    "──────────────────────────" "────────" "────────" "────────"

for i in "${!NAMES[@]}"; do
    before="${HEALTH_BEFORES[$i]}"
    after="${HEALTH_AFTERS[$i]}"
    delta=$(( after - before ))
    sign=""
    if [[ "$delta" -gt 0 ]]; then sign="+"; fi
    printf "%-26s │ %8s │ %8s │ %8s\n" \
        "${NAMES[$i]}" \
        "$before/100" \
        "$after/100" \
        "${sign}${delta}"
done

echo

# ── Timing table ──────────────────────────────────────────────────────────────

printf "%-26s │ %8s │ %8s │ %8s\n" \
    "Fixture" "Safe" "Balanced" "Aggressv"
printf "%-26s─┼─%8s─┼─%8s─┼─%8s\n" \
    "──────────────────────────" "────────" "────────" "────────"

for i in "${!NAMES[@]}"; do
    printf "%-26s │ %6sms │ %6sms │ %6sms\n" \
        "${NAMES[$i]}" \
        "${TIME_SAFES[$i]}" \
        "${TIME_BALANCEDS[$i]}" \
        "${TIME_AGGRESSIVES[$i]}"
done

echo
echo "═══════════════════════════════════════════════════════════════════════════════"

# ── Aggregate stats ───────────────────────────────────────────────────────────

total_orig=0
total_balanced=0
for i in "${!NAMES[@]}"; do
    total_orig=$(( total_orig + ORIGINALS[i] ))
    total_balanced=$(( total_balanced + BALANCEDS[i] ))
done
total_saved=$(( total_orig - total_balanced ))
overall_pct=$(python3 -c "print(f'{$total_saved / $total_orig * 100:.1f}%')")

echo
echo "Total tokens across all fixtures:"
echo "  Original:  $(comma $total_orig)"
echo "  Balanced:  $(comma $total_balanced)"
echo "  Saved:     $(comma $total_saved) ($overall_pct reduction)"
