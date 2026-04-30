#!/usr/bin/env bash
# Performance benchmark: how fast does Tier-0 fingerprinting scale?
#
# Generates synthetic transcripts at 50 / 200 / 500 / 1500 messages
# (the last one approximates the 200K-token compaction trigger), then
# times `cctx fingerprint --tier 0` on each. Five runs per size; the
# median is the reported number so a single GC blip doesn't dominate.
#
# Target (from COMPACTION_DESIGN.md): 1500 messages should fingerprint
# in under 3 seconds on a modern dev laptop.
#
# Output is human-readable with a machine-friendly trailer that the
# Markdown report generator can parse.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/target/release/cctx"

if [ ! -x "$BIN" ]; then
  echo "[bench] building release binary..."
  (cd "$REPO_ROOT" && cargo build --release --features "proxy,embeddings,llm" >/dev/null)
fi

SIZES=(50 200 500 1500)
RUNS=5

WORKDIR="$(mktemp -d -t cctx_perf.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

# Pre-generate every transcript up front so we're not timing
# generation alongside fingerprinting.
echo "[bench] generating transcripts..."
for n in "${SIZES[@]}"; do
  "$BIN" bench generate --messages "$n" --output "$WORKDIR/n${n}.jsonl" >/dev/null 2>&1
done

# Use python3 for millisecond timing — `time` on macOS is awkward to
# script and `date +%N` doesn't work on BSD date.
time_ms() {
  python3 -c '
import subprocess, sys, time
cmd = sys.argv[1:]
start = time.perf_counter()
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
elapsed = (time.perf_counter() - start) * 1000.0
print(f"{elapsed:.1f}")
' "$@"
}

# median <numbers...>
median() {
  python3 -c '
import sys
xs = sorted(float(x) for x in sys.argv[1:])
n = len(xs)
mid = xs[n // 2] if n % 2 == 1 else (xs[n // 2 - 1] + xs[n // 2]) / 2
print(f"{mid:.1f}")
' "$@"
}

echo
echo "================================================================"
echo "Fingerprint performance benchmark (Tier 0)"
echo "================================================================"
printf "%-10s %-10s %-12s %-12s %-12s\n" "Messages" "Bytes" "Median(ms)" "Min(ms)" "Max(ms)"
echo "----------------------------------------------------------------"

# Capture results for the trailer.
declare -a TRAIL

for n in "${SIZES[@]}"; do
  fixture="$WORKDIR/n${n}.jsonl"
  bytes="$(wc -c <"$fixture" | tr -d ' ')"

  runs=()
  for _ in $(seq 1 "$RUNS"); do
    runs+=("$(time_ms "$BIN" fingerprint --tier 0 "$fixture")")
  done

  med="$(median "${runs[@]}")"
  mn="$(printf '%s\n' "${runs[@]}" | python3 -c 'import sys; print(f"{min(float(x) for x in sys.stdin):.1f}")')"
  mx="$(printf '%s\n' "${runs[@]}" | python3 -c 'import sys; print(f"{max(float(x) for x in sys.stdin):.1f}")')"

  printf "%-10s %-10s %-12s %-12s %-12s\n" "$n" "$bytes" "$med" "$mn" "$mx"
  TRAIL+=("N=${n} BYTES=${bytes} MEDIAN_MS=${med} MIN_MS=${mn} MAX_MS=${mx}")
done

echo
# Machine-friendly trailer — one line per size, easy to grep.
for line in "${TRAIL[@]}"; do
  echo "PERF: $line"
done
