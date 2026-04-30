#!/usr/bin/env bash
# Fingerprint accuracy benchmark.
#
# Runs `cctx fingerprint --tier 0` against the curated fixture
# `tests/fixtures/fingerprint_accuracy.jsonl` and tallies recall +
# precision against a known set of 6 fingerprintable facts:
#
#   1. budget $50K constraint
#   2. PostgreSQL (primary store decision)
#   3. port 8443 (auth-service operational fact)
#   4. /etc/myapp/prod.yml (config path)
#   5. HPA scaling delay (root-cause fact)
#   6. CORS header missing (root-cause fact)
#
# Recall      = facts found / 6
# Precision   = "load-bearing" extracted items / total extracted items
#               (where "load-bearing" means an item that matches one of
#               the six fact patterns; everything else is treated as
#               filler for precision purposes)
#
# Output is human-readable; the trailing line is machine-friendly so a
# CI step can grep for `RECALL=` / `PRECISION=` if it wants to gate on
# accuracy regressions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${REPO_ROOT}/tests/fixtures/fingerprint_accuracy.jsonl"
BIN="${REPO_ROOT}/target/release/cctx"

if [ ! -f "$FIXTURE" ]; then
  echo "fixture not found: $FIXTURE" >&2
  exit 1
fi

# Build release if needed — release is what users actually run, so
# benchmark numbers from debug builds would be misleading.
if [ ! -x "$BIN" ]; then
  echo "[bench] building release binary..."
  (cd "$REPO_ROOT" && cargo build --release --features "proxy,embeddings,llm" >/dev/null)
fi

# Run fingerprint and capture JSON to a tempfile.
TMP="$(mktemp -t cctx_accuracy.XXXXXX.json)"
trap 'rm -f "$TMP"' EXIT

"$BIN" fingerprint --tier 0 "$FIXTURE" >"$TMP"

# Parse with python3 — jq is not guaranteed to be installed.
python3 - "$TMP" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

items = data.get("items", [])

# Six known fingerprintable facts. Each entry is (label, regex). Regex
# is case-insensitive and matches the canonical surface form OR a
# close paraphrase that still preserves the load-bearing token (port
# number, file path, framework name, ...).
KNOWN = [
    ("budget $50K",           re.compile(r"\$50K|50K hard cap|budget.*\$50K|budget.*50K", re.I)),
    ("PostgreSQL",            re.compile(r"\bPostgreSQL\b", re.I)),
    ("port 8443",             re.compile(r"\bport 8443\b|\b8443\b", re.I)),
    ("/etc/myapp/prod.yml",   re.compile(r"/etc/myapp/prod\.yml", re.I)),
    ("HPA scaling delay",     re.compile(r"\bHPA\b.*scaling|scaling delay|HPA.*cold start", re.I)),
    ("CORS header missing",   re.compile(r"missing CORS header|CORS header.*missing", re.I)),
]

# RECALL: which of the 6 facts shows up in any extracted item?
found = []
missing = []
for label, rx in KNOWN:
    if any(rx.search(item.get("content", "")) for item in items):
        found.append(label)
    else:
        missing.append(label)

# PRECISION: how many extracted items are load-bearing (contain at
# least one of the known fact patterns)? Everything else is filler.
load_bearing = 0
filler = []
for item in items:
    content = item.get("content", "")
    if any(rx.search(content) for _, rx in KNOWN):
        load_bearing += 1
    else:
        filler.append(item)

total = len(items)
recall_pct = (len(found) / len(KNOWN)) * 100.0
precision_pct = (load_bearing / total) * 100.0 if total else 0.0

print()
print("=" * 64)
print("Fingerprint accuracy benchmark")
print("=" * 64)
print(f"Fixture: tests/fixtures/fingerprint_accuracy.jsonl")
print(f"Items extracted: {total}")
print()
print(f"RECALL: {len(found)}/{len(KNOWN)} known facts ({recall_pct:.0f}%)")
for label in found:
    print(f"  ✓ {label}")
for label in missing:
    print(f"  ✗ {label}  (MISSED)")
print()
print(f"PRECISION: {load_bearing}/{total} extracted items load-bearing ({precision_pct:.0f}%)")
print()
print("Filler items (not matching any of the 6 known facts):")
if not filler:
    print("  (none — every extracted item matched a known fact)")
else:
    for item in filler:
        cat = item.get("category", "?")
        score = item.get("priority_score", 0.0)
        snippet = (item.get("content", "")[:80] + "…") if len(item.get("content", "")) > 80 else item.get("content", "")
        print(f"  [{cat:<14}] score={score:.2f}  {snippet}")
print()
# Machine-friendly trailer for CI.
print(f"RECALL={recall_pct:.0f}% PRECISION={precision_pct:.0f}% FOUND={len(found)}/{len(KNOWN)} ITEMS={total}")
PY
