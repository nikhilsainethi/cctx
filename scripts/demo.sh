#!/usr/bin/env bash
#
# cctx end-to-end demo — record this for the README GIF.
#
# Usage:
#   cargo build --release --features proxy
#   ./scripts/demo.sh
#
# For recording: use `asciinema rec demo.cast` then `./scripts/demo.sh`

set -uo pipefail

CCTX="${CCTX:-target/release/cctx}"
if [[ ! -x "$CCTX" ]]; then
    CCTX="target/debug/cctx"
fi
if [[ ! -x "$CCTX" ]]; then
    echo "Build first: cargo build --release --features proxy" >&2
    exit 1
fi

FIXTURE="tests/fixtures/large_conversation.json"
TECH="tests/fixtures/technical_conversation.json"

# ── Colors ─────────────────────────────────────────────────────────────────────

BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}═══ $1 ═══${RESET}"
    echo ""
}

run() {
    echo -e "${DIM}\$ $*${RESET}"
    "$@"
    echo ""
}

pause() {
    sleep "${1:-1}"
}

# ═══════════════════════════════════════════════════════════════════════════════

clear
echo ""
echo -e "${BOLD}${CYAN}"
echo "   ╭──────────────────────────────────────────╮"
echo "   │  cctx — Context Compiler for LLMs        │"
echo "   │                                          │"
echo "   │  Your LLM is only as smart as the        │"
echo "   │  context you feed it.                    │"
echo "   ╰──────────────────────────────────────────╯"
echo -e "${RESET}"
pause 2

# ── 1. The Problem ─────────────────────────────────────────────────────────────

banner "1. THE PROBLEM"
echo -e "Let's analyze a ${BOLD}39-turn conversation${RESET} (8,000+ tokens)..."
echo -e "This is a real debugging session — a developer asks an AI about"
echo -e "PostgreSQL connection leaks, Redis eviction, and Kubernetes scaling."
echo ""
pause 1

run $CCTX analyze "$FIXTURE"
pause 2

echo -e "${YELLOW}42% of tokens are in the attention dead zone.${RESET}"
echo -e "${YELLOW}The LLM will lose ~30% accuracy on that content.${RESET}"
pause 2

# ── 2. Bookend Reordering ─────────────────────────────────────────────────────

banner "2. FIX: BOOKEND REORDERING"
echo -e "Reorder chunks so high-relevance content sits at the ${BOLD}beginning and end${RESET}"
echo -e "of the context — where LLMs pay the most attention."
echo ""
pause 1

run $CCTX optimize "$FIXTURE" --strategy bookend --output /tmp/demo_step1.json
pause 1

run $CCTX diff "$FIXTURE" /tmp/demo_step1.json
pause 2

# ── 3. Structural Compression ─────────────────────────────────────────────────

banner "3. FIX: STRUCTURAL COMPRESSION"
echo -e "Compress JSON, code, and markdown inside messages."
echo -e "This fixture has an API schema + Python code — let's compress it."
echo ""
pause 1

run $CCTX optimize "$TECH" --strategy structural --output /tmp/demo_structural.json
pause 1

run $CCTX diff "$TECH" /tmp/demo_structural.json
pause 2

# ── 4. Full Pipeline ──────────────────────────────────────────────────────────

banner "4. FULL PIPELINE: BALANCED PRESET"
echo -e "The ${BOLD}balanced${RESET} preset combines bookend + structural compression."
echo ""
pause 1

run $CCTX optimize "$TECH" --preset balanced --output /tmp/demo_balanced.json
pause 1

run $CCTX diff "$TECH" /tmp/demo_balanced.json
pause 2

# ── 5. Budget Compression ─────────────────────────────────────────────────────

banner "5. BUDGET COMPRESSION"
echo -e "Force a ${BOLD}2,000 token budget${RESET} on an 8,000 token conversation."
echo -e "Structural compression first, then smart truncation."
echo -e "System messages and recent user messages are ${GREEN}never dropped${RESET}."
echo ""
pause 1

run $CCTX compress "$FIXTURE" --budget 2000 --output /tmp/demo_compressed.json
pause 1

run $CCTX diff "$FIXTURE" /tmp/demo_compressed.json
pause 2

# ── 6. Token Counting + Piping ────────────────────────────────────────────────

banner "6. PIPING: UNIX COMPOSABILITY"
echo -e "cctx works in shell pipelines — data to stdout, messages to stderr."
echo ""
pause 1

echo -e "${DIM}\$ cctx count tests/fixtures/large_conversation.json${RESET}"
$CCTX count "$FIXTURE"
echo ""
pause 1

echo -e "${DIM}\$ cctx optimize $TECH --preset balanced | cctx count${RESET}"
$CCTX optimize "$TECH" --preset balanced 2>/dev/null | $CCTX count
echo ""
pause 1

echo -e "${DIM}\$ cat tests/fixtures/rag_chunks.json | cctx analyze --format json | python3 -c ...${RESET}"
cat tests/fixtures/rag_chunks.json | $CCTX analyze --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Health: {d[\"health_score\"]}/100, Tokens: {d[\"total_tokens\"]}')"
echo ""
pause 2

# ── 7. Multi-Format Support ───────────────────────────────────────────────────

banner "7. INPUT FORMAT AUTO-DETECTION"
echo -e "cctx auto-detects: OpenAI, Anthropic, RAG chunks, raw text."
echo ""

for f in tests/fixtures/sample_conversation.json tests/fixtures/anthropic_conversation.json tests/fixtures/rag_chunks.json tests/fixtures/raw_document.txt; do
    name=$(basename "$f")
    count=$($CCTX count "$f")
    echo -e "  ${name}: ${BOLD}${count} tokens${RESET}"
done
echo ""
pause 2

# ── 8. Proxy Mode ─────────────────────────────────────────────────────────────

banner "8. PROXY MODE"

if $CCTX help 2>&1 | grep -q proxy; then
    echo -e "Start a transparent proxy that optimizes every API call:"
    echo ""
    echo -e "${DIM}\$ cctx proxy --listen 127.0.0.1:8080 \\"
    echo -e "    --upstream https://api.openai.com \\"
    echo -e "    --strategy bookend --strategy structural \\"
    echo -e "    --dry-run${RESET}"
    echo ""
    pause 1

    $CCTX proxy --listen 127.0.0.1:18099 --upstream https://api.openai.com \
        --strategy bookend --dry-run 2>/dev/null &
    PROXY_PID=$!
    sleep 2

    echo -e "Health check:"
    echo -e "${DIM}\$ curl http://127.0.0.1:18099/cctx/health${RESET}"
    curl -s http://127.0.0.1:18099/cctx/health | python3 -m json.tool
    echo ""

    echo -e "Metrics:"
    echo -e "${DIM}\$ curl http://127.0.0.1:18099/cctx/metrics${RESET}"
    curl -s http://127.0.0.1:18099/cctx/metrics | python3 -m json.tool
    echo ""

    kill $PROXY_PID 2>/dev/null
    wait $PROXY_PID 2>/dev/null
else
    echo -e "${DIM}(proxy not compiled — build with: cargo build --features proxy)${RESET}"
fi
pause 2

# ── Summary ────────────────────────────────────────────────────────────────────

banner "SUMMARY"
echo -e "${BOLD}cctx${RESET} optimizes LLM context at the application layer:"
echo ""
echo -e "  ${GREEN}analyze${RESET}   — health report (tokens, dead zones, duplication)"
echo -e "  ${GREEN}optimize${RESET}  — apply strategies (bookend, structural, dedup, prune)"
echo -e "  ${GREEN}compress${RESET}  — hit a hard token budget"
echo -e "  ${GREEN}count${RESET}     — token counting (pipe-friendly)"
echo -e "  ${GREEN}diff${RESET}      — before/after comparison"
echo -e "  ${GREEN}proxy${RESET}     — transparent optimization proxy"
echo ""
echo -e "Install: ${BOLD}cargo install cctx${RESET}"
echo -e "Source:  ${BOLD}https://github.com/nikhilsainethi/cctx${RESET}"
echo ""

# Cleanup
rm -f /tmp/demo_step1.json /tmp/demo_structural.json /tmp/demo_balanced.json /tmp/demo_compressed.json
