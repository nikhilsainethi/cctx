# VHS tape file — records a terminal GIF from the demo script.
#
# Prerequisites:
#   brew install vhs
#   cargo build --release --features proxy
#
# Record:
#   cd /path/to/cctx
#   vhs scripts/record-demo.tape
#
# Output: demo.gif (in project root)

Output demo.gif
Set Shell bash
Set FontSize 14
Set Width 1200
Set Height 800
Set Padding 20
Set Theme "Molokai"
Set TypingSpeed 30ms

# ── Hidden setup: build and add to PATH ────────────────────────────────────
# VHS runs a fresh shell, so `cctx` won't be in PATH.
# We build the binary (hidden) and prepend target/release to PATH.
Hide
Type "cargo build --release --features proxy 2>/dev/null"
Enter
Sleep 20s
Type "export PATH=$PWD/target/release:$PATH"
Enter
Sleep 500ms
Show

# ── Title ──────────────────────────────────────────────────────────────────
Type "echo ''"
Enter
Sleep 300ms
Type "echo '  cctx — Context Compiler for LLMs'"
Enter
Type "echo '  Your LLM is only as smart as the context you feed it.'"
Enter
Type "echo ''"
Enter
Sleep 2s

# ── Step 1: Analyze ────────────────────────────────────────────────────────
Type "# Step 1: Analyze a 39-turn conversation (8,000+ tokens)"
Enter
Sleep 1s
Type "cctx analyze tests/fixtures/large_conversation.json"
Enter
Sleep 5s

# ── Step 2: Optimize ──────────────────────────────────────────────────────
Type "# Step 2: Optimize with bookend + structural compression"
Enter
Sleep 1s
Type "cctx optimize tests/fixtures/technical_conversation.json --preset balanced --output /tmp/opt.json 2>&1 | tail -5"
Enter
Sleep 3s

# ── Step 3: Diff ──────────────────────────────────────────────────────────
Type "# Step 3: Compare before vs after"
Enter
Sleep 1s
Type "cctx diff tests/fixtures/technical_conversation.json /tmp/opt.json"
Enter
Sleep 4s

# ── Step 4: Budget compression ────────────────────────────────────────────
Type "# Step 4: Compress 8,000 tokens to a 2,000 token budget"
Enter
Sleep 1s
Type "cctx compress tests/fixtures/large_conversation.json --budget 2000 2>&1 | tail -8"
Enter
Sleep 3s

# ── Step 5: Piping ────────────────────────────────────────────────────────
Type "# Step 5: Pipe-friendly — optimize then count tokens"
Enter
Sleep 1s
Type "cctx optimize tests/fixtures/technical_conversation.json --preset balanced 2>/dev/null | cctx count"
Enter
Sleep 2s

# ── Step 6: Proxy ─────────────────────────────────────────────────────────
Type "# Step 6: Transparent proxy — one line to optimize all API calls"
Enter
Sleep 1s
Type "cctx proxy --listen 127.0.0.1:8080 --upstream https://api.openai.com --strategy bookend --dry-run &"
Enter
Sleep 3s
Type "curl -s http://127.0.0.1:8080/cctx/health | python3 -m json.tool"
Enter
Sleep 2s

# Cleanup proxy
Hide
Type "kill %1 2>/dev/null; wait 2>/dev/null"
Enter
Sleep 1s
Show

# ── Closing ────────────────────────────────────────────────────────────────
Type "echo ''"
Enter
Type "echo '  Install: cargo install cctx'"
Enter
Type "echo '  Source:  github.com/nikhilsainethi/cctx'"
Enter
Sleep 3s
