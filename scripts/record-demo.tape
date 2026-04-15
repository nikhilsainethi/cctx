# VHS tape file — records a terminal GIF for the README.
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
Set FontSize 15
Set Width 1000
Set Height 650
Set Padding 15
Set Theme "Molokai"
Set TypingSpeed 50ms
Set CursorBlink false

# ── Hidden setup ───────────────────────────────────────────────────────────
Hide
Type "cargo build --release --features proxy 2>/dev/null"
Enter
Sleep 20s
Type "export PATH=$PWD/target/release:$PATH"
Enter
Type "clear"
Enter
Sleep 500ms
Show

# ── Analyze a conversation ─────────────────────────────────────────────────
Sleep 500ms
Type "cctx analyze tests/fixtures/technical_conversation.json"
Enter
Sleep 6s

# ── Optimize it ────────────────────────────────────────────────────────────
Type "cctx optimize tests/fixtures/technical_conversation.json --preset balanced --output /tmp/optimized.json"
Enter
Sleep 4s

# ── Show the diff ──────────────────────────────────────────────────────────
Type "cctx diff tests/fixtures/technical_conversation.json /tmp/optimized.json"
Enter
Sleep 5s

# ── Compress to a budget ───────────────────────────────────────────────────
Type "cctx compress tests/fixtures/large_conversation.json --budget 2000"
Enter
Sleep 4s

# ── Pipe: optimize then count ──────────────────────────────────────────────
Type "cctx optimize tests/fixtures/technical_conversation.json --preset balanced 2>/dev/null | cctx count"
Enter
Sleep 3s

# ── Start proxy, check health ─────────────────────────────────────────────
Type "cctx proxy --listen 127.0.0.1:8080 --upstream https://api.openai.com --strategy bookend --dry-run &"
Enter
Sleep 3s
Type "curl -s localhost:8080/cctx/health | python3 -m json.tool"
Enter
Sleep 3s

# ── Cleanup (hidden) ──────────────────────────────────────────────────────
Hide
Type "kill %1 2>/dev/null; wait 2>/dev/null"
Enter
Sleep 1s
Show

Sleep 2s
