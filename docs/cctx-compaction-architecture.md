# cctx Compaction Guard — Architecture Design Document

> **Version:** 1.0  
> **Author:** Nikhil Sai Nethi + Claude (Architecture Partner)  
> **Date:** April 15, 2026  
> **Status:** Pre-implementation (Post Day-20 roadmap)  
> **Prerequisite:** cctx core (Days 1–20) must be complete — analyze, optimize, diff, health scoring, bookend/structural/budget strategies, edge case hardening, CLI polish.

---

## 1. Problem Statement

### 1.1 The Two Context Problems

LLM coding agents like Claude Code and Codex maintain a context window — every message, file read, command output, and tool result lives in this shared memory. Two distinct problems degrade quality as sessions grow:

**Problem A — Attention Dead Zone (within a live context):**  
Research from Stanford ("Lost in the Middle," 2023) demonstrated that LLMs pay disproportionate attention to content at the beginning and end of context, with accuracy dropping ~30% for information positioned in the middle. This is the problem cctx's core (Days 1–20) addresses via bookend reordering, dead zone detection, and health scoring.

**Problem B — Compaction Loss (when context overflows):**  
When the context window fills (typically at 85–95% of 200K tokens), the agent summarizes the entire conversation into a compressed form and continues from that summary. This process is *lossy*. Architectural decisions from message 5, debugging insights from message 30, constraints the user stated once — all get paraphrased or dropped entirely. The agent "forgets" and the user re-explains, or worse, the agent contradicts prior decisions without knowing it.

**These problems are correlated:** Content in the attention dead zone during a live session is the same content the LLM is likely to underweight when asked to produce a compaction summary. Dead zone analysis from Problem A is therefore a useful *predictor* for Problem B loss.

### 1.2 Why Existing Solutions Are Insufficient

**What compaction does today:**  
Claude Code's compaction is a three-layer system: (1) microcompaction offloads bulky tool results to disk, keeping only references; (2) auto-compaction triggers at ~95% capacity with a generic summarization prompt; (3) manual /compact lets users trigger with optional focus instructions. The summarization prompt asks Claude to preserve "what was accomplished, current work in progress, files involved, next steps, and key user requests." This is reasonable but generic — it doesn't know which items appeared exactly once (high entropy, irreplaceable) vs. which were mentioned repeatedly (safe to compress).

**What people are building as workarounds:**  
Developers are using PreCompact hooks to dump full transcripts into SQLite databases and SessionStart hooks to re-inject "relevant" past context. This brute-force approach has two failure modes: (a) re-injecting too much burns fresh context window space, and (b) relevance is determined by keyword matching, not by information-theoretic analysis.

### 1.3 What cctx Uniquely Provides

cctx is NOT trying to be a better summarizer than Claude. Claude is better at semantic understanding. What cctx provides is something the LLM fundamentally cannot do:

1. **Persistent external state across compaction boundaries** — the LLM's "memory" resets after compaction; cctx's doesn't.
2. **Information-theoretic analysis** — identifying items with high information entropy (mentioned once = irreplaceable) vs. low entropy (mentioned 5 times = safe to compress).
3. **Selective, budget-constrained re-injection** — rather than dumping everything back, cctx injects only what was actually lost AND has high information value, staying within a configurable token budget so the fresh context window isn't immediately polluted.
4. **Measurable before/after health scoring** — quantifiable proof that post-compaction context is healthier than default.

---

## 2. Architecture Overview

### 2.1 System Boundary

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code / Codex                    │
│                                                          │
│  User Prompt → Agentic Loop → Tool Calls → Response      │
│       │                                        │         │
│       │            Context Window               │         │
│       │    [system + messages + tools]           │         │
│       │                                        │         │
│  ┌────┴────────────────────────────────────────┴───┐     │
│  │              Hook Lifecycle Events               │     │
│  │  SessionStart │ UserPromptSubmit │ PreCompact    │     │
│  │  PostCompact  │ Stop             │ PreToolUse    │     │
│  └──────┬───────────────┬──────────────────┬───────┘     │
└─────────┼───────────────┼──────────────────┼─────────────┘
          │               │                  │
          ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                     cctx Binary (Rust)                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐ │
│  │ Transcript│  │ Fingerprint│  │ Loss      │  │Injection│ │
│  │ Parser   │  │ Engine    │  │ Detector  │  │ Builder│ │
│  └──────────┘  └──────────┘  └───────────┘  └────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              State Store (.cctx/)                  │   │
│  │  session-analysis.json │ compaction-log.json       │   │
│  │  fingerprints/         │ injection-history/        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow — Complete Compaction Lifecycle

```
PHASE 1: PRE-COMPACTION (PreCompact hook)
─────────────────────────────────────────
Trigger: Context at ~85-95% capacity, compaction imminent
Input:   Session transcript (JSONL), trigger type, custom_instructions
Action:  cctx fingerprint <transcript_path>

    1. Parse JSONL transcript into internal representation
    2. Tokenize and score every message/chunk
    3. Build information fingerprint:
       - Extract unique facts (mentioned exactly once)
       - Identify critical constraints and decisions
       - Score each item by: uniqueness × recency × position_risk
       - position_risk = dead zone score from core cctx analysis
    4. Write fingerprint to .cctx/fingerprints/<session_id>.json
    5. stdout → Lightweight hint for compaction (< 200 tokens)
       NOTE: This stdout gets paraphrased. It is a HINT, not a
       guarantee. The real mechanism is Phase 3.

    Time budget: < 3 seconds for 200K token transcript


PHASE 2: POST-COMPACTION (PostCompact hook)
───────────────────────────────────────────
Trigger: Compaction complete
Input:   compact_summary (the generated summary text)
Action:  cctx diff-compact <session_id>

    1. Load pre-compaction fingerprint from disk
    2. Parse the compact_summary
    3. For each fingerprinted item, check if it survived:
       - Exact match → PRESERVED
       - Semantic near-match (fuzzy token overlap) → PARAPHRASED
       - No match → LOST
    4. Score the compaction quality (preservation ratio)
    5. Write loss report to .cctx/loss-reports/<session_id>.json
    6. Build injection payload:
       - Select LOST items ranked by (uniqueness × recency)
       - Truncate to injection_budget tokens (configurable, default 4096)
       - Format as structured context for SessionStart
    7. Write injection payload to .cctx/pending-injection/<session_id>.json

    Time budget: < 2 seconds


PHASE 3: RE-INJECTION (SessionStart hook, source: "compact")
─────────────────────────────────────────────────────────────
Trigger: Session resumes after compaction
Input:   source="compact", session_id
Action:  cctx inject <session_id>

    1. Check for pending injection at .cctx/pending-injection/<session_id>.json
    2. If exists, read the injection payload
    3. stdout → Structured context block with recovered items
       Format:
       ---
       [cctx Context Recovery — Items lost during compaction]
       The following items from earlier in this session were not
       captured in the compaction summary. They are listed in
       priority order:

       1. [CONSTRAINT] User specified budget must not exceed $50K
          (stated in message 3, never repeated)
       2. [DECISION] Chose PostgreSQL over MongoDB for ACID compliance
          (discussed in messages 8-10, conclusion in message 10)
       3. [ARCHITECTURE] Auth service runs on port 8443, separate
          from main API on 3000
          (mentioned once in message 14)
       ---
    4. Log injection event to .cctx/compaction-log.json
    5. Delete pending injection file (one-time use)

    Time budget: < 500ms (just file read + stdout)
```

### 2.3 Hook Configuration Schema

This is the `.claude/settings.json` (or `.claude/settings.local.json`) configuration that `cctx install-hooks` generates:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "cctx hook pre-compact"
          }
        ]
      }
    ],
    "PostCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "cctx hook post-compact"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "cctx hook session-start"
          }
        ]
      }
    ]
  }
}
```

Hook input is received via stdin as JSON. The hook subcommands parse this automatically.

---

## 3. Component Design

### 3.1 Module Structure (additions to existing cctx)

```
src/
├── main.rs                    # CLI entry point (existing)
├── lib.rs                     # Library root (existing)
├── core/
│   ├── mod.rs                 # (existing)
│   ├── context.rs             # Context, Chunk, AttentionZone (existing)
│   └── tokenizer.rs           # tiktoken-rs wrapper (existing)
├── analyzer/
│   ├── mod.rs                 # (existing)
│   └── health.rs              # Health scoring (existing)
├── strategies/
│   ├── mod.rs                 # (existing)
│   ├── bookend.rs             # Bookend reordering (existing)
│   ├── structural.rs          # Structural compression (existing)
│   └── budget.rs              # Budget compression (existing)
│
│   ── NEW MODULES BELOW ──
│
├── transcript/
│   ├── mod.rs                 # Transcript module root
│   ├── parser.rs              # JSONL transcript parser
│   ├── schema.rs              # Transcript data types
│   └── normalizer.rs          # Normalize transcript → internal Context repr
├── fingerprint/
│   ├── mod.rs                 # Fingerprint module root
│   ├── extractor.rs           # Fact/decision/constraint extraction
│   ├── scorer.rs              # Information entropy scoring
│   └── index.rs               # Fingerprint index (serializable)
├── compaction/
│   ├── mod.rs                 # Compaction guard module root
│   ├── loss_detector.rs       # Diff fingerprint vs summary
│   ├── injection_builder.rs   # Build re-injection payload
│   └── budget.rs              # Token budget management for injection
├── hooks/
│   ├── mod.rs                 # Hook handler module root
│   ├── pre_compact.rs         # PreCompact hook handler
│   ├── post_compact.rs        # PostCompact hook handler
│   ├── session_start.rs       # SessionStart hook handler
│   └── input.rs               # Hook input JSON parsing
└── state/
    ├── mod.rs                 # State management module root
    ├── store.rs               # .cctx/ directory manager
    └── history.rs             # Compaction history tracking
```

### 3.2 Transcript Parser (`transcript/`)

#### 3.2.1 Problem

Claude Code transcripts are JSONL files, NOT simple OpenAI chat format arrays. Each line is a JSON object representing a conversation event. The schema includes:

- User messages (role: "user")
- Assistant messages (role: "assistant") with potential tool_use blocks
- Tool results (role: "user" with tool_result content blocks)
- System messages and injected context
- Cache control metadata
- Session management events

cctx's existing core works with `Vec<Message>` where `Message = { role: String, content: String }`. The transcript parser must bridge these two representations.

#### 3.2.2 Data Types

```rust
// transcript/schema.rs

use serde::{Deserialize, Serialize};

/// Raw JSONL transcript entry — loosely typed to handle schema variations
#[derive(Debug, Deserialize)]
pub struct TranscriptEntry {
    #[serde(rename = "type")]
    pub entry_type: String,         // "human", "assistant", "tool_result", etc.
    pub message: Option<TranscriptMessage>,
    pub timestamp: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptMessage {
    pub role: String,
    pub content: TranscriptContent,
}

/// Content can be a string or an array of content blocks
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TranscriptContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: Option<String> },
    #[serde(other)]
    Unknown,
}
```

#### 3.2.3 Normalization Strategy

The normalizer converts transcript entries into cctx's internal `Context` representation:

1. **User messages** → `Chunk { role: User, content: text, position: N }`
2. **Assistant text** → `Chunk { role: Assistant, content: text, position: N }`
3. **Tool use + result pairs** → Merged into a single chunk tagged with `ChunkType::ToolInteraction` containing tool name, summarized input, and output
4. **System/injected context** → `Chunk { role: System, content: text, position: 0 }` (pinned to start)
5. **Ordering** — Chunks are ordered by their position in the transcript, preserving conversation flow

Tool results are often the largest chunks (file contents, grep output, etc.). The normalizer should detect and flag these as `compressible: true` since they can be re-read from disk.

### 3.3 Fingerprint Engine (`fingerprint/`)

The fingerprint is the core intellectual contribution of cctx for compaction. It answers: "What unique, irreplaceable information exists in this conversation?"

#### 3.3.1 Fingerprint Item

```rust
// fingerprint/index.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Fingerprint {
    pub session_id: String,
    pub created_at: String,
    pub total_tokens: usize,
    pub total_items: usize,
    pub items: Vec<FingerprintItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FingerprintItem {
    /// Unique identifier for this item
    pub id: String,

    /// What category this item belongs to
    pub category: ItemCategory,

    /// The actual content (condensed to essential text)
    pub content: String,

    /// Token count of the content
    pub tokens: usize,

    /// How many times this information appears in the conversation
    /// 1 = unique/irreplaceable, 5+ = redundant/safe to lose
    pub occurrence_count: u32,

    /// Which message positions contain this information
    pub source_positions: Vec<usize>,

    /// Composite priority score (higher = more important to preserve)
    /// Formula: uniqueness_score × recency_score × position_risk_score
    pub priority_score: f64,

    /// Component scores for transparency
    pub scores: ItemScores,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ItemScores {
    /// 1.0 / occurrence_count — rarer items score higher
    pub uniqueness: f64,

    /// Exponential decay from most recent message. Recent = higher.
    /// Formula: e^(-decay_rate × (total_messages - last_occurrence_position))
    pub recency: f64,

    /// How deep in the dead zone the primary occurrence sits.
    /// Uses cctx core's attention zone mapping.
    /// 0.0 = start/end (safe), 1.0 = deep middle (high risk)
    pub position_risk: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ItemCategory {
    /// User-stated requirement or constraint
    /// e.g., "budget is $50K", "must use PostgreSQL", "deadline is Friday"
    Constraint,

    /// Architectural or design decision with rationale
    /// e.g., "chose REST over GraphQL because of team familiarity"
    Decision,

    /// Specific technical fact (port number, file path, variable name)
    /// e.g., "auth service on port 8443", "config in /etc/app/config.yml"
    TechnicalFact,

    /// Error or debugging insight
    /// e.g., "the CORS error was caused by missing headers in nginx"
    DebugInsight,

    /// Task status or progress marker
    /// e.g., "files A, B, C are done; D and E remain"
    ProgressMarker,

    /// Uncategorized but unique information
    Other,
}
```

#### 3.3.2 Extraction Algorithm

The extraction is NOT purely heuristic. It combines structural analysis with lightweight semantic patterns:

**Step 1: Chunk and tokenize** (uses existing cctx core)

**Step 2: Extract candidate items** via pattern matching on message text:
- Constraints: Look for phrases like "must", "should not", "requirement", "constraint", "limit", "budget", "deadline", numbers with units
- Decisions: Look for "chose", "decided", "went with", "because", "trade-off", "instead of"
- Technical facts: Look for port numbers, file paths, URLs, IP addresses, variable/function names, version numbers
- Debug insights: Look for "the issue was", "root cause", "fixed by", "error was caused by"
- Progress markers: Look for "done", "completed", "remaining", "TODO", "next step"

**Step 3: Deduplicate** — if the same fact appears multiple times (user mentions budget 3 times), merge into one item with `occurrence_count: 3`

**Step 4: Score** — apply the priority formula: `uniqueness × recency × position_risk`

**Step 5: Sort by priority** — highest first

This is a heuristic extraction system. It will miss some items and occasionally miscategorize. That's acceptable — the goal is catching the highest-value items that compaction is most likely to drop, not perfect recall. Even catching 60% of critical items that would otherwise be lost is a significant improvement over catching 0%.

#### 3.3.3 Performance Target

For a 200K-token transcript (~1500 JSONL lines):
- JSONL parsing: < 200ms
- Tokenization: < 1.5s (tiktoken-rs on full text)
- Pattern extraction: < 500ms
- Scoring: < 100ms
- **Total: < 3 seconds**

Rust is the right choice here. Python would be 10-20x slower on tokenization.

### 3.4 Loss Detector (`compaction/loss_detector.rs`)

#### 3.4.1 Detection Strategy

After compaction, we receive the `compact_summary` text. We need to determine which fingerprinted items survived.

**Approach: Token-overlap scoring (NOT semantic embedding)**

For each `FingerprintItem`, compute a token-level overlap score against the compact summary:

```
overlap_score = |tokens(item.content) ∩ tokens(compact_summary)| / |tokens(item.content)|
```

Classification:
- `overlap_score >= 0.7` → PRESERVED (most tokens survived)
- `0.3 <= overlap_score < 0.7` → PARAPHRASED (partially preserved, may have lost precision)
- `overlap_score < 0.3` → LOST (effectively dropped)

This is a deliberately simple approach. We're NOT doing embedding similarity or LLM-based checking — that would add latency and API cost. Token overlap is fast, deterministic, and good enough for our purpose.

#### 3.4.2 Loss Report Schema

```rust
// compaction/loss_detector.rs

#[derive(Debug, Serialize, Deserialize)]
pub struct LossReport {
    pub session_id: String,
    pub compaction_trigger: String,   // "auto" or "manual"
    pub pre_compaction_tokens: usize,
    pub post_compaction_tokens: usize,
    pub compression_ratio: f64,       // post / pre

    pub total_fingerprinted: usize,
    pub preserved_count: usize,
    pub paraphrased_count: usize,
    pub lost_count: usize,
    pub preservation_ratio: f64,      // preserved / total

    pub lost_items: Vec<LostItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LostItem {
    pub fingerprint_id: String,
    pub category: ItemCategory,
    pub content: String,
    pub tokens: usize,
    pub priority_score: f64,
    pub overlap_score: f64,          // How much survived (0.0–0.3)
}
```

### 3.5 Injection Builder (`compaction/injection_builder.rs`)

#### 3.5.1 Budget-Constrained Selection

The injection builder selects which lost items to re-inject, respecting a configurable token budget.

**Default budget: 4096 tokens** (approximately 2% of a 200K context window)

Algorithm:
1. Take `LossReport.lost_items`, already sorted by `priority_score` descending
2. Greedily select items while `sum(tokens) <= budget`
3. For each selected item, format it as a structured recovery line
4. If the highest-priority item alone exceeds the budget, truncate it to fit

Why 4096 default? It's large enough to recover 10-20 critical items (constraints, decisions, facts) without significantly impacting the fresh context window. Users can configure this up or down.

#### 3.5.2 Injection Format

The injection is output to stdout during SessionStart. Claude Code treats SessionStart stdout as injected context that Claude can see and act on.

```
[cctx] Context items recovered from pre-compaction analysis (4 items, 847 tokens):

CONSTRAINT: User requires PostgreSQL for ACID compliance. MongoDB explicitly ruled out.
(Originally stated in message 8, not found in compaction summary)

DECISION: Authentication uses JWT with RS256 signing, auth service on port 8443.
(Originally decided in messages 12-14, not found in compaction summary)

TECHNICAL: Config file location: /etc/myapp/production.yml
(Originally mentioned in message 19, not found in compaction summary)

DEBUG: The intermittent 503 errors were caused by HPA scaling delay during cold starts.
Fix: Set minReplicas=2 in the HPA spec.
(Originally debugged in messages 22-28, not found in compaction summary)
```

This format is designed to be:
- Scannable by both the LLM and the human
- Categorized so Claude knows what type of information each item is
- Attributed so Claude knows these aren't new instructions but recovered history

### 3.6 State Store (`state/`)

All cctx state lives in `.cctx/` within the project directory (similar to `.git/`):

```
.cctx/
├── config.json                       # cctx configuration
├── compaction-log.json               # History of all compaction events
├── fingerprints/
│   ├── <session_id_1>.json           # Pre-compaction fingerprint
│   └── <session_id_2>.json
├── loss-reports/
│   ├── <session_id_1>.json           # Post-compaction loss analysis
│   └── <session_id_2>.json
├── pending-injection/
│   └── <session_id>.json             # Awaiting next SessionStart
└── injection-history/
    ├── <session_id_1>.json           # Record of what was re-injected
    └── <session_id_2>.json
```

#### 3.6.1 Configuration (`config.json`)

```json
{
  "version": "0.1.0",
  "injection_budget_tokens": 4096,
  "fingerprint_extraction": {
    "min_priority_score": 0.1,
    "max_items": 200
  },
  "loss_detection": {
    "preserved_threshold": 0.7,
    "lost_threshold": 0.3
  },
  "recency_decay_rate": 0.05,
  "enabled": true
}
```

---

## 4. CLI Interface

### 4.1 New Commands

```bash
# ── Hook handlers (called by Claude Code hooks, receive JSON on stdin) ──

cctx hook pre-compact
    # Reads transcript, builds fingerprint, writes to .cctx/, outputs hint to stdout
    # Input: stdin JSON with session_id, transcript_path, trigger, custom_instructions

cctx hook post-compact
    # Loads fingerprint, diffs against compact_summary, builds injection payload
    # Input: stdin JSON with session_id, transcript_path, trigger, compact_summary

cctx hook session-start
    # Checks for pending injection, outputs recovered context to stdout
    # Input: stdin JSON with session_id, source


# ── Standalone commands (user can run directly) ──

cctx fingerprint <transcript_file>
    # Generate a fingerprint from a JSONL transcript file
    # Output: JSON fingerprint to stdout (or --output file)

cctx loss-report <session_id>
    # Display the loss report for a past compaction event
    # Reads from .cctx/loss-reports/

cctx compaction-history
    # Show all compaction events with preservation ratios
    # Reads from .cctx/compaction-log.json

cctx install-hooks [--project | --user | --local]
    # Write hook configuration to the appropriate settings.json
    # --project: .claude/settings.json (shared with team)
    # --user:    ~/.claude/settings.json (personal)
    # --local:   .claude/settings.local.json (personal, not committed)

cctx uninstall-hooks [--project | --user | --local]
    # Remove cctx hooks from settings.json

cctx config [key] [value]
    # View or set configuration values in .cctx/config.json
    # e.g., cctx config injection_budget_tokens 8192
```

### 4.2 Integration with Existing Commands

The existing `cctx analyze` and `cctx health` commands should also work with JSONL transcript files, not just OpenAI chat format JSON. The transcript parser module enables this:

```bash
# Existing command, now also accepts JSONL transcripts
cctx analyze /path/to/session-transcript.jsonl

# Existing command, now works on transcripts
cctx health /path/to/session-transcript.jsonl
```

---

## 5. Hook Input/Output Specifications

### 5.1 PreCompact Hook

**Input (stdin):**
```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/nikhil/.claude/projects/.../abc123.jsonl",
  "cwd": "/Users/nikhil/projects/myapp",
  "hook_event_name": "PreCompact",
  "trigger": "auto",
  "custom_instructions": ""
}
```

**Output (stdout, exit 0):**
```
[cctx] Fingerprinted 47 items (12 high-priority) across 89 messages.
Top items at risk: PostgreSQL decision (msg 8), auth architecture (msg 12-14), HPA fix (msg 25).
Full analysis saved to .cctx/fingerprints/abc123.json
```

**Notes:**
- This stdout text goes INTO the context being summarized. It is a best-effort hint.
- The real preservation mechanism is the fingerprint file on disk.
- If cctx fails or times out, exit with non-zero non-2 code (non-blocking error).
- NEVER exit 2 from PreCompact — that blocks compaction, which can cause the session to error out if context is already at the limit.

### 5.2 PostCompact Hook

**Input (stdin):**
```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/nikhil/.claude/projects/.../abc123.jsonl",
  "cwd": "/Users/nikhil/projects/myapp",
  "hook_event_name": "PostCompact",
  "trigger": "auto",
  "compact_summary": "Summary of the compacted conversation..."
}
```

**Output (stdout, exit 0):**
```
[cctx] Compaction analysis: 47 items fingerprinted → 31 preserved, 7 paraphrased, 9 lost.
Preservation ratio: 66%. Injection payload ready (9 items, 1847 tokens).
```

**Notes:**
- PostCompact has NO decision control. Output is informational only.
- The injection payload is written to `.cctx/pending-injection/abc123.json`, not output here.

### 5.3 SessionStart Hook (source: compact)

**Input (stdin):**
```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/nikhil/.claude/projects/.../abc123.jsonl",
  "cwd": "/Users/nikhil/projects/myapp",
  "hook_event_name": "SessionStart",
  "source": "compact",
  "model": "claude-sonnet-4-6"
}
```

**Output (stdout, exit 0):**
The structured recovery block (see Section 3.5.2). This stdout is added as context that Claude can see and act on.

**Notes:**
- The matcher is "compact" — this hook ONLY fires on sessions resuming after compaction, not on fresh startup or resume.
- If no pending injection exists, output nothing and exit 0.

---

## 6. Testing Strategy

### 6.1 Unit Tests

| Component | Test Cases |
|-----------|------------|
| `transcript/parser` | Valid JSONL, malformed lines (skip gracefully), empty file, single-entry file, entries with tool_use blocks, entries with nested content blocks |
| `transcript/normalizer` | User messages normalize correctly, tool pairs merge, system messages pin to position 0, ordering preserved |
| `fingerprint/extractor` | Constraint patterns detected, decision patterns detected, technical facts (ports, paths, URLs) extracted, progress markers found, edge case: no patterns found → empty fingerprint |
| `fingerprint/scorer` | Unique item (count=1) scores higher than repeated (count=5), recent item scores higher than old, dead-zone item scores higher than edge item, composite score multiplies correctly |
| `compaction/loss_detector` | Item fully preserved (overlap > 0.7), item paraphrased (0.3-0.7), item lost (< 0.3), empty summary → all items lost, identical summary → all preserved |
| `compaction/injection_builder` | Budget respected (never exceeds), items selected by priority, truncation when single item exceeds budget, empty lost list → no injection |
| `hooks/input` | All three hook input schemas parse correctly, missing optional fields handled, malformed JSON → clean error |
| `state/store` | Create .cctx/ if missing, write and read fingerprints, write and read loss reports, pending injection lifecycle (write → read → delete) |

### 6.2 Integration Tests

**Test 1: Full Compaction Lifecycle Simulation**
1. Create a synthetic 200-message JSONL transcript with known items
2. Run `cctx hook pre-compact` (simulated stdin)
3. Generate a synthetic compact_summary that deliberately drops 5 known items
4. Run `cctx hook post-compact` (simulated stdin with the summary)
5. Run `cctx hook session-start` (simulated stdin with source: compact)
6. Assert: stdout contains exactly the 5 lost items
7. Assert: items are in priority order
8. Assert: total injection tokens ≤ budget

**Test 2: Fingerprint Accuracy on Real-ish Transcript**
1. Create a transcript where a user:
   - States a budget constraint once in message 3
   - Makes an architecture decision in messages 8-10
   - Mentions a port number in message 14
   - Discusses the same error 4 times (messages 5, 12, 20, 30)
2. Run fingerprinting
3. Assert: budget constraint has occurrence_count=1, high uniqueness score
4. Assert: repeated error has occurrence_count=4, low uniqueness score
5. Assert: budget constraint has higher priority than repeated error

**Test 3: Edge Cases**
- Empty transcript → clean exit, no fingerprint
- Transcript with only system messages → no user items to fingerprint
- Transcript where compact_summary is empty string → all items marked LOST
- Session with no pending injection → SessionStart outputs nothing
- Very large transcript (simulated 200K tokens) → completes within 3s time budget

### 6.3 Benchmark Tests

```bash
# Generate synthetic transcripts of increasing size
cctx bench generate --messages 50 --output bench_50.jsonl
cctx bench generate --messages 200 --output bench_200.jsonl
cctx bench generate --messages 500 --output bench_500.jsonl
cctx bench generate --messages 1500 --output bench_1500.jsonl

# Benchmark fingerprinting
cctx bench fingerprint bench_50.jsonl bench_200.jsonl bench_500.jsonl bench_1500.jsonl

# Expected output:
#   50 messages  (  ~8K tokens): fingerprint in   120ms
#  200 messages  ( ~35K tokens): fingerprint in   450ms
#  500 messages  ( ~90K tokens): fingerprint in  1100ms
# 1500 messages  (~200K tokens): fingerprint in  2800ms
```

---

## 7. Configuration and Installation

### 7.1 Installation Flow

```bash
# From the cctx project directory
cargo install --path .

# Install hooks into current project
cd /path/to/my-project
cctx install-hooks --local

# Verify
cat .claude/settings.local.json
# Should show PreCompact, PostCompact, SessionStart hooks

# Initialize cctx state directory
cctx init
# Creates .cctx/ with default config.json
```

### 7.2 CLAUDE.md Integration

Add to the project's CLAUDE.md:

```markdown
## Compact Instructions

When compacting this conversation, pay special attention to:
- User-stated constraints and requirements (budgets, deadlines, tech choices)
- Architectural decisions and their rationale
- Specific technical facts (ports, paths, config values)
- Debugging conclusions and root causes

cctx is installed as a compaction guard. After compaction, recovered context
from cctx should be treated as authoritative — these are items that were
verified to be present before compaction and absent after.
```

### 7.3 Cross-Platform Support

**Claude Code:** Full support via hooks system. `cctx install-hooks` configures `.claude/settings.json`.

**Codex CLI:** Codex now has hooks with similar lifecycle events (UserPromptSubmit, SessionStart). The hook configuration format differs slightly. `cctx install-hooks --codex` generates the Codex-compatible configuration.

**API-level (Agent SDK):** For developers building custom agents with the Anthropic Agent SDK, cctx provides a library function that generates custom compaction instructions:

```rust
// In a custom agent built with the Anthropic SDK
let transcript = read_transcript("session.jsonl");
let fingerprint = cctx::fingerprint::analyze(&transcript);
let instructions = cctx::compaction::generate_instructions(&fingerprint);

// Pass to the API's context_management.edits[].instructions
// This replaces the default compaction prompt with one that
// explicitly preserves high-priority fingerprinted items
```

This is the most powerful integration point — it lets cctx directly influence WHAT the compaction preserves, not just recover what was lost after the fact.

---

## 8. Success Metrics

### 8.1 Quantitative

| Metric | Baseline (no cctx) | Target (with cctx) | How Measured |
|--------|--------------------|--------------------|--------------|
| Critical item preservation rate | ~60-70% (estimated) | >90% | Fingerprint vs summary diff |
| Post-compaction health score | Variable | ≥85 | cctx health on post-compaction + injection context |
| Unique constraints preserved | Often lost | >95% | Count of occurrence_count=1 items in CONSTRAINT category |
| Re-injection token overhead | 0 (nothing recovered) | ≤4096 (configurable) | Token count of injection payload |
| PreCompact latency | N/A | <3s for 200K tokens | Benchmark suite |
| PostCompact latency | N/A | <2s | Benchmark suite |
| SessionStart latency | N/A | <500ms | Benchmark suite |

### 8.2 Qualitative

- User should NEVER need to re-state a constraint after compaction
- User should NEVER see the agent contradict a prior architectural decision after compaction
- The injection block should be concise enough that a human can scan it in <10 seconds
- `cctx compaction-history` should give a clear picture of how well context has been preserved across a project's lifetime

---

## 9. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| PreCompact hook times out on large transcripts | Fingerprint not saved, no recovery possible | Medium | Benchmark extensively. Use async hook if needed. Implement early-exit with partial fingerprint for very large transcripts. |
| JSONL transcript schema changes between Claude Code versions | Parser breaks | Medium | Use loose/tolerant deserialization (serde `#[serde(default)]`). Log unknown fields. Test against multiple Claude Code versions. |
| Token overlap detection misclassifies preserved items as lost | Over-injection wastes context | Low | Tune thresholds empirically. Start conservative (0.3/0.7). Add `cctx config` for user tuning. |
| Pattern extraction misses important items | Critical info not fingerprinted, not recovered | Medium | This is acceptable — partial recovery > no recovery. Improve patterns iteratively based on real-world testing. Consider optional LLM-assisted extraction in future version (calls a fast model like Haiku for semantic extraction). |
| User installs hooks but forgets to run `cctx init` | State directory missing, hooks fail | Low | Hooks auto-initialize .cctx/ if missing. |
| Codex hook format diverges from Claude Code | Cross-platform breaks | Medium | Abstract hook configuration behind `cctx install-hooks --platform`. Test against both platforms in CI. |

---

## 10. Future Extensions (Post-MVP)

These are NOT in scope for the initial implementation but represent the product roadmap:

1. **LLM-assisted fingerprinting** — Use Haiku (fast, cheap) to extract semantic items rather than relying on pattern matching. Improves recall at the cost of ~1s latency and ~$0.001/call.

2. **Continuous health monitoring** — A `UserPromptSubmit` hook that runs lightweight dead-zone analysis on every prompt and injects attention hints: "Items at positions 15-25 may be underweighted. Key items in this range: [list]." This is a lightweight version of Idea 1 that IS feasible with current hook capabilities.

3. **MCP server mode** — Package cctx as an MCP server with tools like `cctx_analyze`, `cctx_fingerprint`, `cctx_recover`. Claude Code or Codex can then call these tools explicitly during a session, not just during lifecycle events.

4. **Team-wide compaction analytics** — Aggregate compaction history across a team. Dashboard showing: average preservation rate, most commonly lost item categories, sessions with worst compaction quality. Helps teams tune their CLAUDE.md and compaction strategies.

5. **Custom compaction prompt generation** — For Agent SDK users, generate a context-aware compaction prompt that explicitly lists high-priority items to preserve. This is the API-level integration described in Section 7.3.

---

## 11. Day-by-Day Implementation Plan (Post Day 20)

> **Prerequisites:** cctx core (analyze, optimize, diff, health, all strategies) is complete and stable. `cargo build` succeeds. `cargo test` passes. README exists.

### Day 21: Transcript Parser + State Store Foundation

**Goal:** cctx can read Claude Code JSONL transcripts and has a state directory.

**Tasks:**
1. Create `src/transcript/` module: `schema.rs`, `parser.rs`, `normalizer.rs`
2. Implement JSONL line-by-line parser with tolerant deserialization
3. Implement normalizer: transcript entries → `Vec<Chunk>` (existing cctx type)
4. Create `src/state/store.rs`: `.cctx/` directory creation, config read/write
5. Add `cctx init` command
6. Make existing `cctx analyze` accept JSONL files (detect by extension or content sniffing)
7. Write unit tests for parser (valid, malformed, empty, tool-use entries)
8. Write unit tests for normalizer
9. Write unit tests for state store

**Test fixture:** Create `tests/fixtures/sample_transcript.jsonl` — a realistic 30-message JSONL transcript with user messages, assistant responses, tool calls, and tool results.

**Commit:** `feat: JSONL transcript parser and state store foundation`

### Day 22: Fingerprint Engine

**Goal:** cctx can analyze a transcript and extract prioritized information items.

**Tasks:**
1. Create `src/fingerprint/` module: `extractor.rs`, `scorer.rs`, `index.rs`
2. Implement pattern-based extraction for all 5 categories (Constraint, Decision, TechnicalFact, DebugInsight, ProgressMarker)
3. Implement deduplication (same fact mentioned multiple times → single item with count)
4. Implement scoring: uniqueness (1/count), recency (exponential decay), position_risk (dead zone score from existing core)
5. Implement `Fingerprint` serialization to JSON
6. Add `cctx fingerprint <file>` CLI command
7. Write unit tests for each extraction category
8. Write unit tests for scoring (unique > repeated, recent > old, dead-zone > edge)
9. Write integration test: fingerprint a transcript with known items, verify extraction

**Commit:** `feat: fingerprint engine with information entropy scoring`

### Day 23: Loss Detection + Injection Builder

**Goal:** cctx can diff a fingerprint against a compaction summary and build re-injection payload.

**Tasks:**
1. Create `src/compaction/` module: `loss_detector.rs`, `injection_builder.rs`, `budget.rs`
2. Implement token-overlap loss detection (PRESERVED / PARAPHRASED / LOST classification)
3. Implement `LossReport` generation and serialization
4. Implement budget-constrained injection builder (greedy selection by priority, token budget limit)
5. Implement injection text formatter (the structured recovery block)
6. Add `cctx loss-report <session_id>` CLI command
7. Write unit tests for loss detection (all three classifications)
8. Write unit tests for injection builder (budget respected, priority ordering, truncation)
9. Write integration test: synthetic fingerprint + synthetic summary → verify correct lost items identified

**Commit:** `feat: loss detection and budget-constrained injection builder`

### Day 24: Hook Handlers + CLI Integration

**Goal:** cctx can be called as a Claude Code hook with proper stdin/stdout handling.

**Tasks:**
1. Create `src/hooks/` module: `input.rs`, `pre_compact.rs`, `post_compact.rs`, `session_start.rs`
2. Implement hook input JSON parsing for all three event types
3. Implement `cctx hook pre-compact`: parse stdin → read transcript → fingerprint → save → stdout hint
4. Implement `cctx hook post-compact`: parse stdin → load fingerprint → diff summary → save loss report → save injection payload → stdout summary
5. Implement `cctx hook session-start`: parse stdin → check pending injection → stdout recovery block (or nothing)
6. Add all three hook subcommands to CLI
7. Implement `cctx install-hooks` and `cctx uninstall-hooks`
8. Write unit tests for hook input parsing
9. Write integration test: simulate full lifecycle (pre-compact → post-compact → session-start) with piped stdin

**Commit:** `feat: hook handlers for PreCompact, PostCompact, SessionStart`

### Day 25: End-to-End Integration Testing + Benchmarks

**Goal:** Full lifecycle works end-to-end. Performance meets targets.

**Tasks:**
1. Write the full lifecycle integration test (Section 6.2, Test 1)
2. Write the fingerprint accuracy test (Section 6.2, Test 2)
3. Write all edge case tests (Section 6.2, Test 3)
4. Implement `cctx bench generate` for synthetic transcript generation
5. Implement `cctx bench fingerprint` for performance benchmarking
6. Run benchmarks, record baseline numbers
7. Optimize hot paths if any benchmark exceeds target (tokenization is likely bottleneck)
8. Fix any bugs found during integration testing
9. Add `cctx compaction-history` command

**Commit:** `test: end-to-end integration tests and benchmark suite`

### Day 26: Polish, README, Cross-Platform

**Goal:** Project is presentable. Installation is one-command. Docs are complete.

**Tasks:**
1. Update README.md with Compaction Guard section:
   - Problem statement (with stats)
   - Installation (`cargo install`, `cctx install-hooks`)
   - How it works (the three-phase diagram)
   - Configuration options
   - Example output
2. Add `--codex` flag to `cctx install-hooks` for Codex configuration format
3. Add `.cctx/` to typical .gitignore templates (state is local, not committed)
4. Test installation flow from scratch (fresh project, install cctx, install hooks, verify)
5. Record a demo: long Claude Code session → compaction → show cctx recovery in action
6. Write CHANGELOG entry
7. Final `cargo clippy`, `cargo fmt`, `cargo test`
8. Tag release

**Commit:** `docs: README update, cross-platform install, release polish`

---

## 12. Appendix: Research Sources

This architecture is informed by:

1. **Claude Code Hooks Reference** — Official Anthropic documentation on hook lifecycle events, input/output schemas, and decision control
2. **Claude Code Compaction Deep Dive (Decode Claude)** — Reverse-engineered compaction internals including three-layer design and rehydration sequence
3. **Anthropic Compaction API (Beta)** — Server-side compaction with custom instructions parameter
4. **Context Engineering Cookbook (Anthropic)** — Official patterns for compaction, tool clearing, and memory
5. **"Lost in the Middle" (Stanford, 2023)** — Research paper demonstrating attention degradation in the middle of long contexts
6. **Claude Code System Prompt Analysis (Piebald-AI)** — Extracted system prompt showing how CLAUDE.md, tools, and conversation are assembled
7. **Codex CLI Changelog** — Hook system evolution including UserPromptSubmit and plugin architecture
8. **Community Implementations** — Various PreCompact hook implementations using SQLite, JSONL archiving, and context recovery patterns from GitHub issues and blog posts
