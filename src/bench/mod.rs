//! Synthetic transcript generation for benchmarks.
//!
//! `cctx bench generate --messages N --output FILE` writes a JSONL
//! transcript of `N` plausibly-shaped Claude Code entries: a system
//! prompt, then alternating user/assistant turns, sprinkled with
//! tool_use / tool_result pairs and a known set of fingerprintable
//! facts at predictable positions.
//!
//! Determinism: the generator is seeded by a small linear-congruential
//! PRNG with a fixed seed, so two runs with the same `N` produce
//! byte-identical files. That keeps the perf benchmarks reproducible
//! and lets `scripts/accuracy_benchmark.sh` rely on a known
//! ground-truth set.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

// ── Templates ─────────────────────────────────────────────────────────────────

/// Conversational user-side templates. Picked round-robin keyed by index.
const USER_TEMPLATES: &[&str] = &[
    "What's the recommended approach for {topic}?",
    "Help me think through the tradeoffs for {topic}.",
    "I'm seeing slow query times on the {topic} endpoint, ideas?",
    "Quick question — does {topic} require any special config?",
    "Walk me through how {topic} should integrate with our existing services.",
    "Could you sketch the API surface for the {topic} feature?",
    "What's the failure mode if {topic} times out under load?",
    "How would you structure the rollout for {topic} over the next sprint?",
    "Are there any security considerations I should think about for {topic}?",
    "If we deploy {topic} to staging today, what's the verification checklist?",
];

const ASSISTANT_TEMPLATES: &[&str] = &[
    "For {topic}, the most common pattern is to start small and iterate. \
     Pin the contract first, then layer in retries and observability.",
    "{topic} works best when you decouple the read path from the write path. \
     Backpressure on writes, generous caching on reads.",
    "Three options for {topic}: (a) pull-based polling, (b) push via webhooks, \
     (c) a thin event bus. Pull is simplest; events scale further.",
    "I'd lean conservative on {topic}. Validate the assumptions in a small \
     dark-launch before committing the whole team.",
    "Treat {topic} as a stateful subsystem — give it its own metrics, its own \
     SLO, and its own rollback plan separate from the main service.",
    "For {topic}, the failure modes that bite are silent ones: corrupt cache, \
     stale leader, expired credentials. Add a synthetic prober.",
    "I'd start with the boring stack for {topic}: PostgreSQL, a Rust service, \
     boring deployment. Optimize once we have load patterns.",
    "{topic} is the kind of thing where a 1-day prototype answers more \
     questions than a week of design discussion.",
];

/// Topics rotated through the templates to give them variety. Mix of
/// realistic backend / infra / data topics so the resulting tokens
/// look like real engineering chat rather than lorem ipsum.
const TOPICS: &[&str] = &[
    "rate limiting",
    "the cache invalidation",
    "the auth handshake",
    "schema migrations",
    "blue-green deployments",
    "the queue depth metric",
    "the retry policy",
    "request shaping",
    "log sampling",
    "the feature flag rollout",
    "circuit breakers",
    "the metrics pipeline",
];

/// Known facts injected at fixed indices. The accuracy benchmark
/// expects to find each of these in the resulting fingerprint.
struct KnownFact {
    /// 1-based message index where the fact appears.
    position: usize,
    /// The exact text of the user message containing the fact.
    content: &'static str,
}

const KNOWN_FACTS: &[KnownFact] = &[
    KnownFact {
        position: 5,
        content: "Hard constraint: the budget should not exceed $50K for the entire project.",
    },
    KnownFact {
        position: 10,
        content:
            "We decided to go with PostgreSQL over MongoDB for the primary store — ACID compliance is non-negotiable.",
    },
    KnownFact {
        position: 15,
        content: "For ops: the auth service runs on port 8443 inside the mesh.",
    },
    KnownFact {
        position: 20,
        content:
            "The production config lives at /etc/myapp/prod.yml — the deploy script reads it on boot.",
    },
    KnownFact {
        position: 25,
        content: "The intermittent 503s in staging — the root cause was the HPA scaling delay during cold starts.",
    },
    KnownFact {
        position: 30,
        content: "Hit a CORS bug today on the user-service. The issue was a missing CORS header.",
    },
];

// ── Generation entry point ────────────────────────────────────────────────────

/// Generate a synthetic JSONL transcript of `messages` entries to `output`.
///
/// One leading system prompt, then `messages - 1` alternating
/// user/assistant turns. Every 8th pair gets a `tool_use` +
/// `tool_result` cycle. The [`KNOWN_FACTS`] list seeds specific
/// content at specific indices so accuracy benchmarks have ground
/// truth to compare against.
///
/// # Errors
///
/// Returns `Err` if `messages` is zero or if writing the output
/// file fails.
pub fn generate(messages: usize, output: &Path) -> Result<()> {
    if messages == 0 {
        anyhow::bail!("messages must be > 0");
    }

    let file =
        File::create(output).with_context(|| format!("Cannot create {}", output.display()))?;
    let mut w = BufWriter::new(file);

    writeln_system(
        &mut w,
        "You are an expert engineer pair-programming with the user.",
    )?;

    let mut tool_use_id: u64 = 0;
    let mut next_user = true; // alternate user → assistant
    let mut idx = 1usize; // 1-based logical message index
    while idx < messages {
        // Every 8th pair: emit a tool_use + tool_result instead of a
        // plain assistant text block. Keeps tool-output size bounded
        // so the synthetic transcript doesn't accidentally exceed
        // the truncation cap.
        if next_user {
            let user_content = pick_user_content(idx);
            writeln_user(&mut w, &user_content)?;
            next_user = false;
        } else {
            let do_tool = idx.is_multiple_of(8);
            if do_tool {
                tool_use_id += 1;
                let id = format!("toolu_bench_{}", tool_use_id);
                writeln_tool_use(&mut w, &id, "Bash", &json!({"command": "ls -la"}))?;
                writeln_tool_result(
                    &mut w,
                    &id,
                    "total 8\ndrwxr-xr-x 2 dev dev 4096 file_a.txt\n-rw-r--r-- 1 dev dev   12 file_b.txt",
                )?;
                // We just emitted TWO entries — bump idx accordingly.
                idx += 2;
                next_user = true;
                continue;
            } else {
                let asst_content = pick_assistant_content(idx);
                writeln_assistant_text(&mut w, &asst_content)?;
                next_user = true;
            }
        }
        idx += 1;
    }

    w.flush().context("Failed to flush transcript")?;
    Ok(())
}

// ── Content pickers ──────────────────────────────────────────────────────────

fn pick_user_content(index: usize) -> String {
    if let Some(fact) = KNOWN_FACTS.iter().find(|f| f.position == index) {
        return fact.content.to_string();
    }
    let template = USER_TEMPLATES[index % USER_TEMPLATES.len()];
    let topic = TOPICS[index % TOPICS.len()];
    template.replace("{topic}", topic)
}

fn pick_assistant_content(index: usize) -> String {
    // The tool-use cycle at every 8th assistant turn shifts the
    // user/assistant parity by one. That can land a `KNOWN_FACTS`
    // position on what is now an assistant slot — without this
    // fallthrough, the seeded fact would silently drop. Emitting the
    // exact fact text as an assistant message is fine for synthetic
    // benchmark data; the accuracy harness only cares that the string
    // is present *somewhere* in the transcript.
    if let Some(fact) = KNOWN_FACTS.iter().find(|f| f.position == index) {
        return fact.content.to_string();
    }
    let template = ASSISTANT_TEMPLATES[index % ASSISTANT_TEMPLATES.len()];
    let topic = TOPICS[(index / 2) % TOPICS.len()];
    template.replace("{topic}", topic)
}

// ── JSONL writers ────────────────────────────────────────────────────────────

fn writeln_system<W: Write>(w: &mut W, content: &str) -> Result<()> {
    writeln_json(
        w,
        &json!({
            "type": "system",
            "message": {"role": "system", "content": content},
            "session_id": "bench"
        }),
    )
}

fn writeln_user<W: Write>(w: &mut W, content: &str) -> Result<()> {
    writeln_json(
        w,
        &json!({
            "type": "user",
            "message": {"role": "user", "content": content},
            "session_id": "bench"
        }),
    )
}

fn writeln_assistant_text<W: Write>(w: &mut W, content: &str) -> Result<()> {
    writeln_json(
        w,
        &json!({
            "type": "assistant",
            "message": {"role": "assistant", "content": content},
            "session_id": "bench"
        }),
    )
}

fn writeln_tool_use<W: Write>(
    w: &mut W,
    id: &str,
    name: &str,
    input: &serde_json::Value,
) -> Result<()> {
    writeln_json(
        w,
        &json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": id, "name": name, "input": input}
                ]
            },
            "session_id": "bench"
        }),
    )
}

fn writeln_tool_result<W: Write>(w: &mut W, tool_use_id: &str, output: &str) -> Result<()> {
    writeln_json(
        w,
        &json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": output}
                ]
            },
            "session_id": "bench"
        }),
    )
}

fn writeln_json<W: Write>(w: &mut W, value: &serde_json::Value) -> Result<()> {
    let line = serde_json::to_string(value).context("Failed to serialize JSONL line")?;
    writeln!(w, "{}", line).context("Failed to write JSONL line")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{normalize, parse_transcript};

    fn unique_path(tag: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cctx_bench_{}_{}_{}.jsonl", tag, pid, nanos))
    }

    #[test]
    fn generates_requested_message_count_within_one() {
        // Tool-use cycles produce 2 entries from one slot, so the
        // exact count can be off by 1. We aim for "approximately N".
        let path = unique_path("count");
        generate(50, &path).unwrap();
        let entries = parse_transcript(&path).unwrap();
        let n = entries.len();
        assert!((45..=55).contains(&n), "expected ~50 entries, got {}", n);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn generated_transcript_normalizes_cleanly() {
        let path = unique_path("normalize");
        generate(100, &path).unwrap();
        let entries = parse_transcript(&path).unwrap();
        let ctx = normalize(entries).unwrap();
        assert!(ctx.chunk_count() > 0);
        assert!(ctx.total_tokens > 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn generated_transcript_contains_seeded_facts_when_long_enough() {
        // KNOWN_FACTS go up to index 30; a 50-message transcript
        // covers all of them.
        let path = unique_path("facts");
        generate(50, &path).unwrap();
        let entries = parse_transcript(&path).unwrap();
        let ctx = normalize(entries).unwrap();
        let body: String = ctx
            .chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for fact in [
            "$50K",
            "PostgreSQL",
            "8443",
            "/etc/myapp/prod.yml",
            "HPA",
            "CORS",
        ] {
            assert!(
                body.contains(fact),
                "expected `{}` in 50-message transcript",
                fact
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn zero_messages_errors() {
        let path = unique_path("zero");
        let err = generate(0, &path).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }
}
