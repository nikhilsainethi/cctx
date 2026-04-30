//! End-to-end transcript tests against the realistic 30-entry sample fixture.
//!
//! The unit tests inside `src/transcript/parser.rs` and
//! `src/transcript/normalizer.rs` cover narrow behaviors with hand-rolled
//! inputs. These tests prove the same code works on a realistic Claude
//! Code transcript and that the specific facts seeded into the fixture
//! survive parsing + normalization at their expected positions.

use std::path::Path;

use cctx::transcript::{normalize, parse_transcript};

const FIXTURE: &str = "tests/fixtures/sample_transcript.jsonl";

#[test]
fn fixture_parses_to_thirty_entries() {
    let entries = parse_transcript(Path::new(FIXTURE)).unwrap();
    assert_eq!(
        entries.len(),
        30,
        "fixture should have exactly 30 entries — adjust the test fixture or update this assertion"
    );
}

#[test]
fn fixture_target_facts_appear_at_expected_positions() {
    let entries = parse_transcript(Path::new(FIXTURE)).unwrap();

    // Helper: pull the plain text out of a transcript entry, joining
    // every text/tool block content into one string for substring search.
    let entry_text = |i: usize| -> String {
        use cctx::transcript::{ContentBlock, TranscriptContent};
        let msg = entries[i].message.as_ref().expect("missing message");
        match &msg.content {
            TranscriptContent::Text(s) => s.clone(),
            TranscriptContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => text.clone(),
                    ContentBlock::ToolUse { name, input, .. } => {
                        format!("[tool_use {} {}]", name, input)
                    }
                    ContentBlock::ToolResult { content, .. } => content.clone().unwrap_or_default(),
                    ContentBlock::Unknown => String::new(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    };

    // Indices are 0-based; the spec lists "entry 5/10/15/20/25" 1-based.
    assert!(
        entry_text(4).contains("budget is $50K"),
        "entry 5 should contain the budget constraint"
    );
    assert!(
        entry_text(9).contains("PostgreSQL"),
        "entry 10 should contain the PostgreSQL adoption"
    );
    assert!(
        entry_text(14).contains("port 8443"),
        "entry 15 should mention auth service on port 8443"
    );
    assert!(
        entry_text(19).contains("/etc/myapp/prod.yml"),
        "entry 20 should mention the prod config path"
    );
    assert!(
        entry_text(24).contains("HPA"),
        "entry 25 should describe the HPA-related bug root cause"
    );
}

#[test]
fn fixture_normalizes_into_a_context() {
    let entries = parse_transcript(Path::new(FIXTURE)).unwrap();
    let ctx = normalize(entries).unwrap();

    // Every entry produces at least one chunk; tool_use+tool_result pairs
    // collapse to a single chunk so the chunk count is < 30.
    assert!(ctx.chunk_count() > 0);
    assert!(ctx.chunk_count() <= 30);
    assert!(ctx.total_tokens > 0);

    // System messages must occupy the first slots after pinning.
    assert_eq!(ctx.chunks[0].role, "system");
    assert_eq!(ctx.chunks[1].role, "system");

    // tool_interaction chunks should appear for each tool sequence (4).
    let tool_count = ctx
        .chunks
        .iter()
        .filter(|c| c.role == "tool_interaction")
        .count();
    assert_eq!(
        tool_count, 4,
        "fixture has 4 tool_use/tool_result pairs; normalizer should produce 4 tool_interaction chunks"
    );
}

#[test]
fn fixture_big_tool_output_is_truncated_in_normalized_chunk() {
    let entries = parse_transcript(Path::new(FIXTURE)).unwrap();
    let ctx = normalize(entries).unwrap();

    // The grep TODO output is the largest tool_result and should be the
    // only one that triggers truncation. Find it by searching for the
    // truncation marker the normalizer adds.
    let truncated_count = ctx
        .chunks
        .iter()
        .filter(|c| c.role == "tool_interaction" && c.content.contains("[output truncated"))
        .count();
    assert_eq!(
        truncated_count, 1,
        "exactly one tool output (the grep TODO scan) should be over the 500-token cap"
    );
}

#[test]
fn fixture_targets_survive_normalization() {
    // The five seeded facts must still be findable in the normalized
    // context — they're what fingerprint extraction (Day 22) will look for.
    let entries = parse_transcript(Path::new(FIXTURE)).unwrap();
    let ctx = normalize(entries).unwrap();

    let body: String = ctx
        .chunks
        .iter()
        .map(|c| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    for fact in [
        "budget is $50K",
        "PostgreSQL",
        "port 8443",
        "/etc/myapp/prod.yml",
        "HPA",
    ] {
        assert!(
            body.contains(fact),
            "seeded fact `{}` must survive normalization (was the truncation aggressive?)",
            fact
        );
    }
}
