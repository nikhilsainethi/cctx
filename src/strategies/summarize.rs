//! Hierarchical summarization — compress older context using an LLM.
//!
//! Research basis: Factory.ai anchored summarization, ACON framework (2025),
//! adaptive context compression (2026).
//!
//! Algorithm:
//!   1. Divide turns into zones: RECENT (verbatim), AGING (bullet summaries),
//!      ARCHIVED (merged into one rolling summary)
//!   2. LLM summarizes aging turns into bullet points
//!   3. LLM merges archived turns into a single summary paragraph
//!   4. Reassemble: `[system] + [archived summary] + [aging bullets] + [recent]`
//!
//! Requires an LLM provider (Ollama or OpenAI). Without one, falls back to
//! keeping recent turns and truncating the oldest.

use anyhow::Result;

use crate::core::context::{AttentionZone, Chunk, Context};
use crate::core::tokenizer::Tokenizer;
use crate::llm::LlmProvider;

/// Apply hierarchical summarization to `context`.
///
/// Three tiers based on `recent_n` (typical default 6):
///
/// - **Recent** — last `recent_n` turns kept verbatim.
/// - **Aging** — turns `recent_n+1..=2*recent_n` compressed to bullet lists
///   via the LLM provider.
/// - **Archived** — everything older, merged into a single summary paragraph.
///
/// When `provider` is `None`, falls back to a no-LLM path: keep recent
/// verbatim and drop archived turns entirely. Still effective; just coarser.
///
/// # Errors
///
/// Returns `Err` if the LLM provider fails on an aging- or archived-tier
/// summary call.
pub fn apply(
    context: &Context,
    provider: Option<&dyn LlmProvider>,
    tokenizer: &Tokenizer,
    recent_n: usize,
) -> Result<Vec<Chunk>> {
    let chunks = &context.chunks;
    let n = chunks.len();

    if n == 0 {
        return Ok(vec![]);
    }

    // ── Separate system messages from conversation turns ──────────────────
    // System messages are always preserved at the front, never summarized.
    let mut system_chunks: Vec<Chunk> = Vec::new();
    let mut turns: Vec<&Chunk> = Vec::new();

    for chunk in chunks {
        if chunk.role == "system" {
            system_chunks.push(chunk.clone());
        } else {
            turns.push(chunk);
        }
    }

    let turn_count = turns.len();
    if turn_count == 0 {
        return Ok(system_chunks);
    }

    // ── Divide into zones ────────────────────────────────────────────────
    // RECENT: last `recent_n` turns (kept verbatim)
    // AGING: turns from recent_n+1 to 2*recent_n (summarized to bullets)
    // ARCHIVED: everything older (merged into one summary)
    let recent_start = turn_count.saturating_sub(recent_n);
    let aging_start = turn_count.saturating_sub(recent_n * 2);

    let archived = &turns[..aging_start];
    let aging = &turns[aging_start..recent_start];
    let recent = &turns[recent_start..];

    // ── Process each zone ────────────────────────────────────────────────
    let mut result: Vec<Chunk> = Vec::new();

    // 1. System messages first.
    result.extend(system_chunks);

    // 2. Archived → single summary (or fallback to drop).
    if !archived.is_empty() {
        match provider {
            Some(llm) => {
                let summary = summarize_archived(archived, llm)?;
                result.push(Chunk {
                    index: 0,
                    role: "system".to_string(),
                    content: summary.clone(),
                    token_count: tokenizer.count(&summary),
                    relevance_score: 0.8,
                    attention_zone: AttentionZone::Strong,
                });
            }
            None => {
                // No LLM → drop archived turns entirely (simple truncation).
                // This is the fallback behavior.
            }
        }
    }

    // 3. Aging → bullet-point summaries.
    for turn in aging {
        match provider {
            Some(llm) => match summarize_to_bullets(turn, llm) {
                Ok(bullets) => {
                    result.push(Chunk {
                        content: bullets.clone(),
                        token_count: tokenizer.count(&bullets),
                        ..(*turn).clone()
                    });
                }
                Err(_) => {
                    // LLM failed for this turn → keep original.
                    result.push((*turn).clone());
                }
            },
            None => {
                // No LLM → keep aging turns as-is.
                result.push((*turn).clone());
            }
        }
    }

    // 4. Recent → verbatim.
    for turn in recent {
        result.push((*turn).clone());
    }

    // ── Safety: if result is somehow larger, return original ─────────────
    let result_tokens: usize = result.iter().map(|c| c.token_count).sum();
    if result_tokens >= context.total_tokens {
        return Ok(context.chunks.clone());
    }

    Ok(result)
}

// ── LLM prompts ───────────────────────────────────────────────────────────────

fn summarize_archived(turns: &[&Chunk], llm: &dyn LlmProvider) -> Result<String> {
    let mut history = String::new();
    for turn in turns {
        history.push_str(&format!("[{}]: {}\n\n", turn.role, turn.content));
    }

    let prompt = format!(
        "Create a concise summary of this conversation history. \
         Preserve: key decisions, specific technical details, names, numbers, \
         code references, unresolved questions. \
         Remove: pleasantries, repetition, step-by-step debugging that's already resolved.\n\n\
         Conversation history:\n{}",
        history
    );

    let summary = llm.complete(
        "You are a precise summarizer. Output only the summary, no preamble.",
        &prompt,
    )?;

    Ok(format!("[Conversation summary]\n{}", summary))
}

fn summarize_to_bullets(turn: &Chunk, llm: &dyn LlmProvider) -> Result<String> {
    let prompt = format!(
        "Summarize the following conversation turn into 1-2 bullet points. \
         Preserve: specific names, numbers, decisions, code references, action items. \
         Remove: pleasantries, filler, repetition.\n\n\
         Turn ({role}): {content}",
        role = turn.role,
        content = turn.content
    );

    let bullets = llm.complete(
        "You are a precise summarizer. Output only bullet points, no preamble.",
        &prompt,
    )?;

    // Ensure bullets start with "- " formatting.
    let formatted: String = bullets
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return String::new();
            }
            if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("• ")
            {
                trimmed.to_string()
            } else {
                format!("- {}", trimmed)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(formatted)
}
