// serde traits: Serialize = Rust → JSON, Deserialize = JSON → Rust.
// `use` brings them into scope so #[derive] can find them.
use serde::{Deserialize, Serialize};

// ── Input format ─────────────────────────────────────────────────────────────

/// A single message in an LLM conversation — the wire format for input / output.
///
/// Matches the OpenAI chat completions schema: every message has a `role`
/// (`"system"`, `"user"`, `"assistant"`, `"tool"`, …) and `content`. An
/// optional `relevance_score` drives bookend placement and pruning decisions
/// when it's present on input; it's omitted from output JSON when `None`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    /// The speaker of the message (`"system"`, `"user"`, `"assistant"`, …).
    pub role: String,
    /// The message body.
    pub content: String,
    /// Optional pre-computed relevance score in `[0.0, 1.0]`. When set, it's
    /// used directly by strategies like `bookend`; otherwise a heuristic
    /// assigns a score at context-build time. Omitted from output JSON when
    /// `None` to keep round-tripped data clean.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevance_score: Option<f64>,
}

// ── Attention model ───────────────────────────────────────────────────────────

/// Where a chunk sits on the U-shaped attention curve.
///
/// Models the finding from Liu et al., *Lost in the Middle* (TACL 2024):
/// LLMs attend strongly to the beginning and end of the context window, and
/// lose ~30% recall for information buried in the middle. Strategies like
/// `bookend` use this classification to decide placement.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionZone {
    /// Beginning (~0–25%) or end (~75–100%) of context — high LLM attention.
    Strong,
    /// Middle (~25–75%) of context — the attention dead zone where recall drops.
    DeadZone,
}

// impl block adds methods to a type — like a class's methods in other languages.
impl std::fmt::Display for AttentionZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // match is exhaustive — the compiler forces you to handle every variant.
        match self {
            AttentionZone::Strong => write!(f, "STRONG  "),
            AttentionZone::DeadZone => write!(f, "DEAD ZONE"),
        }
    }
}

// ── Chunk ─────────────────────────────────────────────────────────────────────

/// A single unit of context with pre-computed metadata.
///
/// Built from a [`Message`] by the format parser + tokenizer. Strategies
/// operate on `Chunk`s (not `Message`s) because they need token counts,
/// relevance scores, and attention-zone placement without recomputing on
/// every pass.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Original position in the conversation (0-indexed). Preserved across
    /// reorderings so strategies can refer back to the source ordering.
    pub index: usize,
    /// Speaker role — same values as [`Message::role`].
    pub role: String,
    /// Message body.
    pub content: String,
    /// Pre-computed token count under the tokenizer used to build the context.
    pub token_count: usize,
    /// Relevance in `[0.0, 1.0]`. 0.0 = filler, 1.0 = critical. Used by
    /// bookend, prune, and budget enforcement for ranking decisions.
    pub relevance_score: f64,
    /// Which part of the attention curve this chunk currently sits in.
    /// Recomputed by [`crate::analyzer::health::assign_attention_zones`] after
    /// every pipeline run.
    pub attention_zone: AttentionZone,
}

impl Chunk {
    /// Convert this chunk back to a plain [`Message`] suitable for JSON output.
    ///
    /// Drops the computed metadata (token count, attention zone, relevance
    /// score) — those are implementation details of the optimization pass and
    /// shouldn't leak into the wire format. The `relevance_score` is set to
    /// `None` so it's omitted from JSON.
    ///
    /// # Examples
    ///
    /// ```
    /// use cctx::core::context::{Chunk, AttentionZone};
    ///
    /// let chunk = Chunk {
    ///     index: 0,
    ///     role: "user".into(),
    ///     content: "Hi".into(),
    ///     token_count: 1,
    ///     relevance_score: 0.5,
    ///     attention_zone: AttentionZone::Strong,
    /// };
    /// let msg = chunk.to_message();
    /// assert_eq!(msg.role, "user");
    /// assert!(msg.relevance_score.is_none());
    /// ```
    pub fn to_message(&self) -> Message {
        Message {
            role: self.role.clone(),
            content: self.content.clone(),
            relevance_score: None,
        }
    }
}

// ── Context ───────────────────────────────────────────────────────────────────

/// An ordered sequence of [`Chunk`]s with a cached total token count.
///
/// This is the primary value that flows through the strategy pipeline.
/// Strategies take ownership of a `Context`, transform it, and return a new
/// one — the `Clone` impl exists mostly for benchmarks and tests.
#[derive(Clone)]
pub struct Context {
    /// Chunks in their current order. Reordered by `bookend`, filtered by
    /// `dedup` / `prune`, mutated in place by `structural` / `summarize`.
    pub chunks: Vec<Chunk>,
    /// Sum of `chunks[i].token_count`. Recomputed by `Context::new`; strategies
    /// that mutate chunks re-wrap the result in a new `Context` to keep this
    /// field accurate.
    pub total_tokens: usize,
}

impl Context {
    /// Build a `Context` from a `Vec<Chunk>`, computing `total_tokens`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cctx::core::context::{Chunk, Context, AttentionZone};
    ///
    /// let chunks = vec![Chunk {
    ///     index: 0,
    ///     role: "user".into(),
    ///     content: "Hello".into(),
    ///     token_count: 1,
    ///     relevance_score: 0.5,
    ///     attention_zone: AttentionZone::Strong,
    /// }];
    /// let ctx = Context::new(chunks);
    /// assert_eq!(ctx.total_tokens, 1);
    /// ```
    pub fn new(chunks: Vec<Chunk>) -> Self {
        let total_tokens = chunks.iter().map(|c| c.token_count).sum();
        Context {
            chunks,
            total_tokens,
        }
    }

    /// Number of chunks currently in the context.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Borrow the chunks currently sitting in the attention dead zone.
    ///
    /// The zone is whatever [`crate::analyzer::health::assign_attention_zones`]
    /// last classified — call that again after mutations to refresh.
    pub fn dead_zone_chunks(&self) -> Vec<&Chunk> {
        self.chunks
            .iter()
            .filter(|c| c.attention_zone == AttentionZone::DeadZone)
            .collect()
    }
}
