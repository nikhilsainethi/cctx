// serde traits: Serialize = Rust → JSON, Deserialize = JSON → Rust.
// `use` brings them into scope so #[derive] can find them.
use serde::{Deserialize, Serialize};

// ── Input format ─────────────────────────────────────────────────────────────

// #[derive(...)] auto-generates trait implementations.
// Debug: lets you print with {:?}. Clone: lets you .clone() the value.
// Deserialize: serde can parse JSON into this struct.
// Serialize: serde can turn this struct into JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    // Option<T> is Rust's null-safe type: either Some(value) or None.
    // skip_serializing_if omits the field from JSON output when it's None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevance_score: Option<f64>,
}

// ── Attention model ───────────────────────────────────────────────────────────

// An enum in Rust is a type that can be exactly ONE of its variants.
// This models the U-shaped attention curve from Liu et al. (TACL 2024):
// LLMs strongly attend to the start and end; the middle is a "dead zone".
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionZone {
    /// Beginning (~0–25%) or end (~75–100%) of context — high LLM attention.
    Strong,
    /// Middle (~25–75%) of context — ~30% accuracy drop per research.
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

// A Chunk wraps one Message with computed metadata.
// We derive Clone because the bookend strategy needs to rearrange copies.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Original position in the conversation (0-indexed).
    pub index: usize,
    pub role: String,
    pub content: String,
    pub token_count: usize,
    /// 0.0 = irrelevant, 1.0 = critical. Used by bookend for placement decisions.
    pub relevance_score: f64,
    pub attention_zone: AttentionZone,
}

impl Chunk {
    /// Convert back to a plain Message for JSON output.
    /// &self is an immutable borrow — we read the chunk but don't consume it.
    pub fn to_message(&self) -> Message {
        Message {
            // .clone() deep-copies a String; required because String is owned,
            // not Copy (unlike integers which are trivially copied).
            role: self.role.clone(),
            content: self.content.clone(),
            relevance_score: None,
        }
    }
}

// ── Context ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Context {
    pub chunks: Vec<Chunk>,
    pub total_tokens: usize,
}

impl Context {
    /// Construct a Context and compute total_tokens automatically.
    /// `chunks: Vec<Chunk>` moves ownership of the Vec into this function.
    pub fn new(chunks: Vec<Chunk>) -> Self {
        // Iterator::sum() adds up all the values. map() transforms each element.
        let total_tokens = chunks.iter().map(|c| c.token_count).sum();
        Context {
            chunks,
            total_tokens,
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns references to chunks that sit in the attention dead zone.
    /// Vec<&Chunk> is a vector of *borrowed* references — we don't own the chunks.
    pub fn dead_zone_chunks(&self) -> Vec<&Chunk> {
        self.chunks
            .iter()
            .filter(|c| c.attention_zone == AttentionZone::DeadZone)
            .collect()
    }
}
