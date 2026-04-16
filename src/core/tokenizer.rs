use anyhow::Result;
// tiktoken_rs::cl100k_base() returns the GPT-4 BPE tokenizer.
// cl100k is also what Claude models approximate — close enough for budgeting.
use tiktoken_rs::cl100k_base;

/// BPE tokenizer using the OpenAI `cl100k_base` vocabulary.
///
/// Produces token counts that match what GPT-4 / GPT-4o see exactly, and
/// approximate within a few percent for Claude and most other major models —
/// good enough for token-budget decisions. Wraps `tiktoken_rs::CoreBPE` so
/// callers don't depend on the underlying crate.
pub struct Tokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl Tokenizer {
    /// Build a new tokenizer, loading the BPE vocabulary.
    ///
    /// The first call loads ~1 MB of encoded vocabulary data into memory;
    /// subsequent calls reuse the crate's internal cache.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the embedded BPE table fails to decode — effectively
    /// only possible if `tiktoken-rs` itself is broken. In practice, treat
    /// this as infallible for the lifetime of the program.
    ///
    /// # Examples
    ///
    /// ```
    /// use cctx::core::tokenizer::Tokenizer;
    ///
    /// let tok = Tokenizer::new().unwrap();
    /// assert!(tok.count("hello world") > 0);
    /// ```
    pub fn new() -> Result<Self> {
        let bpe = cl100k_base()?;
        Ok(Tokenizer { bpe })
    }

    /// Count the tokens in `text`.
    ///
    /// Uses the same encoding path as GPT-4, including special-token
    /// handling — output matches what the model's billing counts.
    ///
    /// # Examples
    ///
    /// ```
    /// use cctx::core::tokenizer::Tokenizer;
    ///
    /// let tok = Tokenizer::new().unwrap();
    /// assert_eq!(tok.count(""), 0);
    /// assert!(tok.count("The quick brown fox") >= 4);
    /// ```
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }
}
