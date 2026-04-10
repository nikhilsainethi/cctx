use anyhow::Result;
// tiktoken_rs::cl100k_base() returns the GPT-4 BPE tokenizer.
// cl100k is also what Claude models approximate — close enough for budgeting.
use tiktoken_rs::cl100k_base;

// CoreBPE is tiktoken-rs's tokenizer type. We wrap it so the rest of the
// codebase doesn't need to know which tokenizer crate we're using.
pub struct Tokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl Tokenizer {
    /// Initialize the tokenizer. Loads BPE vocab data — may fail, so returns Result.
    /// Result<T> is either Ok(T) on success or Err(E) on failure.
    pub fn new() -> Result<Self> {
        // The ? operator: if cl100k_base() returns Err, immediately return that
        // Err from *this* function. Like a checked exception that propagates up.
        let bpe = cl100k_base()?;
        Ok(Tokenizer { bpe })
    }

    /// Count tokens in a string.
    /// `text: &str` is a borrowed string slice — we read it without taking ownership.
    pub fn count(&self, text: &str) -> usize {
        // encode_with_special_tokens returns Vec<usize> (one usize per token).
        // .len() gives us the count.
        self.bpe.encode_with_special_tokens(text).len()
    }
}
