//! Embedding providers and vector math for semantic deduplication.
//!
//! # What are embeddings?
//!
//! An embedding model converts text into a dense vector of floats (e.g.,
//! 768 dimensions). Similar meanings → similar vectors, even if the words
//! are completely different. "The car is fast" and "The automobile has great
//! speed" would have nearly identical embeddings, while "The car is fast"
//! and "I like pizza" would be far apart.
//!
//! This is fundamentally different from our Jaccard word-overlap approach
//! (which only catches literal word reuse). Embeddings capture *meaning*.
//!
//! # Cosine similarity
//!
//! Cosine similarity measures the angle between two vectors:
//!   cos(A, B) = (A · B) / (|A| × |B|)
//!
//! Range: -1.0 (opposite) to 1.0 (identical direction).
//! For text embeddings, values are typically 0.0–1.0:
//!   - > 0.85: nearly identical meaning (candidates for dedup)
//!   - 0.5–0.85: related topics
//!   - < 0.5: different subjects
//!
//! # Why the trait is sync, not async
//!
//! The embedding providers make HTTP calls (to Ollama or OpenAI). In Rust,
//! HTTP is usually async. But our Strategy pipeline is sync (it runs on a
//! blocking thread in the proxy, or on the main thread in the CLI).
//!
//! We use `reqwest::blocking::Client` — a sync HTTP client that internally
//! manages its own tokio runtime. This lets the embedding calls block the
//! current thread, which is safe because:
//!   - In the CLI: we're on the main thread, blocking is fine.
//!   - In the proxy: we're inside `spawn_blocking`, which is a dedicated
//!     thread pool for blocking work.
//!
//! An async trait version would look like:
//! ```text
//! // Requires the `async-trait` crate because Rust's built-in async traits
//! // can't be used with `dyn` dispatch — the compiler needs to know the
//! // future's size, but async fns return opaque types whose size varies.
//! // `async-trait` auto-generates a `Pin<Box<dyn Future>>` wrapper.
//! #[async_trait::async_trait]
//! pub trait EmbeddingProvider {
//!     async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
//! }
//! ```
//! For our use case, sync is simpler and equally correct.

use anyhow::Result;

/// TF-IDF mock embedder — always available, no external dependencies.
/// Good for testing and CI. Not as accurate as neural embeddings.
pub mod tfidf;

// Real providers are only compiled with --features embeddings (they need reqwest).
#[cfg(feature = "embeddings")]
pub mod ollama;
#[cfg(feature = "embeddings")]
pub mod openai;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// A provider that converts text into dense embedding vectors.
///
/// Implementations call an external API (Ollama, OpenAI) or compute locally
/// (TF-IDF fallback). `Send + Sync` lets the provider live inside
/// `Arc<dyn EmbeddingProvider>` and cross thread boundaries.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts and return their vector representations.
    ///
    /// Every inner `Vec<f32>` has the same dimensionality (e.g. 768 for
    /// `nomic-embed-text`, 1536 for OpenAI's `text-embedding-3-small`).
    /// Order must match `texts` exactly.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the provider's network call fails, the response is
    /// malformed, or the returned batch size doesn't match `texts.len()`.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// ── Vector math ───────────────────────────────────────────────────────────────

/// Cosine similarity between two vectors: `(A · B) / (|A| × |B|)`.
///
/// Returns `0.0` when either vector has zero magnitude, and the result is
/// clamped to `[-1.0, 1.0]` to guard against floating-point drift.
///
/// # Examples
///
/// ```
/// use cctx::embeddings::cosine_similarity;
///
/// let a = [1.0, 0.0, 0.0];
/// let b = [0.0, 1.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
///
/// let c = [1.0, 1.0, 0.0];
/// let d = [1.0, 1.0, 0.0];
/// assert!((cosine_similarity(&c, &d) - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Dot product: Σ(a_i × b_i)
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    // L2 norms: √(Σ(a_i²))
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}
