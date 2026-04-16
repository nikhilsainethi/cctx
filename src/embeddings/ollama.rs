//! Ollama embedding provider — calls a local Ollama instance.
//!
//! Setup:
//!   brew install ollama        (or download from ollama.ai)
//!   ollama pull nomic-embed-text   (274MB, fast on CPU)
//!   ollama serve               (starts on localhost:11434)
//!
//! Usage:
//!   cctx optimize input.json --strategy dedup --embedding-provider ollama

use anyhow::{Context, Result};
use reqwest::blocking::Client;

use super::EmbeddingProvider;

/// Embedding provider backed by a local Ollama server.
///
/// Calls Ollama's `/api/embeddings` endpoint one text at a time; the trait-
/// level batch interface is preserved but requests are serialized. Uses
/// `reqwest::blocking::Client` so no tokio runtime is required.
pub struct OllamaEmbedder {
    client: Client,
    url: String,
    model: String,
}

impl OllamaEmbedder {
    /// Build an embedder pointing at a specific Ollama URL and model.
    ///
    /// A 60-second timeout is applied; if the timeout-capable client fails
    /// to build, falls back to the default client.
    pub fn new(url: &str, model: &str) -> Self {
        OllamaEmbedder {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| Client::new()),
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Convenience constructor: `http://localhost:11434` + `nomic-embed-text`.
    pub fn default_local() -> Self {
        Self::new("http://localhost:11434", "nomic-embed-text")
    }
}

impl EmbeddingProvider for OllamaEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Ollama's /api/embeddings takes one prompt at a time.
        // Batch support (/api/embed) exists in newer versions but we use
        // the single-prompt endpoint for maximum compatibility.
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let resp: serde_json::Value = self
                .client
                .post(format!("{}/api/embeddings", self.url))
                .json(&serde_json::json!({
                    "model": self.model,
                    "prompt": text
                }))
                .send()
                .with_context(|| {
                    format!(
                        "Cannot reach Ollama at {}. Is `ollama serve` running?",
                        self.url
                    )
                })?
                .json()
                .context("Ollama returned invalid JSON")?;

            let embedding: Vec<f32> = resp
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Ollama response has no 'embedding' field. Model '{}' may not support embeddings.",
                        self.model
                    )
                })?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            results.push(embedding);
        }

        Ok(results)
    }
}
