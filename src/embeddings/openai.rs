//! OpenAI embedding provider — calls the OpenAI embeddings API.
//!
//! Requires an API key (via --openai-api-key or OPENAI_API_KEY env var).
//!
//! Usage:
//!   export OPENAI_API_KEY=sk-...
//!   cctx optimize input.json --strategy dedup --embedding-provider openai

use anyhow::{Context, Result};
use reqwest::blocking::Client;

use super::EmbeddingProvider;

pub struct OpenAIEmbedder {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIEmbedder {
    pub fn new(api_key: &str, model: &str) -> Self {
        OpenAIEmbedder {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| Client::new()),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    /// Default model: text-embedding-3-small (cheap, fast, 1536 dims).
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY not set. Export it or use --embedding-provider ollama.")?;
        Ok(Self::new(&key, "text-embedding-3-small"))
    }
}

impl EmbeddingProvider for OpenAIEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // OpenAI's /v1/embeddings supports batching natively.
        let resp: serde_json::Value = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "input": texts
            }))
            .send()
            .context("Cannot reach OpenAI embeddings API")?
            .json()
            .context("OpenAI returned invalid JSON")?;

        // Check for API errors.
        if let Some(err) = resp.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("OpenAI embeddings error: {}", msg);
        }

        let data = resp
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow::anyhow!("OpenAI response missing 'data' array"))?;

        // Results come back sorted by index, but let's be safe and sort.
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .filter_map(|item| {
                let idx = item.get("index")?.as_u64()? as usize;
                let emb: Vec<f32> = item
                    .get("embedding")?
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                Some((idx, emb))
            })
            .collect();

        indexed.sort_by_key(|(idx, _)| *idx);
        Ok(indexed.into_iter().map(|(_, emb)| emb).collect())
    }
}
