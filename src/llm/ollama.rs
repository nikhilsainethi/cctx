//! Ollama LLM provider — calls a local Ollama instance for text generation.
//!
//! Setup:
//!   brew install ollama
//!   ollama pull llama3.2:3b     (2GB, fast on CPU, good for summarization)
//!   ollama serve                (starts on localhost:11434)
//!
//! Usage:
//!   cctx summarize --llm-provider ollama < input.txt
//!   cctx summarize --llm-provider ollama --llm-model mistral < input.txt

use anyhow::{Context, Result};
use reqwest::blocking::Client;

use super::LlmProvider;

/// LLM provider backed by a local Ollama server.
///
/// Calls `/api/generate` with `stream: false`; the system prompt is
/// prepended to the user prompt because Ollama's generate endpoint takes a
/// single prompt field. Uses `reqwest::blocking::Client`, 120s timeout.
pub struct OllamaLlm {
    client: Client,
    url: String,
    model: String,
    temperature: f64,
}

impl OllamaLlm {
    /// Build a provider with an explicit URL, model, and sampling temperature.
    pub fn new(url: &str, model: &str, temperature: f64) -> Self {
        OllamaLlm {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            temperature,
        }
    }

    /// Convenience constructor: `http://localhost:11434`, `llama3.2:3b`, T=0.3.
    pub fn default_local() -> Self {
        Self::new("http://localhost:11434", "llama3.2:3b", 0.3)
    }

    /// Default localhost URL with a caller-chosen model.
    pub fn with_model(model: &str) -> Self {
        Self::new("http://localhost:11434", model, 0.3)
    }
}

impl LlmProvider for OllamaLlm {
    fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        // Ollama's /api/generate endpoint takes a single prompt.
        // We prepend the system message to the prompt.
        let full_prompt = if system.is_empty() {
            prompt.to_string()
        } else {
            format!("{}\n\n{}", system, prompt)
        };

        let resp: serde_json::Value = self
            .client
            .post(format!("{}/api/generate", self.url))
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": full_prompt,
                "stream": false,
                "options": {
                    "temperature": self.temperature
                }
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

        // Check for errors.
        if let Some(err) = resp.get("error") {
            let msg = err.as_str().unwrap_or("unknown error");
            anyhow::bail!("Ollama error: {}. Try: ollama pull {}", msg, self.model);
        }

        resp.get("response")
            .and_then(|r| r.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("Ollama response missing 'response' field"))
    }
}
