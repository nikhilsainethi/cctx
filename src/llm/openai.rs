//! OpenAI LLM provider — calls the OpenAI chat completions API.
//!
//! Usage:
//!   export OPENAI_API_KEY=sk-...
//!   cctx summarize --llm-provider openai < input.txt
//!   cctx summarize --llm-provider openai --llm-model gpt-4o < input.txt

use anyhow::{Context, Result};
use reqwest::blocking::Client;

use super::LlmProvider;

/// LLM provider backed by OpenAI's `/v1/chat/completions` endpoint.
///
/// Sends system and user messages as separate chat messages. Uses
/// `reqwest::blocking::Client`, 120s timeout.
pub struct OpenAILlm {
    client: Client,
    api_key: String,
    model: String,
    temperature: f64,
}

impl OpenAILlm {
    /// Build a provider with an explicit API key, model, and temperature.
    pub fn new(api_key: &str, model: &str, temperature: f64) -> Self {
        OpenAILlm {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
            api_key: api_key.to_string(),
            model: model.to_string(),
            temperature,
        }
    }

    /// Read `OPENAI_API_KEY` from the environment and use `gpt-4o-mini` at T=0.3.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `OPENAI_API_KEY` is not set.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY not set. Export it or use --llm-provider ollama.")?;
        Ok(Self::new(&key, "gpt-4o-mini", 0.3))
    }

    /// Same as [`Self::from_env`] but with a caller-chosen model name.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `OPENAI_API_KEY` is not set.
    pub fn from_env_with_model(model: &str) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY not set. Export it or use --llm-provider ollama.")?;
        Ok(Self::new(&key, model, 0.3))
    }
}

impl LlmProvider for OpenAILlm {
    fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": system}));
        }
        messages.push(serde_json::json!({"role": "user", "content": prompt}));

        let resp: serde_json::Value = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }))
            .send()
            .context("Cannot reach OpenAI API")?
            .json()
            .context("OpenAI returned invalid JSON")?;

        // Check for API errors.
        if let Some(err) = resp.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("OpenAI error: {}", msg);
        }

        resp.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("OpenAI response missing content"))
    }
}
