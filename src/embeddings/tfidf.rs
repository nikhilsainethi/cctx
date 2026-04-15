//! TF-IDF mock embedding provider — no external dependencies.
//!
//! Converts text to a sparse vector based on word frequencies weighted by
//! inverse document frequency. Texts about the same topic produce similar
//! vectors because they share important words.
//!
//! This is NOT a real embedding model — it can't capture synonyms or
//! rephrasings the way neural embeddings do. But it's close enough for:
//!   - Testing the semantic dedup pipeline without Ollama/OpenAI
//!   - CI environments where no embedding API is available
//!   - Catching conversations that repeat the same key terms

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::EmbeddingProvider;

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "it", "its", "are", "was", "be", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "can", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "my", "your", "his", "her", "our", "their", "me", "him", "us", "them", "what",
    "which", "who", "whom", "if", "not", "no", "so", "as", "from", "about", "into", "through",
    "also", "just", "more", "very", "too",
];

/// TF-IDF embedder for testing. No API calls, pure computation.
pub struct TfIdfEmbedder;

impl EmbeddingProvider for TfIdfEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // ── Build vocabulary across all texts ─────────────────────────────
        let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
        let tokenized: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t, &stop)).collect();

        // Collect all unique words for the vector dimensions.
        let mut vocab: Vec<String> = tokenized
            .iter()
            .flat_map(|words| words.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        vocab.sort(); // deterministic dimension ordering

        let vocab_index: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let dim = vocab.len();
        let n_docs = texts.len() as f32;

        // ── Compute IDF for each word ─────────────────────────────────────
        // IDF = ln(N / (1 + df)) where df = how many documents contain the word.
        let mut df: HashMap<&str, usize> = HashMap::new();
        for words in &tokenized {
            let unique: HashSet<&str> = words.iter().map(|w| w.as_str()).collect();
            for w in unique {
                *df.entry(w).or_default() += 1;
            }
        }

        let idf: HashMap<&str, f32> = vocab
            .iter()
            .map(|w| {
                let d = *df.get(w.as_str()).unwrap_or(&0) as f32;
                (w.as_str(), (n_docs / (1.0 + d)).ln().max(0.0))
            })
            .collect();

        // ── Compute TF-IDF vector per text ────────────────────────────────
        let embeddings: Vec<Vec<f32>> = tokenized
            .iter()
            .map(|words| {
                let mut vec = vec![0.0f32; dim];
                let wc = words.len() as f32;
                if wc == 0.0 {
                    return vec;
                }
                // Count word frequencies in this document.
                let mut tf: HashMap<&str, f32> = HashMap::new();
                for w in words {
                    *tf.entry(w.as_str()).or_default() += 1.0;
                }
                for (word, count) in &tf {
                    if let Some(&idx) = vocab_index.get(word) {
                        let tf_val = count / wc;
                        let idf_val = idf.get(word).copied().unwrap_or(0.0);
                        vec[idx] = tf_val * idf_val;
                    }
                }
                vec
            })
            .collect();

        Ok(embeddings)
    }
}

fn tokenize(text: &str, stop: &HashSet<&str>) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| w.len() > 1 && !stop.contains(w.as_str()))
        .collect()
}
