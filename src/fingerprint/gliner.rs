//! `gline-rs`-backed [`GlinerEngine`] implementation.
//!
//! Compiled only when `--features gliner` is set. Loads a downloaded
//! GLiNER ONNX model + Hugging Face tokenizer from
//! `~/.cctx/models/gliner/` and runs span-mode inference.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context as _, Result};
use gliner::model::input::text::TextInput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::GLiNER;
use orp::params::RuntimeParameters;

use crate::core::context::Chunk;

use super::gliner_iface::{GlinerEngine, GlinerEntity};

// ── Model location ────────────────────────────────────────────────────────────

/// Default subdirectory of `$HOME` where cctx stashes downloaded models.
pub const MODEL_HOME_SUBDIR: &str = ".cctx/models/gliner";

/// Filenames on disk under [`MODEL_HOME_SUBDIR`]. Matches what
/// `cctx model download gliner` writes.
pub const MODEL_FILE: &str = "model.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Hugging Face URLs the downloader fetches from.
pub const HF_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/onnx/model.onnx";
pub const HF_TOKENIZER_URL: &str =
    "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/tokenizer.json";

/// Resolve the model directory under `$HOME`.
pub fn model_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(MODEL_HOME_SUBDIR))
}

/// Verify the GLiNER model files exist on disk; return their paths or
/// fail with a friendly download hint.
///
/// # Errors
///
/// Returns `Err` when either `model.onnx` or `tokenizer.json` is
/// missing under `~/.cctx/models/gliner/`. The error message instructs
/// the user to run `cctx model download gliner`.
pub fn ensure_model() -> Result<(PathBuf, PathBuf)> {
    let dir =
        model_dir().ok_or_else(|| anyhow!("Cannot determine $HOME for GLiNER model directory"))?;
    let model = dir.join(MODEL_FILE);
    let tokenizer = dir.join(TOKENIZER_FILE);

    if !model.is_file() || !tokenizer.is_file() {
        return Err(anyhow!(
            "GLiNER model not found at {}.\n\
             Download it with: cctx model download gliner",
            dir.display()
        ));
    }
    Ok((model, tokenizer))
}

// ── Engine implementation ─────────────────────────────────────────────────────

/// Real `gline-rs`-backed engine. Loads the ONNX model + tokenizer at
/// construction time so subsequent `predict` calls are zero-overhead
/// past the first.
pub struct GlineRsEngine {
    model: GLiNER<SpanMode>,
}

impl GlineRsEngine {
    /// Build an engine, locating model files via [`ensure_model`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the files are missing or the model fails to load
    /// (e.g. ONNX runtime initialization, tokenizer parse error).
    pub fn from_default_dir() -> Result<Self> {
        let (model_path, tokenizer_path) = ensure_model()?;
        Self::from_paths(&model_path, &tokenizer_path)
    }

    /// Build an engine from explicit paths. Useful for tests against
    /// a fixture model in a non-standard location.
    pub fn from_paths(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        // Tier-1 brief specifies 0.4 as the model-side threshold; we
        // additionally filter by config.gliner_threshold post-hoc so
        // callers can tighten further without re-loading the model.
        let params = Parameters {
            threshold: 0.4,
            ..Parameters::default()
        };

        let model = GLiNER::<SpanMode>::new(
            params,
            RuntimeParameters::default(),
            tokenizer_path,
            model_path,
        )
        .map_err(|e| anyhow!("Failed to load GLiNER model: {}", e))?;
        Ok(Self { model })
    }
}

impl GlinerEngine for GlineRsEngine {
    fn predict(
        &self,
        chunks: &[(usize, &Chunk)],
        labels: &[&str],
        threshold: f64,
    ) -> Result<Vec<GlinerEntity>> {
        if chunks.is_empty() || labels.is_empty() {
            return Ok(Vec::new());
        }

        // gline-rs takes a batch of texts and a single label set.
        let texts: Vec<&str> = chunks.iter().map(|(_, c)| c.content.as_str()).collect();
        let input = TextInput::from_str(&texts, labels)
            .map_err(|e| anyhow!("Failed to build GLiNER input: {}", e))?;
        let output = self
            .model
            .inference(input)
            .map_err(|e| anyhow!("GLiNER inference failed: {}", e))?;

        // output.spans[i] = entities found in texts[i]; the i-th text
        // corresponds to chunks[i]. Walk both in lockstep.
        let mut out: Vec<GlinerEntity> = Vec::new();
        for (batch_idx, spans) in output.spans.iter().enumerate() {
            let Some(&(chunk_idx, _)) = chunks.get(batch_idx) else {
                continue;
            };
            for span in spans {
                let confidence = span.probability() as f64;
                if confidence < threshold {
                    continue;
                }
                let (start, end) = span.offsets();
                out.push(GlinerEntity {
                    source_chunk_idx: chunk_idx,
                    label: span.class().to_string(),
                    text: span.text().to_string(),
                    start,
                    end,
                    confidence,
                });
            }
        }
        Ok(out)
    }
}

// ── Model download ────────────────────────────────────────────────────────────

/// Download the GLiNER small v2.1 ONNX model and tokenizer from
/// Hugging Face into `~/.cctx/models/gliner/`.
///
/// Idempotent — if both files already exist, returns immediately. The
/// model is ~188 MB, so this is a one-time cost per machine.
///
/// # Errors
///
/// Returns `Err` on any HTTP failure, filesystem failure, or if the
/// tokenizer/model URL changes upstream. We log progress to stderr.
pub fn download_model() -> Result<()> {
    let dir = model_dir().ok_or_else(|| anyhow!("Cannot determine $HOME"))?;
    std::fs::create_dir_all(&dir).with_context(|| format!("Cannot create {}", dir.display()))?;

    let model_path = dir.join(MODEL_FILE);
    let tokenizer_path = dir.join(TOKENIZER_FILE);

    if model_path.is_file() && tokenizer_path.is_file() {
        eprintln!("[cctx] GLiNER model already present at {}.", dir.display());
        return Ok(());
    }

    eprintln!(
        "[cctx] Downloading GLiNER (small v2.1) into {}",
        dir.display()
    );
    eprintln!("[cctx] This is a one-time ~188 MB download.");

    // Tokenizer first — it's small (~7 MB) and a quick sanity check
    // that the network and HF hostname work before the big download.
    if !tokenizer_path.is_file() {
        download_to_file(HF_TOKENIZER_URL, &tokenizer_path)?;
    }
    if !model_path.is_file() {
        download_to_file(HF_MODEL_URL, &model_path)?;
    }

    eprintln!("[cctx] GLiNER model ready at {}.", dir.display());
    Ok(())
}

fn download_to_file(url: &str, dest: &Path) -> Result<()> {
    eprintln!("[cctx]   GET {}", url);
    let mut resp = reqwest::blocking::get(url)
        .with_context(|| format!("Cannot reach {}", url))?
        .error_for_status()
        .with_context(|| format!("Bad status from {}", url))?;
    // Write to a sibling tmp file then rename, so a partial download
    // doesn't half-populate the model dir on Ctrl-C / crash.
    let tmp = dest.with_extension(format!(
        "{}.partial",
        dest.extension().and_then(|s| s.to_str()).unwrap_or("tmp")
    ));
    {
        let mut file = std::fs::File::create(&tmp)
            .with_context(|| format!("Cannot create {}", tmp.display()))?;
        std::io::copy(&mut resp, &mut file)
            .with_context(|| format!("Failed to write {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("Cannot rename {} -> {}", tmp.display(), dest.display()))?;
    let bytes = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[cctx]     wrote {} ({:.1} MB)",
        dest.display(),
        bytes as f64 / 1_048_576.0
    );
    Ok(())
}

/// List downloaded model directories under `~/.cctx/models/`. Used by
/// `cctx model list`.
pub fn list_installed_models() -> Vec<(String, PathBuf, bool)> {
    let mut out: Vec<(String, PathBuf, bool)> = Vec::new();
    let Some(home) = std::env::var_os("HOME") else {
        return out;
    };
    let root = PathBuf::from(home).join(".cctx/models");
    if !root.is_dir() {
        return out;
    }
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = match entry.file_name().to_str() {
                Some(s) => s.to_string(),
                None => continue,
            };
            // For "gliner", success = both files present.
            let installed = path.join(MODEL_FILE).is_file() && path.join(TOKENIZER_FILE).is_file();
            out.push((name, path, installed));
        }
    }
    out
}
