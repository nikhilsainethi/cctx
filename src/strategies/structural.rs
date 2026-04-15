//! Structural compression — reduces tokens by compressing structured content
//! (JSON, code, markdown) inside messages without losing meaning.
//!
//! Three sub-strategies:
//!   1. **JSON Pruning** — remove timestamps, UUIDs, metadata; collapse deep nesting
//!   2. **Code → Signatures** — keep fn/class signatures + docstrings, drop bodies
//!   3. **Markdown Collapse** — collapse sections irrelevant to the query

use std::collections::HashSet;

use serde_json::Value;

use crate::core::context::{Chunk, Context};
use crate::core::tokenizer::Tokenizer;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compress structured content in every chunk. Returns new chunks with
/// modified content and recounted token totals.
pub fn apply(context: &Context, query: Option<&str>, tokenizer: &Tokenizer) -> Vec<Chunk> {
    context
        .chunks
        .iter()
        .map(|chunk| {
            let compressed = compress_content(&chunk.content, query);
            Chunk {
                index: chunk.index,
                role: chunk.role.clone(),
                content: compressed.clone(),
                token_count: tokenizer.count(&compressed),
                relevance_score: chunk.relevance_score,
                attention_zone: chunk.attention_zone.clone(),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Content block parsing
// ═══════════════════════════════════════════════════════════════════════════════
//
// A message may contain plain text, ```fenced code blocks```, and bare JSON.
// We split the content into typed blocks, compress each one appropriately,
// then reassemble the message.

enum ContentBlock {
    /// Plain text (may contain markdown headers).
    Text(String),
    /// A ``` fenced block with an optional language tag.
    Fenced { lang: String, code: String },
}

/// Split message content into typed blocks at ``` fence boundaries.
fn parse_blocks(content: &str) -> Vec<ContentBlock> {
    let mut blocks: Vec<ContentBlock> = Vec::new();
    let mut text_buf = String::new();
    let mut in_fence = false;
    let mut fence_lang = String::new();
    let mut fence_buf = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Toggle fence state on ``` lines.
        if let Some(after_ticks) = trimmed.strip_prefix("```") {
            if !in_fence {
                // ── Opening fence ──
                if !text_buf.is_empty() {
                    blocks.push(classify_text(std::mem::take(&mut text_buf)));
                }
                // Everything after ``` is the language tag (e.g. "json", "python").
                fence_lang = after_ticks.trim().to_string();
                fence_buf.clear();
                in_fence = true;
            } else {
                // ── Closing fence ──
                blocks.push(ContentBlock::Fenced {
                    lang: std::mem::take(&mut fence_lang),
                    code: std::mem::take(&mut fence_buf),
                });
                in_fence = false;
            }
            continue;
        }

        if in_fence {
            if !fence_buf.is_empty() {
                fence_buf.push('\n');
            }
            fence_buf.push_str(line);
        } else {
            if !text_buf.is_empty() {
                text_buf.push('\n');
            }
            text_buf.push_str(line);
        }
    }

    // Flush remaining buffers.
    if in_fence {
        // Unclosed fence — treat the whole thing as text.
        text_buf.push_str(&format!("\n```{}\n{}", fence_lang, fence_buf));
    }
    if !text_buf.is_empty() {
        blocks.push(classify_text(text_buf));
    }

    blocks
}

/// Promote a text block to BareJson if it parses as JSON.
fn classify_text(text: String) -> ContentBlock {
    // serde_json::from_str attempts to parse the trimmed text.
    // If it succeeds, we know it's valid JSON we can prune.
    let trimmed = text.trim();
    let looks_like_json = (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'));
    if looks_like_json && serde_json::from_str::<Value>(trimmed).is_ok() {
        // Wrap as Fenced so the JSON pruner picks it up.
        return ContentBlock::Fenced {
            lang: "json".to_string(),
            code: trimmed.to_string(),
        };
    }
    ContentBlock::Text(text)
}

/// Compress each block according to its type, then reassemble.
fn compress_content(content: &str, query: Option<&str>) -> String {
    let blocks = parse_blocks(content);

    let parts: Vec<String> = blocks
        .into_iter()
        .map(|block| match block {
            ContentBlock::Fenced { lang, code } => {
                let lang_lower = lang.to_lowercase();
                let line_count = code.lines().count();
                match lang_lower.as_str() {
                    // ── JSON pruning ──
                    "json" => {
                        let pruned = prune_json(&code);
                        format!("```json\n{}\n```", pruned)
                    }
                    // ── Python signature extraction (> 20 lines only) ──
                    "python" | "py" if line_count > 20 => {
                        format!("```python\n{}\n```", extract_python_signatures(&code))
                    }
                    // ── JS/TS signature extraction ──
                    "javascript" | "js" | "typescript" | "ts" if line_count > 20 => {
                        let compressed =
                            extract_signatures_braces(&code, is_js_decl, "/* ... ({}) */");
                        format!("```{}\n{}\n```", lang, compressed)
                    }
                    // ── Rust signature extraction ──
                    "rust" | "rs" if line_count > 20 => {
                        let compressed =
                            extract_signatures_braces(&code, is_rust_decl, "/* ... ({}) */");
                        format!("```rust\n{}\n```", compressed)
                    }
                    // ── Unknown / short — keep as-is ──
                    _ => format!("```{}\n{}\n```", lang, code),
                }
            }
            ContentBlock::Text(text) => {
                // If the text has markdown headers and we have a query,
                // collapse irrelevant sections.
                if text.lines().any(|l| l.trim_start().starts_with('#')) {
                    collapse_markdown(&text, query)
                } else {
                    text
                }
            }
        })
        .collect();

    parts.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. JSON Pruning
// ═══════════════════════════════════════════════════════════════════════════════
//
// Walk a serde_json::Value tree and remove low-signal fields:
//   - Timestamps (keys ending in _at, _time, _timestamp)
//   - UUIDs / GUIDs (by key name or value pattern)
//   - Metadata blocks
//   - Anything deeper than depth 3 (collapsed to a summary string)

fn prune_json(json_str: &str) -> String {
    match serde_json::from_str::<Value>(json_str.trim()) {
        Ok(val) => {
            let pruned = prune_value(val, 0);
            // to_string_pretty re-serializes with 2-space indent.
            serde_json::to_string_pretty(&pruned).unwrap_or_else(|_| json_str.to_string())
        }
        Err(_) => json_str.to_string(),
    }
}

/// Recursively prune a JSON value tree.
///
/// `depth` tracks nesting level. At depth > 3 we collapse entire sub-trees
/// into summary strings like "{...5 fields}" to save tokens.
fn prune_value(value: Value, depth: usize) -> Value {
    match value {
        Value::Object(map) => {
            if depth > 3 {
                // Collapse deep objects to a summary.
                return Value::String(format!("{{...{} fields}}", map.len()));
            }
            // serde_json::Map is an ordered map — insertion order is preserved.
            let pruned: serde_json::Map<String, Value> = map
                .into_iter()
                .filter(|(key, val)| !should_prune_field(key, val))
                .map(|(key, val)| (key, prune_value(val, depth + 1)))
                .collect();

            // Flatten single-child wrapper objects.
            // {"data": {"name": "x", "id": 1}} → {"name": "x", "id": 1}
            // Only flatten if the wrapper key is a generic name.
            if pruned.len() == 1 {
                let (key, val) = pruned.iter().next().unwrap();
                if is_wrapper_key(key) && matches!(val, Value::Object(_)) {
                    return val.clone();
                }
            }

            Value::Object(pruned)
        }
        Value::Array(arr) => {
            if depth > 3 {
                return Value::String(format!("[...{} items]", arr.len()));
            }
            Value::Array(arr.into_iter().map(|v| prune_value(v, depth + 1)).collect())
        }
        other => other,
    }
}

/// Decide whether a JSON field should be pruned.
fn should_prune_field(key: &str, value: &Value) -> bool {
    let k = key.to_lowercase();

    // Timestamp fields: created_at, updated_at, login_time, etc.
    if k.ends_with("_at") || k.ends_with("_time") || k.ends_with("_timestamp") {
        return true;
    }

    // UUID/GUID key names.
    if k == "uuid" || k == "guid" {
        return true;
    }

    // Metadata blocks — often large, rarely useful for reasoning.
    if k == "metadata" || k == "meta" {
        return true;
    }

    // Values that look like UUIDs (8-4-4-4-12 hex pattern).
    if let Value::String(s) = value
        && is_uuid_like(s)
    {
        return true;
    }

    false
}

fn is_uuid_like(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    parts.len() == 5
        && parts[0].len() == 8
        && parts[1].len() == 4
        && parts[2].len() == 4
        && parts[3].len() == 4
        && parts[4].len() == 12
        && parts
            .iter()
            .all(|p| p.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Generic wrapper keys that add nesting without meaning.
fn is_wrapper_key(key: &str) -> bool {
    matches!(
        key.to_lowercase().as_str(),
        "data" | "result" | "results" | "response" | "body" | "payload" | "value" | "item"
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Code → Signatures
// ═══════════════════════════════════════════════════════════════════════════════

// ── Python ────────────────────────────────────────────────────────────────────
//
// Python uses indentation for scoping, so we detect function/class definitions
// and measure their body by indentation depth.

fn extract_python_signatures(code: &str) -> String {
    let lines: Vec<&str> = code.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        let indent = leading_spaces(lines[i]);

        // Detect function/class definitions.
        let is_def = trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("async def ");

        if !is_def {
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }

        // ── Keep the signature line ──
        out.push(lines[i].to_string());
        i += 1;

        // ── Keep docstring if present ──
        // Python docstrings are the first statement in a function body,
        // delimited by triple-quotes (""" or ''').
        if i < lines.len() {
            let first_body = lines[i].trim();
            let is_docstring = first_body.starts_with("\"\"\"") || first_body.starts_with("'''");
            if is_docstring {
                let quote = &first_body[..3];
                out.push(lines[i].to_string());

                // Check if single-line docstring: """text""" all on one line.
                // first_body[3..] is a string slice starting after the opening quotes.
                let rest = &first_body[3..];
                if rest.contains(quote) {
                    // Single-line — already pushed, move on.
                    i += 1;
                } else {
                    // Multi-line — consume until closing triple-quote.
                    i += 1;
                    while i < lines.len() {
                        out.push(lines[i].to_string());
                        if lines[i].trim().ends_with(quote) {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
            }
        }

        // ── Skip the body ──
        // Body = all lines indented deeper than the def/class line, plus blanks.
        let mut body_lines = 0;
        while i < lines.len() {
            let line = lines[i];
            if line.trim().is_empty() || leading_spaces(line) > indent {
                body_lines += 1;
                i += 1;
            } else {
                break;
            }
        }

        if body_lines > 0 {
            let body_indent = indent + 4;
            let label = if body_lines == 1 { "line" } else { "lines" };
            out.push(format!(
                "{}# ... ({} {})",
                " ".repeat(body_indent),
                body_lines,
                label
            ));
        }
    }

    out.join("\n")
}

// ── Brace-counted languages (JS, TS, Rust) ───────────────────────────────────
//
// For languages that use { } for scoping we:
//   1. Find declaration lines (matched by `is_decl`)
//   2. Keep everything up to and including the opening {
//   3. Count brace depth to find the matching }
//   4. Replace the body with a summary comment

/// Generic signature extractor for brace-delimited languages.
///
/// `is_decl` is a *function pointer* — `fn(&str) -> bool`. Unlike a closure,
/// a function pointer is a concrete type (no generics, no monomorphization).
/// We use it here because the detection logic lives in standalone functions
/// (`is_js_decl`, `is_rust_decl`) and doesn't capture any environment.
///
/// `comment_fmt` is a template like `"/* ... ({}) */"` where `{}` gets replaced
/// with the line count (e.g. `"/* ... (12 lines) */"`).
fn extract_signatures_braces(code: &str, is_decl: fn(&str) -> bool, comment_fmt: &str) -> String {
    let lines: Vec<&str> = code.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        if !is_decl(trimmed) {
            // Not a declaration — pass through (includes doc comments, imports, etc.)
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }

        // ── Found a declaration — consume up to the opening { ──
        let mut depth: i32 = 0;
        out.push(lines[i].to_string());
        for ch in lines[i].chars() {
            if ch == '{' {
                depth += 1;
            }
            if ch == '}' {
                depth -= 1;
            }
        }
        i += 1;

        // If { wasn't on the declaration line, keep consuming (up to 5 lines)
        // to find it. Handles multi-line signatures like:
        //   fn long_name(
        //       arg1: T,
        //   ) -> R {
        if depth == 0 {
            let limit = (i + 5).min(lines.len());
            while i < limit {
                out.push(lines[i].to_string());
                for ch in lines[i].chars() {
                    if ch == '{' {
                        depth += 1;
                    }
                    if ch == '}' {
                        depth -= 1;
                    }
                }
                i += 1;
                if depth > 0 {
                    break; // Found the opening {
                }
            }
        }

        if depth <= 0 {
            // One-liner (e.g. `fn foo() {}`) or no body — already in output.
            continue;
        }

        // ── Skip body lines until matching } ──
        let mut body_lines = 0;
        while i < lines.len() && depth > 0 {
            for ch in lines[i].chars() {
                if ch == '{' {
                    depth += 1;
                }
                if ch == '}' {
                    depth -= 1;
                }
            }
            body_lines += 1;
            i += 1;
        }

        // body_lines includes the closing-} line. The *implementation* is one fewer.
        let impl_lines = if body_lines > 0 { body_lines - 1 } else { 0 };
        if impl_lines > 0 {
            let label = if impl_lines == 1 { "line" } else { "lines" };
            let comment = comment_fmt.replace("{}", &format!("{} {}", impl_lines, label));
            out.push(format!("    {}", comment));
        }
        out.push("}".to_string());
    }

    out.join("\n")
}

// ── Declaration detectors ─────────────────────────────────────────────────────

fn is_js_decl(trimmed: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "function ",
        "async function ",
        "class ",
        "export function ",
        "export async function ",
        "export class ",
        "export default function ",
        "export default class ",
    ];
    KEYWORDS.iter().any(|kw| trimmed.starts_with(kw))
        || ((trimmed.starts_with("const ") || trimmed.starts_with("let "))
            && trimmed.contains("=>"))
}

fn is_rust_decl(trimmed: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "fn ",
        "pub fn ",
        "pub(crate) fn ",
        "pub(super) fn ",
        "async fn ",
        "pub async fn ",
        "unsafe fn ",
        "pub unsafe fn ",
    ];
    KEYWORDS.iter().any(|kw| trimmed.starts_with(kw))
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Markdown Collapse
// ═══════════════════════════════════════════════════════════════════════════════
//
// Split text at markdown headers (lines starting with #). For each section,
// check word overlap with the query. Relevant sections stay verbatim;
// irrelevant ones collapse to: `# Header [collapsed: N paragraphs, ~M tokens]`

fn collapse_markdown(text: &str, query: Option<&str>) -> String {
    // Without a query we can't judge relevance — keep everything.
    let query = match query {
        Some(q) if !q.is_empty() => q,
        _ => return text.to_string(),
    };

    let query_words: HashSet<String> = query
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect();

    // ── Split into sections at # headers ──
    let mut sections: Vec<(String, String)> = Vec::new(); // (header, body)
    let mut cur_header = String::new();
    let mut cur_body = String::new();

    for line in text.lines() {
        if line.trim_start().starts_with('#') {
            // Flush previous section.
            if !cur_header.is_empty() || !cur_body.is_empty() {
                sections.push((
                    std::mem::take(&mut cur_header),
                    std::mem::take(&mut cur_body),
                ));
            }
            cur_header = line.to_string();
        } else {
            if !cur_body.is_empty() {
                cur_body.push('\n');
            }
            cur_body.push_str(line);
        }
    }
    if !cur_header.is_empty() || !cur_body.is_empty() {
        sections.push((cur_header, cur_body));
    }

    // ── Decide per section: keep or collapse ──
    let mut result: Vec<String> = Vec::new();

    for (header, body) in &sections {
        // Combine header + body words for relevance check.
        let section_words: HashSet<String> = format!("{} {}", header, body)
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();

        let overlap = query_words.intersection(&section_words).count();

        if overlap > 0 || header.is_empty() {
            // Relevant — keep verbatim.
            if !header.is_empty() {
                result.push(header.clone());
            }
            if !body.is_empty() {
                result.push(body.clone());
            }
        } else if body.trim().is_empty() {
            // Header with no body — keep as-is (nothing to collapse).
            result.push(header.clone());
        } else {
            // Irrelevant with content — collapse.
            let paragraphs = body.split("\n\n").filter(|p| !p.trim().is_empty()).count();
            // Rough token estimate: word count ÷ 0.75 (average BPE ratio).
            let approx_tokens = (body.split_whitespace().count() as f64 / 0.75).round() as usize;
            result.push(format!(
                "{} [collapsed: {} paragraphs, ~{} tokens]",
                header, paragraphs, approx_tokens
            ));
        }
    }

    result.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Count leading space characters (for Python indent detection).
fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start().len()
}
