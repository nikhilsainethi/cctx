//! Live-updating terminal dashboard for the proxy.
//!
//! Renders a box to stderr every few seconds with current stats.
//! Uses ANSI escape codes to overwrite previous output in place:
//!   - \x1b[{n}A  — move cursor up n lines
//!   - \x1b[2K    — clear entire line
//!
//! When --dashboard is NOT set, the proxy uses per-request log lines instead.

use std::sync::Arc;

use super::metrics::Metrics;

/// Number of lines the dashboard renders (for cursor-up calculation).
const DASHBOARD_LINES: usize = 9;

/// Render one frame of the dashboard to stderr. On subsequent calls,
/// moves the cursor up to overwrite the previous frame.
pub fn render(metrics: &Arc<Metrics>, first: bool) {
    let snap = metrics.snapshot();

    let w = 51;

    // Move cursor up to overwrite previous frame (skip on first render).
    if !first {
        eprint!("\x1b[{}A", DASHBOARD_LINES);
    }

    let clear_line = "\x1b[2K"; // ANSI: clear entire current line

    // Format uptime as HHh MMm SSs.
    let h = snap.uptime_seconds / 3600;
    let m = (snap.uptime_seconds % 3600) / 60;
    let s = snap.uptime_seconds % 60;
    let uptime = format!("{:02}h {:02}m {:02}s", h, m, s);

    let saved_pct = if snap.tokens.input_original > 0 {
        format!(
            "{:.1}%",
            snap.tokens.saved as f64 / snap.tokens.input_original as f64 * 100.0
        )
    } else {
        "—".to_string()
    };

    let last_req = match &snap.last_request {
        Some(lr) => {
            if lr.original_tokens > 0 {
                let pct = (lr.original_tokens as f64 - lr.optimized_tokens as f64)
                    / lr.original_tokens as f64
                    * 100.0;
                format!(
                    "{} | {}->{}  ({:.1}%)",
                    lr.model, lr.original_tokens, lr.optimized_tokens, -pct
                )
            } else {
                format!("{} | passthrough", lr.model)
            }
        }
        None => "none yet".to_string(),
    };

    eprintln!("{clear_line}╭{}╮", "─".repeat(w));
    eprintln!(
        "{clear_line}│  {:<width$}│",
        "cctx proxy — live stats",
        width = w - 2
    );
    eprintln!("{clear_line}├{}┤", "─".repeat(w));
    eprintln!(
        "{clear_line}│  {:<width$}│",
        format!(
            "Uptime: {}    Requests: {}",
            uptime,
            format_number(snap.requests.total)
        ),
        width = w - 2
    );
    eprintln!(
        "{clear_line}│  {:<width$}│",
        format!(
            "Tokens saved: {} ({})",
            format_number(snap.tokens.saved),
            saved_pct
        ),
        width = w - 2
    );
    eprintln!(
        "{clear_line}│  {:<width$}│",
        format!("Est. cost saved: ${:.2}", snap.cost.estimated_saved_usd),
        width = w - 2
    );
    eprintln!(
        "{clear_line}│  {:<width$}│",
        format!(
            "Avg optimization latency: {:.1}ms",
            snap.avg_optimize_latency_ms
        ),
        width = w - 2
    );
    eprintln!(
        "{clear_line}│  {:<width$}│",
        format!("Last: {}", truncate(&last_req, w - 10)),
        width = w - 2
    );
    eprintln!("{clear_line}╰{}╯", "─".repeat(w));
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let chars: Vec<char> = s.chars().collect();
    let mut r = String::with_capacity(s.len() + s.len() / 3);
    for (i, &ch) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            r.push(',');
        }
        r.push(ch);
    }
    r
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}
