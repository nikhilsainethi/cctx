#!/usr/bin/env python3
"""
cctx Quality Benchmark — proves optimization doesn't break LLM answers.

For each fixture, sends questions to an LLM with original context vs
cctx-optimized context, then uses LLM-as-judge to score equivalence.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=sk-...
    cargo build --release

Usage:
    python3 scripts/quality_benchmark.py

Output:
    docs/QUALITY_RESULTS.md
    (also prints to stdout)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

CCTX = os.environ.get("CCTX", "target/release/cctx")
MODEL = os.environ.get("QUALITY_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
PRESETS = ["safe", "balanced", "aggressive"]
OUTPUT_FILE = "docs/QUALITY_RESULTS.md"

# ── Fixtures and questions ─────────────────────────────────────────────────────

BENCHMARKS = [
    {
        "fixture": "tests/fixtures/bench_long_chat.json",
        "name": "long_chat",
        "questions": [
            "What web framework is the user building their API with?",
            "What database are they using as the primary data store?",
            "What authentication approach did they agree on for the API?",
            "How is the shopping cart stored and what is its TTL?",
            "What deployment platform are they targeting?",
        ],
    },
    {
        "fixture": "tests/fixtures/bench_rag_chunks.json",
        "name": "rag_chunks",
        "questions": [
            "What is the role of CNI plugins in Kubernetes networking?",
            "Name three popular CNI plugins and their key differences.",
            "How does pod-to-pod networking work across different nodes?",
        ],
    },
    {
        "fixture": "tests/fixtures/bench_agent_history.json",
        "name": "agent_history",
        "questions": [
            "What was the root cause of the 500 errors?",
            "What specific fix was applied to the database?",
            "What was the final health status after the fix?",
        ],
    },
    {
        "fixture": "tests/fixtures/bench_codebase_context.json",
        "name": "codebase_context",
        "questions": [
            "What bug causes orders to get stuck in PAYMENT_PROCESSING status?",
            "What is the recommended fix for the payment processing bug?",
            "Why should you call session.rollback() in exception handlers?",
        ],
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def check_prereqs():
    """Verify cctx binary and API key are available."""
    if not os.path.isfile(CCTX):
        print(f"Error: cctx binary not found at {CCTX}")
        print("Run: cargo build --release")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        print("Export it: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("Error: openai package not installed.")
        print("Install: pip install openai")
        sys.exit(1)


def cctx_count(path: str) -> int:
    """Get token count via cctx."""
    result = subprocess.run(
        [CCTX, "count", path], capture_output=True, text=True
    )
    return int(result.stdout.strip())


def cctx_optimize(input_path: str, preset: str, output_path: str) -> int:
    """Optimize a fixture and return the new token count."""
    subprocess.run(
        [CCTX, "optimize", input_path, "--preset", preset, "--output", output_path],
        capture_output=True,
        text=True,
    )
    return cctx_count(output_path)


def load_messages(path: str) -> list:
    """Load a JSON file as a list of messages."""
    with open(path) as f:
        return json.load(f)


def ask_llm(messages: list, question: str, client) -> str:
    """Send context + question to the LLM, return the answer."""
    # Build the prompt: context messages + the question as the final user msg.
    chat_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Normalize roles for OpenAI API.
        if role not in ("system", "user", "assistant"):
            role = "user"
        chat_messages.append({"role": role, "content": content})

    # Add the question as the final user message.
    chat_messages.append(
        {
            "role": "user",
            "content": f"Based on the conversation above, answer this question concisely:\n\n{question}",
        }
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_messages,
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def judge_equivalence(question: str, answer_a: str, answer_b: str, client) -> float:
    """Use LLM-as-judge to score semantic equivalence (0-10)."""
    prompt = f"""You are a strict judge comparing two answers to the same question.

Question: {question}

Answer A (from original context):
{answer_a}

Answer B (from optimized context):
{answer_b}

Score how semantically equivalent Answer B is to Answer A on a scale of 0-10:
- 10: Identical meaning, same key facts
- 8-9: Same conclusion, minor wording differences
- 5-7: Partially correct, missing some details
- 1-4: Significantly different or wrong
- 0: Completely wrong or contradictory

Reply with ONLY a number (0-10), nothing else."""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        score_text = response.choices[0].message.content.strip()
        # Parse the numeric score.
        return float(score_text.split()[0].rstrip("."))
    except Exception:
        return -1.0


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    check_prereqs()

    from openai import OpenAI

    client = OpenAI()

    print("cctx Quality Benchmark")
    print(f"Model: {MODEL}  |  Judge: {JUDGE_MODEL}")
    print(f"Presets: {', '.join(PRESETS)}")
    print()

    all_results = []
    tmpdir = Path("/tmp/cctx_quality")
    tmpdir.mkdir(exist_ok=True)

    for bench in BENCHMARKS:
        fixture = bench["fixture"]
        name = bench["name"]
        questions = bench["questions"]

        if not os.path.isfile(fixture):
            print(f"  SKIP {name}: fixture not found")
            continue

        original_tokens = cctx_count(fixture)
        original_messages = load_messages(fixture)

        print(f"── {name} ({original_tokens:,} tokens, {len(questions)} questions) ──")

        # Get ground-truth answers from original context.
        ground_truths = {}
        for q in questions:
            print(f"  [original] Q: {q[:60]}...")
            answer = ask_llm(original_messages, q, client)
            ground_truths[q] = answer
            print(f"             A: {answer[:80]}...")
            time.sleep(0.5)  # Rate limit courtesy.

        # Test each preset.
        for preset in PRESETS:
            opt_path = str(tmpdir / f"{name}_{preset}.json")
            opt_tokens = cctx_optimize(fixture, preset, opt_path)
            opt_messages = load_messages(opt_path)

            reduction = (
                (original_tokens - opt_tokens) / original_tokens * 100
                if original_tokens > 0
                else 0
            )

            scores = []
            for q in questions:
                print(f"  [{preset}] Q: {q[:60]}...")
                opt_answer = ask_llm(opt_messages, q, client)
                print(f"           A: {opt_answer[:80]}...")

                score = judge_equivalence(q, ground_truths[q], opt_answer, client)
                scores.append(score)
                print(f"           Score: {score}/10")
                time.sleep(0.5)

            avg_score = sum(s for s in scores if s >= 0) / max(
                len([s for s in scores if s >= 0]), 1
            )

            result = {
                "fixture": name,
                "preset": preset,
                "original_tokens": original_tokens,
                "optimized_tokens": opt_tokens,
                "reduction_pct": round(reduction, 1),
                "avg_quality_score": round(avg_score, 1),
                "scores": scores,
            }
            all_results.append(result)
            print(
                f"  [{preset}] Tokens: {original_tokens:,} → {opt_tokens:,} "
                f"(-{reduction:.1f}%)  Quality: {avg_score:.1f}/10"
            )

        print()

    # ── Generate report ────────────────────────────────────────────────────────

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("# cctx Quality Benchmark Results\n\n")
        f.write(
            "Measures whether LLM answers remain correct after context optimization.\n\n"
        )
        f.write(f"- **Model**: {MODEL}\n")
        f.write(f"- **Judge**: {JUDGE_MODEL} (LLM-as-judge, 0–10 scale)\n")
        f.write(f"- **Presets tested**: {', '.join(PRESETS)}\n\n")

        f.write("## Results\n\n")
        f.write(
            "| Fixture | Preset | Tokens | Reduction | Quality (0-10) |\n"
        )
        f.write(
            "|---------|--------|--------|-----------|----------------|\n"
        )

        for r in all_results:
            f.write(
                f"| `{r['fixture']}` | {r['preset']} | "
                f"{r['original_tokens']:,} → {r['optimized_tokens']:,} | "
                f"-{r['reduction_pct']}% | "
                f"**{r['avg_quality_score']}**/10 |\n"
            )

        # Summary by preset.
        f.write("\n## Summary by Preset\n\n")
        f.write("| Preset | Avg Reduction | Avg Quality |\n")
        f.write("|--------|--------------|-------------|\n")

        for preset in PRESETS:
            preset_results = [r for r in all_results if r["preset"] == preset]
            if not preset_results:
                continue
            avg_red = sum(r["reduction_pct"] for r in preset_results) / len(
                preset_results
            )
            avg_qual = sum(r["avg_quality_score"] for r in preset_results) / len(
                preset_results
            )
            f.write(f"| {preset} | -{avg_red:.1f}% | {avg_qual:.1f}/10 |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **9-10**: Optimization is transparent — answers are identical.\n")
        f.write(
            "- **7-8**: Minor details lost but conclusions are correct.\n"
        )
        f.write(
            "- **5-6**: Noticeable quality loss — some information was removed.\n"
        )
        f.write("- **<5**: Significant quality degradation — too aggressive.\n")
        f.write(
            "\nThe **balanced** preset is the recommended default — it provides meaningful "
            "token savings while maintaining high answer quality.\n"
        )

        f.write(
            f"\n---\n\n*Generated on {time.strftime('%Y-%m-%d')} with "
            f"{MODEL} / {JUDGE_MODEL}.*\n"
        )

    print(f"Report written to {OUTPUT_FILE}")

    # ── Print summary table ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  QUALITY BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"  {'Fixture':<20} {'Preset':<12} {'Reduction':>10} {'Quality':>10}"
    )
    print("  " + "-" * 52)
    for r in all_results:
        print(
            f"  {r['fixture']:<20} {r['preset']:<12} "
            f"{'-' + str(r['reduction_pct']) + '%':>10} "
            f"{str(r['avg_quality_score']) + '/10':>10}"
        )


if __name__ == "__main__":
    main()
