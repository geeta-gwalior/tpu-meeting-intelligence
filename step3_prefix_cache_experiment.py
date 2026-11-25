"""
STEP 3: Research Experiment 1 — Prefix Caching Across Context Lengths (1k to 6k Tokens)

Tests how Cloud TPU v6e's 256-token KV blocks scale as meeting transcripts grow
from short 10-minute standups (~1k tokens) to full 1-hour executive reviews (~6k tokens),
comparing Naive (`Task-First`) vs. Prefix-Cached (`Transcript-First`) agent prompts.

Saves results to `prefix_cache_results.json`.
"""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
import httpx
from step2_adk_pipeline import run_pipeline

METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")


def read_vllm_cache_counters() -> tuple[float, float] | None:
    try:
        text = httpx.get(METRICS_URL, timeout=2.0).text
    except Exception:
        return None

    queries = hits = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if re.match(r"^vllm:prefix_cache_queries(_total)?\b", line):
            queries += float(line.split()[-1])
        elif re.match(r"^vllm:prefix_cache_hits(_total)?\b", line):
            hits += float(line.split()[-1])
    return queries, hits


async def main() -> None:
    mode = os.getenv("MODE", "tpu")
    base = Path("sample_meeting.txt").read_text(encoding="utf-8")

    # Multipliers create ~1k, ~2k, ~4k, and ~6k token transcripts (within 8,192 max_model_len)
    scales = [
        ("~1k tokens (10-min sync)", 1),
        ("~2k tokens (20-min sync)", 2),
        ("~4k tokens (40-min review)", 4),
        ("~6k tokens (60-min board)", 6),
    ]

    all_results = []
    for scale_label, mult in scales:
        transcript = "\n\n".join([base] * mult)

        # Warmup TPU XLA prefill shape for this length once before timing
        await run_pipeline(
            f"[Warmup-{uuid.uuid4().hex[:6]}]\n{transcript}",
            mode=mode,
            transcript_first=True,
        )

        entry = {"scale_label": scale_label, "multiplier": mult}
        for key, transcript_first in [("naive", False), ("cached", True)]:
            trial_text = f"[Trial-{uuid.uuid4().hex[:8]}]\n{transcript}"
            before = read_vllm_cache_counters() if mode == "tpu" else None
            out = await run_pipeline(trial_text, mode=mode, transcript_first=transcript_first)
            after = read_vllm_cache_counters() if mode == "tpu" else None

            q_tok = int(after[0] - before[0]) if (before and after) else 0
            h_tok = int(after[1] - before[1]) if (before and after) else 0
            overall_hit_pct = round((h_tok / q_tok) * 100, 1) if q_tok > 0 else 0.0
            # Agent 1 warms cache (0 hits); Agents 2-4 receive all `h_tok` hits
            downstream_hit_pct = round((h_tok / (q_tok * 0.75)) * 100, 1) if q_tok > 0 else 0.0

            entry[f"{key}_latency_s"] = out["latency_seconds"]
            entry[f"{key}_overall_hit_pct"] = overall_hit_pct
            entry[f"{key}_agents_2_4_hit_pct"] = min(100.0, downstream_hit_pct)
            entry[f"{key}_hit_tokens"] = h_tok
            entry[f"{key}_query_tokens"] = q_tok

        entry["speedup_x"] = round(
            entry["naive_latency_s"] / max(entry["cached_latency_s"], 0.01), 2
        )
        all_results.append(entry)
        print(
            f"{scale_label:<28} | Naive: {entry['naive_latency_s']:>5.2f}s ({entry['naive_overall_hit_pct']:>4.1f}%) "
            f"| Cached: {entry['cached_latency_s']:>5.2f}s ({entry['cached_overall_hit_pct']:>4.1f}% total / "
            f"{entry['cached_agents_2_4_hit_pct']:>4.1f}% Agt2-4) | Speedup: {entry['speedup_x']}x"
        )

    Path("prefix_cache_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print("\nSaved scaling results to prefix_cache_results.json")


if __name__ == "__main__":
    asyncio.run(main())
