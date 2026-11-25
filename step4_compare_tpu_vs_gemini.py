"""
STEP 4: Research Experiment 2 — Multi-Model TPU Tradeoff & Hosted API Benchmark

Evaluates across all 5 synthetic meetings in `meetings/*.txt` (each containing
3 ground-truth decisions, 4 risks, and 4 action items = 11 structured items/meeting):
  - `tpu_4b` : Cloud TPU v6e-1 serving `google/gemma-3-4b-it`
  - `tpu_12b`: Cloud TPU v6e-1 serving `google/gemma-3-12b-it`
  - `gemini` : Hosted Vertex AI Gemini (`gemini-3.5-flash`)

Updates `research_comparison.json` incrementally so each TPU model can be tested
one at a time on the same single 32 GB `v6e-1` chip.

Usage:
  .venv/bin/python step4_compare_tpu_vs_gemini.py --target tpu_4b
  .venv/bin/python step4_compare_tpu_vs_gemini.py --target tpu_12b
  .venv/bin/python step4_compare_tpu_vs_gemini.py --target gemini
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
import httpx
from step2_adk_pipeline import run_pipeline

TPU_V6E_1_SPOT_USD_PER_HOUR = 0.85
TPU_V6E_1_ONDEMAND_USD_PER_HOUR = 2.70
GEMINI_FLASH_USD_PER_1K_MEETINGS = 3.20
METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")

TARGETS = {
    "tpu_4b": ("tpu", "google/gemma-3-4b-it", "TPU v6e-1: Gemma 3 4B"),
    "tpu_12b": ("tpu", "google/gemma-3-12b-it", "TPU v6e-1: Gemma 3 12B"),
    "gemini": ("gemini", "gemini-3.5-flash", "Hosted API: Gemini 3.5 Flash"),
}


def read_tpu_kv_capacity_tokens() -> int | None:
    try:
        text = httpx.get(METRICS_URL, timeout=2.0).text
        m = re.search(r'kv_cache_size_tokens="(\d+)"', text)
        return int(m.group(1)) if m else None
    except Exception:
        return None


async def evaluate_target(target_key: str) -> dict:
    mode, model_id, label = TARGETS[target_key]
    meeting_files = sorted(Path("meetings").glob("*.txt"))

    latencies = []
    completeness_scores = []
    valid_json_runs = 0
    per_meeting = []

    # Warmup first meeting once on TPU so XLA prefill shape is hot
    if mode == "tpu" and meeting_files:
        await run_pipeline(
            meeting_files[0].read_text(encoding="utf-8"),
            mode=mode,
            model_id=model_id,
        )

    for mf in meeting_files:
        text = mf.read_text(encoding="utf-8")
        try:
            res = await run_pipeline(text, mode=mode, transcript_first=True, model_id=model_id)
        except Exception:
            # Retry once if SSH tunnel keep-alive socket resets between meetings
            res = await run_pipeline(text, mode=mode, transcript_first=True, model_id=model_id)
        latencies.append(res["latency_seconds"])

        d_cnt = len(res.get("decisions", []))
        r_cnt = len(res.get("risks", []))
        a_cnt = len(res.get("business_actions", []))
        i_cnt = len(res.get("key_insights", []))

        # Each synthetic meeting has 3 decisions, 4 risks, and 4 actions
        score = (min(d_cnt, 3) / 3 + min(r_cnt, 4) / 4 + min(a_cnt, 4) / 4) / 3.0
        completeness_scores.append(score)
        if i_cnt > 0 and d_cnt > 0 and r_cnt > 0 and a_cnt > 0:
            valid_json_runs += 1

        per_meeting.append(
            {
                "meeting": mf.stem,
                "latency_s": res["latency_seconds"],
                "insights": i_cnt,
                "decisions": d_cnt,
                "risks": r_cnt,
                "actions": a_cnt,
            }
        )
        print(
            f"  [{label}] {mf.name}: {res['latency_seconds']}s | "
            f"D={d_cnt}/3 R={r_cnt}/4 A={a_cnt}/4"
        )

    n = max(len(meeting_files), 1)
    avg_sec = round(sum(latencies) / n, 2)
    avg_completeness_pct = round((sum(completeness_scores) / n) * 100, 1)
    schema_valid_pct = round((valid_json_runs / n) * 100, 1)

    if mode == "tpu":
        meetings_per_hr = 3600.0 / max(avg_sec, 0.1)
        spot_cost_1k = round((TPU_V6E_1_SPOT_USD_PER_HOUR / meetings_per_hr) * 1000, 2)
        ondemand_cost_1k = round((TPU_V6E_1_ONDEMAND_USD_PER_HOUR / meetings_per_hr) * 1000, 2)
        kv_tokens = read_tpu_kv_capacity_tokens()
        privacy = "100% Private VPC"
    else:
        spot_cost_1k = GEMINI_FLASH_USD_PER_1K_MEETINGS
        ondemand_cost_1k = GEMINI_FLASH_USD_PER_1K_MEETINGS
        kv_tokens = None
        privacy = "External Cloud API"

    return {
        "target": target_key,
        "label": label,
        "model_id": model_id,
        "meetings_tested": len(meeting_files),
        "avg_latency_s": avg_sec,
        "schema_valid_pct": schema_valid_pct,
        "extraction_completeness_pct": avg_completeness_pct,
        "tpu_kv_cache_capacity_tokens": kv_tokens,
        "cost_per_1k_spot_usd": spot_cost_1k,
        "cost_per_1k_ondemand_usd": ondemand_cost_1k,
        "privacy": privacy,
        "per_meeting": per_meeting,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["tpu_4b", "tpu_12b", "gemini"],
        required=True,
    )
    args = parser.parse_args()

    out_path = Path("research_comparison.json")
    existing = {}
    if out_path.exists():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "models" in loaded:
                existing = loaded["models"]
        except Exception:
            existing = {}

    summary = await evaluate_target(args.target)
    existing[args.target] = summary
    out_path.write_text(json.dumps({"models": existing}, indent=2), encoding="utf-8")

    print("\n=== Multi-Model Benchmark Summary (5 Enterprise Meetings) ===")
    print(
        f"{'Model':<30} | {'Avg Latency':>11} | {'Completeness':>12} | "
        f"{'TPU KV Cap':>12} | {'Spot $/1k':>9} | {'Privacy'}"
    )
    print("-" * 102)
    for k in ("tpu_4b", "tpu_12b", "gemini"):
        if k not in existing:
            continue
        m = existing[k]
        kv_str = f"{m['tpu_kv_cache_capacity_tokens']:,}" if m["tpu_kv_cache_capacity_tokens"] else "Managed"
        print(
            f"{m['label']:<30} | {str(m['avg_latency_s']) + 's':>11} | "
            f"{str(m['extraction_completeness_pct']) + '%':>12} | {kv_str:>12} | "
            f"{'$' + str(m['cost_per_1k_spot_usd']):>9} | {m['privacy']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
