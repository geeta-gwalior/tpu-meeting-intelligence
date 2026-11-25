"""
STEP 6: Research Experiment 3 — Concurrency / Throughput on a Single Cloud TPU v6e-1

Question: TPU ki asli taakat batching hai. Agar bahut saari meetings ek saath aayein,
toh ek hi v6e-1 chip kitni meetings per hour process kar sakti hai, aur per-meeting cost
kitni girti hai?

Method:
  - Same 4-agent ADK pipeline (step2), Gemma 3 4B on vLLM-TorchTPU.
  - Har concurrency level (1, 4, 8, 16) par 32 meetings process ki jaati hain, jisme
    ek waqt par sirf `concurrency` meetings in-flight hoti hain.
  - Har meeting ke shuru mein unique ID lagaya jaata hai, taaki alag meetings ek doosre
    ka prefix cache reuse na karein (real-world jaisa, koi cheating nahi).
  - vLLM /metrics se prompt/generation tokens aur prefix-cache hits measure hote hain.

Outputs: `concurrency_results.json` and `charts/3_concurrency_throughput.png`

Usage (TPU tunnel on :8000 must be live):
  .venv/bin/python step6_concurrency_benchmark.py
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
from step2_adk_pipeline import run_pipeline

METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
TPU_V6E_1_SPOT_USD_PER_HOUR = 0.85
TPU_V6E_1_ONDEMAND_USD_PER_HOUR = 2.70


def read_counters() -> dict[str, float]:
    text = httpx.get(METRICS_URL, timeout=5.0).text
    vals = {"prompt": 0.0, "gen": 0.0, "q": 0.0, "h": 0.0}
    names = {
        "vllm:prompt_tokens_total": "prompt",
        "vllm:generation_tokens_total": "gen",
        "vllm:prefix_cache_queries_total": "q",
        "vllm:prefix_cache_hits_total": "h",
    }
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in names:
            vals[names[name]] += float(line.split()[-1])
    return vals


async def run_level(concurrency: int, meetings: list[str], total: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0
    retries = 0
    errors: list[str] = []

    async def one(i: int) -> None:
        nonlocal failures, retries
        text = f"[Meeting-ID {uuid.uuid4().hex[:10]}]\n{meetings[i % len(meetings)]}"
        async with sem:
            for attempt in range(2):
                try:
                    res = await run_pipeline(text, mode="tpu", transcript_first=True)
                    latencies.append(res["latency_seconds"])
                    return
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
                    if attempt == 0:
                        # SSH tunnel can drop an idle keep-alive socket; retry once
                        retries += 1
                    else:
                        failures += 1

    before = read_counters()
    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(total)))
    wall_s = time.perf_counter() - t0
    after = read_counters()

    done = len(latencies)
    meetings_per_hour = (done / wall_s) * 3600.0 if wall_s > 0 else 0.0
    d_prompt = after["prompt"] - before["prompt"]
    d_gen = after["gen"] - before["gen"]
    d_q = after["q"] - before["q"]
    d_h = after["h"] - before["h"]

    return {
        "concurrency": concurrency,
        "meetings_completed": done,
        "failures": failures,
        "retries": retries,
        "error_samples": sorted(set(errors))[:3],
        "wall_time_s": round(wall_s, 2),
        "meetings_per_hour": round(meetings_per_hour, 1),
        "avg_meeting_latency_s": round(sum(latencies) / max(done, 1), 2),
        "max_meeting_latency_s": round(max(latencies), 2) if latencies else 0.0,
        "prompt_tokens_per_s": round(d_prompt / wall_s, 1),
        "generation_tokens_per_s": round(d_gen / wall_s, 1),
        "prefix_cache_hit_pct": round((d_h / d_q) * 100, 1) if d_q > 0 else 0.0,
        "cost_per_1k_meetings_spot_usd": round(TPU_V6E_1_SPOT_USD_PER_HOUR / meetings_per_hour * 1000, 3) if meetings_per_hour else None,
        "cost_per_1k_meetings_ondemand_usd": round(TPU_V6E_1_ONDEMAND_USD_PER_HOUR / meetings_per_hour * 1000, 3) if meetings_per_hour else None,
    }


def plot(results: list[dict]) -> None:
    conc = [str(r["concurrency"]) for r in results]
    mph = [r["meetings_per_hour"] for r in results]
    lat = [r["avg_meeting_latency_s"] for r in results]
    cost = [r["cost_per_1k_meetings_ondemand_usd"] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.suptitle(
        "Single Cloud TPU v6e-1 (Gemma 3 4B, vLLM-TorchTPU) — Throughput vs. Concurrent Meetings",
        fontsize=12,
        fontweight="bold",
    )

    b1 = ax1.bar(conc, mph, color="#0b57d0", width=0.55)
    ax1.set_title("Meetings Processed per Hour (Higher = Better)")
    ax1.set_xlabel("Meetings In-Flight at Once")
    ax1.set_ylim(0, max(mph) * 1.18)
    ax1.grid(axis="y", alpha=0.3)
    ax1.bar_label(b1, fmt="%.0f", padding=3, fontsize=9)

    ax2.plot(conc, lat, marker="o", linewidth=2.5, color="#e67c00")
    ax2.set_title("Avg Latency per Meeting (4 Agents)")
    ax2.set_xlabel("Meetings In-Flight at Once")
    ax2.set_ylabel("Seconds")
    ax2.set_ylim(0, max(lat) * 1.25)
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(lat):
        ax2.annotate(f"{v:.1f}s", (conc[i], v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    b3 = ax3.bar(conc, cost, color="#137333", width=0.55)
    ax3.set_title("TPU Cost / 1,000 Meetings (On-Demand)")
    ax3.set_xlabel("Meetings In-Flight at Once")
    ax3.set_ylabel("USD ($)")
    ax3.set_ylim(0, max(cost) * 1.18)
    ax3.grid(axis="y", alpha=0.3)
    ax3.bar_label(b3, fmt="$%.2f", padding=3, fontsize=9)

    fig.tight_layout()
    Path("charts").mkdir(exist_ok=True)
    out = Path("charts/3_concurrency_throughput.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,4,8,16")
    parser.add_argument("--meetings-per-level", type=int, default=32)
    parser.add_argument("--plot-only", action="store_true", help="Re-draw chart from concurrency_results.json")
    args = parser.parse_args()

    if args.plot_only:
        plot(json.loads(Path("concurrency_results.json").read_text(encoding="utf-8")))
        return

    levels = [int(x) for x in args.levels.split(",")]
    meetings = [f.read_text(encoding="utf-8") for f in sorted(Path("meetings").glob("*.txt"))]

    print("Warmup (not recorded): 8 meetings at concurrency 8 ...")
    await run_level(8, meetings, 8)

    results = []
    for c in levels:
        r = await run_level(c, meetings, args.meetings_per_level)
        results.append(r)
        print(
            f"Concurrency {c:>2} | {r['meetings_completed']}/{args.meetings_per_level} ok "
            f"(retries {r['retries']}) | "
            f"wall {r['wall_time_s']:>6.1f}s | {r['meetings_per_hour']:>7.1f} meetings/hr | "
            f"avg {r['avg_meeting_latency_s']:>5.2f}s/meeting | gen {r['generation_tokens_per_s']:>6.1f} tok/s | "
            f"KV hit {r['prefix_cache_hit_pct']:>4.1f}% | ${r['cost_per_1k_meetings_ondemand_usd']}/1k (on-demand)"
        )

    Path("concurrency_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved concurrency_results.json")
    plot(results)


if __name__ == "__main__":
    asyncio.run(main())
