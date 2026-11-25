"""
STEP 5: Generate Publication-Ready Research Charts (PNG)

Reads `prefix_cache_results.json` and `research_comparison.json` and generates:
  1. `charts/1_prefix_cache_scaling.png` — Latency & KV Hit % across 1k-6k tokens
  2. `charts/2_model_comparison.png`     — Gemma 4B vs Gemma 12B vs Gemini API

Usage:
  .venv/bin/python step5_generate_charts.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CHARTS_DIR = Path("charts")


def plot_prefix_scaling() -> None:
    data = json.loads(Path("prefix_cache_results.json").read_text(encoding="utf-8"))
    labels = [d["scale_label"].split("(")[0].strip() for d in data]
    naive_s = [d["naive_latency_s"] for d in data]
    cached_s = [d["cached_latency_s"] for d in data]
    agt_hits = [d["cached_agents_2_4_hit_pct"] for d in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        "Cloud TPU v6e-1 (vLLM-TorchTPU, 256-Token KV Blocks) — Multi-Agent Prefix Caching",
        fontsize=12,
        fontweight="bold",
    )

    b1 = ax1.bar(x - width / 2, naive_s, width, label="Naive (Task First — 0% Hit)", color="#d9534f")
    b2 = ax1.bar(x + width / 2, cached_s, width, label="Prefix-Cached (Transcript First)", color="#0b57d0")
    ax1.set_ylabel("4-Agent Pipeline Latency (seconds)")
    ax1.set_title("End-to-End Latency vs. Meeting Length")
    ax1.set_xticks(x, labels)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.bar_label(b1, fmt="%.2fs", padding=2, fontsize=8)
    ax1.bar_label(b2, fmt="%.2fs", padding=2, fontsize=8)

    bars = ax2.plot(labels, agt_hits, marker="o", linewidth=2.5, color="#137333", label="Agents 2–4 KV Hit % (estimated)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Downstream KV Cache Hit Rate (%, est.)")
    ax2.set_title("TPU HBM KV Reuse on Parallel Agents (256-Token Blocks)")
    ax2.grid(True, alpha=0.3)
    for i, val in enumerate(agt_hits):
        ax2.annotate(f"{val:.1f}%", (labels[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    out = CHARTS_DIR / "1_prefix_cache_scaling.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def plot_model_comparison() -> None:
    raw = json.loads(Path("research_comparison.json").read_text(encoding="utf-8"))["models"]
    keys = [k for k in ("tpu_4b", "tpu_12b", "gemini") if k in raw]
    names = [raw[k]["label"].replace(": ", "\n") for k in keys]
    lat = [raw[k]["avg_latency_s"] for k in keys]
    qual = [raw[k]["extraction_completeness_pct"] for k in keys]
    cost = [raw[k]["cost_per_1k_spot_usd"] for k in keys]
    colors = ["#0b57d0", "#137333", "#e67c00"][: len(keys)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.suptitle(
        "5-Meeting Benchmark: Self-Hosted Cloud TPU v6e-1 (Gemma) vs. Hosted Gemini API",
        fontsize=12,
        fontweight="bold",
    )

    b1 = ax1.bar(names, lat, color=colors, width=0.52)
    ax1.set_ylim(0, max(lat) * 1.16)
    ax1.set_title("Avg 4-Agent Latency (Lower = Faster)")
    ax1.set_ylabel("Seconds")
    ax1.grid(axis="y", alpha=0.3)
    ax1.bar_label(b1, fmt="%.2fs", padding=3, fontsize=9)

    b2 = ax2.bar(names, qual, color=colors, width=0.52)
    ax2.set_ylim(0, 115)
    ax2.set_title("Structured Extraction Completeness")
    ax2.set_ylabel("Ground-Truth Coverage (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)

    b3 = ax3.bar(names, cost, color=colors, width=0.52)
    ax3.set_ylim(0, max(cost) * 1.16)
    ax3.set_title("Cost per 1,000 Meetings (USD)")
    ax3.set_ylabel("USD ($)")
    ax3.grid(axis="y", alpha=0.3)
    ax3.bar_label(b3, fmt="$%.2f", padding=3, fontsize=9)

    fig.tight_layout()
    out = CHARTS_DIR / "2_model_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    CHARTS_DIR.mkdir(exist_ok=True)
    plot_prefix_scaling()
    plot_model_comparison()
