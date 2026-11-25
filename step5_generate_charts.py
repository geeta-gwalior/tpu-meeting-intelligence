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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

CHARTS_DIR = Path("charts")


def plot_architecture() -> None:
    """Draw a clean, Medium-ready system architecture diagram PNG."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        6.5, 4.85,
        "Private Meeting Intelligence: 2-Stage Google ADK Pipeline on Cloud TPU v6e-1",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111827",
    )

    def box(x, y, w, h, title, body, fc, ec, title_color="#111827"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
                fontsize=9.8, fontweight="bold", color=title_color)
        if body:
            ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center",
                    fontsize=8.3, color="#374151", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", color="#374151", rad=0.0, label_dy=0.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35",
                            color=color, lw=1.6, connectionstyle=f"arc3,rad={rad}"),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + label_dy, label, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.95))

    # 1. Input Transcript
    box(0.3, 2.35, 2.1, 1.85, "1. Meeting Input",
        "Confidential transcript\n(M&A, HIPAA, Security)\n\nStays 100% private",
        "#f3f4f6", "#6b7280")

    # 2. Stage 1: InsightsAgent
    box(3.0, 2.55, 2.3, 1.45, "2. Stage 1 (Warm Cache)",
        "InsightsAgent\nRuns first on its own\nWarms 256-token KV blocks",
        "#e8f0fe", "#0b57d0", "#0b57d0")

    # 3. Stage 2: Parallel Agents container
    p_rect = patches.FancyBboxPatch(
        (5.9, 1.95), 3.1, 2.55, boxstyle="round,pad=0.08,rounding_size=0.14",
        facecolor="#e6f4ea", edgecolor="#137333", linewidth=1.6, linestyle="--",
    )
    ax.add_patch(p_rect)
    ax.text(7.45, 4.30, "3. Stage 2 (Parallel Fan-Out)", ha="center", va="center",
            fontsize=9.6, fontweight="bold", color="#137333")
    box(6.1, 3.45, 2.7, 0.58, "DecisionsAgent", "Approved / Rejected / Deferred", "white", "#137333")
    box(6.1, 2.75, 2.7, 0.58, "RisksAgent", "Category + Severity", "white", "#137333")
    box(6.1, 2.05, 2.7, 0.58, "ActionsAgent", "Owner + Deadline", "white", "#137333")

    # 4. Output JSON
    box(9.65, 2.45, 2.95, 1.65, "4. Structured JSON",
        "Validated Pydantic output:\n• Executive summary & insights\n• Decisions, risks, action items",
        "#fef7e0", "#b06000", "#b06000")

    # 5. Cloud TPU v6e-1 Box at bottom
    box(2.5, 0.25, 8.0, 1.15,
        "Private Cloud TPU v6e-1 (32 GB HBM) — vLLM-TorchTPU + Gemma 3 (4B / 12B)",
        "Bound to 127.0.0.1:8000 (SSH / VPC only)  •  256-Token KV Prefix Cache (fp8_e5m2)  •  Up to 69% overall / ~92% downstream reuse",
        "#f8fafc", "#1e3a8a", "#1e3a8a")

    # Flow arrows
    arrow(2.45, 3.28, 2.95, 3.28)
    arrow(5.35, 3.28, 5.85, 3.28)
    arrow(9.05, 3.28, 9.60, 3.28)

    # TPU connection arrows
    arrow(4.15, 2.47, 4.15, 1.48, "Writes KV blocks", "#0b57d0", label_dy=0.0)
    arrow(7.45, 1.87, 7.45, 1.48, "Reuses cached KV blocks", "#137333", label_dy=-0.02)

    fig.tight_layout()
    out = CHARTS_DIR / "0_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


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
    plot_architecture()
    plot_prefix_scaling()
    plot_model_comparison()
