# Blog Tutorial & Video Walkthrough Guide

## Blog draft: Private meeting intelligence on Cloud TPU v6e with vLLM-TorchTPU and Google ADK

### 1. The problem
Board reviews, M&A due diligence, clinical syncs, and security postmortems are some of the most sensitive conversations a company has. Many compliance teams won't allow these transcripts to go to a third-party API.

Meeting intelligence is also not one prompt. It's several agents: one writes the summary, one pulls out decisions, one finds risks, one lists action items. If each agent reads the same long transcript from scratch, the hardware does the same work four times.

This project puts both pieces together:
- **Gemma 3 (4B and 12B)** served with **`vllm-torchtpu` 0.29.0** on a single **Cloud TPU v6e-1**, inside our own GCP project.
- **Google ADK** running four `LlmAgent`s against that endpoint through `LiteLlm`.
- A prompt layout and agent order that lets the agents **reuse the transcript's KV cache** on the TPU.

### 2. From zero to a live Gemma endpoint (`step1_tpu_setup_and_serve.sh`)
The first surprise was the Python version. The `v2-alpha-tpuv6e` image ships Python 3.10.12, while `vllm-torchtpu` wheels need 3.12. `uv` solves it in under a minute:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/vllm-env
uv pip install --python ~/vllm-env/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  vllm-torchtpu huggingface_hub

~/vllm-env/bin/vllm serve google/gemma-3-4b-it \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 \
  --enable-prefix-caching
```

The server only listens on `127.0.0.1`. The laptop reaches it through an SSH tunnel (`-L 8000:127.0.0.1:8000`), so nothing is exposed publicly.

### 3. Designing the agents for TPU prefix caching (`step2_adk_pipeline.py`)
The `/metrics` endpoint showed two TPU-specific details:
- KV blocks are **256 tokens** (GPU vLLM uses 16).
- **`fp8_e5m2` KV cache** is turned on automatically.

Two easy mistakes kill cache reuse:
1. **A different instruction at the start of each prompt.** The very first block differs between agents, so nothing can be reused (we measured 0%).
2. **Starting all agents at once on a cold cache.** They all compute the transcript before any of it is cached.

The fix:
- Every agent's instruction starts with the same preamble + full transcript and ends with `YOUR TASK: ...`.
- A `SequentialAgent` runs `InsightsAgent` first, then a `ParallelAgent` runs `DecisionsAgent`, `RisksAgent`, and `ActionsAgent` against the warm cache.

### 4. Results

**Experiment 1: prefix caching vs. meeting length (`charts/1_prefix_cache_scaling.png`)**
- With the task first, cache reuse was 0% at every length.
- With the transcript first, overall reuse measured from vLLM's counters was 45% (~1k tokens), 61% (~2k), 67% (~4k), and 69% (~6k).
- On the three parallel agents alone, we estimate ~60% → ~92%.
- End-to-end, the 4-agent run was up to 1.29x faster.

**Experiment 2: 4B vs. 12B on TPU vs. hosted Gemini (`charts/2_model_comparison.png`)**, 5 meetings, 55 expected items:
- **Gemma 3 4B on v6e-1:** 3.99 s per meeting, 54/55 items, 213K tokens of KV cache. That is about 4x faster than the hosted API's median of 16.2 s.
- **Gemma 3 12B on the same single chip:** 11.91 s, 55/55 items, 20.8K tokens of KV cache left.
- **Hosted Gemini 3.5 Flash:** 55/55 items, 16.2 s median, with one 84.9 s outlier.
- Note: these item counts check how many items came back, not whether each one is correct.

**Experiment 3: many meetings at once (`charts/3_concurrency_throughput.png`)**
- This is where the TPU earns its place. Running 16 meetings at a time instead of 1 raised throughput from **915 to 8,686 meetings per hour (~9.5x)** on the same chip.
- Each meeting only got slower by 2.4 s (3.9 s → 6.3 s).
- On-demand TPU cost per 1,000 meetings fell from **$2.95 to $0.31**, or $0.10 on Spot pricing.

### 5. What we'd tell someone starting out
- Use `uv` for Python 3.12 on the TPU image.
- Put shared context first in every agent prompt, and warm the cache before fanning out.
- Size `--max-num-seqs` to your real concurrency; it cuts startup compile time.
- vLLM's worker is called `VLLM::EngineCore`. If the TPU stays "in use", free it with `sudo fuser -k /dev/vfio/0`.
- Don't rely on Spot alone, and don't run production traffic over an SSH tunnel.

---

## Video walkthrough outline (~3–4 minutes)
1. **0:00–0:40.** The problem: confidential meetings plus multi-agent extraction. Show the README architecture diagram.
2. **0:40–1:30.** `step2_adk_pipeline.py`: shared transcript first, `SequentialAgent` → `ParallelAgent`.
3. **1:30–2:30.** `streamlit run demo_ui.py`. Paste a meeting, click **Run 4 Google ADK Agents Live**, and walk through the decision, risk, and action cards.
4. **2:30–3:20.** The three charts: prefix caching (0% → 69%), 4B vs. 12B vs. Gemini, and throughput (915 → 8,686 meetings/hour).
5. **3:20–4:00.** Field notes and limitations: synthetic data, count-based quality, estimated costs.
