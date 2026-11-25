#!/usr/bin/env bash
# Step 1: Create a Cloud TPU v6e-1, install vllm-torchtpu, and serve Gemma.
#
# Parts A, B and D run on your laptop. Part C runs inside the TPU VM.
# Set these first:
#   export PROJECT_ID=your-gcp-project-id
#   export ZONE=europe-west4-a

# --- A. Create the TPU VM (laptop) ---
# gcloud compute tpus tpu-vm create meeting-tpu \
#   --project=$PROJECT_ID \
#   --zone=$ZONE \
#   --accelerator-type=v6e-1 \
#   --version=v2-alpha-tpuv6e
#
# Add --spot for a much cheaper VM. It can be shut down at any time (this happened to me once).

# --- B. SSH in and forward port 8000 to your laptop (laptop) ---
# gcloud compute tpus tpu-vm ssh meeting-tpu \
#   --project=$PROJECT_ID \
#   --zone=$ZONE \
#   -- -L 8000:127.0.0.1:8000

# --- C. Install and serve (inside the TPU VM) ---
# The TPU image has Python 3.10, but vllm-torchtpu needs Python 3.12.
# uv installs a separate Python 3.12 in under a minute.
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-4b-it}"   # or google/gemma-3-12b-it

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/vllm-env
uv pip install --python ~/vllm-env/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  vllm-torchtpu huggingface_hub

# Gemma weights are gated on Hugging Face. Accept the license, then log in:
# export HF_TOKEN="hf_..."
# ~/vllm-env/bin/huggingface-cli login --token "$HF_TOKEN"

export VLLM_TORCHTPU_BACKEND=tpu

# --max-num-seqs: 16 is enough for one-at-a-time runs.
# Use 64 for the concurrency benchmark (step 6).
~/vllm-env/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching

# --- D. Delete the TPU when you are done (laptop) ---
# gcloud compute tpus tpu-vm delete meeting-tpu --project=$PROJECT_ID --zone=$ZONE --quiet
