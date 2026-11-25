"""
STEP 2: Simple Multi-Agent Meeting Intelligence Pipeline (Google ADK + vLLM-TorchTPU)

Reads a meeting transcript, runs 4 specialist Google ADK agents that share the
same transcript prefix (so vLLM caches the 256-token KV blocks on TPU), and
outputs clean structured JSON (`meeting_output.json`).

Usage:
  .venv/bin/python step2_adk_pipeline.py
  MODE=gemini .venv/bin/python step2_adk_pipeline.py
"""

import asyncio
import json
import os
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from google.genai import types
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner

load_dotenv()


# ------------------------------------------------------------------------------
# 1. Structured JSON Schemas (one simple schema per agent)
# ------------------------------------------------------------------------------
class InsightsOutput(BaseModel):
    executive_summary: str
    key_insights: list[str]


class Decision(BaseModel):
    decision: str
    status: str  # Approved / Deferred / Rejected
    owner: str


class DecisionsOutput(BaseModel):
    decisions: list[Decision]


class Risk(BaseModel):
    risk: str
    category: str
    severity: str  # High / Medium / Low


class RisksOutput(BaseModel):
    risks: list[Risk]


class ActionItem(BaseModel):
    action: str
    owner: str
    deadline: str


class ActionsOutput(BaseModel):
    business_actions: list[ActionItem]


# ------------------------------------------------------------------------------
# 2. Choose Model: Self-Hosted TPU vLLM ("tpu") OR Hosted Gemini ("gemini")
# ------------------------------------------------------------------------------
def get_model(mode: str, model_id: str | None = None):
    if mode == "gemini":
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                raise RuntimeError("Set GCP_PROJECT_ID in .env (see .env.example) or set GEMINI_API_KEY.")
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEX_LOCATION", "global")
        return model_id or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    tpu_model = model_id or os.getenv("TPU_MODEL", "google/gemma-3-4b-it")
    api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    return LiteLlm(model=f"hosted_vllm/{tpu_model}", api_base=api_base, api_key="local")


# ------------------------------------------------------------------------------
# 3. Build & Run the 4-Agent Google ADK Pipeline
# ------------------------------------------------------------------------------
async def run_pipeline(
    transcript: str,
    mode: str = "tpu",
    transcript_first: bool = True,
    model_id: str | None = None,
) -> dict:
    model = get_model(mode, model_id=model_id)

    shared_transcript = (
        "You are a private meeting intelligence assistant. "
        "Use ONLY facts from the meeting transcript below.\n\n"
        f"=== MEETING TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ==="
    )

    def build_instruction(task_text: str) -> str:
        # Putting `shared_transcript` FIRST makes tokens 0..N identical across all 4 agents!
        if transcript_first:
            return f"{shared_transcript}\n\nYOUR TASK: {task_text}"
        return f"YOUR TASK: {task_text}\n\n{shared_transcript}"

    # Stage 1: InsightsAgent runs first so vLLM caches transcript KV blocks in TPU HBM
    insights_agent = LlmAgent(
        name="InsightsAgent",
        model=model,
        instruction=build_instruction("Write a short executive_summary and 3-5 key_insights."),
        include_contents="none",
        output_schema=InsightsOutput,
        output_key="insights",
    )

    # Stage 2: Remaining 3 agents run in parallel and reuse the warm KV prefix cache!
    decisions_agent = LlmAgent(
        name="DecisionsAgent",
        model=model,
        instruction=build_instruction("Extract all explicit decisions with status and owner."),
        include_contents="none",
        output_schema=DecisionsOutput,
        output_key="decisions",
    )
    risks_agent = LlmAgent(
        name="RisksAgent",
        model=model,
        instruction=build_instruction("Extract all explicit risks with category and severity."),
        include_contents="none",
        output_schema=RisksOutput,
        output_key="risks",
    )
    actions_agent = LlmAgent(
        name="ActionsAgent",
        model=model,
        instruction=build_instruction("Extract concrete business_actions with owner and deadline."),
        include_contents="none",
        output_schema=ActionsOutput,
        output_key="actions",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parallel_stage = ParallelAgent(
            name="ParallelSpecialists",
            sub_agents=[decisions_agent, risks_agent, actions_agent],
        )
        workflow = SequentialAgent(
            name="MeetingIntelligencePipeline",
            sub_agents=[insights_agent, parallel_stage],
        )

    t0 = time.perf_counter()
    runner = InMemoryRunner(agent=workflow, app_name="meeting_intel")
    session = await runner.session_service.create_session(
        app_name="meeting_intel", user_id="researcher"
    )
    msg = types.Content(role="user", parts=[types.Part.from_text(text="Analyze transcript.")])

    async for _ in runner.run_async(user_id="researcher", session_id=session.id, new_message=msg):
        pass

    elapsed_sec = round(time.perf_counter() - t0, 2)
    final_session = await runner.session_service.get_session(
        app_name="meeting_intel", user_id="researcher", session_id=session.id
    )
    await runner.close()

    def parse_state(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            clean = val.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        return {}

    state = final_session.state
    return {
        "mode": mode,
        "model_id": model_id or (os.getenv("TPU_MODEL", "google/gemma-3-4b-it") if mode == "tpu" else "gemini-3.5-flash"),
        "transcript_first_prefix_cache": transcript_first,
        "latency_seconds": elapsed_sec,
        **parse_state(state.get("insights")),
        **parse_state(state.get("decisions")),
        **parse_state(state.get("risks")),
        **parse_state(state.get("actions")),
    }


if __name__ == "__main__":
    active_mode = os.getenv("MODE", "tpu")
    transcript_text = Path("sample_meeting.txt").read_text(encoding="utf-8")

    print(f"Running 4-Agent ADK Pipeline (MODE={active_mode})...")
    result = asyncio.run(run_pipeline(transcript_text, mode=active_mode, transcript_first=True))

    Path("meeting_output.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved structured JSON to meeting_output.json (took {result['latency_seconds']}s)")
