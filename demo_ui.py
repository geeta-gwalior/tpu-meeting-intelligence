"""
Live Interactive Multi-Agent App — Private Meeting Intelligence on Cloud TPU v6e

100% Dynamic Execution:
  - Paste ANY custom meeting transcript, edit text, or upload a `.txt` file
  - Click "Run 4 Google ADK Agents Live" to execute the real 2-stage ADK pipeline live!
  - Automatically detects if your Cloud TPU v6e SSH tunnel (`127.0.0.1:8000`) is online;
    when TPU is off to save credits, runs the exact same 4 ADK agents live via Vertex Gemini.

Run:
  .venv/bin/streamlit run demo_ui.py
"""

import asyncio
import json
import os
import time
import warnings
from pathlib import Path
import httpx
import pandas as pd
import streamlit as st
from google.genai import types
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from step2_adk_pipeline import (
    ActionsOutput,
    DecisionsOutput,
    InsightsOutput,
    RisksOutput,
    get_model,
)

st.set_page_config(
    page_title="Private Meeting Intelligence — Cloud TPU v6e",
    layout="wide",
)

st.markdown(
    """
    <style>
    .step-box {
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.28);
        background: rgba(11, 87, 208, 0.05);
        height: 100%;
    }
    .step-num {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #0b57d0;
        letter-spacing: 0.04em;
    }
    .step-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin: 4px 0 6px 0;
    }
    .step-desc {
        font-size: 0.87rem;
        opacity: 0.88;
        line-height: 1.45;
    }
    .item-card {
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 10px;
    }
    .pill {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .pill-green { background: #dafbe1; color: #116329; }
    .pill-red { background: #ffebe9; color: #a40e26; }
    .pill-amber { background: #fff8c5; color: #7d4e00; }
    .pill-blue { background: #ddf4ff; color: #0550ae; }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_tpu_tunnel(api_base: str) -> tuple[bool, str | None]:
    """Check if live vLLM-TorchTPU server is reachable on the local SSH tunnel."""
    try:
        r = httpx.get(f"{api_base}/models", timeout=1.0)
        if r.status_code == 200:
            data = r.json().get("data", [])
            model_name = data[0]["id"] if data else "google/gemma-3-4b-it"
            return True, model_name
    except Exception:
        pass
    return False, None


def read_vllm_hits(metrics_url: str) -> tuple[float, float] | None:
    try:
        txt = httpx.get(metrics_url, timeout=1.0).text
        q = h = 0.0
        for ln in txt.splitlines():
            if ln.startswith("vllm:prefix_cache_queries_total"):
                q += float(ln.split()[-1])
            elif ln.startswith("vllm:prefix_cache_hits_total"):
                h += float(ln.split()[-1])
        return q, h
    except Exception:
        return None


async def run_live_with_stage_events(
    transcript: str,
    mode: str,
    model_id: str | None,
    transcript_first: bool,
    status_box,
) -> dict:
    """Run the real 4-Agent Google ADK pipeline and stream live stage updates to UI."""
    model = get_model(mode, model_id=model_id)
    metrics_url = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1").replace("/v1", "/metrics")
    before_m = read_vllm_hits(metrics_url) if mode == "tpu" else None

    shared_transcript = (
        "You are a private meeting intelligence assistant. "
        "Use ONLY facts from the meeting transcript below.\n\n"
        f"=== MEETING TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ==="
    )

    def build_inst(task_text: str) -> str:
        if transcript_first:
            return f"{shared_transcript}\n\nYOUR TASK: {task_text}"
        return f"YOUR TASK: {task_text}\n\n{shared_transcript}"

    insights_agent = LlmAgent(
        name="InsightsAgent",
        model=model,
        instruction=build_inst("Write a short executive_summary and 3-5 key_insights."),
        include_contents="none",
        output_schema=InsightsOutput,
        output_key="insights",
    )
    decisions_agent = LlmAgent(
        name="DecisionsAgent",
        model=model,
        instruction=build_inst("Extract all explicit decisions with status and owner."),
        include_contents="none",
        output_schema=DecisionsOutput,
        output_key="decisions",
    )
    risks_agent = LlmAgent(
        name="RisksAgent",
        model=model,
        instruction=build_inst("Extract all explicit risks with category and severity."),
        include_contents="none",
        output_schema=RisksOutput,
        output_key="risks",
    )
    actions_agent = LlmAgent(
        name="ActionsAgent",
        model=model,
        instruction=build_inst("Extract concrete business_actions with owner and deadline."),
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
    stage_timings: dict[str, float] = {}

    status_box.write("Stage 1/2: Running **InsightsAgent** (reading transcript & warming KV prefix blocks)...")
    runner = InMemoryRunner(agent=workflow, app_name="meeting_intel_live")
    session = await runner.session_service.create_session(
        app_name="meeting_intel_live", user_id="live_user"
    )
    msg = types.Content(role="user", parts=[types.Part.from_text(text="Analyze transcript.")])

    stage2_announced = False
    async for event in runner.run_async(user_id="live_user", session_id=session.id, new_message=msg):
        author = getattr(event, "author", "")
        if author:
            stage_timings[author] = round(time.perf_counter() - t0, 2)
            if author == "InsightsAgent" and not stage2_announced:
                stage2_announced = True
                status_box.write(
                    f"Done **InsightsAgent** in `{stage_timings[author]}s` — KV prefix warm! "
                    "Now launching **DecisionsAgent + RisksAgent + ActionsAgent** in parallel..."
                )
            elif author in ("DecisionsAgent", "RisksAgent", "ActionsAgent"):
                status_box.write(f"Finished **{author}** at `{stage_timings[author]}s`")

    total_sec = round(time.perf_counter() - t0, 2)
    final_session = await runner.session_service.get_session(
        app_name="meeting_intel_live", user_id="live_user", session_id=session.id
    )
    await runner.close()

    after_m = read_vllm_hits(metrics_url) if mode == "tpu" else None
    kv_hit_str = None
    if before_m and after_m and (after_m[0] - before_m[0]) > 0:
        q_d = int(after_m[0] - before_m[0])
        h_d = int(after_m[1] - before_m[1])
        kv_hit_str = f"{(h_d / q_d) * 100:.1f}% ({h_d}/{q_d} tok)"

    def parse_val(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            clean = val.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        return {}

    st_state = final_session.state
    return {
        "mode": mode,
        "model_id": model_id or ("google/gemma-3-4b-it" if mode == "tpu" else "gemini-3.5-flash"),
        "transcript_first_prefix_cache": transcript_first,
        "latency_seconds": total_sec,
        "live_vllm_kv_hit": kv_hit_str,
        "agent_completion_timestamps_s": stage_timings,
        **parse_val(st_state.get("insights")),
        **parse_val(st_state.get("decisions")),
        **parse_val(st_state.get("risks")),
        **parse_val(st_state.get("actions")),
    }


# ------------------------------------------------------------------------------
# Header + Live TPU Connection Badge
# ------------------------------------------------------------------------------
api_base_default = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
tpu_online, live_tpu_model = check_tpu_tunnel(api_base_default)

head_l, head_r = st.columns([3, 1.2])
with head_l:
    st.title("Private Meeting Intelligence on Cloud TPU v6e")
    st.caption("Live 4-Agent Google ADK Pipeline • vLLM-TorchTPU (256-Token KV Prefix Caching) • Open-Weight Gemma")
with head_r:
    if tpu_online:
        st.success(f"🟢 Cloud TPU v6e Tunnel LIVE\n`{live_tpu_model}`")
    else:
        st.warning("🟡 TPU VM Off (Saving Credits)\nLive ADK runs via Vertex Gemini")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        """<div class="step-box">
        <div class="step-num">Step 1 • Private Input</div>
        <div class="step-title">Write, Paste, or Upload Any Transcript</div>
        <div class="step-desc">Type your own meeting notes, upload a <code>.txt</code> file, or pick a sample enterprise meeting below.</div>
        </div>""",
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        """<div class="step-box">
        <div class="step-num">Step 2 • Stage 1 Cache Warmup</div>
        <div class="step-title">InsightsAgent Warms TPU HBM</div>
        <div class="step-desc"><code>InsightsAgent</code> runs first so the shared transcript is cached in 256-token KV blocks inside TPU memory.</div>
        </div>""",
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        """<div class="step-box">
        <div class="step-num">Step 3 • Stage 2 Parallel Fan-Out</div>
        <div class="step-title">3 Specialist Agents Run Simultaneously</div>
        <div class="step-desc"><code>DecisionsAgent</code>, <code>RisksAgent</code>, and <code>ActionsAgent</code> run in parallel and return structured JSON live.</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ------------------------------------------------------------------------------
# Sample Templates (for convenience, plus Custom Blank / Upload!)
# ------------------------------------------------------------------------------
TEMPLATES = {"✨ Write / Paste Custom Meeting Transcript": ""}
if Path("sample_meeting.txt").exists():
    TEMPLATES["Sample: Healthcare M&A & HIPAA Sync"] = Path("sample_meeting.txt").read_text(encoding="utf-8")
for mf in sorted(Path("meetings").glob("*.txt")):
    TEMPLATES[f"Sample: {mf.stem.replace('_', ' ').title()}"] = mf.read_text(encoding="utf-8")


def status_pill(status: str) -> str:
    s = status.lower()
    if "approv" in s:
        return '<span class="pill pill-green">APPROVED</span>'
    if "reject" in s:
        return '<span class="pill pill-red">REJECTED</span>'
    return '<span class="pill pill-amber">DEFERRED</span>'


def severity_pill(sev: str) -> str:
    s = sev.lower()
    if "high" in s:
        return '<span class="pill pill-red">HIGH RISK</span>'
    if "med" in s:
        return '<span class="pill pill-amber">MEDIUM RISK</span>'
    return '<span class="pill pill-green">LOW RISK</span>'


tab_live, tab_cache, tab_models, tab_scale, tab_notes = st.tabs(
    [
        "1. Live Multi-Agent Analyzer",
        "2. Experiment 1: TPU Prefix Caching",
        "3. Experiment 2: TPU (4B vs 12B) vs Gemini API",
        "4. Experiment 3: TPU Throughput at Scale",
        "5. Behind the Scenes: TPU Lessons",
    ]
)

with tab_live:
    col_in, col_out = st.columns([1.05, 1.35], gap="large")

    with col_in:
        st.subheader("1. Meeting Input & Live Engine")

        backend_options = [
            "Cloud TPU v6e — vLLM-TorchTPU (Private Tunnel :8000)",
            "Hosted Gemini 3.5 Flash — Live ADK (Works When TPU is Off)",
        ]
        default_idx = 0 if tpu_online else 1
        chosen_backend = st.radio("Select Live Inference Engine", backend_options, index=default_idx)
        active_mode = "tpu" if chosen_backend.startswith("Cloud TPU") else "gemini"

        c1, c2 = st.columns(2)
        chosen_model = (
            c1.text_input("TPU Model ID", value=live_tpu_model or "google/gemma-3-4b-it")
            if active_mode == "tpu"
            else c1.text_input("Gemini Model ID", value="gemini-3.5-flash")
        )
        prefix_aligned = c2.checkbox("Prefix-Aligned Prompt (Transcript First)", value=True)

        template_choice = st.selectbox(
            "Load a Sample Transcript or Write Your Own",
            options=list(TEMPLATES.keys()),
            index=1 if len(TEMPLATES) > 1 else 0,
        )
        uploaded_file = st.file_uploader("Or upload any .txt / .md meeting transcript", type=["txt", "md"])

        initial_text = (
            uploaded_file.read().decode("utf-8")
            if uploaded_file is not None
            else TEMPLATES[template_choice]
        )

        transcript_box = st.text_area(
            "Meeting Transcript (Edit any line or paste new dialogue — agents analyze this live!)",
            value=initial_text,
            height=300,
            placeholder="Paste any meeting transcript here, e.g.:\nRahul: Let's approve moving our database to Mumbai region by Friday.\nNeha: Risk is 2 hours of downtime during migration...",
        )

        b_run, b_demo = st.columns([1.6, 1])
        run_live_clicked = b_run.button(
            "🚀 Run 4 Google ADK Agents Live",
            type="primary",
            width="stretch",
        )
        load_last_tpu = b_demo.button(
            "Load Last TPU Run",
            help="View the saved JSON from our Cloud TPU v6e hardware run without making a new API call.",
            width="stretch",
        )

    with col_out:
        st.subheader("2. Live Multi-Agent Output")

        if "current_output" not in st.session_state:
            st.session_state.current_output = None

        if load_last_tpu and Path("meeting_output.json").exists():
            st.session_state.current_output = json.loads(
                Path("meeting_output.json").read_text(encoding="utf-8")
            )

        if run_live_clicked:
            if not transcript_box.strip():
                st.warning("Please enter or select a meeting transcript first.")
            else:
                with st.status(
                    f"Executing 4 Google ADK Agents Live on `{chosen_model}`...",
                    expanded=True,
                ) as status_box:
                    try:
                        st.session_state.current_output = asyncio.run(
                            run_live_with_stage_events(
                                transcript=transcript_box,
                                mode=active_mode,
                                model_id=chosen_model,
                                transcript_first=prefix_aligned,
                                status_box=status_box,
                            )
                        )
                        status_box.update(
                            label=f"✅ All 4 ADK Agents Finished in {st.session_state.current_output['latency_seconds']}s!",
                            state="complete",
                            expanded=False,
                        )
                    except Exception as exc:
                        status_box.update(label="❌ Connection Error", state="error", expanded=True)
                        st.error(
                            f"Failed to connect to `{active_mode}` (`{chosen_model}`): {exc}\n\n"
                            "If your Cloud TPU VM is turned off right now to save credits, select "
                            "**'Hosted Gemini 3.5 Flash — Live ADK'** on the left to test live!"
                        )

        res = st.session_state.current_output
        if res is None:
            st.info(
                "👈 **Ready for Live Execution!**\n\n"
                "Edit the transcript on the left (or paste your own meeting notes) and click "
                "**'🚀 Run 4 Google ADK Agents Live'** to watch `InsightsAgent`, `DecisionsAgent`, "
                "`RisksAgent`, and `ActionsAgent` run in real time."
            )
        else:
            is_priv = res.get("mode") == "tpu"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Latency", f"{res.get('latency_seconds', '—')} s")
            m2.metric("Model Used", str(res.get("model_id", "")).split("/")[-1])
            m3.metric(
                "vLLM KV Cache Hit",
                res.get("live_vllm_kv_hit") or ("Warm Prefix" if is_priv else "N/A (Cloud API)"),
            )
            m4.metric("Privacy", "100% Private TPU" if is_priv else "External API")

            if res.get("agent_completion_timestamps_s"):
                ts_str = " → ".join(
                    f"`{k}` ({v}s)" for k, v in res["agent_completion_timestamps_s"].items()
                )
                st.caption(f"**Live Agent Execution Timeline:** {ts_str}")

            st.info(f"**Executive Summary:** {res.get('executive_summary', '')}")

            t1, t2, t3, t4, t5 = st.tabs(
                [
                    f"Decisions ({len(res.get('decisions', []))})",
                    f"Risks ({len(res.get('risks', []))})",
                    f"Action Items ({len(res.get('business_actions', []))})",
                    f"Key Insights ({len(res.get('key_insights', []))})",
                    "Raw JSON",
                ]
            )
            with t1:
                for d in res.get("decisions", []):
                    st.markdown(
                        f"""<div class="item-card">
                        {status_pill(d.get('status', ''))}
                        <span class="pill pill-blue">Owner: {d.get('owner', 'Unassigned')}</span>
                        <div style="margin-top:6px;font-weight:500;">{d.get('decision', '')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            with t2:
                for r in res.get("risks", []):
                    st.markdown(
                        f"""<div class="item-card">
                        {severity_pill(r.get('severity', ''))}
                        <span class="pill pill-blue">{r.get('category', 'General')}</span>
                        <div style="margin-top:6px;font-weight:500;">{r.get('risk', '')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            with t3:
                acts = res.get("business_actions", [])
                if acts:
                    st.dataframe(
                        pd.DataFrame(acts).rename(
                            columns={"action": "Business Action Item", "owner": "Owner", "deadline": "Deadline"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )
            with t4:
                for i, ins in enumerate(res.get("key_insights", []), start=1):
                    st.markdown(f"**{i}.** {ins}")
            with t5:
                st.json(res)

with tab_cache:
    st.subheader("Why Prompt Order Matters: 0% vs ~69% KV Cache Reuse on Cloud TPU v6e")
    p1, p2 = st.columns(2)
    with p1:
        st.error(
            "**Naive Multi-Agent Prompt (0.0% Cache Hit)**\n\n"
            "`[Unique Agent Task]` -> `[Long Meeting Transcript]`\n\n"
            "Token 0 differs per agent, breaking every 256-token block hash."
        )
    with p2:
        st.success(
            "**Our Prefix-Aligned Prompt (~69% overall hit on 6k-token meetings)**\n\n"
            "`[Shared Meeting Transcript]` -> `[Unique Agent Task]`\n\n"
            "`InsightsAgent` warms 256-token KV blocks in TPU HBM; parallel agents reuse them "
            "(~90% estimated on Agents 2–4)."
        )
    if Path("charts/1_prefix_cache_scaling.png").exists():
        st.image("charts/1_prefix_cache_scaling.png", width="stretch")
    if Path("prefix_cache_results.json").exists():
        st.dataframe(pd.DataFrame(json.loads(Path("prefix_cache_results.json").read_text())), width="stretch", hide_index=True)

with tab_models:
    st.subheader("Single Cloud TPU v6e-1 (Gemma 4B vs. 12B) vs. Hosted Gemini API")
    if Path("charts/2_model_comparison.png").exists():
        st.image("charts/2_model_comparison.png", width="stretch")
    if Path("research_comparison.json").exists():
        mods = json.loads(Path("research_comparison.json").read_text())["models"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": mods[k]["label"],
                        "Avg Latency (s)": mods[k]["avg_latency_s"],
                        "Completeness (%)": mods[k]["extraction_completeness_pct"],
                        "TPU KV Cache Capacity": mods[k]["tpu_kv_cache_capacity_tokens"] or "Managed",
                        "Spot Cost / 1k ($)": mods[k]["cost_per_1k_spot_usd"],
                        "Privacy": mods[k]["privacy"],
                    }
                    for k in ("tpu_4b", "tpu_12b", "gemini")
                    if k in mods
                ]
            ),
            width="stretch",
            hide_index=True,
        )

with tab_scale:
    st.subheader("How Many Meetings Can One Cloud TPU v6e-1 Chip Handle?")
    st.markdown(
        "A TPU's real strength is **batching**: handling many requests together. We ran the same 4-agent "
        "pipeline with 1, 4, 8, and 16 meetings at once (32 meetings per level, Gemma 3 4B, one `v6e-1` chip). "
        "With 16 at once, the chip processed **~9.5x more meetings per hour**, while each meeting took only "
        "3.9s → 6.3s."
    )
    if Path("charts/3_concurrency_throughput.png").exists():
        st.image("charts/3_concurrency_throughput.png", width="stretch")
    if Path("concurrency_results.json").exists():
        conc = json.loads(Path("concurrency_results.json").read_text(encoding="utf-8"))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Meetings In-Flight": r["concurrency"],
                        "Meetings / Hour": r["meetings_per_hour"],
                        "Avg Latency / Meeting (s)": r["avg_meeting_latency_s"],
                        "Generation Tokens / s": r["generation_tokens_per_s"],
                        "On-Demand Cost / 1k Meetings ($)": r["cost_per_1k_meetings_ondemand_usd"],
                        "Spot Cost / 1k Meetings ($)": r["cost_per_1k_meetings_spot_usd"],
                    }
                    for r in conc
                ]
            ),
            width="stretch",
            hide_index=True,
        )

with tab_notes:
    st.subheader("Behind the Scenes: What We Learned Running This on Cloud TPU")
    st.markdown(
        """
        - **Old Python on the TPU image.** The TPU came with Python 3.10, but the serving software needs 3.12.
          We installed Python 3.12 separately with `uv`, which took under a minute.
        - **TPU caches text in big chunks.** The TPU stores processed text in 256-token blocks (GPUs use 16).
          That's why longer meetings get more reuse than short ones.
        - **Prompt order made a huge difference.** When each agent's task came first, the agents shared nothing (0% reuse).
          Putting the transcript first lifted reuse to about 69%.
        - **A 12B model fit on one TPU chip.** The TPU automatically compressed its memory cache, so the larger,
          more accurate model ran on a single chip.
        - **Cheap Spot TPUs can disappear.** Our discounted Spot TPU was shut down by Google Cloud mid-test,
          so a real deployment needs at least one regular (on-demand) TPU.
        """
    )
    st.caption("Full technical notes, limitations, and setup commands are in the project README and blog post.")
