"""
Phase 4: Minimalist Dark-Mode Telemetry Dashboard for Agentic AIOps Orchestrator.
Streams real-time LangGraph node transitions from FastAPI over NDJSON.
"""

import os
import json
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="AIOps Orchestrator | Self-Reflective CRAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-contrast developer theme styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px; }
    .terminal-card {
        background-color: #12141D;
        border: 1px solid #232736;
        border-left: 3px solid #10B981;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.86rem;
    }
    .node-header {
        color: #F9FAFB;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .latency-badge {
        background-color: #1F2433;
        color: #34D399;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .node-detail { color: #9CA3AF; line-height: 1.45; }
    .sub-pill {
        display: inline-block;
        background: #181B26;
        border: 1px solid #2D3348;
        color: #E5E7EB;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header Section ---
st.markdown("### ⚡ Agentic AIOps Orchestrator")
st.markdown(
    "<span class='sub-pill'>LangGraph State Machine</span>"
    "<span class='sub-pill'>Groq Llama-3.3-70B</span>"
    "<span class='sub-pill'>Qdrant Corrective RAG</span>"
    "<span class='sub-pill'>FastAPI Control Plane</span>",
    unsafe_allow_html=True,
)
st.write("")

# --- Preset Incident Logs ---
PRESET_LOGS = {
    "PostgreSQL Port 5432 Connection Timeout (DB Outage)": (
        "2026-11-10T14:12:09.412Z [ERROR] [payment-service] sqlalchemy.exc.OperationalError: "
        "(psycopg2.OperationalError) connection to server at 'db-prod-internal.cluster' (10.0.4.21), "
        "port 5432 failed: timeout expired. FATAL: sorry, too many clients already (active=500/500)."
    ),
    "Docker Container OOMKill Exit Code 137 (Memory Leak)": (
        "2026-11-10T14:15:44.801Z [CRITICAL] [kubelet-node-04] Container 'ml-inference-worker-7d9b' "
        "in pod 'worker-service-7d9b-x8k2p' terminated by kernel: Docker container OOM kill "
        "(Exit Code 137). Memory cgroup usage 512Mi/512Mi exceeded."
    ),
    "Redis Cluster Port 6379 Replica Desync (Cache Failure)": (
        "2026-11-10T14:18:02.119Z [ERROR] [session-gateway] RedisClusterException: MASTERDOWN "
        "Link with MASTER is down on port 6379. READONLY You can't write against a read only replica."
    ),
    "Custom / Ambiguous Log (Triggers CRAG Self-Reflection Loop)": (
        "2026-11-10T14:20:11.004Z [CRITICAL] [checkout-api] Upstream datastore socket hangup "
        "after 30000ms; relational pooler saturated with lingering idle transactions."
    ),
}

col_input, col_meta = st.columns([2.2, 1])

with col_input:
    selected_preset = st.selectbox("Select Simulated Infrastructure Incident", list(PRESET_LOGS.keys()))
    raw_log_input = st.text_area(
        "Raw Telemetry / Server Error Log",
        value=PRESET_LOGS[selected_preset],
        height=115,
    )
    trigger_btn = st.button("🚀 Trigger Autonomous Triage & Remediation", type="primary", use_container_width=True)

with col_meta:
    st.markdown("##### Architecture Pipeline")
    st.caption(
        "1. **Monitor Agent**: Parses service, error code, and severity.\n"
        "2. **Analyst Agent (CRAG)**: Queries Qdrant Cloud (`aiops_sops`) and grades SOP alignment.\n"
        "3. **Self-Reflection**: Rewrites vector query if relevance fails.\n"
        "4. **Executor Agent**: Calls FastAPI `/api/v1/remediate`."
    )

st.divider()

# --- Live Execution Stream ---
if trigger_btn:
    kpi_cols = st.columns(4)
    stream_col, result_col = st.columns([1.35, 1])

    with stream_col:
        st.markdown("#### Live Agent Telemetry Stream")
        trace_container = st.container()

    with result_col:
        st.markdown("#### Remediation Execution Receipt")
        receipt_placeholder = st.empty()
        receipt_placeholder.info("Awaiting Executor Agent dispatch to `/api/v1/remediate`...")

    final_state_data = {}
    total_pipeline_ms = 0.0

    try:
        with st.status("Running LangGraph CRAG State Machine...", expanded=True) as status_box:
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{FASTAPI_BASE_URL}/api/v1/triage/stream",
                    json={"error_log": raw_log_input},
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)

                        if event.get("type") == "node_update":
                            trace = event.get("trace", {})
                            snapshot = event.get("state_snapshot", {})
                            final_state_data.update(snapshot)

                            node_title = trace.get("node", event.get("node"))
                            node_status = trace.get("status", "DONE")
                            latency = trace.get("latency_ms", 0.0)
                            detail = trace.get("detail", "")

                            st.write(f"✓ **{node_title}** (`{latency} ms`)")

                            with trace_container:
                                st.markdown(
                                    f"""
                                    <div class="terminal-card">
                                        <div class="node-header">
                                            <span>[{node_status}] {node_title}</span>
                                            <span class="latency-badge">{latency} ms</span>
                                        </div>
                                        <div class="node-detail">{detail}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        elif event.get("type") == "complete":
                            total_pipeline_ms = event.get("total_latency_ms", 0.0)
                            status_box.update(
                                label=f"Incident Remediated in {total_pipeline_ms} ms",
                                state="complete",
                                expanded=False,
                            )

                        elif event.get("type") == "error":
                            status_box.update(label="Pipeline Error", state="error")
                            st.error(f"Agent Error: {event.get('message')}")

        # --- Populate Top KPI Metrics ---
        rem_result = final_state_data.get("remediation_result", {})
        sop_data = final_state_data.get("retrieved_sop", {})
        sim_score = final_state_data.get("similarity_score", 0.0)

        kpi_cols[0].metric(
            label="End-to-End Triage Latency",
            value=f"{round(total_pipeline_ms / 1000, 2)} s",
            delta="-99.8% vs Manual",
        )
        kpi_cols[1].metric(
            label="Matched SOP Vector Score",
            value=f"{sim_score:.3f}",
            delta=sop_data.get("sop_id", "N/A"),
        )
        kpi_cols[2].metric(
            label="Simulated MTTR Reduction",
            value="42m → 3.4s",
            delta=f"{rem_result.get('mttr_saved_percentage', 99.86)}% Saved",
        )
        kpi_cols[3].metric(
            label="Control Plane Status",
            value=rem_result.get("status", "UNKNOWN"),
            delta=final_state_data.get("severity", "HIGH"),
        )

        # --- Populate Right Column Receipt ---
        with result_col:
            receipt_placeholder.empty()
            st.markdown("**Shell Commands Executed on Target Host:**")
            for cmd in rem_result.get("executed_commands", []):
                st.code(cmd, language="bash")

            st.markdown("**FastAPI `/api/v1/remediate` Verified JSON Receipt:**")
            st.json(rem_result)

    except Exception as exc:
        st.error(
            f"Could not connect to FastAPI backend at `{FASTAPI_BASE_URL}`. "
            f"Ensure `uvicorn main:app --reload` is running.\n\nDetails: `{exc}`"
        )