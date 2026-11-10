"""
Phase 3B: FastAPI Backend hosting Remediation Tools & LangGraph Streaming Endpoint.
Enforces strict Pydantic v2 validation across all payloads.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Literal, Generator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator

from agent import aiops_graph, AgentState

app = FastAPI(
    title="Agentic AIOps Orchestrator API",
    description="Autonomous Incident Triage & Remediation via Self-Reflective CRAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Strict Pydantic v2 Schemas ---

class IncidentLogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    error_log: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Raw server or container error log line to triage.",
    )

    @field_validator("error_log")
    @classmethod
    def validate_non_empty_log(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("error_log cannot be blank or whitespace only.")
        return v


class RemediationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_service: str = Field(..., min_length=2, max_length=100)
    error_code: str = Field(..., min_length=2, max_length=100)
    severity: Literal["CRITICAL", "HIGH", "MEDIUM"]
    sop_id: str = Field(..., pattern=r"^SOP-[A-Z0-9]+-\d{3}$")
    action_type: str = Field(..., min_length=3, max_length=120)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    commands_to_run: List[str] = Field(..., min_length=1)


class RemediationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    timestamp_utc: str
    status: Literal["SUCCESS", "ESCALATED"]
    incident_service: str
    sop_applied: str
    action_executed: str
    executed_commands: List[str]
    mttr_baseline_minutes: float
    mttr_actual_seconds: float
    mttr_saved_percentage: float


# --- Endpoints ---

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "aiops-orchestrator-api"}


@app.post("/api/v1/remediate", response_model=RemediationResponse)
async def execute_remediation(payload: RemediationRequest) -> RemediationResponse:
    """
    Simulated Infrastructure Control Plane Endpoint.
    Called autonomously by the LangGraph Executor Agent.
    """
    exec_id = f"rem-{uuid.uuid4().hex[:10]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Calculate simulated MTTR savings vs. 42.0 minute manual SRE baseline
    baseline_minutes = 42.0
    actual_seconds = 3.4
    saved_pct = round(((baseline_minutes * 60 - actual_seconds) / (baseline_minutes * 60)) * 100, 2)

    status_val: Literal["SUCCESS", "ESCALATED"] = (
        "ESCALATED" if payload.sop_id == "SOP-FALLBACK-000" else "SUCCESS"
    )

    return RemediationResponse(
        execution_id=exec_id,
        timestamp_utc=now_iso,
        status=status_val,
        incident_service=payload.incident_service,
        sop_applied=payload.sop_id,
        action_executed=payload.action_type,
        executed_commands=payload.commands_to_run,
        mttr_baseline_minutes=baseline_minutes,
        mttr_actual_seconds=actual_seconds,
        mttr_saved_percentage=saved_pct,
    )


@app.post("/api/v1/triage/stream")
def stream_incident_triage(request: IncidentLogRequest) -> StreamingResponse:
    """
    Streams LangGraph execution state transitions as NDJSON so the Streamlit
    frontend can render live agent telemetry node-by-node.
    """
    initial_state: AgentState = {
        "raw_log": request.error_log,
        "service": "",
        "error_code": "",
        "severity": "HIGH",
        "search_query": "",
        "retrieved_sop": {},
        "similarity_score": 0.0,
        "relevance_grade": "",
        "retry_count": 0,
        "remediation_payload": {},
        "remediation_result": {},
        "telemetry_trace": [],
    }

    def event_generator() -> Generator[str, None, None]:
        t_start = time.perf_counter()
        try:
            for event in aiops_graph.stream(initial_state):
                for node_name, state_update in event.items():
                    latest_trace = (
                        state_update.get("telemetry_trace", [])[-1]
                        if state_update.get("telemetry_trace")
                        else {}
                    )
                    chunk = {
                        "type": "node_update",
                        "node": node_name,
                        "trace": latest_trace,
                        "state_snapshot": {
                            k: v for k, v in state_update.items() if k != "telemetry_trace"
                        },
                    }
                    yield json.dumps(chunk) + "\n"

            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            yield json.dumps({"type": "complete", "total_latency_ms": total_ms}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")