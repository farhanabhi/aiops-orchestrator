"""
Phase 3A: LangGraph State Machine with Self-Reflective Corrective RAG (CRAG).
"""

import os
import time
import logging
from typing import TypedDict, List, Dict, Any, Literal, Optional
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

load_dotenv()
logger = logging.getLogger("aiops_agent")

# Singleton resources to avoid reloading models across requests (saves RAM on Render)
_embedder: Optional[TextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=20,
        )
    return _qdrant_client


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_retries=2,
    )


# --- Structured Output Schemas for LLM Reasoning ---

class LogMonitorOutput(BaseModel):
    service: str = Field(description="Target infrastructure service (e.g., postgresql, docker-runtime, redis-cluster)")
    error_code: str = Field(description="Extracted error code, port, or signal (e.g., PORT_5432_TIMEOUT, EXIT_137_OOM)")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM"] = Field(description="Incident severity level")
    initial_search_query: str = Field(description="Concise technical search query to retrieve the remediation SOP")
    summary: str = Field(description="One-sentence technical diagnosis of the log")


class SOPGradeOutput(BaseModel):
    is_relevant: Literal["relevant", "irrelevant"] = Field(
        description="Whether the retrieved SOP directly addresses the root cause in the error log"
    )
    reasoning: str = Field(description="Brief explanation of why the SOP matches or fails to match")


class QueryRewriteOutput(BaseModel):
    rewritten_query: str = Field(
        description="Improved search query using alternative infrastructure keywords, ports, and failure signatures"
    )
    rationale: str = Field(description="Why the query was rewritten this way")


# --- LangGraph State Schema ---

class AgentState(TypedDict):
    raw_log: str
    service: str
    error_code: str
    severity: str
    search_query: str
    retrieved_sop: Dict[str, Any]
    similarity_score: float
    relevance_grade: str
    retry_count: int
    remediation_payload: Dict[str, Any]
    remediation_result: Dict[str, Any]
    telemetry_trace: List[Dict[str, Any]]


# --- Graph Nodes ---

def monitor_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    llm = get_llm().with_structured_output(LogMonitorOutput)

    prompt = (
        "You are an elite Site Reliability Engineer (Monitor Agent). "
        "Parse the following raw infrastructure error log and extract the affected service, "
        "error code/signature, severity, and an optimal vector search query for our SOP knowledge base.\n\n"
        f"RAW LOG:\n{state['raw_log']}"
    )
    parsed: LogMonitorOutput = llm.invoke(prompt)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    trace_entry = {
        "node": "Monitor Agent",
        "status": "COMPLETED",
        "latency_ms": latency_ms,
        "detail": (
            f"Extracted service='{parsed.service}', code='{parsed.error_code}', "
            f"severity='{parsed.severity}'. Diagnosis: {parsed.summary}"
        ),
        "metadata": parsed.model_dump(),
    }

    return {
        "service": parsed.service,
        "error_code": parsed.error_code,
        "severity": parsed.severity,
        "search_query": parsed.initial_search_query,
        "retry_count": state.get("retry_count", 0),
        "telemetry_trace": state.get("telemetry_trace", []) + [trace_entry],
    }


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state["search_query"]
    collection_name = os.getenv("QDRANT_COLLECTION", "aiops_sops")

    embedder = get_embedder()
    query_vector = list(embedder.embed([query]))[0].tolist()

    client = get_qdrant()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1,
        with_payload=True,
    )

    if response.points:
        top_hit = response.points[0]
        sop_payload = top_hit.payload or {}
        score = round(float(top_hit.score), 4)
    else:
        sop_payload = {"sop_id": "NONE", "title": "No SOP Found", "content": ""}
        score = 0.0

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace_entry = {
        "node": "Analyst Agent (Qdrant Retrieval)",
        "status": "COMPLETED",
        "latency_ms": latency_ms,
        "detail": f"Queried Qdrant with '{query}' → Matched {sop_payload.get('sop_id')} ({sop_payload.get('title')}) [Cosine Score: {score}]",
        "metadata": {"query": query, "sop_id": sop_payload.get("sop_id"), "score": score},
    }

    return {
        "retrieved_sop": sop_payload,
        "similarity_score": score,
        "telemetry_trace": state.get("telemetry_trace", []) + [trace_entry],
    }


def grade_sop_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    llm = get_llm().with_structured_output(SOPGradeOutput)
    sop = state["retrieved_sop"]

    prompt = (
        "You are a Self-Reflective CRAG Evaluator for an AIOps system.\n"
        "Determine if the retrieved Standard Operating Procedure (SOP) accurately resolves the incident.\n\n"
        f"INCIDENT LOG: {state['raw_log']}\n"
        f"EXTRACTED SERVICE: {state['service']} | ERROR CODE: {state['error_code']}\n"
        f"RETRIEVED SOP ({sop.get('sop_id')}): {sop.get('content')}\n"
    )
    grade: SOPGradeOutput = llm.invoke(prompt)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    trace_entry = {
        "node": "Analyst Agent (Self-Reflective Grader)",
        "status": "VERIFIED" if grade.is_relevant == "relevant" else "REJECTED",
        "latency_ms": latency_ms,
        "detail": f"Relevance Verdict: {grade.is_relevant.upper()} — {grade.reasoning}",
        "metadata": grade.model_dump(),
    }

    return {
        "relevance_grade": grade.is_relevant,
        "telemetry_trace": state.get("telemetry_trace", []) + [trace_entry],
    }


def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    llm = get_llm(temperature=0.2).with_structured_output(QueryRewriteOutput)

    prompt = (
        "The previous vector search query did not retrieve a matching SOP for this infrastructure incident.\n"
        "Formulate a sharper search query targeting specific daemon names, port numbers (e.g., 5432, 6379), "
        "kernel signals (e.g., OOMKilled, exit code 137), or database pool states.\n\n"
        f"RAW LOG: {state['raw_log']}\n"
        f"PREVIOUS FAILED QUERY: {state['search_query']}"
    )
    rewrite: QueryRewriteOutput = llm.invoke(prompt)
    new_retry_count = state.get("retry_count", 0) + 1
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    trace_entry = {
        "node": f"Analyst Agent (CRAG Query Rewriter #{new_retry_count})",
        "status": "RETRYING",
        "latency_ms": latency_ms,
        "detail": f"Rewrote query to: '{rewrite.rewritten_query}' ({rewrite.rationale})",
        "metadata": rewrite.model_dump(),
    }

    return {
        "search_query": rewrite.rewritten_query,
        "retry_count": new_retry_count,
        "telemetry_trace": state.get("telemetry_trace", []) + [trace_entry],
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sop = state["retrieved_sop"]
    fastapi_base = os.getenv("FASTAPI_BASE_URL", "https://aiops-orchestrator-api.onrender.com").rstrip("/")

    payload = {
        "incident_service": state["service"],
        "error_code": state["error_code"],
        "severity": state["severity"],
        "sop_id": sop.get("sop_id", "SOP-FALLBACK-000"),
        "action_type": sop.get("action_type", "escalate_to_oncall_human"),
        "parameters": sop.get("default_parameters", {"reason": "No matching automated SOP found"}),
        "commands_to_run": sop.get("commands", ["echo 'Escalating incident to PagerDuty'"]),
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{fastapi_base}/api/v1/remediate", json=payload)
            resp.raise_for_status()
            remediation_result = resp.json()
            exec_status = "REMEDIATED"
    except Exception as exc:
        remediation_result = {
            "status": "error",
            "message": f"FastAPI remediation call failed: {str(exc)}",
            "mttr_saved_minutes": 0,
        }
        exec_status = "FAILED"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace_entry = {
        "node": "Executor Agent (REST Remediation)",
        "status": exec_status,
        "latency_ms": latency_ms,
        "detail": (
            f"Dispatched POST /api/v1/remediate with action='{payload['action_type']}' "
            f"(SOP: {payload['sop_id']}). Response status: {remediation_result.get('status')}."
        ),
        "metadata": {"request_payload": payload, "api_response": remediation_result},
    }

    return {
        "remediation_payload": payload,
        "remediation_result": remediation_result,
        "telemetry_trace": state.get("telemetry_trace", []) + [trace_entry],
    }


# --- Conditional Edge Routing ---

def decide_to_execute_or_rewrite(state: AgentState) -> Literal["executor_node", "rewrite_query_node"]:
    if state.get("relevance_grade") == "relevant":
        return "executor_node"
    if state.get("retry_count", 0) >= 2:
        # Cap corrective retries at 2 and proceed to executor with best-effort/escalation
        return "executor_node"
    return "rewrite_query_node"


def build_aiops_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("monitor_node", monitor_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_sop_node", grade_sop_node)
    workflow.add_node("rewrite_query_node", rewrite_query_node)
    workflow.add_node("executor_node", executor_node)

    workflow.set_entry_point("monitor_node")
    workflow.add_edge("monitor_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_sop_node")
    workflow.add_conditional_edges(
        "grade_sop_node",
        decide_to_execute_or_rewrite,
        {
            "executor_node": "executor_node",
            "rewrite_query_node": "rewrite_query_node",
        },
    )
    workflow.add_edge("rewrite_query_node", "retrieve_node")
    workflow.add_edge("executor_node", END)

    return workflow.compile()


aiops_graph = build_aiops_graph()