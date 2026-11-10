"""
Phase 2: SOP Vector Ingestion Pipeline for Qdrant Cloud.
Uses FastEmbed (ONNX CPU inference) to stay lightweight and idempotent.
"""

import os
import uuid
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("sop_ingestor")

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "aiops_sops")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384

MOCK_SOPS: List[Dict[str, Any]] = [
    {
        "sop_id": "SOP-DB-001",
        "title": "PostgreSQL Connection Pool Exhaustion & Port 5432 Timeout",
        "service": "postgresql",
        "error_signatures": ["connection timeout", "port 5432", "FATAL: sorry, too many clients already", "pool exhausted"],
        "action_type": "restart_pgbouncer_and_terminate_idle",
        "default_parameters": {
            "port": 5432,
            "idle_timeout_seconds": 300,
            "max_client_conn": 500,
            "pool_mode": "transaction",
        },
        "commands": [
            "psql -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND state_change < now() - interval '5 minutes';\"",
            "systemctl reload pgbouncer",
        ],
        "content": (
            "SOP-DB-001: PostgreSQL Connection Timeout on Port 5432 / Pool Exhaustion. "
            "Symptoms: Application services throw 'Postgres connection timeout on port 5432' or "
            "'FATAL: too many clients already'. Root Cause: Stale idle-in-transaction connections "
            "saturating max_connections. Remediation Action: Execute 'restart_pgbouncer_and_terminate_idle' "
            "to terminate idle backends exceeding 300s and reload the PgBouncer transaction pool."
        ),
    },
    {
        "sop_id": "SOP-K8S-002",
        "title": "Docker / Kubernetes Container OOMKill (Exit Code 137)",
        "service": "docker-runtime",
        "error_signatures": ["OOMKilled", "exit code 137", "Out of memory", "cgroup memory limit exceeded"],
        "action_type": "scale_memory_limit_and_rolling_restart",
        "default_parameters": {
            "previous_memory_limit": "512Mi",
            "new_memory_limit": "1024Mi",
            "heap_dump_Captured": True,
            "restart_policy": "rolling",
        },
        "commands": [
            "kubectl set resources deployment/worker-service --limits=memory=1024Mi --requests=memory=512Mi",
            "kubectl rollout restart deployment/worker-service",
        ],
        "content": (
            "SOP-K8S-002: Docker Container OOMKill / Exit Code 137 Memory Leak Remediation. "
            "Symptoms: Container terminated by Linux kernel OOM killer with exit code 137 or "
            "'Docker container OOM kill'. Root Cause: Unbounded worker heap growth under spike traffic. "
            "Remediation Action: Execute 'scale_memory_limit_and_rolling_restart' to bump container cgroup "
            "memory limit from 512Mi to 1024Mi and trigger a zero-downtime rolling restart."
        ),
    },
    {
        "sop_id": "SOP-CACHE-003",
        "title": "Redis Cluster Master-Replica Desync & High Eviction Latency",
        "service": "redis-cluster",
        "error_signatures": ["READONLY You can't write against a read only replica", "MASTERDOWN", "eviction spike", "port 6379"],
        "action_type": "failover_promote_and_purge_volatile_keys",
        "default_parameters": {
            "port": 6379,
            "maxmemory_policy": "allkeys-lru",
            "replica_quorum": 2,
        },
        "commands": [
            "redis-cli -p 6379 CONFIG SET maxmemory-policy allkeys-lru",
            "redis-cli -p 26379 SENTINEL failover mymaster",
        ],
        "content": (
            "SOP-CACHE-003: Redis Cluster Master-Replica Desynchronization & Port 6379 Failover. "
            "Symptoms: 'MASTERDOWN Link with MASTER is down' or 'READONLY replica error on port 6379'. "
            "Root Cause: Primary node split-brain or memory saturation blocking replication backlog. "
            "Remediation Action: Execute 'failover_promote_and_purge_volatile_keys' via Redis Sentinel "
            "and enforce 'allkeys-lru' eviction policy."
        ),
    },
]


def ingest_sops() -> None:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")

    logger.info("Connecting to Qdrant Cloud at %s", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    # Recreate or verify collection
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        logger.info("Creating collection '%s' (dim=%d, Cosine)", COLLECTION_NAME, VECTOR_DIM)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        logger.info("Collection '%s' already exists. Upserting points idempotently.", COLLECTION_NAME)

    logger.info("Loading FastEmbed model: %s", EMBEDDING_MODEL_NAME)
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    documents = [sop["content"] for sop in MOCK_SOPS]
    embeddings = list(embedder.embed(documents))

    points: List[models.PointStruct] = []
    for sop, vector in zip(MOCK_SOPS, embeddings):
        # Deterministic UUID5 from sop_id ensures idempotent upserts
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, sop["sop_id"]))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=sop,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Successfully ingested %d SOP documents into '%s'.", len(points), COLLECTION_NAME)


if __name__ == "__main__":
    ingest_sops()