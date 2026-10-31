"""Ablation: exhaustive FP32 MaxSim vs binary-only vs binary + FP32 re-rank.

Synthetic clustered vectors; not a vidore/ColPali result. Run:
    python3 benchmarks/bench_index_ablation.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colpali_edge.index import MultiVectorIndex
from colpali_edge.maxsim import maxsim

SEED = 11
DOCS = 300
TOKENS = 10
DIM = 24
CLUSTERS = 8
QUERIES = 12
TOP_K = 5
CANDIDATE_K = 30


def vector(center, noise, rng):
    return [value + rng.uniform(-noise, noise) for value in center]


def latency(values):
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
    }


def main():
    rng = random.Random(SEED)
    centers = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(CLUSTERS)]
    index = MultiVectorIndex()
    documents = {}
    labels = {}

    for doc_index in range(DOCS):
        cluster = doc_index % CLUSTERS
        doc_id = f"doc-{doc_index:04d}"
        embeddings = [vector(centers[cluster], 0.35, rng) for _ in range(TOKENS)]
        index.add(doc_id, embeddings)
        documents[doc_id] = embeddings
        labels[doc_id] = cluster

    query_latencies = {"exhaustive": [], "binary": [], "binary_rerank": []}
    recall = {"exhaustive": 0, "binary": 0, "binary_rerank": 0}

    for query_index in range(QUERIES):
        cluster = query_index % CLUSTERS
        query = [vector(centers[cluster], 0.25, rng) for _ in range(TOKENS)]

        started = time.perf_counter()
        exhaustive = sorted(
            ((doc_id, maxsim(query, embeddings)) for doc_id, embeddings in documents.items()),
            key=lambda item: (-item[1], item[0]),
        )[:TOP_K]
        query_latencies["exhaustive"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        binary = index.search_coarse(query, top_k=TOP_K)
        query_latencies["binary"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        reranked = index.search(query, top_k=TOP_K, candidate_k=CANDIDATE_K)
        query_latencies["binary_rerank"].append((time.perf_counter() - started) * 1000)

        recall["exhaustive"] += int(any(labels[doc_id] == cluster for doc_id, _ in exhaustive))
        recall["binary"] += int(any(labels[hit.doc_id] == cluster for hit in binary))
        recall["binary_rerank"] += int(any(labels[hit.doc_id] == cluster for hit in reranked))

    result = {
        "kind": "synthetic-index-ablation",
        "not_a_real_dataset": True,
        "seed": SEED,
        "documents": DOCS,
        "tokens_per_document": TOKENS,
        "dimension": DIM,
        "clusters": CLUSTERS,
        "queries": QUERIES,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "recall_at_k": {
            name: count / QUERIES for name, count in recall.items()
        },
        "latency_ms": {
            name: latency(values) for name, values in query_latencies.items()
        },
        "note": "Synthetic algorithm ablation; not vidore NDCG@5.",
    }
    out = Path(__file__).with_name("index_ablation_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
