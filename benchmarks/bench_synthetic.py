"""Synthetic two-stage retrieval benchmark.

This measures the algorithm core, not ColPali model quality. The corpus is
clustered random data and the labels come from cluster membership; these are
not NDCG@5 results on vidore/esg_reports or any real document set.
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

SEED = 7
DOCS = 500
TOKENS = 12
DIM = 32
CLUSTERS = 10
QUERIES = 20
TOP_K = 5


def vector(center: list[float], noise: float, rng: random.Random) -> list[float]:
    return [value + rng.uniform(-noise, noise) for value in center]


def main() -> None:
    rng = random.Random(SEED)
    centers = [
        [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
        for _ in range(CLUSTERS)
    ]
    index = MultiVectorIndex()
    labels: dict[str, int] = {}

    for doc_index in range(DOCS):
        cluster = doc_index % CLUSTERS
        doc_id = f"doc-{doc_index:04d}"
        labels[doc_id] = cluster
        index.add(
            doc_id,
            [vector(centers[cluster], 0.35, rng) for _ in range(TOKENS)],
        )

    latencies: list[float] = []
    recall = 0
    for query_index in range(QUERIES):
        cluster = query_index % CLUSTERS
        query = [vector(centers[cluster], 0.25, rng) for _ in range(TOKENS)]
        started = time.perf_counter()
        hits = index.search(query, top_k=TOP_K, candidate_k=50)
        latencies.append((time.perf_counter() - started) * 1000)
        if any(labels[hit.doc_id] == cluster for hit in hits):
            recall += 1

    result = {
        "kind": "synthetic-clustered-vectors",
        "not_a_real_dataset": True,
        "seed": SEED,
        "documents": DOCS,
        "tokens_per_document": TOKENS,
        "dimension": DIM,
        "clusters": CLUSTERS,
        "queries": QUERIES,
        "top_k": TOP_K,
        "candidate_k": 50,
        "recall_at_k": recall / QUERIES,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "note": "Algorithm-core benchmark only; not ColPali/vidore NDCG@5.",
    }
    out = Path(__file__).with_name("synthetic_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
