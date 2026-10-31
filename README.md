# colpali-edge

Dependency-free core primitives for ColPali-style visual document retrieval:
MaxSim late interaction, in-batch contrastive loss, hierarchical token pooling,
binary quantization, and a two-stage multi-vector index.

This repository is the deterministic algorithm layer. It does **not** download
PaliGemma/ColQwen2/ModernVBERT weights, run a vision encoder, train on a GPU, or
claim a vidore NDCG@5 result. Those adapters are separate work.

## What is implemented

### MaxSim late interaction (arXiv:2407.01449)

```
LI(q, d) = sum_{j=1..Nq} max_{i=1..Nd} <E_q[j], E_d[i]>
```

Raw page images are embedded into query-token and document-patch matrices in a
shared dimension `D`; the score sums the best patch match for each query token.
No OCR, layout detection, or chunking is involved in this operator.

### In-batch contrastive loss

```
s_k+ = LI(q_k, d_k)
s_k- = max_{l != k} LI(q_k, d_l)
L    = -1/b * sum_k log(exp(s_k+) / (exp(s_k+) + exp(s_k-)))
```

The implementation uses the stable `softplus(s_k- - s_k+)` form and rejects a
batch of one (no negative).

### Hierarchical token pooling

Mean-pools consecutive patch vectors by a factor, with a short final group
averaged over its actual length. This reduces the number of vectors the MaxSim
operator scores.

### Binary quantization and two-stage search

Each float vector is reduced to one sign bit per dimension, packed LSB-first.
The coarse score approximates dot products with Hamming distance:

```
dot(a, b) ~= D - 2 * hamming(bits(a), bits(b))
```

`MultiVectorIndex` scans the binary representation, keeps `candidate_k`, then
re-ranks candidates with FP32 MaxSim. Every hit keeps **both** `coarse_score` and
`score` so ranking changes are attributable.

## Install

```bash
python3 -m pip install -e .
```

No runtime dependencies.

## Usage

```python
from colpali_edge import (
    maxsim,
    in_batch_contrastive_loss,
    hierarchical_pool,
    MultiVectorIndex,
)

query = [[1.0, 0.0], [0.0, 1.0]]
document = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
print(maxsim(query, document))  # 2.0

loss = in_batch_contrastive_loss([query, query], [document, document])
print(loss.loss, loss.positives, loss.negatives)

pooled = hierarchical_pool(document, factor=2)

index = MultiVectorIndex()
index.add("page-1", document)
index.add("page-2", [[-1.0, 0.0]])
for hit in index.search(query, top_k=2, candidate_k=2):
    print(hit.doc_id, hit.score, hit.coarse_score)
```

## Test

```bash
python3 -m unittest discover -s tests -v
```

16 tests cover MaxSim, loss math, pooling, quantization, and index validation.

## Synthetic benchmark

```bash
python3 benchmarks/bench_synthetic.py
```

Measured on 2026-10-31 with CPython 3.14 on Apple Silicon, seed 7, 500 documents
× 12 tokens × 32 dimensions, 20 queries, `top_k=5`, `candidate_k=50`:

| Metric | Value |
|---|---:|
| Recall@5 | 1.000 |
| p50 latency | 66.531 ms |
| p95 latency | 130.350 ms |

These are **synthetic clustered vectors**, not vidore/esg_reports_v2,
vidore/biomedical_lectures_v2, or `vidore/colpali_train_set`. The latency is pure
Python; it is not a Qdrant/Ampere/WebGPU result and makes no sub-millisecond
claim.

## Boundaries

- No model weights, image preprocessing, or PaliGemma/ColQwen2/ModernVBERT
  adapters in this repository.
- No GPU training, QLoRA, or contrastive fine-tuning run.
- No Qdrant deployment or binary-quantization benchmark.
- No vidore NDCG@5 evaluation, no HF Space, no trained checkpoint.
- The package is MIT; model weights have their own licenses.

## Next layer

1. Add a `ColPaliEncoder` adapter that consumes page images and produces the
   `E_q`/`E_d` matrices with the correct projection dimension.
2. Export a Qdrant binary-quantized collection and compare coarse/fine ranking.
3. Run NDCG@5 on the vidore datasets when GPU/model access exists.
4. Build the PDF patch-heatmap UI on top of the index hits.
