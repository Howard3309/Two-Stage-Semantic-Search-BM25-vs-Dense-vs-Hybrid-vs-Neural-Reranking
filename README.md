# Two-Stage Semantic Search: BM25 vs Dense vs Hybrid vs Neural Reranking

## Problem
Given a natural-language query, retrieve the most relevant documents. This project compares lexical search, dense retrieval, hybrid fusion, and cross-encoder reranking on a public benchmark, with both quality and latency measured.

## Pipeline
1. **First stage:** BM25 (lexical), bi-encoder + FAISS (dense), and Reciprocal Rank Fusion of the two (hybrid)
2. **Second stage:** cross-encoder reranking of the top 50 candidates
3. **Evaluation:** NDCG@10, MRR@10, Recall@100, latency (ms/query)

## Setup
- Dataset: BEIR SciFact, 5,183 documents, 300 test queries
- Bi-encoder: `all-MiniLM-L6-v2` | Cross-encoder: `ms-marco-MiniLM-L-6-v2`
- RRF constant k = 60; 100 first-stage candidates; top 50 reranked

## Results

| system                     |   NDCG@10 |   MRR@10 |   Recall@100 |   ms/query |
|:---------------------------|----------:|---------:|-------------:|-----------:|
| BM25                       |     0.652 |    0.619 |        0.873 |     19.865 |
| Dense (bi-encoder + FAISS) |     0.648 |    0.607 |        0.925 |      0.716 |
| Hybrid (BM25 + dense, RRF) |     0.685 |    0.654 |        0.958 |     20.667 |
| Dense + cross-encoder      |     0.688 |    0.658 |        0.889 |    261.084 |
| Hybrid + cross-encoder     |     0.692 |    0.659 |        0.940 |    290.822 |

Note: the reranked rows only reorder the top 50 candidates, so their Recall@100 is capped by that cutoff and is not comparable to the first-stage rows.

NDCG@10 of the best system by query length: short (<=8 words) 0.687, medium (9-14) 0.673, long (15+) 0.719.

## Findings
- BM25 and the dense model score almost the same NDCG@10 (0.652 vs 0.648), but dense has much higher Recall@100 (0.925 vs 0.873) and is far faster per query.
- Fusing them with RRF gave the largest gain: NDCG@10 rose to 0.685 and Recall@100 to 0.958, at about the same latency as BM25 alone.
- The cross-encoder added only +0.007 NDCG@10 over hybrid (0.692) while raising latency from about 21 ms to 291 ms per query, so on this dataset most of the quality gain came from hybrid retrieval, not reranking.

## Failure analysis
<add 2-3 patterns you see in the worst-scoring queries from failures.md, in your own words>

## Run it
```
pip install sentence-transformers rank_bm25 datasets faiss-cpu numpy pandas tabulate
python project.py
```

## Limitations / next steps
- Zero-shot models with no domain fine-tuning (next: contrastive fine-tuning with hard negatives)
- Single dataset; testing on FiQA would check generalization
- Latency numbers depend on hardware and were measured in a single notebook run
