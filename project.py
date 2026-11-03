import time, math, re, sys
import numpy as np, pandas as pd, faiss
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

DATASET = "scifact"                      # "scifact" or "fiqa"
BI_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_FIRST = 100        # candidates from first stage
K_RERANK = 50        # candidates sent to cross-encoder
RRF_K = 60           # reciprocal rank fusion constant

# ---------------- data ----------------
corpus = load_dataset(f"BeIR/{DATASET}", "corpus", split="corpus")
queries_ds = load_dataset(f"BeIR/{DATASET}", "queries", split="queries")
qrels_ds = load_dataset(f"BeIR/{DATASET}-qrels", split="test")

doc_ids = [str(d["_id"]) for d in corpus]
doc_text = [((d["title"] or "") + ". " + d["text"]).strip() for d in corpus]
id2text = dict(zip(doc_ids, doc_text))
qrels = {}
for r in qrels_ds:
    if int(r["score"]) > 0:
        qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
queries = {str(q["_id"]): q["text"] for q in queries_ds if str(q["_id"]) in qrels}
qids = list(queries)
print(f"[{DATASET}] docs={len(doc_ids)} test_queries={len(qids)}")

# ---------------- metrics ----------------
def ndcg(ranked, rel, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0
def mrr(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in rel: return 1 / (i + 1)
    return 0.0
def recall(ranked, rel, k=100):
    return len(set(ranked[:k]) & rel) / len(rel)

rows, runs = [], {}
def record(name, run, per_query_s):
    runs[name] = run
    rows.append({
        "system": name,
        "NDCG@10": np.mean([ndcg(run[q], qrels[q]) for q in qids]),
        "MRR@10": np.mean([mrr(run[q], qrels[q]) for q in qids]),
        "Recall@100": np.mean([recall(run[q], qrels[q]) for q in qids]),
        "ms/query": 1000 * per_query_s,
    })
    print(rows[-1])

tok = lambda s: re.findall(r"\w+", s.lower())

# ---------------- 1) BM25 ----------------
bm25 = BM25Okapi([tok(t) for t in doc_text])
t0 = time.time(); run_bm25 = {}
for q in qids:
    s = bm25.get_scores(tok(queries[q]))
    run_bm25[q] = [doc_ids[i] for i in np.argsort(-s)[:K_FIRST]]
record("BM25", run_bm25, (time.time() - t0) / len(qids))

# ---------------- 2) Dense retrieval with FAISS ----------------
bi = SentenceTransformer(BI_MODEL)
emb = bi.encode(doc_text, batch_size=64, show_progress_bar=True,
                normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
t0 = time.time()
qe = bi.encode([queries[q] for q in qids], normalize_embeddings=True).astype("float32")
_, I = index.search(qe, K_FIRST)
run_dense = {q: [doc_ids[j] for j in I[i]] for i, q in enumerate(qids)}
record("Dense (bi-encoder + FAISS)", run_dense, (time.time() - t0) / len(qids))

# ---------------- 3) Hybrid: reciprocal rank fusion ----------------
def rrf(*rankings, k=RRF_K, top=K_FIRST):
    score = {}
    for rk in rankings:
        for r, d in enumerate(rk):
            score[d] = score.get(d, 0) + 1 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]
t0 = time.time()
run_hybrid = {q: rrf(run_bm25[q], run_dense[q]) for q in qids}
record("Hybrid (BM25 + dense, RRF)", run_hybrid,
       rows[0]["ms/query"] / 1000 + rows[1]["ms/query"] / 1000 + (time.time() - t0) / len(qids))

# ---------------- 4) Cross-encoder rerank ----------------
ce = CrossEncoder(CE_MODEL)
def rerank(run, name):
    t0 = time.time(); out = {}
    for q in qids:
        c = run[q][:K_RERANK]
        sc = ce.predict([(queries[q], id2text[d]) for d in c], batch_size=32)
        out[q] = [c[i] for i in np.argsort(-sc)]
    return out, (time.time() - t0) / len(qids)
base_ms = {r["system"]: r["ms/query"] / 1000 for r in rows}
rd, t = rerank(run_dense, "dense"); record("Dense + cross-encoder", rd, base_ms["Dense (bi-encoder + FAISS)"] + t)
rh, t = rerank(run_hybrid, "hybrid"); record("Hybrid + cross-encoder", rh, base_ms["Hybrid (BM25 + dense, RRF)"] + t)

# ---------------- outputs ----------------
df = pd.DataFrame(rows).round(3)
df.to_csv("results.csv", index=False)
with open("results.md", "w") as f:
    f.write(df.to_markdown(index=False))
print("\n" + df.to_markdown(index=False))

# failure analysis on the best system (by NDCG@10)
best = df.sort_values("NDCG@10", ascending=False).iloc[0]["system"]
best_run = runs[best]
per_q = {q: ndcg(best_run[q], qrels[q]) for q in qids}
worst = sorted(qids, key=lambda q: per_q[q])[:10]
lens = pd.Series({q: len(tok(queries[q])) for q in qids})
bucket = pd.cut(lens, [0, 8, 14, 1000], labels=["short(<=8w)", "medium(9-14w)", "long(15w+)"])
by_len = pd.Series(per_q).groupby(bucket).mean().round(3)
with open("failures.md", "w") as f:
    f.write(f"Best system: {best}\n\nNDCG@10 by query length:\n{by_len.to_markdown()}\n\nWorst queries:\n")
    for q in worst:
        f.write(f"- ({per_q[q]:.2f}) {queries[q]}\n")
print(f"\nBest system: {best}\nNDCG@10 by query length:\n{by_len}")
print("Saved results.md, results.csv, failures.md")
