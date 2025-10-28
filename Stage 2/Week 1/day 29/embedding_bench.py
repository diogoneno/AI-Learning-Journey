import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CORPUS_PATH = DATA / "mini_corpus.txt"
QUERIES_PATH = DATA / "queries.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

def load_corpus(path: Path):
    docs, ids = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        doc_id, text = line.split(":", 1)
        ids.append(doc_id.strip())
        docs.append(text.strip())
    return ids, docs

def load_queries(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def accuracy_at_k(results, docs, expected_keywords, k=TOP_K):
    # success if any of top-k docs contain any expected keyword (case-insensitive)
    top_texts = [docs[i] for i in results[:k]]
    t = " ".join(top_texts).lower()
    return any(kw.lower() in t for kw in expected_keywords)

def fit_nn(embeds: np.ndarray, metric: str):
    # map 'dot' to cosine by normalized vectors but we can emulate dot via neg-dist trick; simpler: normalize for dot/cos
    # We'll handle metrics explicitly:
    if metric == "cosine":
        nn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine").fit(embeds)
    elif metric == "euclidean":
        nn = NearestNeighbors(n_neighbors=TOP_K, metric="euclidean").fit(embeds)
    elif metric == "dot":
        # dot similarity = -euclidean distance on normalized vectors approx; use cosine by negating distances
        # We'll compute dot scores manually during query.
        nn = NearestNeighbors(n_neighbors=len(embeds), metric="cosine").fit(embeds)  # reuse cosine for candidate pool
    else:
        raise ValueError(metric)
    return nn

def rank_dot(nn_cosine, q_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int):
    # compute dot products and return top_k indices
    scores = (doc_vecs @ q_vec)
    idxs = np.argsort(-scores)[:top_k]
    return idxs

def main():
    ids, docs = load_corpus(CORPUS_PATH)
    queries = load_queries(QUERIES_PATH)
    model = SentenceTransformer(MODEL_NAME)
    doc_vecs = model.encode(docs, normalize_embeddings=True)

    metrics = ["cosine", "dot", "euclidean"]
    rows = []
    for metric in metrics:
        # prepare index
        nn = fit_nn(doc_vecs, metric)
        correct = 0
        examples = []
        for q in queries:
            q_vec = model.encode([q["question"]], normalize_embeddings=True)[0]
            if metric == "dot":
                idxs = rank_dot(nn, q_vec, doc_vecs, TOP_K)
            else:
                dists, idxs = nn.kneighbors([q_vec], n_neighbors=TOP_K)
                idxs = idxs[0]
            ok = accuracy_at_k(idxs, docs, q["expected_keywords"], k=TOP_K)
            correct += int(ok)
            examples.append({
                "question": q["question"],
                "top_ids": [ids[i] for i in idxs],
                "hit": bool(ok)
            })
        acc = correct / len(queries)
        rows.append({"metric": metric, "accuracy_at_3": acc})

        # write examples per metric
        (OUT / f"day29_{metric}_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows).sort_values("accuracy_at_3", ascending=False)
    out_csv = OUT / "day29_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}")
    print(df)

if __name__ == "__main__":
    main()
