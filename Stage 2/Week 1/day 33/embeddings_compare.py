import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-small-v2"
]
TOP_K = 3

def load_corpus():
    ids, docs = [], []
    for line in (DATA / "mini_corpus.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        i, t = line.split(":", 1)
        ids.append(i.strip()); docs.append(t.strip())
    return ids, docs

def queries():
    return json.loads((DATA / "queries.json").read_text(encoding="utf-8"))

def accuracy(nn, q_vec, docs, expected_keywords):
    dists, idxs = nn.kneighbors([q_vec], n_neighbors=TOP_K)
    txt = " ".join(docs[i] for i in idxs[0]).lower()
    return any(kw.lower() in txt for kw in expected_keywords)

def main():
    ids, docs = load_corpus()
    qs = queries()
    rows = []
    for model_name in MODELS:
        model = SentenceTransformer(model_name)
        doc_vecs = model.encode(docs, normalize_embeddings=True)
        nn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine").fit(doc_vecs)

        hits = 0
        for q in qs:
            q_vec = model.encode([q["question"]], normalize_embeddings=True)[0]
            hits += int(accuracy(nn, q_vec, docs, q["expected_keywords"]))
        acc = hits / len(qs)
        rows.append({"model": model_name, "accuracy_at_3": acc})

    df = pd.DataFrame(rows).sort_values("accuracy_at_3", ascending=False)
    out_csv = OUT / "day33_embeddings_compare.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}\n", df)

if __name__ == "__main__":
    main()
