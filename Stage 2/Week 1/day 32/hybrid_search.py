import argparse, json, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def load_corpus():
    ids, docs = [], []
    for line in (DATA / "mini_corpus.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        i, t = line.split(":", 1)
        ids.append(i.strip()); docs.append(t.strip())
    return ids, docs

def load_queries():
    return json.loads((DATA / "queries.json").read_text(encoding="utf-8"))

def normalise(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0: return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def main(alpha: float, top_k: int):
    ids, docs = load_corpus()
    tokenised = [re.findall(r"\w+", d.lower()) for d in docs]
    bm25 = BM25Okapi(tokenised)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_vecs = model.encode(docs, normalize_embeddings=True)

    examples = []
    for q in load_queries():
        toks = re.findall(r"\w+", q["question"].lower())
        bm25_scores = bm25.get_scores(toks)

        q_vec = model.encode([q["question"]], normalize_embeddings=True)[0]
        cos_scores = (doc_vecs @ q_vec)  # cosine since normalized

        # normalise for fusion
        b = normalise(bm25_scores)
        c = normalise(cos_scores)
        fused = alpha * b + (1 - alpha) * c

        idxs = np.argsort(-fused)[:top_k]
        examples.append({
            "question": q["question"],
            "ranked": [{"id": ids[i], "score": float(fused[i])} for i in idxs]
        })
        print("\nQ:", q["question"])
        for i in idxs:
            print(f"  - {ids[i]} | {fused[i]:.3f} | {docs[i][:70]}")

    (OUT / "day32_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Wrote {OUT/'day32_examples.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    main(args.alpha, args.top_k)
