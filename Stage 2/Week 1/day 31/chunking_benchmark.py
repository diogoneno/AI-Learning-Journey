import os, re, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def load_text():
    return (DATA / "long_text.txt").read_text(encoding="utf-8")

def split_fixed(text, size=250, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0: start = 0
    return [c for c in chunks if c]

def split_recursive(text, size=300, overlap=60):
    # simplistic recursive-like splitter using paragraphs/sentences fallbacks
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + " " + p).strip()
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    # add overlaps
    out2 = []
    for i, c in enumerate(out):
        prev_tail = out[i-1][-overlap:] if i > 0 else ""
        out2.append((prev_tail + " " + c).strip())
    return out2

def split_sentences(text, size=300, overlap=40):
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= size:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunks.append(buf)
            buf = s
    if buf: chunks.append(buf)
    # overlaps
    out = []
    for i,c in enumerate(chunks):
        prev_tail = chunks[i-1][-overlap:] if i>0 else ""
        out.append((prev_tail + " " + c).strip())
    return out

def eval_strategy(name, chunks, queries, model):
    # embed chunks
    chunk_vecs = model.encode(chunks, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(chunk_vecs)
    hits = 0
    for q in queries:
        q_vec = model.encode([q["question"]], normalize_embeddings=True)[0]
        _, idxs = nn.kneighbors([q_vec], n_neighbors=3)
        top_text = " ".join(chunks[i] for i in idxs[0]).lower()
        ok = any(kw in top_text for kw in [k.lower() for k in q["keywords"]])
        hits += int(ok)
    return hits / len(queries)

def main():
    text = load_text()
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    strategies = {
        "fixed_250_50": split_fixed(text, 250, 50),
        "recursive_300_60": split_recursive(text, 300, 60),
        "sentence_300_40": split_sentences(text, 300, 40),
    }
    rows = []
    for name, chunks in strategies.items():
        acc = eval_strategy(name, chunks, queries, model)
        rows.append({"strategy": name, "accuracy_at_3": acc, "num_chunks": len(chunks)})

    df = pd.DataFrame(rows).sort_values("accuracy_at_3", ascending=False)
    out_csv = OUT / "day31_chunking_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}\n", df)

if __name__ == "__main__":
    main()
