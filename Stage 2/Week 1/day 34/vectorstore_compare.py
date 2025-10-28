import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

EMB = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = ROOT / "chroma_db_day34"

def load():
    ids, docs = [], []
    for line in (DATA / "mini_corpus.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        i, t = line.split(":", 1)
        ids.append(i.strip()); docs.append(t.strip())
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    return ids, docs, queries

def hits_from_results(results, expected_keywords):
    txt = " ".join([r.page_content for r in results]).lower()
    return any(kw.lower() in txt for kw in expected_keywords)

def main():
    ids, docs, qs = load()
    embeds = HuggingFaceEmbeddings(model_name=EMB)

    # reset chroma collection
    vs_chroma = Chroma(collection_name="day34", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    if "day34" in [c.name for c in vs_chroma._client.list_collections()]:
        vs_chroma._client.delete_collection("day34")
    vs_chroma = Chroma(collection_name="day34", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    vs_chroma.add_texts(docs, metadatas=[{"id": i} for i in ids])
    vs_chroma.persist()

    vs_faiss = FAISS.from_texts(docs, embedding=embeds, metadatas=[{"id": i} for i in ids])

    rows = []
    for name, store in [("chroma", vs_chroma), ("faiss", vs_faiss)]:
        t_start = time.perf_counter()
        hits = 0
        for q in qs:
            res = store.similarity_search(q["question"], k=3)
            hits += int(hits_from_results(res, q["expected_keywords"]))
        t_end = time.perf_counter()
        rows.append({
            "vectorstore": name,
            "accuracy_at_3": hits / len(qs),
            "elapsed_seconds": round(t_end - t_start, 4)
        })

    df = pd.DataFrame(rows).sort_values("accuracy_at_3", ascending=False)
    out_csv = OUT / "day34_vectorstore_compare.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}\n", df)

if __name__ == "__main__":
    main()
