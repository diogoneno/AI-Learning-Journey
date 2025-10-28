import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db_day30"

def load_docs():
    texts, metas = [], []
    for line in (DATA / "mini_corpus.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        doc_id, text = line.split(":", 1)
        texts.append(text.strip())
        metas.append({"id": doc_id.strip()})
    return texts, metas

def build_store():
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="day30", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    return vs

def main():
    texts, metas = load_docs()
    # fresh collection: drop if exists
    vs0 = build_store()
    if "day30" in [c.name for c in vs0._client.list_collections()]:
        vs0._client.delete_collection("day30")
    vs = build_store()

    print("Adding documents...")
    ids = vs.add_texts(texts, metadatas=metas)
    vs.persist()
    print(f"Added {len(ids)} items. Persisted at {CHROMA_DIR}")

    print("Re-opening the store...")
    vs_reload = build_store()
    res = vs_reload.similarity_search("what is the cia triad", k=3)
    print("Top-3:", [r.page_content[:60] for r in res])

    # Demonstrate delete by ID (use first id)
    print("Deleting first document...")
    vs_reload.delete(ids=[ids[0]])
    vs_reload.persist()

    res2 = vs_reload.similarity_search("what is the cia triad", k=3)
    print("Top-3 after delete:", [r.page_content[:60] for r in res2])

if __name__ == "__main__":
    main()
