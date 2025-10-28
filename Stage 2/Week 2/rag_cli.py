import os, argparse, json
from pathlib import Path
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db_day36"

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")

def build_index():
    text = (DATA / "knowledge.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="day36", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    if "day36" in [c.name for c in vs._client.list_collections()]:
        vs._client.delete_collection("day36")
    vs = Chroma(collection_name="day36", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    vs.add_texts(chunks)
    vs.persist()
    print(f"✅ Indexed {len(chunks)} chunks into {CHROMA_DIR}")

def ask_once(query: str) -> str:
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="day36", embedding_function=embeds, persist_directory=str(CHROMA_DIR))
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "Index empty. Run with --build first."
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "Answer ONLY using the CONTEXT. If not found, reply 'I do not know based on the provided context.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": 220, "temperature": 0.2}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(Re)build the vector index")
    ap.add_argument("--ask", type=str, default=None, help="Ask one question")
    args = ap.parse_args()

    if args.build:
        build_index()

    if args.ask:
        print(ask_once(args.ask))
        return

    # interactive
    print("RAG CLI (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break
        print("\nAI:", ask_once(q))

if __name__ == "__main__":
    main()
