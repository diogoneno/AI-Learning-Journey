
## 🐍 rag_system.py
```python
import os
import requests
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CHROMA_DIR = "chroma_db_day16"

def load_knowledge(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")

def build_vector_store(text: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="day16", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    # Clean old index for reproducibility
    vs._client.delete_collection("day16") if "day16" in [c.name for c in vs._client.list_collections()] else None
    vs = Chroma(collection_name="day16", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    vs.add_texts(chunks)
    vs.persist()
    return vs

def lmstudio_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["text"].strip()

def build_prompt(context: str, question: str) -> str:
    return (
        "You are a concise assistant. Answer the QUESTION **using only** the CONTEXT. "
        "If the answer is not in the context, reply: 'I do not know based on the provided context.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )

def main():
    kb = load_knowledge("knowledge.txt")
    vs = build_vector_store(kb)

    print("📚 RAG assistant ready (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            print("AI: Goodbye!")
            break

        docs = vs.similarity_search(q, k=4)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = build_prompt(context, q)
        try:
            ans = lmstudio_complete(prompt)
        except Exception as e:
            ans = f"Error contacting LM Studio: {e}"
        print(f"\nAI: {ans}")

if __name__ == "__main__":
    main()
