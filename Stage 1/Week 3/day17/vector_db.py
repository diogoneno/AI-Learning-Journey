
## 🐍 vector_db.py
```python
import os
import requests
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CHROMA_DIR = "chroma_db_day17"

def ingest(path: str) -> Chroma:
    text = Path(path).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="day17", embedding_function=embed, persist_directory=CHROMA_DIR)
    # Reset collection for clean runs
    vs._client.delete_collection("day17") if "day17" in [c.name for c in vs._client.list_collections()] else None
    vs = Chroma(collection_name="day17", embedding_function=embed, persist_directory=CHROMA_DIR)
    vs.add_texts(chunks)
    vs.persist()
    return vs

def lmstudio(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    r = requests.post(
        API_URL,
        json={"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def answer_with_context(vs: Chroma, question: str) -> str:
    docs = vs.similarity_search(question, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "Use only the CONTEXT to answer the QUESTION. If unknown, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    try:
        return lmstudio(prompt)
    except Exception as e:
        return f"Error contacting LM Studio: {e}"

def main():
    vs = ingest("knowledge.txt")
    print("📊 Vector DB ready (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            print("AI: Goodbye!")
            break
        print("\nAI:", answer_with_context(vs, q))

if __name__ == "__main__":
    main()
