
## 🐍 search_app.py
```python
import os
import requests
import gradio as gr
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db_day20"

vector_store = None

def ingest_folder(folder_path: str) -> str:
    global vector_store
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        return "❌ Invalid folder."
    texts = []
    for fp in p.rglob("*.txt"):
        try:
            texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    if not texts:
        return "❌ No .txt files found."

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeds = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name="day20", embedding_function=embeds, persist_directory=CHROMA_DIR)
    # reset for clean session
    for c in [c.name for c in vs._client.list_collections()]:
        if c == "day20":
            vs._client.delete_collection("day20")
    vector_store = Chroma(collection_name="day20", embedding_function=embeds, persist_directory=CHROMA_DIR)
    vector_store.add_texts(chunks)
    vector_store.persist()

    return f"✅ Ingested {len(chunks)} chunks from {folder_path}"

def lmstudio_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def ask(query: str) -> str:
    if vector_store is None:
        return "Please ingest a folder first."
    docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "Use only the CONTEXT to answer the QUESTION concisely. If not in context, say you do not know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )
    try:
        return lmstudio_complete(prompt)
    except Exception as e:
        return f"Error contacting LM Studio: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🔎 Local Search Engine (Folder RAG)")
    folder = gr.Textbox(label="Folder path with .txt files", placeholder="./docs")
    ingest_btn = gr.Button("Ingest Folder")
    status = gr.Markdown()

    query = gr.Textbox(label="Search / Question")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer")

    ingest_btn.click(ingest_folder, inputs=folder, outputs=status)
    ask_btn.click(ask, inputs=query, outputs=answer)

if __name__ == "__main__":
    demo.launch()
