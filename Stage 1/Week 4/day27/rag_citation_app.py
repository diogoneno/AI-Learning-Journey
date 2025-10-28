
## rag_citation_app.py
```python
import os
import gradio as gr
import requests
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db_day27"

emb = HuggingFaceEmbeddings(model_name=EMBED)
vs = Chroma(collection_name="day27", embedding_function=emb, persist_directory=CHROMA_DIR)

def extract_text(file) -> str:
    suffix = Path(file.name).suffix.lower()
    if suffix == ".txt":
        return Path(file.name).read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        pdf = PdfReader(file.name)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])
    return ""

def ingest(files: List[gr.File]) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    # reset collection
    for c in [c.name for c in vs._client.list_collections()]:
        if c == "day27":
            vs._client.delete_collection("day27")
    local_vs = Chroma(collection_name="day27", embedding_function=emb, persist_directory=CHROMA_DIR)

    count = 0
    for f in files or []:
        text = extract_text(f)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        metadatas = [{"source": Path(f.name).name} for _ in chunks]
        local_vs.add_texts(chunks, metadatas=metadatas)
        count += len(chunks)

    local_vs.persist()
    return f"✅ Ingested {count} chunks from {len(files or [])} file(s)."

def lmstudio(prompt: str) -> str:
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": 220, "temperature": 0.2}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def ask(q: str) -> str:
    local_vs = Chroma(collection_name="day27", embedding_function=emb, persist_directory=CHROMA_DIR)
    docs = local_vs.similarity_search(q, k=5)
    if not docs:
        return "Index empty. Please upload files first."
    context = "\n\n".join(f"[{i+1}] ({d.metadata.get('source','?')}) {d.page_content}" for i, d in enumerate(docs))
    prompt = (
        "Use only the CONTEXT to answer the QUESTION. Include bracketed source numbers where relevant.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {q}\nANSWER (with sources):"
    )
    try:
        ans = lmstudio(prompt)
    except Exception as e:
        ans = f"Error contacting LM Studio: {e}"
    return ans

with gr.Blocks() as demo:
    gr.Markdown("# 🏁 Capstone — Multi-file RAG with Citations")
    files = gr.File(file_types=[".txt", ".pdf"], file_count="multiple", label="Upload files")
    ingest_btn = gr.Button("Ingest")
    status = gr.Markdown()

    query = gr.Textbox(label="Ask a question")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8)

    ingest_btn.click(ingest, inputs=files, outputs=status)
    ask_btn.click(ask, inputs=query, outputs=answer)

if __name__ == "__main__":
    demo.launch()
