
## 🐍 research_assistant.py
```python
import os
import requests
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import trafilatura

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CHROMA_DIR = "chroma_db_day21"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

vector_store = None

def fetch_clean(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded, include_tables=False, include_comments=False, favor_recall=True) or ""
    return text.strip()

def ingest_urls(urls_text: str) -> str:
    global vector_store
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not urls:
        return "❌ Please provide at least one URL."

    texts = []
    for u in urls:
        try:
            txt = fetch_clean(u)
            if txt:
                texts.append(txt)
        except Exception:
            pass

    if not texts:
        return "❌ Could not extract text from provided URLs."

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeds = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name="day21", embedding_function=embeds, persist_directory=CHROMA_DIR)
    # reset for clean session
    for c in [c.name for c in vs._client.list_collections()]:
        if c == "day21":
            vs._client.delete_collection("day21")
    vector_store = Chroma(collection_name="day21", embedding_function=embeds, persist_directory=CHROMA_DIR)
    vector_store.add_texts(chunks)
    vector_store.persist()
    return f"✅ Ingested {len(chunks)} chunks from {len(urls)} URL(s)."

def lmstudio_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def ask(query: str) -> str:
    if vector_store is None:
        return "Please ingest URLs first."
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
    gr.Markdown("# 🧪 AI Research Assistant (URL → RAG)")
    urls = gr.Textbox(label="URLs (one per line)", lines=6, placeholder="https://example.com/article-1\nhttps://example.com/article-2")
    ingest_btn = gr.Button("Ingest URLs")
    status = gr.Markdown()

    query = gr.Textbox(label="Ask a question")
    ask_btn = gr.Button("Get Answer")
    answer = gr.Textbox(label="Answer")

    ingest_btn.click(ingest_urls, inputs=urls, outputs=status)
    ask_btn.click(ask, inputs=query, outputs=answer)

if __name__ == "__main__":
    demo.launch()
