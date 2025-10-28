
## 🐍 qa_system.py
```python
import os
import io
import requests
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CHROMA_DIR = "chroma_db_day18"

# Build (or reuse) embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(collection_name="day18", embedding_function=embeddings, persist_directory=CHROMA_DIR)

def reset_index():
    # Drop and recreate the collection for a clean session
    all_names = [c.name for c in vector_store._client.list_collections()]
    if "day18" in all_names:
        vector_store._client.delete_collection("day18")
    return Chroma(collection_name="day18", embedding_function=embeddings, persist_directory=CHROMA_DIR)

def process_document(file) -> str:
    """
    Accept a .txt file, chunk + embed it into Chroma.
    """
    global vector_store
    vector_store = reset_index()

    raw = file.read()
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)
    vector_store.add_texts(chunks)
    vector_store.persist()
    return f"✅ Ingested {len(chunks)} chunks."

def lmstudio_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    payload = {"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def ask_question(query: str) -> str:
    docs = vector_store.similarity_search(query, k=5)
    if not docs:
        return "No data indexed yet. Please upload a text file first."
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
    gr.Markdown("# ❓ AI-Powered Q&A with Vector Search (Local)")
    gr.Markdown("Upload a `.txt` file, then ask questions grounded in your content.")

    with gr.Row():
        file = gr.File(label="Upload .txt", file_types=[".txt"])
        ingest_btn = gr.Button("Process Document")
    status = gr.Markdown()

    with gr.Row():
        question = gr.Textbox(label="Ask a question")
        ask_btn = gr.Button("Get Answer")
    answer = gr.Textbox(label="Answer")

    ingest_btn.click(process_document, inputs=file, outputs=status)
    ask_btn.click(ask_question, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
