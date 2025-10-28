
## fastapi_rag_service.py
```python
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db_day26"

app = FastAPI(title="RAG Service")

emb = HuggingFaceEmbeddings(model_name=EMBED)
vector_store = Chroma(collection_name="day26", embedding_function=emb, persist_directory=CHROMA_DIR)

class IngestBody(BaseModel):
    text: str

class AskBody(BaseModel):
    query: str

def lmstudio_complete(prompt: str, max_tokens: int = 220, temperature: float = 0.2) -> str:
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

@app.post("/ingest")
def ingest(body: IngestBody):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(body.text or "")
    if not chunks:
        return {"status": "no text"}
    # reset collection for clean demo
    for c in [c.name for c in vector_store._client.list_collections()]:
        if c == "day26":
            vector_store._client.delete_collection("day26")
    vs = Chroma(collection_name="day26", embedding_function=emb, persist_directory=CHROMA_DIR)
    vs.add_texts(chunks)
    vs.persist()
    return {"status": "ok", "chunks": len(chunks)}

@app.post("/ask")
def ask(body: AskBody):
    vs = Chroma(collection_name="day26", embedding_function=emb, persist_directory=CHROMA_DIR)
    docs = vs.similarity_search(body.query, k=5)
    if not docs:
        return {"answer": "No data indexed yet."}
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "Use only the CONTEXT to answer the QUESTION concisely. If not in context, say you do not know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {body.query}\nANSWER:"
    )
    try:
        ans = lmstudio_complete(prompt)
    except Exception as e:
        ans = f"Error contacting LM Studio: {e}"
    return {"answer": ans}
