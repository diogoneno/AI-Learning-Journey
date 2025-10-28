# 🧩 Day 26: Productisation — RAG as a FastAPI Service

## 🎯 Learning Objectives
- Expose your RAG pipeline as **HTTP endpoints**.
- Provide `/ingest` and `/ask` routes.
- Keep it **local + private** (LM Studio for generation).

## 🔧 Setup
```bash
pip install -r requirements.txt
# (optional) export LMSTUDIO_API_URL="http://localhost:1234/v1/completions"
# (optional) export LMSTUDIO_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
uvicorn fastapi_rag_service:app --reload --port 8000
