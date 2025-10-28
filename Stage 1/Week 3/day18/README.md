# ❓ Day 18: AI-Powered Q&A (Gradio UI + Vector Search)

## 🎯 Learning Objectives
- Build an **interactive Q&A web app** with **Gradio**.
- Ingest uploaded `.txt` content into **ChromaDB** with **HF embeddings**.
- Retrieve top-k chunks and **ground answers** with a local LLM via **LM Studio**.
- Keep the whole flow **local and private**.

## 🧩 What You’ll Build
A browser app where you:
1) Upload a `.txt` document,
2) It’s embedded + indexed in Chroma,
3) You ask questions; answers are produced using retrieved context.

## 🔧 Setup
```bash
pip install -r requirements.txt
# Start LM Studio, load a model, enable local API server.
