# 📚 Day 16: Retrieval-Augmented Generation (RAG) — Local + Private

## 🎯 Learning Objectives
- Understand **RAG** and why retrieval improves factual accuracy.
- Build a **local** retrieval pipeline with **ChromaDB** + **HuggingFace embeddings**.
- Query a **local LLM** (via LM Studio) using retrieved context.
- Keep everything **offline / privacy-preserving** (no OpenAI key needed).

## 🧩 What You’ll Build
A CLI RAG assistant that:
1) Indexes `knowledge.txt` into a local vector DB,
2) Retrieves the most relevant chunks for a question,
3) Calls your **LM Studio** model with a grounded prompt.

## 🔧 Setup
```bash
pip install -r requirements.txt
# Start LM Studio, load a model (e.g. mistralai/Mistral-7B-Instruct-v0.1),
# enable the local API server (default: http://localhost:1234/v1)
