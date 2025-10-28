# 📊 Day 17: Vector Databases (ChromaDB) — Fast Semantic Search

## 🎯 Learning Objectives
- Understand **vector databases** and semantic retrieval.
- Ingest text into **ChromaDB** with **HuggingFace embeddings**.
- Perform **similarity search** and answer questions using retrieved chunks.
- Keep LLM calls **local** with **LM Studio**.

## 🧩 What You’ll Build
A simple ingestion + query CLI:
- Ingest `knowledge.txt` → vector DB
- Ask questions → retrieve top-k chunks → answer with LM Studio

## 🔧 Setup
```bash
pip install -r requirements.txt
# Start LM Studio, load a model, enable local API server.
