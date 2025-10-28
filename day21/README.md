# 🧪 Day 21: Mini Project — AI Research Assistant (URL Ingest + RAG)

## 🎯 Learning Objectives
- Fetch and clean web pages into **clean text**.
- Embed + index with **ChromaDB** and **HF embeddings**.
- Ask **grounded questions** with a local LLM (LM Studio).
- Provide an **interactive Gradio UI**.

> Note: This script fetches URLs you provide. Use only content you have the right to download/process.

## 🧩 What You’ll Build
- Paste one or more URLs,
- We fetch the pages, extract text, embed + index,
- You ask questions; answers are produced with retrieved context.

## 🔧 Setup
```bash
pip install -r requirements.txt
# Start LM Studio, load a model, enable local API server.
