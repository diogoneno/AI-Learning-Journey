# 📚 Day 36 — RAG v1 (CLI)

## Overview
A minimal Retrieval-Augmented Generation pipeline:
- Index `data/knowledge.txt` into **Chroma**
- Retrieve top-k chunks for a user query
- Call **LM Studio** locally to generate an answer grounded in context

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional env (defaults shown):
# export LMSTUDIO_API_URL="http://localhost:1234/v1/completions"
# export LMSTUDIO_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
