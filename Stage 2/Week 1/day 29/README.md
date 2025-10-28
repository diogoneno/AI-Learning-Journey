# 📈 Day 29 — Embedding Metrics & NN Bench

## Overview
Benchmark semantic search quality across distance metrics:
- Cosine, Dot, Euclidean
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`)
- Tiny corpus + queries with keyword-based relevance checks

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
