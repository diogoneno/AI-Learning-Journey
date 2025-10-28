# 🔀 Day 32 — Hybrid Search (BM25 + Vector)

## Overview
- Keyword relevance via **BM25** (rank_bm25)
- Semantic similarity via **Sentence-Transformers**
- Weighted fusion: `score = α*bm25 + (1-α)*cosine`

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
