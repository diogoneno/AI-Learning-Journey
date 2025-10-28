# 🔁 Day 33 — Embedding Model Swap

## Overview
Compare embedding models for retrieval accuracy@3:
- MiniLM-L6-v2 (fast, small)
- all-mpnet-base-v2 (higher quality, slower)
- intfloat/e5-small-v2 (modern, high quality)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
