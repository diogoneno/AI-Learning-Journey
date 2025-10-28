# 🗄️ Day 34 — Vector Store Compare (Chroma vs FAISS)

## Overview
- Ingest same corpus into **Chroma** and **FAISS**
- Measure **accuracy@3** (keyword hit) and **latency** per query
- Single-run CSV with side-by-side metrics

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
