# 💾 Day 30 — Chroma Persistence & CRUD

## Overview
Demonstrates:
- Build a Chroma vector index (LangChain wrapper) with `all-MiniLM-L6-v2`
- **Persist** to disk, re-open later
- **Add** and **delete** documents by ID
- Simple similarity queries

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
