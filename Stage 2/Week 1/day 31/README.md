# ✂️ Day 31 — Chunking Strategies Benchmark

## Overview
Compare three chunking strategies for retrieval usefulness:
- Fixed-size windows
- RecursiveCharacter (LangChain-like)
- Naive sentence split on punctuation

Evaluate by simple accuracy@3 vs keyword-based expectations.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
