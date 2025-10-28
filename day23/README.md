# ⚡ Day 23: Performance Optimisation (Caching + Batching)

## 🎯 Learning Objectives
- Reduce **latency & cost** with a simple **disk cache**.
- Implement **batch generation** (multiple prompts in one run).
- Add **timeouts & retries** for resilience.

## 🧩 What You’ll Build
A small utility that:
- Reads prompts from a file or list,
- Checks a **JSON cache**,
- Calls **LM Studio** only for misses,
- Writes responses back to the cache.

## 🔧 Setup
```bash
pip install -r requirements.txt
# Optional envs:
# export LMSTUDIO_API_URL="http://localhost:1234/v1/completions"
# export LMSTUDIO_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
