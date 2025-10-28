
## optimizer_demo.py
```python
import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CACHE_PATH = Path("day23_cache.json")

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(d: dict):
    CACHE_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def lmstudio_complete(prompt: str, max_tokens: int = 200, temperature: float = 0.4, retries: int = 3, timeout: int = 60) -> str:
    payload = {"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    last_err = None
    for _ in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"LM Studio request failed after retries: {last_err}")

def batch_generate(prompts: List[str]) -> List[str]:
    """
    Simple sequential batching that respects cache and prints timing.
    """
    cache = load_cache()
    outputs = []
    t0 = time.perf_counter()
    for p in prompts:
        if p in cache:
            outputs.append(cache[p])
            continue
        out = lmstudio_complete(p)
        cache[p] = out
        outputs.append(out)
    save_cache(cache)
    t1 = time.perf_counter()
    print(f"[Batch] processed {len(prompts)} prompts in {t1 - t0:.2f}s (cached: {sum(1 for p in prompts if p in cache)})")
    return outputs

def main():
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        prompts = [line.strip() for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        prompts = [
            "Summarise Zero Trust in 3 bullet points.",
            "Explain defence-in-depth in one paragraph.",
            "List 4 typical controls from ISO 27001 Annex A."
        ]
    outs = batch_generate(prompts)
    for i, (p, o) in enumerate(zip(prompts, outs), 1):
        print(f"\n[{i}] PROMPT: {p}\nRESPONSE: {o}")

if __name__ == "__main__":
    main()
