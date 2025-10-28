
## secure_rag_qa.py
```python
import os
import re
import json
import time
import requests
from dataclasses import dataclass
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
CHROMA_DIR = "chroma_db_day24"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOG_PATH = "day24_security_log.jsonl"

SUS_PATTERNS = [
    r"ignore previous instructions",
    r"disregard.*(rules|instructions)",
    r"reveal.*(system|developer).*prompt",
    r"bypass.*(safety|guard)",
    r"format the disk", r"rm -rf", r"shutdown",
    r"prompt injection", r"jailbreak"
]

@dataclass
class QAConfig:
    k: int = 4
    max_context_chars: int = 3000
    json_mode: bool = False

def log_event(event: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def suspicious(user_input: str) -> bool:
    text = user_input.lower()
    return any(re.search(pat, text) for pat in SUS_PATTERNS)

def lmstudio_complete(prompt: str, max_tokens: int = 220, temperature: float = 0.2) -> str:
    r = requests.post(API_URL, json={"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def build_index(corpus: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(corpus)
    embeds = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name="day24", embedding_function=embeds, persist_directory=CHROMA_DIR)
    # reset
    for c in [c.name for c in vs._client.list_collections()]:
        if c == "day24":
            vs._client.delete_collection("day24")
    vs = Chroma(collection_name="day24", embedding_function=embeds, persist_directory=CHROMA_DIR)
    vs.add_texts(chunks)
    vs.persist()
    return vs

def secure_answer(vs: Chroma, question: str, cfg: QAConfig) -> str:
    t0 = time.time()
    if suspicious(question):
        msg = "⚠️ Refused: prompt appears to contain injection or unsafe instructions."
        log_event({"ts": t0, "type": "refusal", "question": question, "reason": "suspected_injection"})
        return msg

    docs = vs.similarity_search(question, k=cfg.k)
    context = "\n\n".join(d.page_content for d in docs)[: cfg.max_context_chars]

    base_prompt = (
        "You must answer **only** using the CONTEXT below. If the answer is not present, say "
        "'I do not know based on the provided context.'\n\nCONTEXT:\n"
        f"{context}\n\nQUESTION: {question}\n"
    )

    if cfg.json_mode:
        base_prompt += (
            "Return a single-line JSON object with keys: 'answer' and 'confidence' (0-1). "
            "No extra text."
        )
    else:
        base_prompt += "ANSWER:"

    try:
        raw = lmstudio_complete(base_prompt)
    except Exception as e:
        log_event({"ts": time.time(), "type": "error", "err": str(e)})
        return f"Error contacting LM Studio: {e}"

    if cfg.json_mode:
        # Validate JSON
        line = raw.strip().splitlines()[0]
        try:
            obj = json.loads(line)
            if not all(k in obj for k in ("answer", "confidence")):
                raise ValueError("Missing keys")
            return line
        except Exception:
            log_event({"ts": time.time(), "type": "validation_fail", "raw": raw})
            return "Validation error: model did not return valid JSON."

    log_event({"ts": time.time(), "type": "ok", "question": question})
    return raw

def main():
    corpus = (
        "The CIA triad comprises Confidentiality, Integrity, and Availability. "
        "Zero Trust requires verification for every request. "
        "RAG retrieves relevant context and feeds it to an LLM for grounded answers. "
        "ISO/IEC 27001 focuses on an information security management system and risk-based controls."
    )
    vs = build_index(corpus)
    cfg = QAConfig(k=4, max_context_chars=3000, json_mode=False)

    print("🔒 Secure RAG QA (type 'exit' to quit); type 'json' to toggle JSON mode.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            print("Bye.")
            break
        if q.lower() == "json":
            cfg.json_mode = not cfg.json_mode
            print(f"JSON mode: {cfg.json_mode}")
            continue
        print("\nAI:", secure_answer(vs, q, cfg))

if __name__ == "__main__":
    main()
