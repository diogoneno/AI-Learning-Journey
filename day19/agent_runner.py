
## 🐍 agent_runner.py
```python
import os
import re
import glob
import time
import math
import requests
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")

# --- Optional vector search config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db_day19"

def lmstudio_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    payload = {"model": MODEL_ID, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def tool_calc(expr: str) -> str:
    # Safe arithmetic evaluator
    allowed = "0123456789.+-*/()% "
    if any(ch not in allowed for ch in expr):
        return "Error: expression contains disallowed characters."
    try:
        # Only arithmetic via eval on a restricted dict
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def tool_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def index_texts_from_dir(path: str) -> Chroma:
    path = Path(path)
    texts = []
    for p in path.rglob("*.txt"):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    if not texts:
        raise RuntimeError(f"No .txt files found in: {path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name="day19", embedding_function=embed, persist_directory=CHROMA_DIR)
    # reset collection for clean run
    for c in [c.name for c in vs._client.list_collections()]:
        if c == "day19":
            vs._client.delete_collection("day19")
    vs = Chroma(collection_name="day19", embedding_function=embed, persist_directory=CHROMA_DIR)
    vs.add_texts(chunks)
    vs.persist()
    return vs

def tool_search(path: str, question: str) -> str:
    try:
        vs = index_texts_from_dir(path)
    except Exception as e:
        return f"Search index error: {e}"
    docs = vs.similarity_search(question, k=5)
    return "\n\n".join(d.page_content for d in docs)

INSTRUCTIONS = """You are a helpful local agent. You can think then choose tools.
Use at most 4 steps. Available tools:

- CALC[<arith expr>] → evaluate arithmetic, return number
- TIME → returns current local time
- SEARCH[path:<folder_path>] question: <text> → semantic search in .txt files under path

Format your steps strictly as ONE of:
CALC[...]
TIME
SEARCH[path:... ] question: ...
FINAL: <your final concise answer>

If you do not need a tool, go directly to FINAL.
"""

def run_agent(user_query: str) -> str:
    transcript = INSTRUCTIONS + "\nUSER: " + user_query + "\nTHINK+ACT:\n"
    for _ in range(4):
        model_out = lmstudio_complete(transcript, max_tokens=220)
        line = model_out.splitlines()[0].strip() if model_out else ""
        # Normalize
        if line.startswith("FINAL:"):
            return line.replace("FINAL:", "").strip()

        if line.startswith("CALC["):
            m = re.match(r"CALC\[(.+)\]", line)
            expr = m.group(1).strip() if m else ""
            tool_result = tool_calc(expr)
            transcript += f"{line}\nOBSERVATION: {tool_result}\nTHINK+ACT:\n"
            continue

        if line.startswith("TIME"):
            tool_result = tool_time()
            transcript += f"{line}\nOBSERVATION: {tool_result}\nTHINK+ACT:\n"
            continue

        if line.startswith("SEARCH["):
            # Expect: SEARCH[path:... ] question: ...
            m = re.match(r"SEARCH\[path:(.+?)\]\s*question:\s*(.+)", line)
            if not m:
                # try a slightly different pattern
                m = re.match(r"SEARCH\[path:(.+?)\]\s*?(.*)", line)
            if m:
                path = m.group(1).strip()
                question = (m.group(2) or "").strip()
            else:
                path, question = ".", ""
            tool_result = tool_search(path, question or user_query)
            transcript += f"{line}\nOBSERVATION:\n{tool_result}\nTHINK+ACT:\n"
            continue

        # Fallback: if model doesn't follow format, force final
        return lmstudio_complete(
            INSTRUCTIONS + f"\nCONTEXT: (no tools used)\nUSER: {user_query}\nFINAL: ",
            max_tokens=150
        )
    # If loop ends without FINAL, force one
    return lmstudio_complete(transcript + "FINAL:", max_tokens=120)

def main():
    print("🤖 Agent ready (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            print("Agent: Goodbye!")
            break
        print("\nAgent:", run_agent(q))

if __name__ == "__main__":
    main()
