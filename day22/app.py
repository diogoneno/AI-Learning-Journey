
## app.py
```python
import os
import gradio as gr
import requests

BACKEND = os.getenv("BACKEND", "hf").lower().strip()  # 'hf' (default) or 'lmstudio'

# --- LM Studio backend (local/offline)
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/completions")
LMSTUDIO_MODEL_ID = os.getenv("LMSTUDIO_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")

def lmstudio_generate(prompt: str, max_tokens: int = 180, temperature: float = 0.6) -> str:
    payload = {
        "model": LMSTUDIO_MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(LMSTUDIO_API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

# --- HF Transformers backend (for Spaces)
# Lightweight, CPU-friendly text2text model
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-small")

def hf_generate(prompt: str, max_tokens: int = 180, temperature: float = 0.6) -> str:
    # Lazy import so Spaces can fetch transformers at runtime
    from transformers import pipeline
    pipe = pipeline("text2text-generation", model=HF_MODEL_ID)
    out = pipe(prompt, max_length=max(16, min(256, max_tokens)), do_sample=True, temperature=temperature)
    return out[0]["generated_text"].strip()

def generate(user_text: str, preset: str, temp: float, max_toks: int):
    prompt = user_text or preset
    if not prompt:
        return "Please enter text or pick a preset."
    try:
        if BACKEND == "lmstudio":
            return lmstudio_generate(prompt, max_tokens=max_toks, temperature=temp)
        return hf_generate(prompt, max_tokens=max_toks, temperature=temp)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## ✨ Day 22 — Minimal Text App (Local or Spaces)")
    with gr.Row():
        preset = gr.Dropdown(
            label="Preset prompts",
            choices=[
                "Explain Zero Trust in one paragraph.",
                "Summarise the CIA triad in 4 bullet points.",
                "Give a concise definition of Retrieval-Augmented Generation."
            ],
            value=None
        )
    user_text = gr.Textbox(label="Or type your own prompt")
    with gr.Row():
        temp = gr.Slider(0.0, 1.2, value=0.6, step=0.1, label="Temperature")
        max_toks = gr.Slider(32, 256, value=180, step=8, label="Max tokens")
    btn = gr.Button("Generate")
    out = gr.Textbox(label="Output")

    btn.click(generate, inputs=[user_text, preset, temp, max_toks], outputs=out)

if __name__ == "__main__":
    demo.launch()
