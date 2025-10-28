# 🧪 Day 25: Fine-tuning with LoRA (Tiny Demo)

## 🎯 Learning Objectives
- Understand the **LoRA** approach for parameter-efficient fine-tuning.
- Run a **small, fast demo** on CPU/GPU using a tiny dataset.
- Save and load the **LoRA adapter** for inference.

> Note: This is a **toy** run (minutes) to learn the mechanics. For real results, use a larger dataset and train longer.

## 🔧 Setup
```bash
pip install -r requirements.txt
# Optional env:
# export MODEL_ID="gpt2"   # default; tiny and quick
# export OUTPUT_DIR="outputs/lora-gpt2"
