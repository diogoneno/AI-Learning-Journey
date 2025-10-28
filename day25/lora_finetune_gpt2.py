
## lora_finetune_gpt2.py
```python
import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--model_id", type=str, default=os.getenv("MODEL_ID", "gpt2"))
    ap.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "outputs/lora-gpt2"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    return ap.parse_args()

def load_jsonl_dataset(path):
    # Hugging Face datasets can load local JSONL directly
    return load_dataset("json", data_files=path, split="train")

def format_example(ex):
    # Expect keys: instruction, input, output (see sample)
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    prompt = f"### Instruction:\n{instruction}\n"
    if inp:
        prompt += f"### Input:\n{inp}\n"
    prompt += "### Response:\n"
    return prompt + out

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_jsonl_dataset(args.train_file)
    def to_text(ex): return {"text": format_example(ex)}
    ds = raw.map(to_text)

    def tokenize_fn(ex):
        return tok(ex["text"], truncation=True, max_length=512)
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    lconf = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()
