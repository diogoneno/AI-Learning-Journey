
## generate_progress.py
```python
from pathlib import Path

root = Path(__file__).resolve().parents[1]  # repo root
days = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("day")],
              key=lambda p: int(''.join(ch for ch in p.name if ch.isdigit()) or 0))

lines = ["# 📘 AI Engineering — Stage 1 Progress\n"]
lines.append("| Day | Folder | Notes |")
lines.append("|---:|:-------|:------|")

for d in days:
    num = ''.join(ch for ch in d.name if ch.isdigit())
    lines.append(f"| {num} | [{d.name}]({d.name}/) |  |")

(root / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")
print("✅ PROGRESS.md generated at repo root.")
