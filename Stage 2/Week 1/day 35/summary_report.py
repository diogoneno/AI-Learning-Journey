from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def maybe_read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None

def main():
    # Locate CSVs (assuming sibling folders in repo)
    repo_root = ROOT.parent
    day29_csv = repo_root / "day29" / "outputs" / "day29_results.csv"
    day33_csv = repo_root / "day33" / "outputs" / "day33_embeddings_compare.csv"
    day34_csv = repo_root / "day34" / "outputs" / "day34_vectorstore_compare.csv"

    df29 = maybe_read_csv(day29_csv)
    df33 = maybe_read_csv(day33_csv)
    df34 = maybe_read_csv(day34_csv)

    lines = ["# Week 1 Summary (Days 29–35)\n"]
    # Day 29
    if df29 is not None:
        lines.append("## Day 29 — Metrics bench\n")
        lines.append(df29.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Day 29 — Metrics bench: (no CSV found)\n")

    # Day 33
    if df33 is not None:
        lines.append("\n## Day 33 — Embedding models\n")
        lines.append(df33.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("\n## Day 33 — Embedding models: (no CSV found)\n")

    # Day 34
    if df34 is not None:
        lines.append("\n## Day 34 — Vector stores\n")
        lines.append(df34.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("\n## Day 34 — Vector stores: (no CSV found)\n")

    # Simple chart (if at least one df exists)
    plt.figure()
    plotted = False
    if df29 is not None:
        plt.plot(df29["metric"], df29["accuracy_at_3"], marker="o", label="Day29 metrics")
        plotted = True
    if df33 is not None:
        plt.plot([m.split("/")[-1] for m in df33["model"]], df33["accuracy_at_3"], marker="o", label="Day33 models")
        plotted = True
    if df34 is not None:
        plt.plot(df34["vectorstore"], df34["accuracy_at_3"], marker="o", label="Day34 stores")
        plotted = True

    if plotted:
        plt.title("Accuracy@3 comparison")
        plt.xlabel("Category")
        plt.ylabel("Accuracy@3")
        plt.legend()
        fig_path = OUT / "day35_summary.png"
        plt.savefig(fig_path, bbox_inches="tight")
        lines.append(f"\n![Summary chart](day35_summary.png)")
        print(f"✅ Wrote {fig_path}")

    md_path = OUT / "day35_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote {md_path}")

if __name__ == "__main__":
    main()
