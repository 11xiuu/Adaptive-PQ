#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


MIN_RECALL = 0.7


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_dataset(result_subdir: Path) -> None:
    """Plot all JSON results in a single dataset directory."""
    json_files = sorted(result_subdir.glob("*.json"))
    if not json_files:
        return

    out_path = result_subdir / "time_recall_curve.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    for jf in json_files:
        data = load_json(jf)
        name = data.get("name", jf.stem)
        alpha = data.get("alpha")

        rows = data["results"]
        rows.sort(key=lambda r: r["nprobe"])

        # Truncate at MIN_RECALL
        cut = 0
        for i, r in enumerate(rows):
            if r["recall"] >= MIN_RECALL:
                cut = i
                break
        rows = rows[cut:]
        if not rows:
            continue

        time_ms = [r["time_ms"] for r in rows]
        recall = [r["recall"] for r in rows]

        avg_bits = data.get("average_bits")
        bits_str = f" {avg_bits:.0f}b" if avg_bits else ""

        if alpha is not None:
            label = f"{name}{bits_str}  α={alpha}"
        else:
            label = f"{name}{bits_str}"

        marker = "s" if "faiss" in name.lower() else "o"
        ls = "--" if "faiss" in name.lower() else "-"
        ax.plot(time_ms, recall, marker=marker, linestyle=ls, label=label)

    ax.set_title(f"{result_subdir.name}  (recall ≥ {MIN_RECALL})")
    ax.set_xlabel("avg time per query (ms)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(MIN_RECALL - 0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "result"
    subdirs = sorted([d for d in result_dir.iterdir() if d.is_dir()])

    if not subdirs:
        print("no dataset subdirectories found in result/")
        return

    for d in subdirs:
        print(f"plotting {d.name}")
        plot_dataset(d)


if __name__ == "__main__":
    main()
