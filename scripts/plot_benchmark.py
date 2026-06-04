#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "result"
    json_path = result_dir / "benchmark_results.json"
    out_path = result_dir / "benchmark_results.png"

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = [row for cfg in data["configs"] for row in cfg["results"]]
    first_probe = min(row["nprobe"] for row in all_rows)
    last_probe = max(row["nprobe"] for row in all_rows)
    min_recall = max(row["recall"] for row in all_rows if row["nprobe"] == first_probe)
    max_time = min(row["time_ms"] for row in all_rows if row["nprobe"] == last_probe)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for cfg in data["configs"]:
        alphas = sorted({row["alpha"] for row in cfg["results"]}, reverse=True)
        for alpha in alphas:
            rows = [row for row in cfg["results"] if row["alpha"] == alpha]
            rows.sort(key=lambda row: row["nprobe"])
            rows = [
                row for row in rows
                if row["recall"] >= min_recall and row["time_ms"] <= max_time
            ]
            if not rows:
                continue
            nprobe = [row["nprobe"] for row in rows]
            time_ms = [row["time_ms"] for row in rows]
            recall = [row["recall"] for row in rows]
            exact_ratio = [row["exact_ratio"] for row in rows]
            skipped_ratio = [row["skipped_ratio"] for row in rows]
            avg_bits = rows[0].get("average_bits", cfg.get("average_bits", 0.0))
            label = f"{cfg['name']} {avg_bits:.1f}b alpha={alpha}"

            axes[0].plot(time_ms, recall, marker="o", label=label)
            axes[1].plot(nprobe, exact_ratio, marker="o", label=f"{label} exact")
            axes[1].plot(nprobe, skipped_ratio, marker="x", linestyle="--", label=f"{label} skipped")

    axes[0].set_title(f"Time-Recall Curve @ {data['topk']}")
    axes[0].set_xlabel("avg time per query (ms)")
    axes[0].set_ylabel(f"recall@{data['topk']}")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Refine Ratio")
    axes[1].set_xlabel("nprobe")
    axes[1].set_ylabel("ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"IVF={data['ivf_list_size']} base={data['base_n']} query={data['query_n']} "
        f"recall>={min_recall:.4f} time<={max_time:.4f}ms"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
