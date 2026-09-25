"""Plot archived aggregate Copying results; no model loading or GPU work.

Run from any directory with the repository's matplotlib environment.
The LogKV 1b evaluation does not record its checkpoint or evaluation seed.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    root = Path(__file__).resolve().parent / "comparison-1b"
    provenance = json.loads((root / "provenance.json").read_text())
    for filename, metadata in provenance["files"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == metadata["sha256"], filename

    logkv = json.loads((root / "logkv-results.json").read_text())
    config = json.loads((root / "logkv-run_config.json").read_text())
    mamba = json.loads((root.parent / "results/metrics.json").read_text())
    assert logkv["train_max_t"] == config["max_t"] == 2028
    assert logkv["samples"] == 8 and logkv["precision"] == "bf16"
    logkv_points = sorted((int(t), row) for t, row in logkv["results"].items())
    expected_grid = sorted(set(range(1, 15)) | {2**k for k in range(4, 31)} | {3 * 2**k for k in range(3, 29)})
    assert [t for t, _ in logkv_points] == expected_grid
    assert all(row["n"] == 8 and row["string_acc"] == row["token_acc"] == 1 for _, row in logkv_points)
    groups = {ckpt: sorted((r for r in mamba if r["checkpoint"] == ckpt), key=lambda r: r["T"])
              for ckpt in ("best", "final")}
    for rows in groups.values():
        assert [r["T"] for r in rows] == [t for t in expected_grid if t <= 131072]
        for r in rows:
            assert r["samples"] == 256
            assert r["string_acc"] == r["string_correct"] / 256
            assert r["token_acc"] == r["token_correct"] / 2560

    # A flat table makes the plotted values accessible without matplotlib.
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["model", "checkpoint", "T", "samples", "string_acc", "token_acc"])
        writer.writerows(["LogKV (1b)", "unspecified", t, r["n"], r["string_acc"], r["token_acc"]]
                         for t, r in logkv_points)
        for ckpt, rows in groups.items():
            writer.writerows(["Mamba-2", ckpt, r["T"], r["samples"], r["string_acc"], r["token_acc"]]
                             for r in rows)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, metric, title in zip(axes, ("string_acc", "token_acc"),
                                 ("Exact match (all 10 digits)", "Digit accuracy")):
        ax.axvspan(131072, 2e9, color="#f1f3f5", zorder=0)
        for ckpt, color, marker, style in (("best", "#0072B2", "o", "-"),
                                            ("final", "#D55E00", "s", "--")):
            rows = groups[ckpt]
            ax.plot([r["T"] for r in rows], [100 * r[metric] for r in rows],
                    color=color, marker=marker, markersize=3.5, linewidth=1.8,
                    linestyle=style, label=f"Mamba-2 {ckpt} (n=256)", zorder=3)
        ax.plot([t for t, _ in logkv_points], [100 * r[metric] for _, r in logkv_points],
                color="#009E73", linewidth=1.7, linestyle=":", marker="^", markersize=4,
                markerfacecolor="white", label="LogKV 1b run (n=8)", zorder=4)
        ax.axvline(2028, color="#777777", linewidth=1, linestyle=":")
        ax.axvline(131072, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.text(2028, 5, "Train max T = 2,028", rotation=90, va="bottom", ha="right", fontsize=9, color="#555555")
        ax.text(1.4e7, 49, "Mamba-2\nnot evaluated", ha="center", va="center", color="#606870")
        ax.annotate("LogKV: 8/8 exact\nT = 1,073,741,824", xy=(2**30, 100),
                    xytext=(0.98, 0.77), textcoords="axes fraction", ha="right", fontsize=10,
                    color="#007c59", arrowprops={"arrowstyle": "-", "color": "#009E73"})
        ax.set_xscale("log")
        ax.set_xlim(0.85, 2e9)
        ax.set_ylim(-3, 108)
        ax.set_xticks([1, 1e3, 1e6, 1e9], ["1", r"$10^3$", r"$10^6$", r"$10^9$"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Copying horizon T (log scale)")
        ax.set_title(title, fontsize=13)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in (2, 0, 1)], [labels[i] for i in (2, 0, 1)],
               loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3, frameon=False)
    fig.suptitle("Fixed-10 Copying: LogKV through 2\u00b3\u2070 vs Mamba-2 through 2\u00b9\u2077", fontsize=16, y=0.985)
    fig.text(0.5, 0.075, "Both runs: 50,000 training steps; training T \u2264 2,028. LogKV: 5.79M parameters; Mamba-2: 3.45M.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.035, "Separate runs; n is per horizon. LogKV evaluation checkpoint/seed unspecified. Lines connect measured points only.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.065, right=0.98, top=0.81, bottom=0.21, wspace=0.13)
    for ext in ("png", "svg"):
        fig.savefig(root / f"comparison.{ext}", dpi=180, metadata={"Date": None} if ext == "svg" else None)
    svg = root / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Validated {len(logkv_points)} LogKV + {len(mamba)} Mamba-2 points; saved PNG, SVG and CSV to {root}")


if __name__ == "__main__":
    main()
