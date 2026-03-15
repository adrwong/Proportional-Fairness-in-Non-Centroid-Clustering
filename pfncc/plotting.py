from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


METRIC_COLUMNS = {
    "appr_fjr": (1, "Approx FJR Violation", "upper left"),
    "fjr": (2, "FJR Violation", "upper left"),
    "core": (3, "Core Violation", "upper left"),
    "cost": (4, "Cost", "lower left"),
    "kmeans": (5, "k-means Objective", "upper right"),
    "kmedoids": (6, "k-medoids Objective", "lower left"),
}


def _extract_method_series(results: np.ndarray, col: int) -> tuple[float, float, float, float, float, float]:
    return (
        float(results[0, col]),
        float(results[1, col]),
        float(results[2, col]),
        float(results[3, col]),
        float(results[4, col]),
        float(results[5, col]),
    )


def plot_metric_from_csvs(
    dataset: str,
    k_values: list[int],
    result_dir: str,
    metric_key: str,
    out_dir: str,
    repeats: int,
) -> Path:
    if metric_key not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported metric: {metric_key}")

    col, ylabel, legend_loc = METRIC_COLUMNS[metric_key]

    gc = np.zeros((len(k_values), 2), dtype=float)
    km = np.zeros((len(k_values), 2), dtype=float)
    kmed = np.zeros((len(k_values), 2), dtype=float)

    dataset_l = dataset.lower()
    for idx, k in enumerate(k_values):
        csv_path = Path(result_dir) / f"k={k}-{dataset_l}-max.csv"
        results = pd.read_csv(csv_path, header=None).to_numpy(dtype=float)
        gc[idx] = _extract_method_series(results, col)[:2]
        km[idx] = _extract_method_series(results, col)[2:4]
        kmed[idx] = _extract_method_series(results, col)[4:6]

    x = np.array(k_values)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, gc[:, 0], label="Greedy Capture", color="g", marker="o", linewidth=0.8, markersize=4)
    ci = 1.96 * gc[:, 1] / np.sqrt(max(1, repeats))
    ax.fill_between(x, gc[:, 0] - ci, gc[:, 0] + ci, color="g", alpha=0.1)

    ax.plot(x, km[:, 0], label="k-means++", color="r", marker="o", linewidth=0.8, markersize=4)
    ci = 1.96 * km[:, 1] / np.sqrt(max(1, repeats))
    ax.fill_between(x, km[:, 0] - ci, km[:, 0] + ci, color="r", alpha=0.1)

    ax.plot(x, kmed[:, 0], label="k-medoids", color="y", marker="o", linewidth=0.8, markersize=4)
    ci = 1.96 * kmed[:, 1] / np.sqrt(max(1, repeats))
    ax.fill_between(x, kmed[:, 0] - ci, kmed[:, 0] + ci, color="y", alpha=0.1)

    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(min(k_values) - 1, max(k_values) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if metric_key in {"cost", "kmeans", "kmedoids"}:
        ax.set_ylim(bottom=0)

    ax.legend(loc=legend_loc, fontsize=12)
    fig.tight_layout()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{metric_key}_{dataset_l}.png"
    out_pdf = output_dir / f"{metric_key}_{dataset_l}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def plot_all_metrics(dataset: str, k_values: list[int], result_dir: str, out_dir: str, repeats: int) -> None:
    for metric in METRIC_COLUMNS:
        output = plot_metric_from_csvs(dataset, k_values, result_dir, metric, out_dir, repeats)
        print(f"Saved: {output}")
