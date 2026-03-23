from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .algorithms import run_baselines, run_fgc
from .data import load_full_dataset, sample_dataset
from .metrics import kmeans_objective, kmedoids_objective, within_cluster
from .violations import appr_fjr_violation, core_violation, fjr_violation


@dataclass(slots=True)
class ExperimentParams:
    dataset: str
    k: int
    dist_function: str = "euclidean"
    sample_n: int | None = None
    outer_repeats: int = 1
    inner_repeats: int = 20
    seed: int = 0


def _score_labels(k: int, dist_vector: np.ndarray, dist_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    theta = appr_fjr_violation(n=n, k=k, dist_matrix=dist_matrix, labels=labels)
    fjr = fjr_violation(n=n, k=k, dist_matrix=dist_matrix, labels=labels, theta=theta)
    core = core_violation(n=n, k=k, dist_matrix=dist_matrix, labels=labels, theta=theta)
    wcc = within_cluster(dist_vector, labels)
    km_obj = kmeans_objective(dist_vector, labels)
    kmed_obj = kmedoids_objective(dist_vector, labels)
    return np.array([k, theta, fjr, core, wcc, km_obj, kmed_obj], dtype=float)


def run_k_experiment(params: ExperimentParams) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    full_dataset = load_full_dataset(params.dataset)

    gc_scores = np.zeros((params.outer_repeats, 7), dtype=float)
    means_scores = np.zeros((params.outer_repeats, 7), dtype=float)
    medoids_scores = np.zeros((params.outer_repeats, 7), dtype=float)

    for outer in range(params.outer_repeats):
        data = sample_dataset(full_dataset, n=params.sample_n, rng=rng)
        n = data.shape[0]

        dist_vector = pdist(data, metric=params.dist_function)
        dist_matrix = squareform(dist_vector)

        gc_labels = run_fgc(n=n, k=params.k, dist_matrix=dist_matrix)
        gc_scores[outer] = _score_labels(params.k, dist_vector, dist_matrix, gc_labels)

        tmp_means = np.zeros((params.inner_repeats, 7), dtype=float)
        tmp_medoids = np.zeros((params.inner_repeats, 7), dtype=float)

        for inner in range(params.inner_repeats):
            pair = run_baselines(data=data, k=params.k, dist_matrix=dist_matrix, seed=int(rng.integers(1, 10_000_000)))
            tmp_means[inner] = _score_labels(params.k, dist_vector, dist_matrix, pair.kmeans_labels)
            tmp_medoids[inner] = _score_labels(params.k, dist_vector, dist_matrix, pair.kmedoids_labels)

        means_scores[outer] = np.mean(tmp_means, axis=0)
        medoids_scores[outer] = np.mean(tmp_medoids, axis=0)

    result = np.vstack(
        [
            np.mean(gc_scores, axis=0),
            1.96 * np.std(gc_scores, axis=0, ddof=0) / math.sqrt(params.outer_repeats),
            np.mean(means_scores, axis=0),
            1.96 * np.std(means_scores, axis=0, ddof=0) / math.sqrt(params.outer_repeats),
            np.mean(medoids_scores, axis=0),
            1.96 * np.std(medoids_scores, axis=0, ddof=0) / math.sqrt(params.outer_repeats),
        ]
    )
    return result


def run_dataset_experiment(
    dataset: str,
    k_values: list[int],
    out_dir: str,
    dist_function: str = "euclidean",
    seed: int = 0,
) -> None:
    dataset_l = dataset.lower()
    if dataset_l in {"iris", "wine", "student"}:
        sample_n = None
        outer_repeats = 1
        inner_repeats = 20
    else:
        sample_n = 100
        outer_repeats = 40
        inner_repeats = 20

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset, k in enumerate(k_values):
        params = ExperimentParams(
            dataset=dataset,
            k=k,
            dist_function=dist_function,
            sample_n=sample_n,
            outer_repeats=outer_repeats,
            inner_repeats=inner_repeats,
            seed=seed + offset,
        )
        result = run_k_experiment(params)
        out_file = output_dir / f"k={k}-{dataset_l}-max.csv"
        np.savetxt(out_file, result, delimiter=",", fmt="%.8f")
        print(f"Saved: {out_file}")
