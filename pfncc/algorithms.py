from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(slots=True)
class ClusteringPair:
    kmeans_labels: np.ndarray
    kmedoids_labels: np.ndarray


def run_fgc(n: int, k: int, dist_matrix: np.ndarray) -> np.ndarray:
    labels = np.full(n, -1, dtype=int)
    d_local = np.array(dist_matrix, copy=True)
    block_size = math.ceil(n / k)
    cluster_id = 0

    while True:
        if d_local.shape[0] < block_size:
            break
        row_idx = int(np.argmin(np.partition(d_local, block_size - 1, axis=1)[:, block_size - 1]))
        row = d_local[row_idx]
        threshold = float(np.max(np.sort(row)[:block_size]))
        if np.isinf(threshold):
            break
        cluster = np.argsort(row)[:block_size]
        labels[cluster] = cluster_id

        d_local[:, cluster] = np.inf
        d_local[cluster, :] = np.inf
        cluster_id += 1

    labels[labels == -1] = cluster_id
    return labels


def _pam_kmedoids(dist_matrix: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100) -> np.ndarray:
    n = dist_matrix.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)

    medoids = np.array(rng.choice(n, size=k, replace=False), dtype=int)

    def assign(meds: np.ndarray) -> tuple[np.ndarray, float]:
        d = dist_matrix[:, meds]
        assignment = np.argmin(d, axis=1)
        total = float(np.sum(np.min(d, axis=1)))
        return assignment, total

    assignment, best_cost = assign(medoids)

    for _ in range(max_iter):
        improved = False
        medoid_set = set(medoids.tolist())

        for mi, medoid in enumerate(medoids):
            for cand in range(n):
                if cand in medoid_set:
                    continue
                candidate_medoids = medoids.copy()
                candidate_medoids[mi] = cand
                candidate_assignment, candidate_cost = assign(candidate_medoids)
                if candidate_cost + 1e-12 < best_cost:
                    medoids = candidate_medoids
                    assignment = candidate_assignment
                    best_cost = candidate_cost
                    medoid_set = set(medoids.tolist())
                    improved = True
        if not improved:
            break

    return assignment.astype(int)


def run_baselines(data: np.ndarray, k: int, dist_matrix: np.ndarray, seed: int) -> ClusteringPair:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    kmeans_labels = kmeans.fit_predict(data)

    rng = np.random.default_rng(seed)
    kmedoids_labels = _pam_kmedoids(dist_matrix, k=k, rng=rng)

    return ClusteringPair(kmeans_labels=kmeans_labels.astype(int), kmedoids_labels=kmedoids_labels)
