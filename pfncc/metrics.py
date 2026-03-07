from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform


def _to_square_distance(dist: np.ndarray) -> np.ndarray:
    if dist.ndim == 1:
        return squareform(dist)
    return dist


def within_cluster(dist: np.ndarray, clustering: np.ndarray) -> float:
    dist_matrix = _to_square_distance(dist)
    cost = 0.0
    for cluster_id in np.unique(clustering):
        in_cluster = clustering == cluster_id
        size = int(np.sum(in_cluster))
        if size > 1:
            cost += float(np.sum(dist_matrix[np.ix_(in_cluster, in_cluster)]) / (2 * size))
    return float(cost)


def kmeans_objective(dist: np.ndarray, clustering: np.ndarray) -> float:
    dist_matrix = _to_square_distance(dist)
    cost = 0.0
    for cluster_id in np.unique(clustering):
        in_cluster = clustering == cluster_id
        size = int(np.sum(in_cluster))
        if size > 1:
            cost += float(np.sum(dist_matrix[np.ix_(in_cluster, in_cluster)] ** 2) / (2 * size))
    return float(cost)


def kmedoids_objective(dist: np.ndarray, clustering: np.ndarray) -> float:
    dist_matrix = _to_square_distance(dist)
    cost = 0.0
    for cluster_id in np.unique(clustering):
        members = np.where(clustering == cluster_id)[0]
        if len(members) == 0:
            continue
        intra_sum = np.sum(dist_matrix[np.ix_(members, members)], axis=1)
        cost += float(np.min(intra_sum))
    return float(cost)
