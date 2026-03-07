from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def max_loss_per_agent(dist_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    costs = np.zeros(n, dtype=float)
    for i in range(n):
        same_cluster = labels == labels[i]
        if np.any(same_cluster):
            costs[i] = float(np.max(dist_matrix[i, same_cluster]))
    return costs


def _is_feasible_core_alpha(
    alpha: float,
    n: int,
    k: int,
    dist_matrix: np.ndarray,
    costs: np.ndarray,
    big_m: float,
) -> bool:
    var_count = n
    c = np.zeros(var_count)
    integrality = np.ones(var_count, dtype=int)
    bounds = Bounds(lb=np.zeros(var_count), ub=np.ones(var_count))

    rows = []
    lb = []
    ub = []

    row = np.ones(var_count)
    rows.append(row)
    lb.append(n / k)
    ub.append(np.inf)

    for i in range(n):
        for j in range(n):
            row = np.zeros(var_count)
            row[j] = alpha * dist_matrix[i, j]
            row[i] += big_m
            rows.append(row)
            lb.append(-np.inf)
            ub.append(costs[i] + big_m)

    constraints = LinearConstraint(np.asarray(rows), np.asarray(lb), np.asarray(ub))
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    return bool(result.success)


def core_violation(
    n: int,
    k: int,
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    theta: float,
    epsilon: float = 0.01,
) -> float:
    costs = max_loss_per_agent(dist_matrix, labels)
    big_m = max(1e4, float(np.max(dist_matrix)) * 10)

    lower = 1.0
    upper = max(1.0, 4.0 * theta)
    found = False

    while upper - lower > epsilon:
        alpha = (upper + lower) / 2.0
        if _is_feasible_core_alpha(alpha, n, k, dist_matrix, costs, big_m):
            lower = alpha
            found = True
        else:
            upper = alpha

    return float(lower if found else 1.0)


def _is_feasible_fjr_alpha(
    alpha: float,
    n: int,
    k: int,
    dist_matrix: np.ndarray,
    costs: np.ndarray,
    big_m1: float,
    big_m2: float,
) -> bool:
    # Variables: x_0..x_{n-1} binary, z continuous
    var_count = n + 1
    z_idx = n

    c = np.zeros(var_count)
    integrality = np.array([1] * n + [0], dtype=int)
    upper_z = max(1.0, float(np.max(dist_matrix)))
    bounds = Bounds(lb=np.zeros(var_count), ub=np.array([1.0] * n + [upper_z]))

    rows = []
    lb = []
    ub = []

    row = np.zeros(var_count)
    row[:n] = 1.0
    rows.append(row)
    lb.append(n / k)
    ub.append(np.inf)

    for i in range(n):
        for j in range(n):
            row = np.zeros(var_count)
            row[j] = dist_matrix[i, j]
            row[i] += big_m1
            row[z_idx] = -1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(big_m1)

    for i in range(n):
        row = np.zeros(var_count)
        row[z_idx] = alpha
        row[i] = -big_m2
        rows.append(row)
        lb.append(-np.inf)
        ub.append(big_m2 - costs[i])

    constraints = LinearConstraint(np.asarray(rows), np.asarray(lb), np.asarray(ub))
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    return bool(result.success)


def fjr_violation(
    n: int,
    k: int,
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    theta: float,
    epsilon: float = 0.01,
) -> float:
    costs = max_loss_per_agent(dist_matrix, labels)
    scale = max(1.0, float(np.max(dist_matrix)))
    big_m1 = max(1e4, 10.0 * scale)
    big_m2 = max(1e4, 10.0 * scale)

    lower = 1.0
    upper = max(1.0, 4.0 * theta)
    found = False

    while upper - lower > epsilon:
        alpha = (upper + lower) / 2.0
        if _is_feasible_fjr_alpha(alpha, n, k, dist_matrix, costs, big_m1, big_m2):
            lower = alpha
            found = True
        else:
            upper = alpha

    return float(lower if found else 1.0)


def appr_fjr_violation(n: int, k: int, dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    theta = 0.0
    d_local = np.array(dist_matrix, copy=True)
    l = int(np.ceil(n / k))

    costs = max_loss_per_agent(dist_matrix, labels)

    while len(d_local) >= l:
        row_idx = int(np.argmin(np.partition(d_local, l - 1, axis=1)[:, l - 1]))
        row = d_local[row_idx]
        cluster = np.argsort(row)[:l]

        new_cost = np.zeros(len(cluster), dtype=float)
        for r, i in enumerate(cluster):
            new_cost[r] = float(np.max(d_local[i, cluster]))

        larger_new_cost = float(np.max(new_cost))
        smallest_old_cost = float(np.min(costs[cluster]))

        if larger_new_cost > 0:
            theta = max(theta, smallest_old_cost / larger_new_cost)

        remove_agent = int(cluster[np.argmin(costs[cluster])])
        d_local = np.delete(np.delete(d_local, remove_agent, axis=0), remove_agent, axis=1)
        costs = np.delete(costs, remove_agent)

    return float(max(1.0, theta))
