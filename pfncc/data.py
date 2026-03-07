from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_diabetes, load_iris
from sklearn.preprocessing import scale


@dataclass(slots=True)
class DatasetConfig:
    name: str
    dist_function: str = "euclidean"


@dataclass(slots=True)
class FullDataset:
    data: np.ndarray
    weights: np.ndarray | None = None


def load_full_dataset(name: str) -> FullDataset:
    dataset = name.lower()

    if dataset == "iris":
        iris = load_iris()
        return FullDataset(data=iris.data.astype(float), weights=None)

    if dataset == "diabetes":
        # Paper code uses a local diabetes file with 768 rows.
        # Here we use sklearn's numeric diabetes dataset for portability.
        diabetes = load_diabetes()
        return FullDataset(data=diabetes.data.astype(float), weights=None)

    if dataset in {"adult", "adults"}:
        adult = fetch_openml("adult", version=2, as_frame=True)
        frame = adult.frame
        numeric = pd.DataFrame(
            {
                "age": pd.to_numeric(frame["age"], errors="coerce"),
                "education-num": pd.to_numeric(frame["education-num"], errors="coerce"),
                "sex": (frame["sex"].astype(str).str.strip().str.lower() == "male").astype(float),
                "capital-gain": pd.to_numeric(frame["capital-gain"], errors="coerce"),
                "capital-loss": pd.to_numeric(frame["capital-loss"], errors="coerce"),
                "hours-per-week": pd.to_numeric(frame["hours-per-week"], errors="coerce"),
            }
        )
        numeric = numeric.fillna(numeric.median())

        if "fnlwgt" in frame.columns:
            weights = pd.to_numeric(frame["fnlwgt"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        else:
            weights = np.ones(len(numeric), dtype=float)

        return FullDataset(data=numeric.to_numpy(dtype=float), weights=weights)

    raise ValueError(f"Unsupported dataset: {name}")


def sample_dataset(
    full_dataset: FullDataset,
    n: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    data = full_dataset.data
    if n is None or n >= len(data):
        sampled = data.copy()
    else:
        if full_dataset.weights is None:
            idx = rng.integers(0, len(data), size=n)
        else:
            p = full_dataset.weights / np.sum(full_dataset.weights)
            idx = rng.choice(len(data), size=n, replace=True, p=p)
        sampled = data[idx]

    return scale(sampled)
