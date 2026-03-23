from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_diabetes, load_iris, load_wine
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

    if dataset == "wine":
        wine = load_wine()
        return FullDataset(data=wine.data.astype(float), weights=None)

    if dataset in {"student", "students"}:
        # Load from UCI (OpenML student-mat no longer available)
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        with urlopen(url) as resp:
            with zipfile.ZipFile(BytesIO(resp.read())) as outer:
                with outer.open("student.zip") as f:
                    with zipfile.ZipFile(BytesIO(f.read())) as inner:
                        with inner.open("student-mat.csv") as csv_f:
                            frame = pd.read_csv(csv_f, sep=";", encoding="utf-8")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            numeric = pd.DataFrame(
                {col: pd.to_numeric(frame[col], errors="coerce") for col in frame.columns}
            ).dropna(axis=1, how="all")
        numeric = numeric.fillna(numeric.median())
        return FullDataset(data=numeric.to_numpy(dtype=float), weights=None)

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
