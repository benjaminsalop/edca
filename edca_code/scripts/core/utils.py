from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def data_path(base_dir: str | Path, *parts: str) -> Path:
    return Path(base_dir).joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "assembly_id",
        "component_id",
        "component_type",
        "family_id",
        "variant_id",
        "role",
        "passes",
        "utilization_ratio",
        "governing_case",
        "embodied_carbon",
        "cost",
    ]
    cols = list(df.columns)
    new_order = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return df.loc[:, new_order]
