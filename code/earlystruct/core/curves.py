# earlystruct/core/curves.py
from __future__ import annotations
import math
import pandas as pd
from .units import depth_to_m

def _to_m(val, unit: str) -> float | None:
    """Convert a depth value to meters; treat blank/NaN/0 as None."""
    if val in (None, "", " ", "nan", "NaN"):
        return None
    try:
        x = float(val)
    except Exception:
        return None
    if math.isnan(x) or abs(x) < 1e-12:
        return None
    return depth_to_m(x, unit)

def get_intensities(curves_df: pd.DataFrame, system_id: str) -> dict:
    """
    For a given system_id, return per-m² material intensities and depth.

    Uses columns from system_curves.csv:
      - swt              [kN/m²]
      - concrete_volume  [m³/m²]
      - steel_volume     [m³/m²]
      - timber_volume    [m³/m²]
      - depth            [m]
    """
    rows = curves_df[curves_df["system_id"] == system_id]
    if rows.empty:
        return {
            "concrete_m3_per_m2": 0.0,
            "steel_m3_per_m2": 0.0,
            "timber_m3_per_m2": 0.0,
            "swt": 0.0,
            "depth": 0.0,
        }

    r = rows.iloc[0]

    def _f(name: str, default: float = 0.0) -> float:
        v = r.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    return {
        "concrete_m3_per_m2": _f("concrete_volume", 0.0),
        "steel_m3_per_m2":    _f("steel_volume", 0.0),
        "timber_m3_per_m2":   _f("timber_volume", 0.0),
        "swt":                _f("swt", 0.0),
        "depth":              _f("depth", 0.0),
    }

