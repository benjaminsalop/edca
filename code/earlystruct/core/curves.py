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
    Returns per-m² intensities for a system:
      - concrete_m3_per_m2, steel_m3_per_m2, timber_m3_per_m2
      - swt (kN/m²)  [already entered per-m² in your CSVs]
      - depth (m)    if 'depth' present, use it; else sum component depths
    Handles new columns: slab_depth, beam_depth, screed_depth, steel_depth.
    """
    rows = curves_df[curves_df['system_id'] == system_id]
    if rows.empty:
        # default zeros if missing
        return dict(concrete_m3_per_m2=0.0, steel_m3_per_m2=0.0, timber_m3_per_m2=0.0,
                    swt=0.0, depth=0.0)

    r = rows.iloc[0]
    unit = str(r.get('unit', 'metric'))

    conc = float(r.get('concrete_volume', 0.0) or 0.0)
    steel = float(r.get('steel_volume', 0.0) or 0.0)
    timber = float(r.get('timber_volume', 0.0) or 0.0)
    swt = float(r.get('swt', 0.0) or 0.0)

    # Depth: prefer explicit 'depth'; else sum new components
    depth_explicit = _to_m(r.get('depth', None), unit)
    if depth_explicit is not None:
        depth_m = depth_explicit
    else:
        parts = [
            _to_m(r.get('slab_depth', None), unit),
            _to_m(r.get('beam_depth', None), unit),
            _to_m(r.get('screed_depth', None), unit),
            _to_m(r.get('steel_depth', None), unit),
        ]
        depth_m = sum([p for p in parts if p is not None]) if any(p is not None for p in parts) else 0.0

    return dict(
        concrete_m3_per_m2=conc,
        steel_m3_per_m2=steel,
        timber_m3_per_m2=timber,
        swt=swt,
        depth=depth_m
    )
