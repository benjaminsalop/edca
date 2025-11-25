# earlystruct/core/quantities.py
from __future__ import annotations

def totals_from_intensity(per_m2: dict, area_m2: float) -> dict:
    """
    Multiply per-m² intensities by total area.
    Assumes per_m2 contains:
      - concrete_m3_per_m2, steel_m3_per_m2, timber_m3_per_m2
      - depth (m) — carried through for output convenience
    """
    conc = float(per_m2.get('concrete_m3_per_m2', 0.0) or 0.0)
    steel = float(per_m2.get('steel_m3_per_m2', 0.0) or 0.0)
    timber = float(per_m2.get('timber_m3_per_m2', 0.0) or 0.0)
    depth_m = float(per_m2.get('depth', 0.0) or 0.0)

    return dict(
        concrete_m3 = conc * area_m2,
        steel_m3    = steel * area_m2,
        timber_m3   = timber * area_m2,
        depth_m     = depth_m
    )
