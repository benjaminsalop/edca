
from __future__ import annotations

def totals_from_intensity(per_m2: dict, area_m2: float):
    conc_m3   = (per_m2.get('concrete_volume') or 0.0) * area_m2
    steel_m3  = (per_m2.get('steel_volume') or 0.0) * area_m2
    timber_m3 = (per_m2.get('timber_volume') or 0.0) * area_m2
    swt       = (per_m2.get('swt') or 0.0)                 # kN/m^2 (per m^2 basis)
    depth_m   = (per_m2.get('depth') or 0.0)
    return {
        "concrete_m3": conc_m3,
        "steel_m3": steel_m3,
        "timber_m3": timber_m3,
        # 'steel_kg' intentionally omitted; impacts will convert m3->kg via density
        "swt_knm2": swt,
        "depth_m": depth_m
    }
