
from __future__ import annotations

PSF_TO_KNM2 = 0.04788025898      # psf -> kN/m^2
FT_TO_M     = 0.3048             # ft -> m
IN_TO_M     = 0.0254             # in -> m
LB_TO_KN    = 0.0044482216152605 # lb -> kN
KG_TO_TON   = 0.001              # metric ton

def load_to_knm2(v: float, unit_flag: str) -> float:
    if v is None or v == "": return 0.0
    return float(v) * PSF_TO_KNM2 if unit_flag == 'imperial' else float(v)

def span_to_m(v: float, unit_flag: str) -> float:
    if v is None or v == "": return 0.0
    return float(v) * FT_TO_M if unit_flag == 'imperial' else float(v)

def depth_to_m(v: float, unit_flag: str) -> float:
    """Depth is often inches for imperial tables; assume inches when unit=imperial."""
    if v is None or v == "": return 0.0
    return float(v) * IN_TO_M if unit_flag == 'imperial' else float(v)

def mm_to_m(v: float) -> float:
    if v is None or v == "": return 0.0
    return float(v) / 1000.0

def area_to_m2(v: float, unit_flag: str) -> float:
    if v is None or v == "": return 0.0
    if unit_flag == 'imperial':
        # ft^2 -> m^2
        return float(v) * (FT_TO_M**2)
    return float(v)

def parse_spans_arg(spans_str: str | None) -> list[tuple[float,str]]:
    """Parse --spans like '9,10.5' (meters) or '28ft, 32ft' (imperial).
    Returns list of (value, unit_flag) where unit_flag in {'metric','imperial'}.
    """
    if not spans_str:
        return []
    out = []
    for raw in spans_str.split(','):
        s = raw.strip().lower()
        if s.endswith('ft'):
            out.append((float(s[:-2]), 'imperial'))
        elif s.endswith('m'):
            out.append((float(s[:-1]), 'metric'))
        else:
            # assume meters
            out.append((float(s), 'metric'))
    return out
