
from __future__ import annotations
import pandas as pd
from .units import load_to_knm2, depth_to_m

NUMERIC_COLS = [
    "concrete_volume", "steel_volume", "timber_volume", "swt", "depth"
]

def get_intensities(curves_df: pd.DataFrame, system_id: str) -> dict:
    """Return per-m² intensities for a system (no span dependency in new schema).
    Assumptions:
      - volumes are in metric per m²: m3/m2 for concrete/timber; m3/m2 for steel (convert later via density)
      - swt respects 'unit' column (imperial psf or metric kN/m^2)
      - depth respects 'unit' (inches if imperial, meters if metric)
    """
    df = curves_df[curves_df['system_id'] == system_id]
    if df.empty:
        return {k: 0.0 for k in NUMERIC_COLS}
    r = df.iloc[0].to_dict()
    unit = r.get('unit','metric')
    out = {
        'concrete_volume': float(r.get('concrete_volume', 0.0) or 0.0),
        'steel_volume': float(r.get('steel_volume', 0.0) or 0.0),
        'timber_volume': float(r.get('timber_volume', 0.0) or 0.0),
        'swt': load_to_knm2(r.get('swt', 0.0), unit),
        'depth': depth_to_m(r.get('depth', 0.0), unit)
    }
    return out
