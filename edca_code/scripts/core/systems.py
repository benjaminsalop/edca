# edca_code/scripts/core/systems.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from edca_code.scripts.core.utils import data_path
from edca_code.scripts.core.parse import ControlFile
import pandas as pd

# --- Public API --------------------------------------------------------------
def load_systems_catalog(path: str) -> pd.DataFrame:
    """
    Load systems catalog from parquet or csv. Returns a DataFrame.
    Expecting columns (at least some of): max_span, ll, sdl, slab_depth, beam_depth, screed_depth
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Systems catalog not found: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # ensure numeric columns exist
    for c in ("max_span", "ll", "sdl", "slab_depth", "beam_depth", "screed_depth"):
        if c not in df.columns:
            df[c] = 0.0
    return df

def compute_required_max_loads_from_occupancies(occupancies_path: str, unit_filter: Optional[str] = None) -> Dict[str, float]:
    """
    Read occupancies CSV and return the required maximum loads across entries.
    If unit_filter is provided (e.g., 'imperial' or 'metric'), only consider rows with that unit.
    Returns: {"max_ll": float, "max_sdl": float}
    """
    cf = cf = ControlFile.from_path("setup/control_files/control_file.yaml")
    occupancies_path = data_path(cf.data_dir, "source", "presets", "loads", "occupancies.csv")
    p = Path(occupancies_path)
    if not p.exists():
        raise FileNotFoundError(f"Occupancies file not found: {p}")

    occ = pd.read_csv(p, dtype={"use": str})
    if unit_filter:
        if "unit" in occ.columns:
            occ = occ[occ["unit"].fillna("").str.lower() == unit_filter.lower()]
        # if no unit column, assume all rows apply

    # coerce numeric columns
    occ["ll"] = pd.to_numeric(occ.get("ll", 0.0), errors="coerce").fillna(0.0)
    # sum sdl + sdl_partition for total SDL requirement
    occ["sdl_partition"] = pd.to_numeric(occ.get("sdl_partition", 0.0), errors="coerce").fillna(0.0)
    occ["sdl"] = pd.to_numeric(occ.get("sdl", 0.0), errors="coerce").fillna(0.0)
    occ["total_sdl"] = occ["sdl"] + occ["sdl_partition"]

    max_ll = float(occ["ll"].max()) if not occ.empty else 0.0
    max_sdl = float(occ["total_sdl"].max()) if not occ.empty else 0.0

    return {"max_ll": max_ll, "max_sdl": max_sdl}

def _compute_total_depth_mm(row: pd.Series) -> float:
    """
    Sum depths (slab + beam + screed) assuming provided values are in mm (or unit-consistent).
    If values look small (e.g. < 1), assume they are in metres and convert to mm.
    """
    slab = float(row.get("slab_depth", 0.0) or 0.0)
    beam = float(row.get("beam_depth", 0.0) or 0.0)
    screed = float(row.get("screed_depth", 0.0) or 0.0)
    total = slab + beam + screed

    # heuristic: if depths are given in metres (values < 5) convert to mm
    if total > 0 and total < 5:
        total_mm = total * 1000.0
    else:
        total_mm = total
    return total_mm


def filter_systems(df: pd.DataFrame,
                   min_span_required: float,
                   required_loads: Dict[str, float],
                   depth_limit_enabled: bool = False,
                   depth_limit_mm: Optional[float] = None) -> pd.DataFrame:
    """
    Filter the systems DataFrame according to:
      - df['max_span'] >= min_span_required
      - df['ll'] >= required_loads['max_ll']
      - df['sdl'] >= required_loads['max_sdl']
      - if depth_limit_enabled: (slab_depth + beam_depth + screed_depth) <= depth_limit_mm

    Notes:
      - 'min_span_required' uses the same units as df['max_span'] (user must ensure consistency)
      - if depth columns appear to be in metres, small heuristic converts to mm
    """
    # Ensure numeric
    df = df.copy()
    for c in ("max_span", "ll", "sdl"):
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)

    # span filter (systems must have capacity at or above the requested minimum span)
    span_mask = df["max_span"] >= float(min_span_required)

    # loads filter (system capacities must be >= required building loads)
    ll_req = float(required_loads.get("max_ll", 0.0))
    sdl_req = float(required_loads.get("max_sdl", 0.0))
    load_mask = (df["ll"] >= ll_req) & (df["sdl"] >= sdl_req)

    mask = span_mask & load_mask

    # depth limit (if enabled)
    if depth_limit_enabled and depth_limit_mm is not None:
        # compute total depth per row
        df["_total_depth_mm"] = df.apply(_compute_total_depth_mm, axis=1)
        depth_mask = df["_total_depth_mm"] <= float(depth_limit_mm)
        mask = mask & depth_mask

    filtered = df[mask].reset_index(drop=True)
    return filtered

def select_candidate_systems(control, systems_catalog_path: str, occupancies_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience wrapper used by the pipeline.

    Parameters:
      - control: instance of ControlFile (from parse.py)
      - systems_catalog_path: path to systems_variants parquet/csv
      - occupancies_csv_path: optional path to occupancies CSV; if omitted will use control.data_dir + '/occupancies.csv'

    Returns:
      - DataFrame of filtered candidate systems
    """
    occ_path = occupancies_csv_path or (Path(control.data_dir) / "occupancies.csv")
    systems_df = load_systems_catalog(systems_catalog_path)

    # compute required loads from occupancies; use control.unit to filter occupancy rows if present
    required_loads = compute_required_max_loads_from_occupancies(str(occ_path), unit_filter=getattr(control, "unit", None))

    # choose minimum span from control.spans (expects iterable)
    spans = getattr(control, "spans", None)
    if spans:
        try:
            min_span = float(min(spans))
        except Exception:
            # fallback: default to 0.0 if control.spans malformed
            min_span = 0.0
    else:
        min_span = 0.0

    depth_enabled = bool(getattr(control, "depth_limit_enabled", False))
    depth_limit_val = getattr(control, "depth_limit", None) if depth_enabled else None

    filtered = filter_systems(systems_df, min_span_required=min_span, required_loads=required_loads,
                              depth_limit_enabled=depth_enabled, depth_limit_mm=depth_limit_val)
    return filtered

# --- Example usage (not executed on import) ----------------------------------
if __name__ == "__main__":
    # tiny smoke test when executed directly
    from edca_tool.core.parse import ControlFile
    cf = ControlFile.from_path("control_file.yaml")
    candidates = select_candidate_systems(cf, "data/systems_variants.parquet", None)
    print(f"Found {len(candidates)} candidate systems")
    print(candidates[["system_variant", "system_family", "max_span", "ll", "sdl"]].head())
