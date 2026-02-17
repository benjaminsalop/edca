# edca_code/scripts/core/systems.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, Mapping, List
import logging

import pandas as pd

logger = logging.getLogger("systems")


# -------------------------
# helper functions
# -------------------------
def load_systems_catalog(
    path_or_fp: str | Path,
    unit_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load systems, system_families, system_variants (parquet/csv). If unit_filter provided,
    keep only rows where the 'unit' column matches (case-insensitive).
    Returns: (systems_df, families_df_or_None, variants_df_or_None)
    """
    p = Path(path_or_fp)
    parent = p.parent
    systems_df = None
    families_df = None
    variants_df = None

    def load_try(fp: Path) -> Optional[pd.DataFrame]:
        if not fp.exists():
            return None
        if fp.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(fp)
        return pd.read_csv(fp)

    # try to load exactly the provided path
    systems_df = load_try(p)

    # also try common sibling files
    for name in ("system_families", "system_variants", "systems"):
        for ext in (".parquet", ".pq", ".csv"):
            f = parent / (name + ext)
            if f.exists():
                df = load_try(f)
                if df is None:
                    continue
                if name == "system_families":
                    families_df = df
                elif name == "system_variants":
                    variants_df = df
                elif name == "systems" and systems_df is None:
                    systems_df = df

    if systems_df is None:
        raise FileNotFoundError(f"Could not find systems data at {p} or siblings")

    def apply_unit_filter(df: pd.DataFrame) -> pd.DataFrame:
        if unit_filter is None:
            return df
        unit_cols = [c for c in df.columns if c.lower() == "unit"]
        if not unit_cols:
            return df
        ucol = unit_cols[0]
        return df[df[ucol].astype(str).str.lower() == str(unit_filter).strip().lower()].copy()

    systems_df = apply_unit_filter(systems_df)
    if families_df is not None:
        families_df = apply_unit_filter(families_df)
    if variants_df is not None:
        variants_df = apply_unit_filter(variants_df)

    logger.info(
        "[systems] load_systems_catalog: systems=%d rows, families=%s, variants=%s (unit_filter=%s)",
        len(systems_df),
        (len(families_df) if families_df is not None else "N/A"),
        (len(variants_df) if variants_df is not None else "N/A"),
        unit_filter,
    )
    return systems_df, families_df, variants_df

def compute_total_depth_mm(row: pd.Series) -> float:
    """
    Sum depths (slab + screed) and return mm.
    Heuristic: if total is very small (< 10), assume metres and convert to mm.
    """
    slab = float(row.get("slab_depth", 0.0) or 0.0)
    screed = float(row.get("screed_depth", 0.0) or 0.0)
    total = slab + screed

    # minimal, necessary guardrail: avoid silent unit mismatch
    if 0 < total < 10:
        total = total * 1000.0
    return float(total)


def normalize_unit_column(
    df: pd.DataFrame,
    unit_col_candidates: Tuple[str, ...] = ("unit", "units", "measurement"),
) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, canonical_unit_col_name). Ensures df has a column named 'unit'
    whose values are normalized (stripped & lowercased).
    """
    detected = None
    lower_map = {c.lower(): c for c in df.columns}

    for cand in unit_col_candidates:
        if cand.lower() in lower_map:
            detected = lower_map[cand.lower()]
            break

    if detected is None:
        df = df.copy()
        df["unit"] = ""
        detected = "unit"

    df = df.copy()
    df[detected] = df[detected].astype(str).str.strip().str.lower()

    if detected != "unit":
        df = df.rename(columns={detected: "unit"})
    return df, "unit"


def filter_systems(
    df: pd.DataFrame,
    min_span_required: Optional[float] = None,
    required_loads: Optional[Dict[str, float]] = None,
    unit: Optional[str] = None,
    depth_limit_enabled: bool = False,
    depth_limit_mm: Optional[float] = None,
    require_separate_checks: bool = False) -> pd.DataFrame:
    """
    Robust system filter:
      - normalizes & enforces `unit` strictly (if provided)
      - if min_span_required is None, span test is skipped
      - required_loads expected keys (examples): 'max_total', 'max_sdl', 'max_ll'
      - require_separate_checks=True enforces BOTH sdl_total >= max_sdl and ll >= max_ll
    """
    if required_loads is None:
        required_loads = {}

    df, _unit_col = normalize_unit_column(df)

    # Strict unit enforcement (always normalized)
    if unit is not None:
        wanted = str(unit).strip().lower()
        before_rows = len(df)
        df = df[df["unit"] == wanted].copy()
        after_rows = len(df)
        if after_rows == 0:
            raise RuntimeError(
                f"filter_systems: unit='{wanted}' eliminated all systems ({before_rows} -> 0). "
                "Check the systems catalog 'unit' values and your control file 'unit' setting."
            )

    # Ensure numeric columns exist (coerce missing -> 0.0)
    numeric_cols = ["max_span", "ll", "sdl", "sdl_partition"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.copy()
    df["sdl_total"] = df["sdl"].astype(float) + df["sdl_partition"].astype(float)
    df["total_capacity"] = df["sdl_total"] + df["ll"].astype(float)

    # SPAN MASK
    if min_span_required is None:
        span_mask = pd.Series(True, index=df.index)
    else:
        try:
            span_val = float(min_span_required)
        except Exception:
            span_val = None

        if span_val is None:
            span_mask = pd.Series(True, index=df.index)
        else:
            span_mask = df["max_span"].astype(float) >= span_val if "max_span" in df.columns else pd.Series(True, index=df.index)

    # LOAD MASK
    max_sdl_req = float(required_loads.get("max_sdl", 0.0) or 0.0)
    max_ll_req = float(required_loads.get("max_ll", 0.0) or 0.0)

    if require_separate_checks:
        load_mask = (df["sdl_total"].astype(float) >= max_sdl_req) & (df["ll"].astype(float) >= max_ll_req)
    else:
        if "max_total" in required_loads:
            max_total_req = float(required_loads.get("max_total", 0.0) or 0.0)
        else:
            max_total_req = (max_sdl_req + max_ll_req) if (max_sdl_req or max_ll_req) else None

        load_mask = pd.Series(True, index=df.index) if max_total_req is None else (df["total_capacity"].astype(float) >= float(max_total_req))

    mask = span_mask & load_mask

    # DEPTH LIMIT
    if depth_limit_enabled and depth_limit_mm is not None:
        if "_total_depth_mm" not in df.columns:
            df["_total_depth_mm"] = df.apply(compute_total_depth_mm, axis=1)
        depth_mask = df["_total_depth_mm"].astype(float) <= float(depth_limit_mm)
        mask = mask & depth_mask

    filtered = df[mask].reset_index(drop=True)
    logger.debug("[systems] filter_systems: started with %d rows, returning %d rows", len(df), len(filtered))
    return filtered

def select_candidate_systems_per_floor(
    systems_catalog_path: str,
    floor_loads: Mapping[int, Any],
    *,
    control: Optional[Any] = None,
    occupancies_csv_path: Optional[str] = None,   # kept for signature compatibility; unused
    default_min_span: float = 0.0,
    span_requirements: Optional[Mapping[int, float]] = None,
    depth_limit_enabled: bool = False,
    depth_limit_mm: Optional[float] = None,
    require_separate_checks: bool = False,
    unit_conversion_fn: Optional[Callable[[float, str, str], float]] = None,
    catalog_unit: Optional[str] = None,
) -> Dict[int, pd.DataFrame]:
    systems_df, _families_df, _variants_df = load_systems_catalog(systems_catalog_path)

    # Determine the unit we will enforce on the catalog (if any)
    enforced_unit = None
    if catalog_unit is not None:
        enforced_unit = catalog_unit
    elif control is not None:
        enforced_unit = getattr(control, "unit", None)

    def min_span_for_floor(floor: int) -> float:
        if span_requirements and int(floor) in span_requirements:
            return float(span_requirements[int(floor)])
        if control is not None:
            spans = getattr(control, "spans", None)
            if spans:
                try:
                    return float(min(spans))
                except Exception:
                    pass
        return float(default_min_span)

    results: Dict[int, pd.DataFrame] = {}

    for floor, fl in floor_loads.items():
        if hasattr(fl, "as_dict"):
            rec = fl.as_dict()
            src_unit = getattr(fl, "unit", None)
        else:
            rec = dict(fl)
            src_unit = rec.get("unit", None)

        sdl = float(rec.get("sdl", 0.0))
        sdl_part = float(rec.get("sdl_partition", 0.0))
        ll = float(rec.get("ll", 0.0))

        # Convert loads to catalog units if requested
        if unit_conversion_fn is not None and catalog_unit is not None and src_unit is not None:
            try:
                sdl = float(unit_conversion_fn(sdl, src_unit, catalog_unit))
                sdl_part = float(unit_conversion_fn(sdl_part, src_unit, catalog_unit))
                ll = float(unit_conversion_fn(ll, src_unit, catalog_unit))
            except Exception:
                logger.exception(
                    "Unit conversion failed for floor %s (from %s to %s). Using unconverted loads.",
                    floor, src_unit, catalog_unit
                )

        sdl_total = sdl + sdl_part
        total_required = sdl_total + ll

        required_loads = {
            "max_ll": float(ll),
            "max_sdl": float(sdl_total),
            "max_total": float(total_required),
        }

        min_span = min_span_for_floor(int(floor))

        filtered = filter_systems(
            systems_df,
            min_span_required=min_span,
            required_loads=required_loads,
            unit=enforced_unit,
            depth_limit_enabled=depth_limit_enabled,
            depth_limit_mm=depth_limit_mm,
            require_separate_checks=require_separate_checks,
        ).copy()

        filtered["_for_floor"] = int(floor)
        filtered["_floor_use"] = rec.get("use", "")
        results[int(floor)] = filtered.reset_index(drop=True)

    return results
