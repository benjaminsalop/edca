# edca_code/scripts/core/reporting.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from pathlib import Path
import glob
import json


# Prefer canonical helpers when available, but keep fallbacks so reporting.py
# doesn’t crash if your utils module changes.
try:
    from edca_code.scripts.core.utils import standardize_schema as _standardize_schema  # type: ignore
except Exception:  # pragma: no cover
    def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        D = df.copy()
        D.columns = [str(c).strip() for c in D.columns]
        return D

standardize_schema = _standardize_schema

try:
    from edca_code.scripts.code_checks.code_runner import run_code_checks_if_requested  # type: ignore
except Exception:  # pragma: no cover
    def run_code_checks_if_requested(*args, **kwargs):
        return pd.DataFrame()

logger = logging.getLogger("reporting")

# ------------------------
# Utilities
# ------------------------

def _get_span_col(df):
    if "max_span" in df.columns:
        return "max_span"
    return None

def _ensure_carbon(df):
    # prefer carbon_per_m2, fallback to carbon_total_kgCO2 (assumed per m2)
    if "carbon_per_m2" in df.columns:
        df = df.copy()
        df["carbon_per_m2"] = pd.to_numeric(df["carbon_per_m2"], errors="coerce")
        return df
    if "carbon_total_kgCO2" in df.columns:
        df = df.copy()
        df["carbon_per_m2"] = pd.to_numeric(df["carbon_total_kgCO2"], errors="coerce")
        return df
    raise KeyError("No carbon column found (expected carbon_per_m2 or carbon_total_kgCO2)")

def _infer_total_load_col(df: pd.DataFrame) -> Optional[str]:
    """Best-effort choice of a 'total load' column.

    Priority:
      1) existing 'total_load'
      2) synthesize total_load = sdl_total + ll, else sdl + ll
      3) fall back to other common single columns (capacity, etc.)

    Note: if synthesis is possible, this function adds a 'total_load' column
    to the passed dataframe so downstream code can safely index it.
    """
    if df is None or df.empty:
        return None
    if "total_load" in df.columns:
        return "total_load"
    if "sdl_total" in df.columns and "ll" in df.columns:
        df["total_load"] = pd.to_numeric(df["sdl_total"], errors="coerce") + pd.to_numeric(df["ll"], errors="coerce")
        return "total_load"
    if "sdl" in df.columns and "ll" in df.columns:
        df["total_load"] = pd.to_numeric(df["sdl"], errors="coerce") + pd.to_numeric(df["ll"], errors="coerce")
        return "total_load"
    for c in ("total_capacity", "capacity", "sdl_total", "sdl", "ll"):
        if c in df.columns:
            return c
    return None


def _ensure_success_mask(df):
    # reuse your existing _success_mask if present; otherwise infer pass_overall
    if "_success" in df.columns:
        return df["_success"]
    if "pass_overall" in df.columns:
        return df["pass_overall"].astype(bool).fillna(False)
    # fallback: mark all as True if no checks exist
    return pd.Series(True, index=df.index)


def load_summary_ranked_all(out_dir: str | Path) -> pd.DataFrame:
    """
    Prefer summary_ranked_all_long.csv (long-form) for reporting.
    Fall back to summary_ranked_all.csv (wide) if needed.
    """
    out_dir = Path(out_dir)

    p_long = out_dir / "summary_ranked_all_long.csv"
    if p_long.exists():
        return pd.read_csv(p_long)

    p_wide = out_dir / "summary_ranked_all.csv"
    if p_wide.exists():
        return pd.read_csv(p_wide)

    # fallback search
    for fname in ["summary_ranked_all_long.csv", "summary_ranked_all.csv"]:
        for p in out_dir.rglob(fname):
            try:
                return pd.read_csv(p)
            except Exception:
                continue

    raise FileNotFoundError(
        f"Could not find summary_ranked_all_long.csv or summary_ranked_all.csv under {out_dir}"
    )

def _group_by_system_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a combined summary table to one row per system_variant.

    - Numeric columns: min
    - Non-numeric columns: first non-null
    - Drops internal provenance columns (_source_*)
    """
    if df is None or df.empty:
        return df

    if "system_variant" not in df.columns:
        return df

    D = df.copy()

    # drop provenance columns
    drop_cols = [c for c in D.columns if c.startswith("_source_")]
    if drop_cols:
        D = D.drop(columns=drop_cols)

    # build aggregation map
    agg: dict[str, str] = {}
    for c in D.columns:
        if c == "system_variant":
            continue
        if pd.api.types.is_numeric_dtype(D[c]):
            agg[c] = "min"
        else:
            agg[c] = "first"

    grouped = (
        D
        .groupby("system_variant", dropna=False, as_index=False)
        .agg(agg)
    )

    return grouped

def to_numeric_safe(series: Any) -> pd.Series:
    """Convert a Series-like to numeric, coercing errors to NaN."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * len(series)) if hasattr(series, "__len__") else pd.Series([np.nan])

def add_typology(df: pd.DataFrame, out_col: str = "typology") -> pd.DataFrame:
    """
    Add a coarse typology label if missing.

    User-defined mapping (highest priority):
      - steel:  composite_deck
      - timber: clt_floor, lvl_joist
      - everything else: concrete

    Falls back to text heuristics if 'system_family' is unavailable.
    """
    if df is None or df.empty:
        return df
    if out_col in df.columns:
        return df

    D = df.copy()

    # Prefer explicit mapping by system_family if available
    if "system_family" in D.columns:
        fam = D["system_family"].astype(str).str.lower()
        D[out_col] = "concrete"
        D.loc[fam.eq("composite_deck"), out_col] = "steel"
        D.loc[fam.isin(["clt_floor", "lvl_joist"]), out_col] = "timber"
        return D

    # Fallback: text heuristics
    text_cols = [c for c in ("category", "type", "manufacturer") if c in D.columns]
    if not text_cols:
        D[out_col] = None
        return D

    def infer(row: pd.Series) -> str:
        s = " ".join(str(row.get(c, "")) for c in text_cols).lower()
        if "timber" in s or "clt" in s or "lvl" in s or "glulam" in s:
            return "timber"
        if "steel" in s:
            return "steel"
        return "concrete"

    D[out_col] = D.apply(infer, axis=1)
    return D


def best_row_by_metric(df: pd.DataFrame, metric: str, feasible_col: Optional[str] = "feasible") -> Optional[pd.Series]:
    """
    Return the best row (lowest metric) optionally filtering to feasible.
    """
    if df is None or df.empty or metric not in df.columns:
        return None
    D = df.copy()
    if feasible_col and feasible_col in D.columns:
        D = D[_coerce_bool_series(D[feasible_col])]
    D[metric] = to_numeric_safe(D[metric])
    D = D.dropna(subset=[metric]).sort_values(metric, ascending=True)
    if D.empty:
        return None
    return D.iloc[0]


def concat_candidates_input(candidates: Any) -> pd.DataFrame:
    """
    Accept a DataFrame or dict[floor -> DataFrame], and return one concatenated DataFrame.
    If dict is provided, adds a '_for_floor' column.
    """
    if candidates is None:
        return pd.DataFrame()

    if isinstance(candidates, pd.DataFrame):
        return candidates.copy()

    if isinstance(candidates, dict):
        frames = []
        for floor, df in candidates.items():
            if df is None or len(df) == 0:
                continue
            sub = df.copy()
            sub["_for_floor"] = int(floor)
            frames.append(sub)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    raise TypeError(f"Unsupported candidates type: {type(candidates)}")

# ------------------------
# Pareto utilities
# ------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maximize_x: bool = True,
    minimize_y: bool = True,
) -> pd.DataFrame:
    """
    Compute a 2D Pareto frontier.

    Default use case here:
      - maximize x (e.g., span)
      - minimize y (e.g., carbon)

    Returns frontier points sorted by x ascending for plotting.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[x_col, y_col])

    D = df[[x_col, y_col]].copy()
    D[x_col] = pd.to_numeric(D[x_col], errors="coerce")
    D[y_col] = pd.to_numeric(D[y_col], errors="coerce")
    D = D.dropna(subset=[x_col, y_col]).drop_duplicates([x_col, y_col])

    if D.empty:
        return pd.DataFrame(columns=[x_col, y_col])

    # Common case for your plot: maximize span, minimize carbon
    if maximize_x and minimize_y:
        # Collapse exact duplicate x values to the best (lowest) y to avoid vertical zig-zags
        D = D.groupby(x_col, as_index=False)[y_col].min()

        # Sweep from largest x to smallest x, keeping running minimum y
        D = D.sort_values([x_col, y_col], ascending=[False, True]).reset_index(drop=True)

        frontier_rows = []
        best_y = float("inf")
        tol = 1e-12

        for _, row in D.iterrows():
            y = row[y_col]
            if y <= best_y + tol:
                frontier_rows.append(row)
                if y < best_y:
                    best_y = y

        F = pd.DataFrame(frontier_rows)
        # Plot left-to-right
        F = F.sort_values([x_col, y_col], ascending=[True, True]).reset_index(drop=True)
        return F

    # Generic fallback (O(n^2)) for other objective directions if needed
    X = D[x_col].to_numpy()
    Y = D[y_col].to_numpy()

    keep = np.ones(len(D), dtype=bool)
    for i in range(len(D)):
        xi, yi = X[i], Y[i]

        if maximize_x:
            x_better_eq = X >= xi
            x_strict_better = X > xi
        else:
            x_better_eq = X <= xi
            x_strict_better = X < xi

        if minimize_y:
            y_better_eq = Y <= yi
            y_strict_better = Y < yi
        else:
            y_better_eq = Y >= yi
            y_strict_better = Y > yi

        dominated = (x_better_eq & y_better_eq & (x_strict_better | y_strict_better))
        dominated[i] = False  # ignore self

        if dominated.any():
            keep[i] = False

    F = D.loc[keep].copy()
    F = F.sort_values([x_col, y_col], ascending=[True, True]).reset_index(drop=True)
    return F

    
# ------------------------
# Global color mapping (consistent across all plots)
# ------------------------

# These globals are initialized once per report run via `init_global_color_maps(df_all)`.
_GLOBAL_COLOR_MAPS: Dict[str, Dict[str, Any]] = {}
_GLOBAL_COLOR_PALETTES_READY: bool = False


def _stable_tab_palette(n: int) -> List[Any]:
    """Return a deterministic list of n colors using matplotlib tab palettes."""
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors: List[Any] = []
    for cmap in cmaps:
        for i in range(cmap.N):
            colors.append(cmap(i))
            if len(colors) >= n:
                return colors
    # fallback: repeat
    if not colors:
        return [plt.get_cmap("tab20")(i) for i in range(n)]
    while len(colors) < n:
        colors.extend(colors)
    return colors[:n]


def _shaded_palette(cmap_name: str, n: int, lo: float = 0.35, hi: float = 0.85) -> List[Any]:
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap((lo + hi) / 2)]
    xs = np.linspace(lo, hi, n)
    return [cmap(float(x)) for x in xs]


def _infer_material_family(label: str) -> str:
    """Map typology/type label to a material family bucket.

    User-defined mapping:
      - steel: composite_deck
      - timber: clt_floor, lvl_joist
      - concrete: everything else (default)
    """
    s = (label or "").strip().lower()
    if s == "composite_deck":
        return "steel"
    if s in {"clt_floor", "lvl_joist"}:
        return "timber"
    # Default everything else to concrete (even if the string doesn't say 'concrete')
    return "concrete"


def init_global_color_maps(df: Optional[pd.DataFrame]) -> None:
    """Initialize global color maps for consistent output across all figures.

    We build stable mappings for:
      - typology / type (material-family themed palettes)
      - manufacturer (categorical palette)
      - material_family (explicit 3-bucket mapping)

    Material family mapping (per your clarification):
      - steel: composite_deck
      - timber: clt_floor, lvl_joist
      - concrete: everything else
    """
    global _GLOBAL_COLOR_MAPS, _GLOBAL_COLOR_PALETTES_READY

    if _GLOBAL_COLOR_PALETTES_READY:
        return

    _GLOBAL_COLOR_MAPS = {}

    # Always define material_family colors so downstream code can rely on it,
    # even if df is None/empty at init time.
    _GLOBAL_COLOR_MAPS["material_family"] = {
        "steel": plt.get_cmap("Blues")(0.65),
        "concrete": plt.get_cmap("Greens")(0.65),
        "timber": plt.get_cmap("Reds")(0.65),
    }

    if df is None or df.empty:
        _GLOBAL_COLOR_PALETTES_READY = True
        return

    def _make_material_map(values: List[str]) -> Dict[str, Any]:
        values = [str(v) for v in values if str(v).strip() != ""]
        uniq = sorted(set(values))

        buckets: Dict[str, List[str]] = {"steel": [], "concrete": [], "timber": [], "other": []}
        for u in uniq:
            buckets[_infer_material_family(u)].append(u)

        out: Dict[str, Any] = {}

        # Assign themed shades within each bucket
        themed = {
            "steel": ("Blues", buckets["steel"]),
            "concrete": ("Greens", buckets["concrete"]),
            "timber": ("Reds", buckets["timber"]),
        }
        for fam, (cmap, labs) in themed.items():
            cols = _shaded_palette(cmap, len(labs))
            for lab, col in zip(labs, cols):
                out[lab] = col

        # Remaining categories (should be rare with the explicit mapping): tab palette
        other_labs = buckets["other"]
        other_cols = _stable_tab_palette(len(other_labs))
        for lab, col in zip(other_labs, other_cols):
            out[lab] = col

        return out

    # type / typology share the same material-inspired mapping
    for col in ["type", "typology"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            _GLOBAL_COLOR_MAPS[col] = _make_material_map(vals)

    # manufacturer mapping: stable categorical palette (not material-based)
    if "manufacturer" in df.columns:
        uniq = sorted(df["manufacturer"].dropna().astype(str).unique().tolist())
        cols = _stable_tab_palette(len(uniq))
        _GLOBAL_COLOR_MAPS["manufacturer"] = {u: c for u, c in zip(uniq, cols)}

    _GLOBAL_COLOR_PALETTES_READY = True

def get_color_map_for(column: str, categories: List[str]) -> Dict[str, Any]:
    """Get a deterministic color map for `column` over `categories`."""
    init_global_color_maps(None)  # no-op if already ready
    cats = [str(c) for c in categories]
    base = _GLOBAL_COLOR_MAPS.get(column, {})
    missing = [c for c in cats if c not in base]
    if not missing:
        return base
    # extend with a stable palette for missing categories
    ext_cols = _stable_tab_palette(len(missing))
    out = dict(base)
    for lab, col in zip(missing, ext_cols):
        out[lab] = col
    # cache extension for future plots
    if column:
        _GLOBAL_COLOR_MAPS[column] = out
    return out


def colors_for_series(series: pd.Series, *, column_name: str = "") -> Tuple[List[Any], List[Any]]:
    """Return colors and legend handles for a categorical series."""
    cats = series.astype(str).fillna("Unknown")
    uniq = sorted(cats.unique())
    cmap = get_color_map_for(column_name, uniq) if column_name else get_color_map_for("_generic", uniq)
    colors = [cmap[v] for v in cats]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=cmap[u], label=u) for u in uniq]
    return colors, handles


# ------------------------
# Optional: load full system catalogue points (to show success + failure curves)
# ------------------------

def _read_first_existing(paths: List[Path]) -> Optional[pd.DataFrame]:
    for p in paths:
        try:
            if p.exists() and p.is_file():
                if p.suffix.lower() == ".parquet":
                    return pd.read_parquet(p)
                if p.suffix.lower() == ".csv":
                    return pd.read_csv(p)
        except Exception:
            logger.exception("[reporting] Failed reading %s", p)
    return None


def load_system_curve_points(out_dir: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Best-effort loader for the full system catalogue points.

    Tries (in order):
      1) systems.parquet / systems.csv
      2) system_variants.parquet / system_variants.csv (and joins manufacturer/family if possible)
    """
    od = Path(out_dir)
    if not od.exists():
        return None

    df = _read_first_existing([od / "systems.parquet", od / "systems.csv"])
    if df is not None and not df.empty:
        return df

    variants = _read_first_existing([od / "system_variants.parquet", od / "system_variants.csv"])
    if variants is None or variants.empty:
        return None

    families = _read_first_existing([od / "system_families.parquet", od / "system_families.csv"])
    if families is None or families.empty:
        return variants

    # Heuristic joins (keep it robust to schema differences)
    v = variants.copy()
    f = families.copy()

    join_keys = []
    for k in ["system_family", "family", "family_id", "system_family_id"]:
        if k in v.columns and k in f.columns:
            join_keys = [k]
            break

    if join_keys:
        try:
            merged = v.merge(f, on=join_keys, how="left", suffixes=("", "_fam"))
            return merged
        except Exception:
            logger.exception("[reporting] Failed joining system_variants to system_families")
            return variants

    return variants

# ------------------------
# IO containers
# ------------------------
@dataclass
class ReportArtifacts:
    """
    Saved report artifact paths + in-memory tables.
    """
    tables: Dict[str, pd.DataFrame]
    table_paths: Dict[str, str]
    figure_paths: Dict[str, str]


def _ensure_report_dirs(out_dir: Union[str, Path]) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return {"root": out_dir, "tables": tables, "figures": figs}


# ------------------------
# Internal helpers
# ------------------------
def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.upper().isin({"Y", "YES", "TRUE", "T", "1", "PASS", "OK"})


def _success_mask(df: pd.DataFrame) -> pd.Series:
    """
    Define 'successful' as passing code checks if available, else feasible if available,
    else everything.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=bool)

    for c in ("pass_overall", "code_check_pass", "code_pass", "pass"):
        if c in df.columns:
            return _coerce_bool_series(df[c])

    if "feasible" in df.columns:
        return _coerce_bool_series(df["feasible"])

    return pd.Series([True] * len(df), index=df.index)

def _join_system_families_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join canonical system_families metadata onto the reporting dataframe.

    Looks for:
      - inputs/canonical/system_families.parquet
      - inputs/canonical/system_families.csv   (fallback)

    Expected canonical columns:
      - system_family (join key)
      - manufacturer  (what we want)
      - category -> typology (optional)
      - type     -> type (optional)

    Reporting-side columns take precedence (non-null).
    """
    if df is None or df.empty or "system_family" not in df.columns:
        return df

    try:
        # --- locate canonical file ---
        sf_parquet = Path("inputs/canonical/system_families.parquet")
        sf_csv = Path("inputs/canonical/system_families.csv")

        if sf_parquet.exists():
            sf = pd.read_parquet(sf_parquet)
        elif sf_csv.exists():
            sf = pd.read_csv(sf_csv)
        else:
            logger.warning("[reporting] system_families not found at %s or %s", sf_parquet, sf_csv)
            return df

        if "system_family" not in sf.columns:
            logger.warning("[reporting] system_families missing 'system_family' column")
            return df

        # --- normalize join keys (prevents whitespace/case mismatches) ---
        D = df.copy()
        D["system_family"] = D["system_family"].astype(str).str.strip()

        sf = sf.copy()
        sf["system_family"] = sf["system_family"].astype(str).str.strip()

        # --- keep/rename canonical cols ---
        keep_cols = ["system_family"]
        rename_map = {}
        if "category" in sf.columns:
            keep_cols.append("category")
            rename_map["category"] = "typology"
        if "typology" in sf.columns:
            keep_cols.append("typology")  # if already named typology
        if "type" in sf.columns:
            keep_cols.append("type")
        if "manufacturer" in sf.columns:
            keep_cols.append("manufacturer")

        sf = sf[keep_cols].rename(columns=rename_map).drop_duplicates(subset=["system_family"])

        # --- left join; prefer existing df values if present ---
        out = D.merge(sf, on="system_family", how="left", suffixes=("", "_canon"))

        for col in ("typology", "type", "manufacturer"):
            canon_col = f"{col}_canon"
            if canon_col in out.columns:
                out[col] = out.get(col).combine_first(out[canon_col])
                out = out.drop(columns=[canon_col])

        logger.info("[reporting] Joined system_families metadata (typology/type/manufacturer)")
        return out

    except Exception:
        logger.exception("[reporting] Failed to join system_families metadata")
        return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    D = standardize_schema(df)

    # numeric coercions
    for c in ("carbon_per_m2", "cost_per_m2", "max_span"):
        if c in D.columns:
            D[c] = to_numeric_safe(D[c])

    # join canonical system_families metadata
    D = _join_system_families_metadata(D)

    # only fall back to heuristic inference if typology absent/empty
    if "typology" not in D.columns or D["typology"].isna().all():
        logger.warning("[reporting] typology missing from canonical data — falling back to heuristic inference")
        D = add_typology(D, "typology")

    return D

def _group_floors_by_category(floor_assignments: Dict[int, str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for f, cat in floor_assignments.items():
        out.setdefault(str(cat), []).append(int(f))
    for k in out:
        out[k] = sorted(out[k])
    return out


def _best_variant_that_works_for_all_floors(
    df: pd.DataFrame,
    floors: List[int],
    *,
    metric: str = "carbon_per_m2",
    variant_col: str = "system_variant",
) -> Optional[pd.Series]:
    """
    Pick ONE system_variant that is successful for ALL floors in the group.
    Conservative aggregation across floors:
      - metric: max across floors (worst-case)
      - cost_per_m2: max across floors
      - success: AND across floors
    """
    if df is None or df.empty:
        return None

    # If no per-floor detail exists, pick best overall
    if "_for_floor" not in df.columns:
        return best_row_by_metric(df, metric, feasible_col="feasible")

    sub = df[df["_for_floor"].isin(floors)].copy()
    if sub.empty:
        return None

    sub["_success"] = _success_mask(sub)

    if variant_col not in sub.columns:
        return best_row_by_metric(sub, metric, feasible_col=None)

    agg_spec: Dict[str, Any] = {"_success": "all"}
    for c in ("carbon_per_m2", "cost_per_m2", "max_span"):
        if c in sub.columns:
            agg_spec[c] = "max"
    for c in ("system_family", "type", "category", "typology"):
        if c in sub.columns:
            agg_spec[c] = "first"

    G = sub.groupby(variant_col, dropna=False).agg(agg_spec).reset_index()
    G = G[G["_success"] == True].copy()
    if G.empty:
        return None

    G["_metric"] = to_numeric_safe(G.get(metric))
    G = G.dropna(subset=["_metric"]).sort_values("_metric", ascending=True)
    if G.empty:
        return None

    return G.iloc[0]


def _lowest_carbon_family(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or "system_family" not in df.columns or "carbon_per_m2" not in df.columns:
        return None
    D = df.copy()
    D["_success"] = _success_mask(D)
    D = D[D["_success"] == True].copy()
    if D.empty:
        return None
    fam_min = D.groupby("system_family")["carbon_per_m2"].min().sort_values()
    return fam_min.index[0] if not fam_min.empty else None


# ------------------------
# Tables
# ------------------------
def table_ideal_per_floor_category(
    df_all: pd.DataFrame,
    floor_assignments: Dict[int, str],
    floor_area_lookup: Optional[Dict[int, float]] = None,
    *,
    metric: str = "carbon_per_m2",
) -> pd.DataFrame:
    """
    Priority (normal+verbose):
    One row per floor-category group with the best single variant that works for all floors,
    including per-m² and totals.
    """
    D = _normalize_columns(df_all)
    area = floor_area_lookup or {}
    cat2floors = _group_floors_by_category(floor_assignments)

    rows: List[Dict[str, Any]] = []
    for cat, floors in cat2floors.items():
        best = _best_variant_that_works_for_all_floors(D, floors, metric=metric)
        if best is None:
            rows.append({
                "category": cat,
                "floors": floors,
                "n_floors": len(floors),
                "best_variant": None,
                "note": "no variant succeeds on all floors",
            })
            continue

        total_area = float(sum(float(area.get(f, 0.0)) for f in floors))
        cpm2 = float(best.get("carbon_per_m2") or 0.0)
        costpm2 = float(best.get("cost_per_m2") or 0.0)

        rows.append({
            "category": cat,
            "floors": floors,
            "n_floors": len(floors),
            "best_variant": best.get("system_variant", best.get("system_id")),
            "system_family": best.get("system_family"),
            "type": best.get("type"),
            "typology": best.get("typology"),
            "carbon_per_m2": cpm2,
            "cost_per_m2": costpm2,
            "total_area_m2": total_area,
            "total_carbon_kgCO2e": cpm2 * total_area,
            "total_cost": costpm2 * total_area,
        })

    out = pd.DataFrame(rows)

    # building totals row
    if not out.empty and "total_area_m2" in out.columns:
        build_area = float(out["total_area_m2"].fillna(0).sum())
        build_carbon = float(out.get("total_carbon_kgCO2e", pd.Series([0]*len(out))).fillna(0).sum())
        build_cost = float(out.get("total_cost", pd.Series([0]*len(out))).fillna(0).sum())

        out = pd.concat([out, pd.DataFrame([{
            "category": "WHOLE_BUILDING",
            "floors": "ALL",
            "n_floors": int(len(floor_assignments)),
            "best_variant": None,
            "system_family": None,
            "type": None,
            "typology": None,
            "carbon_per_m2": (build_carbon / build_area) if build_area > 0 else None,
            "cost_per_m2": (build_cost / build_area) if build_area > 0 else None,
            "total_area_m2": build_area,
            "total_carbon_kgCO2e": build_carbon,
            "total_cost": build_cost,
        }])], ignore_index=True)

    # sort non-total by carbon
    if "category" in out.columns and "carbon_per_m2" in out.columns:
        mask_tot = out["category"].astype(str) == "WHOLE_BUILDING"
        out_non = out[~mask_tot].sort_values(by=["carbon_per_m2"], na_position="last")
        out_tot = out[mask_tot]
        out = pd.concat([out_non, out_tot], ignore_index=True)

    return out


def table_next_best_type_per_category(
    df_all: pd.DataFrame,
    floor_assignments: Dict[int, str],
    *,
    metric: str = "carbon_per_m2",
) -> pd.DataFrame:
    """
    Verbose:
    For each category, show best variant + best variant from the next best *type*.
    """
    D = _normalize_columns(df_all)
    cat2floors = _group_floors_by_category(floor_assignments)

    rows: List[Dict[str, Any]] = []
    for cat, floors in cat2floors.items():
        best = _best_variant_that_works_for_all_floors(D, floors, metric=metric)
        if best is None:
            continue
        best_type = best.get("type")

        # filter out best_type rows then re-run selection
        if "_for_floor" in D.columns:
            sub2 = D[D["_for_floor"].isin(floors)].copy()
        else:
            sub2 = D.copy()

        sub2["_success"] = _success_mask(sub2)
        sub2 = sub2[sub2["_success"] == True].copy()
        if "type" in sub2.columns and best_type is not None:
            sub2 = sub2[sub2["type"] != best_type]

        second = _best_variant_that_works_for_all_floors(sub2, floors, metric=metric) if "_for_floor" in sub2.columns else best_row_by_metric(sub2, metric, feasible_col=None)

        rows.append({
            "category": cat,
            "floors": floors,
            "best_variant": best.get("system_variant"),
            "best_type": best.get("type"),
            "best_family": best.get("system_family"),
            "best_carbon_per_m2": float(best.get("carbon_per_m2") or 0.0),
            "second_best_variant": (second.get("system_variant") if second is not None else None),
            "second_best_type": (second.get("type") if second is not None else None),
            "second_best_family": (second.get("system_family") if second is not None else None),
            "second_best_carbon_per_m2": (float(second.get("carbon_per_m2") or 0.0) if second is not None else None),
        })

    return pd.DataFrame(rows)


def table_families_ranked_by_carbon(df_all: pd.DataFrame, *, metric: str = "carbon_per_m2") -> pd.DataFrame:
    """
    Verbose:
    Every successful system_family ranked by minimum embodied carbon.
    """
    D = _normalize_columns(df_all)
    if "system_family" not in D.columns or metric not in D.columns:
        return pd.DataFrame()
    D["_success"] = _success_mask(D)
    D = D[D["_success"] == True].copy()
    if D.empty:
        return pd.DataFrame()
    fam = D.groupby("system_family")[metric].min().reset_index().sort_values(metric, ascending=True)
    return fam


def table_best_variant_per_group(df_all: pd.DataFrame, group_col: str, *, metric: str = "carbon_per_m2") -> pd.DataFrame:
    """
    Best successful variant per group_col ('type' or 'typology').
    """
    D = _normalize_columns(df_all)
    if group_col not in D.columns or metric not in D.columns:
        return pd.DataFrame()

    D["_success"] = _success_mask(D)
    D = D[D["_success"] == True].copy()
    if D.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for gval, sub in D.groupby(group_col, dropna=False):
        br = best_row_by_metric(sub, metric, feasible_col=None)
        if br is None:
            continue
        d = br.to_dict()
        d[group_col] = gval
        rows.append(d)

    out = pd.DataFrame(rows)
    out[metric] = to_numeric_safe(out.get(metric))
    out = out.sort_values(metric, ascending=True, na_position="last")
    return out


def table_code_checks(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extract code check results (overall pass + deflection/shear/flexure) if present.
    Supports either flat columns or a nested 'code_outputs' dict column.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    D = df_all.copy()

    if "code_outputs" in D.columns:
        def safe_get(d: Any, k: str) -> Any:
            return d.get(k) if isinstance(d, dict) else None

        for k in ("pass_overall", "deflection_ok", "shear_ok", "flex_ok", "flexure_ok",
                  "deflection_util", "shear_util", "flexure_util", "moment_util"):
            if k not in D.columns:
                D[k] = D["code_outputs"].apply(lambda x: safe_get(x, k))

    if "flexure_ok" in D.columns and "flex_ok" not in D.columns:
        D["flex_ok"] = D["flexure_ok"]

    cols_want = [
        "system_variant", "system_family", "type", "typology",
        "pass_overall", "deflection_ok", "shear_ok", "flex_ok",
        "deflection_util", "shear_util", "flexure_util", "moment_util",
    ]
    cols = [c for c in cols_want if c in D.columns]
    if not cols:
        return pd.DataFrame()

    out = D[cols].copy()
    if "pass_overall" in out.columns:
        out["pass_overall"] = _coerce_bool_series(out["pass_overall"])
    return out.drop_duplicates(subset=[c for c in ("system_variant", "system_family", "type") if c in out.columns])

def write_enriched_summary_ranked_all(
    df_all: pd.DataFrame,
    out_dir: Union[str, Path],
    *,
    filename: str = "summary_ranked_all.csv",
) -> str:
    """
    Writes a refreshed/enriched summary_ranked_all.csv at the root of out_dir,
    after schema standardization + canonical joins (e.g., manufacturer).
    Returns filepath as string.
    """
    out_dir = Path(out_dir)
    out_fp = out_dir / filename

    D = df_all.copy()
    D = standardize_schema(D)
    D = _join_system_families_metadata(D)  # adds manufacturer (and possibly typology/type)

    # Ensure join key is present and clean
    if "system_family" in D.columns:
        D["system_family"] = D["system_family"].astype(str).str.strip()
    if "manufacturer" in D.columns:
        D["manufacturer"] = D["manufacturer"].astype(str).str.strip()

    D.to_csv(out_fp, index=False)
    logger.info("[reporting] wrote enriched %s (%d rows)", out_fp.name, len(D))
    return str(out_fp)


# ------------------------
# Figures
# ------------------------
def _component_carbon_columns(df: pd.DataFrame) -> List[str]:
    """Return available per-m2 component carbon columns in a stable order."""
    preferred = [
        "carbon_concrete_per_m2",
        "carbon_steel_per_m2",
        "carbon_timber_per_m2",
        "carbon_screed_per_m2",
    ]
    cols = [c for c in preferred if c in df.columns]
    return cols


def _pick_group_col(df: pd.DataFrame) -> str:
    """Prefer 'type', then 'typology', else fallback to 'system_family'."""
    if "type" in df.columns:
        return "type"
    if "typology" in df.columns:
        return "typology"
    return "system_family" if "system_family" in df.columns else "system_variant"


def plot_carbon_composition_stacked(
    df_all: pd.DataFrame,
    fig_path: Path,
    *,
    carbon_total_col: str = "carbon_per_m2",
    success_only: bool = True,
) -> Optional[Path]:
    """
    Stacked bar: kgCO2e/m2 contributions by component, using the *lowest-carbon* variant per group.
    Group = type (preferred) or typology fallback.
    """
    if df_all is None or df_all.empty:
        return None

    df = df_all.copy()
    df = _ensure_carbon(df)

    comp_cols = _component_carbon_columns(df)
    if not comp_cols:
        return None

    # Optional success filtering (keep consistent with other plots)
    if success_only and "pass_overall" in df.columns:
        df = df[df["pass_overall"].astype(bool)]

    group_col = _pick_group_col(df)
    if group_col not in df.columns:
        return None

    # choose the lowest total carbon per group
    df[carbon_total_col] = pd.to_numeric(df.get(carbon_total_col, df["carbon_per_m2"]), errors="coerce")
    winners = df.loc[df.groupby(group_col)[carbon_total_col].idxmin()].copy()
    winners = winners.dropna(subset=[group_col])

    if winners.empty:
        return None

    # Make sure comp cols are numeric
    for c in comp_cols:
        winners[c] = pd.to_numeric(winners[c], errors="coerce").fillna(0.0)

    winners = winners.sort_values(carbon_total_col, ascending=True)
    X = winners[group_col].astype(str).tolist()

    bottom = np.zeros(len(winners), dtype=float)

    plt.figure(figsize=(max(8, 0.5 * len(winners)), 5))
    for c in comp_cols:
        vals = winners[c].values.astype(float)
        plt.bar(X, vals, bottom=bottom, label=c.replace("carbon_", "").replace("_per_m2", ""))
        bottom += vals

    plt.ylabel("kgCO₂e / m²")
    plt.title(f"Embodied carbon composition (lowest-carbon per {group_col})")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def plot_carbon_composition_share(
    df_all: pd.DataFrame,
    fig_path: Path,
    *,
    carbon_total_col: str = "carbon_per_m2",
    success_only: bool = True,
) -> Optional[Path]:
    """
    100% stacked bar: % shares of component carbon (lowest-carbon per group).
    """
    if df_all is None or df_all.empty:
        return None

    df = df_all.copy()
    df = _ensure_carbon(df)

    comp_cols = _component_carbon_columns(df)
    if not comp_cols:
        return None

    if success_only and "pass_overall" in df.columns:
        df = df[df["pass_overall"].astype(bool)]

    group_col = _pick_group_col(df)
    if group_col not in df.columns:
        return None

    df[carbon_total_col] = pd.to_numeric(df.get(carbon_total_col, df["carbon_per_m2"]), errors="coerce")
    winners = df.loc[df.groupby(group_col)[carbon_total_col].idxmin()].copy()
    winners = winners.dropna(subset=[group_col])

    if winners.empty:
        return None

    for c in comp_cols:
        winners[c] = pd.to_numeric(winners[c], errors="coerce").fillna(0.0)

    totals = winners[comp_cols].sum(axis=1).replace(0.0, np.nan)
    shares = winners[comp_cols].div(totals, axis=0).fillna(0.0) * 100.0

    winners = winners.sort_values(carbon_total_col, ascending=True)
    shares = shares.loc[winners.index]

    X = winners[group_col].astype(str).tolist()
    bottom = np.zeros(len(winners), dtype=float)

    plt.figure(figsize=(max(8, 0.5 * len(winners)), 5))
    for c in comp_cols:
        vals = shares[c].values.astype(float)
        plt.bar(X, vals, bottom=bottom, label=c.replace("carbon_", "").replace("_per_m2", ""))
        bottom += vals

    plt.ylabel("Share of embodied carbon (%)")
    plt.ylim(0, 100)
    plt.title(f"Embodied carbon shares (lowest-carbon per {group_col})")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def build_carbon_per_material_table(
    df_all: pd.DataFrame,
    materials_df: Optional[pd.DataFrame] = None,
    *,
    json_col: str = "carbon_by_material_id_json",
) -> pd.DataFrame:
    """
    Explode carbon_by_material_id_json -> long table aggregated by material_id.
    Returns columns like: material_id, sum_kgco2e_per_m2, mean_kgco2e_per_m2, min, max, count (+ optional metadata).
    """
    if df_all is None or df_all.empty or json_col not in df_all.columns:
        return pd.DataFrame()

    # Parse json dict per row into long records
    records: List[Dict[str, Any]] = []
    for _, r in df_all[[json_col]].dropna().iterrows():
        try:
            d = json.loads(r[json_col]) if isinstance(r[json_col], str) else {}
        except Exception:
            d = {}
        if not isinstance(d, dict):
            continue
        for mat_id, val in d.items():
            try:
                v = float(val)
            except Exception:
                v = 0.0
            records.append({"material_id": str(mat_id), "kgco2e_per_m2": v})

    if not records:
        return pd.DataFrame()

    long_df = pd.DataFrame.from_records(records)

    agg = (
        long_df.groupby("material_id")["kgco2e_per_m2"]
        .agg(["count", "sum", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "sum": "sum_kgco2e_per_m2",
                "mean": "mean_kgco2e_per_m2",
                "min": "min_kgco2e_per_m2",
                "max": "max_kgco2e_per_m2",
            }
        )
        .sort_values("sum_kgco2e_per_m2", ascending=False)
    )

    # Optional enrichment from materials_df
    if materials_df is not None and not materials_df.empty and "material_id" in materials_df.columns:
        meta_cols = [c for c in ["category", "name", "material_name", "unit", "ec_kgco2_per_unit"] if c in materials_df.columns]
        meta = materials_df[["material_id"] + meta_cols].copy()
        meta["material_id"] = meta["material_id"].astype(str)

        # Normalize name column if needed
        if "material_name" in meta.columns and "name" not in meta.columns:
            meta = meta.rename(columns={"material_name": "name"})

        agg = agg.merge(meta.drop_duplicates("material_id"), on="material_id", how="left")

    return agg

def plot_span_vs_carbon_by_type(df: pd.DataFrame, out_fp: Path, *, label_prefix="", success_only=False):
    """
    Scatter of span vs carbon_per_m2 colored by 'type'. If type missing, color by system_family.
    success_only: if True, plot only successful variants (uses pass_overall/_success mask).
    """
    try:
        df = _ensure_carbon(df)
    except KeyError:
        return None

    span_col = _get_span_col(df)
    if span_col is None:
        return None

    df = df.copy()
    df["_carbon"] = pd.to_numeric(df["carbon_per_m2"], errors="coerce")
    df["_span"] = pd.to_numeric(df[span_col], errors="coerce")
    if success_only:
        mask = _ensure_success_mask(df)
        df = df[mask.fillna(False)]
    df = df.dropna(subset=["_carbon", "_span"])
    if df.empty:
        print("[INFO] No passing rows — plotting all candidates instead.")
        df = df.copy()

    color_by = "type" if "type" in df.columns else "system_family" if "system_family" in df.columns else None
    if color_by is None:
        color_by = "_const_type"
        df[color_by] = "all"

    groups = list(df.groupby(color_by))
    names = sorted([str(n) for n, _ in groups])
    colmap = get_color_map_for(color_by, names)
    fig, ax = plt.subplots(figsize=(8,5))
    for name, g in groups:
        ax.scatter(g["_span"], g["_carbon"], label=str(name), s=28, alpha=0.8,
                   edgecolors="none", color=colmap.get(str(name)))
    ax.set_title(f"{label_prefix}Span vs Carbon (colored by {color_by})")
    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied Carbon (kgCO2e/m²)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=1)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out_fp)



def plot_bar_best_by_group(
    best_df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    out_path: Union[str, Path],
    *,
    title: str = "",
    orientation: str = "v",
    add_value_labels: bool = True,
    color_by: Optional[str] = None,
    show_legend: bool = True,
) -> Optional[str]:
    """Bar chart: one bar per group, value=metric.

    Parameters
    ----------
    orientation:
        "v" for vertical bars (default) or "h" for horizontal bars.
    add_value_labels:
        If True, prints bar-end value labels.
    color_by:
        Optional column used to color-code bars (e.g., typology). If not provided or missing,
        bars use matplotlib defaults.
    """
    if best_df is None or best_df.empty:
        return None
    D = best_df.copy()
    if group_col not in D.columns or metric_col not in D.columns:
        return None

    D[metric_col] = to_numeric_safe(D[metric_col])
    D = D.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
    if D.empty:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = D[group_col].astype(str).tolist()
    values = D[metric_col].astype(float).values

    # Build color mapping if requested
    colors = None
    legend_handles = None
    if color_by and color_by in D.columns:
        cats = D[color_by].astype(str).fillna("Unknown")
        colors, legend_handles = colors_for_series(cats, column_name=color_by)
        # legend_handles already built

    if orientation not in {"v", "h"}:
        orientation = "v"

    if orientation == "h":
        fig_h = max(4.2, 0.42 * len(D) + 1.6)
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        y = np.arange(len(values))
        bars = ax.barh(y, values, color=colors)
        ax.set_yticks(y, labels)
        ax.set_xlabel(metric_col)
        ax.set_title(title or f"Best by {group_col} ({metric_col})")
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)
        if add_value_labels:
            for rect, v in zip(bars, values):
                if np.isnan(v):
                    continue
                ax.text(v, rect.get_y() + rect.get_height() / 2, f" {v:.0f}", va="center", fontsize=9)
        if legend_handles and show_legend:
            ax.legend(handles=legend_handles, title=color_by, loc="lower right", frameon=False)
    else:
        fig_w = max(7.2, 0.32 * len(D) + 3.2)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        x = np.arange(len(values))
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(metric_col)
        ax.set_title(title or f"Best by {group_col} ({metric_col})")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        if add_value_labels:
            for rect, v in zip(bars, values):
                if np.isnan(v):
                    continue
                ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        if legend_handles and show_legend:
            ax.legend(handles=legend_handles, title=color_by, loc="upper right", frameon=False)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_path)



def plot_span_vs_carbon_colored(
    df_in: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    success_only: bool,
    color_by: str = "type",
    span_col: str = "max_span",
    carbon_col: str = "carbon_per_m2",
    title: str = "",
    positive_only: bool = False,
    highlight_variants: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Scatter: x=span, y=carbon, colored by `color_by`.

    Enhancements for reporting:
      - positive_only: filter carbon > 0
      - highlight_variants: color-code all points (low alpha/small), then overlay highlighted variants (high alpha/large)
        (expects a column like 'system_variant' or 'variant_id').
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_in is None or df_in.empty:
        _write_placeholder_figure(out_path, "No data to plot.")
        return str(out_path)

    D = _normalize_columns(df_in)
    D["_span"] = to_numeric_safe(D.get(span_col))
    D["_carbon"] = to_numeric_safe(D.get(carbon_col))
    D = D.dropna(subset=["_span", "_carbon"])

    if positive_only:
        D = D[D["_carbon"] > 0].copy()

    if success_only:
        D["_success"] = _success_mask(D)
        D = D[D["_success"] == True].copy()

    if D.empty:
        _write_placeholder_figure(out_path, "No points after filtering.")
        return str(out_path)

    # Determine id column for highlighting
    id_col = None
    for c in ["system_variant", "variant_id", "variant", "id", "name"]:
        if c in D.columns:
            id_col = c
            break

    highlight_set = set([str(x) for x in highlight_variants]) if highlight_variants else set()
    has_highlight = bool(highlight_set) and (id_col is not None)

    if color_by not in D.columns:
        color_by = "typology" if "typology" in D.columns else ""

    cats: List[str] = []
    colmap: Dict[str, Any] = {}
    if color_by:
        cats = sorted(D[color_by].dropna().astype(str).unique())
        colmap = get_color_map_for(color_by, cats)

    fig, ax = plt.subplots(figsize=(9, 6))

    if has_highlight:
        # All points (low alpha/small), still color-coded
        if color_by and cats:
            for gv in cats:
                sub_all = D[D[color_by].astype(str) == gv]
                if not sub_all.empty:
                    ax.scatter(sub_all["_span"], sub_all["_carbon"], s=12, alpha=0.12,
                               color=colmap.get(str(gv)), edgecolors="none")
        else:
            ax.scatter(D["_span"], D["_carbon"], s=12, alpha=0.12, edgecolors="none")

        # Highlighted overlay
        H = D[D[id_col].astype(str).isin(highlight_set)].copy()
        if not H.empty:
            if color_by and cats:
                for gv in cats:
                    sub = H[H[color_by].astype(str) == gv]
                    if sub.empty:
                        continue
                    ax.scatter(sub["_span"], sub["_carbon"], s=55, alpha=1.0,
                               color=colmap.get(str(gv)), label=str(gv), edgecolors="none")
                ax.legend(title=color_by, loc="best", frameon=True)
            else:
                ax.scatter(H["_span"], H["_carbon"], s=55, alpha=1.0, edgecolors="none", label="highlighted")
                ax.legend(loc="best", frameon=True)
    else:
        # Regular colored scatter
        if color_by and cats:
            for gv in cats:
                sub = D[D[color_by].astype(str) == gv]
                ax.scatter(sub["_span"], sub["_carbon"], s=22, alpha=0.85,
                           color=colmap.get(str(gv)), label=str(gv), edgecolors="none")
            ax.legend(title=color_by, loc="best", frameon=True)
        else:
            ax.scatter(D["_span"], D["_carbon"], s=22, alpha=0.85, edgecolors="none", label="variants")
            ax.legend(loc="best", frameon=True)

    ax.set_title(title or "Span vs embodied carbon")
    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e/m²)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(out_path)

def plot_span_vs_carbon_pareto(
    df_all: pd.DataFrame,
    fig_dir: Path,
    carbon_col: str = "carbon_per_m2",
    *,
    span_bin_width: float = 0.25,
    max_frontier_points: int = 60,
) -> Dict[str, Path]:
    """
    Span vs carbon scatter with a "balanced" Pareto frontier.

    Frontier definition:
      - Start from passing rows if pass_overall exists; otherwise all rows.
      - Bin span values to reduce jaggedness (span_bin_width, in meters).
      - For each bin, take the minimum carbon point.
      - Connect those minima in increasing span order (optionally downsampled).
    """
    span_col = "max_span" if "max_span" in df_all.columns else ("span" if "span" in df_all.columns else None)
    if span_col is None or carbon_col not in df_all.columns:
        return {}

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = df_all[[span_col, carbon_col]].copy()
    df[span_col] = pd.to_numeric(df[span_col], errors="coerce")
    df[carbon_col] = pd.to_numeric(df[carbon_col], errors="coerce")
    df = df.dropna(subset=[span_col, carbon_col])

    has_pass = "pass_overall" in df_all.columns
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all candidates faint
    ax.scatter(df[span_col], df[carbon_col], alpha=0.18, label="All candidates")

    base = df
    if has_pass:
        df_pass = df_all[_coerce_bool_series(df_all["pass_overall"]).fillna(False)][[span_col, carbon_col]].copy()
        df_pass[span_col] = pd.to_numeric(df_pass[span_col], errors="coerce")
        df_pass[carbon_col] = pd.to_numeric(df_pass[carbon_col], errors="coerce")
        df_pass = df_pass.dropna(subset=[span_col, carbon_col])
        if not df_pass.empty:
            ax.scatter(df_pass[span_col], df_pass[carbon_col], alpha=0.8, label="Passing")
            base = df_pass

    # Compute binned minima to make the frontier visually stable
    if base.empty:
        frontier = pd.DataFrame(columns=[span_col, carbon_col])
    else:
        bw = float(span_bin_width) if span_bin_width and span_bin_width > 0 else 0.25
        tmp = base.copy()
        tmp["_span_bin"] = (np.round(tmp[span_col] / bw) * bw).astype(float)
        frontier = (
            tmp.groupby("_span_bin", as_index=False)[carbon_col].min()
               .rename(columns={"_span_bin": span_col})
               .sort_values(span_col)
        )

        # Downsample evenly if still too dense
        if max_frontier_points and len(frontier) > int(max_frontier_points):
            k = int(np.ceil(len(frontier) / int(max_frontier_points)))
            frontier = frontier.iloc[::k, :].copy()

    if not frontier.empty:
        ax.plot(frontier[span_col], frontier[carbon_col], linewidth=2.4, label="Pareto frontier")

    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied Carbon (kgCO₂/m²)")
    ax.legend()

    out_path = fig_dir / "span_vs_carbon_pareto.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"span_vs_carbon_pareto": out_path}


def _write_placeholder_figure(out_path: Path, message: str) -> None:
    """Write a minimal placeholder image for empty/invalid plots."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.6, message, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_span_vs_carbon_global(df, out_fp: Path):
    try:
        col = "carbon_per_m2" if "carbon_per_m2" in df.columns else "carbon_total_kgCO2"
        span = "max_span" if "max_span" in df.columns else None
        if span is None or col not in df.columns:
            return None
        df2 = df.copy()
        df2["_span"] = pd.to_numeric(df2[span], errors="coerce")
        df2["_carb"] = pd.to_numeric(df2[col], errors="coerce")
        df2 = df2.dropna(subset=["_span","_carb"])
        if df2.empty:
            return None
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df2["_span"], df2["_carb"], s=18, alpha=0.7, label="variants")
        ax.legend(loc="best", frameon=False)
        ax.set_xlabel("Maximum Grid Span (m)")
        ax.set_ylabel("Embodied Carbon (kgCO₂e / m²)")
        ax.set_title("Maximum Grid Span vs Embodied Carbon (all variants)")
        ax.grid(True, linestyle=":", alpha=0.4)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fp, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return str(out_fp)
    except Exception:
        logger.exception("plot_span_vs_carbon_global failed")
        return None
    
def plot_depth_vs_carbon(
    df_all: pd.DataFrame,
    fig_dir: Optional[Union[str, Path]] = None,
    *,
    outpath: Optional[Union[str, Path]] = None,
    carbon_col: str = "carbon_per_m2",
) -> Dict[str, Path]:
    """Scatter: total structural depth vs carbon.

    Back-compat behavior:
      - Old signature: plot_depth_vs_carbon(df_all, fig_dir)
      - New behavior: you may pass outpath=... to control the exact file.

    Returns a dict mapping a short key -> Path.
    """
    if df_all is None or df_all.empty:
        return {}

    # Determine output location
    if outpath is not None:
        out_path = Path(outpath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if fig_dir is None:
            return {}
        p = Path(fig_dir)
        # If a file path was mistakenly passed instead of a directory, treat its parent as the directory.
        if p.suffix:
            p = p.parent
        p.mkdir(parents=True, exist_ok=True)
        out_path = p / "depth_vs_carbon.png"

    depth_cols = ["slab_depth", "beam_depth", "steel_depth", "screed_depth"]
    depth_cols = [c for c in depth_cols if c in df_all.columns]
    if not depth_cols:
        return {}

    df = df_all.copy()
    for c in depth_cols + [carbon_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["total_depth"] = df[depth_cols].sum(axis=1, skipna=True)
    df = df.dropna(subset=["total_depth", carbon_col])
    if df.empty:
        return {}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["total_depth"], df[carbon_col], alpha=0.3)
    ax.set_xlabel("Total structural depth")
    ax.set_ylabel("Embodied Carbon (kgCO₂/m²)")
    ax.set_title("Depth vs embodied carbon")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"depth_vs_carbon": out_path}

def plot_carbon_distribution_by_type(
    df_all: pd.DataFrame,
    fig_dir: Optional[Union[str, Path]] = None,
    *,
    outpath: Optional[Union[str, Path]] = None,
    carbon_col: str = "carbon_per_m2",
) -> Dict[str, Path]:

    if df_all is None or df_all.empty:
        return {}
    if "type" not in df_all.columns:
        return {}

    # Determine output path
    if outpath is not None:
        out_path = Path(outpath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if fig_dir is None:
            return {}
        p = Path(fig_dir)
        if p.suffix:
            p = p.parent
        p.mkdir(parents=True, exist_ok=True)
        out_path = p / "carbon_distribution_by_type.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    df = df_all.copy()
    if carbon_col not in df.columns:
        # try to ensure carbon column
        try:
            df = _ensure_carbon(df)
        except Exception:
            return {}
    df[carbon_col] = pd.to_numeric(df[carbon_col], errors="coerce")

    # Use pandas boxplot grouped by type
    df.boxplot(column=carbon_col, by="type", ax=ax)
    ax.set_ylabel("Embodied Carbon (kgCO₂/m²)")
    ax.set_title("Embodied carbon distribution by type")
    plt.suptitle("")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"carbon_distribution_by_type": out_path}

def plot_feasibility_heatmap(
    df_all: pd.DataFrame,
    fig_dir: Path,
    *,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Writes TWO heatmaps:
      - Option B: pass rate among evaluated variants (post-systems.py) -> uses df_all
      - Option A: pass rate among generated variants (system_variants) -> loads system_variants and marks pass if variant appears in df_all passes

    Returns figure paths in a dict.
    """
    if df_all is None or df_all.empty:
        return {}

    if "pass_overall" not in df_all.columns:
        return {}

    # --- identify id column (df_all in write_edca_reports requires system_variant)
    id_col = "system_variant" if "system_variant" in df_all.columns else None
    if id_col is None:
        return {}

    df_eval = df_all.copy()
    df_eval[id_col] = df_eval[id_col].astype(str).str.strip()
    df_eval["pass_overall"] = _coerce_bool_series(df_eval["pass_overall"])

    # --- span + total load (evaluated)
    span_col_eval = "max_span" if "max_span" in df_eval.columns else ("span" if "span" in df_eval.columns else None)
    load_col_eval = _infer_total_load_col(df_eval)

    if span_col_eval is None or load_col_eval is None:
        logger.warning("[heatmap] Missing span/load columns for evaluated heatmap.")
        return {}

    df_eval["_span"] = pd.to_numeric(df_eval[span_col_eval], errors="coerce")
    df_eval["_total_load"] = pd.to_numeric(df_eval[load_col_eval], errors="coerce")

    # Passing set (used to score generated catalogue)
    passing_set = set(
        df_eval.loc[df_eval["pass_overall"], id_col]
        .astype(str).str.strip()
        .unique()
        .tolist()
    )

    # --- try to load system_variants (generated)
    df_gen = None
    sys_source = None
    if out_dir is not None:
        od = Path(out_dir)

        # search a few plausible roots (repo root is often od.parent.parent)
        roots = []
        for r in [od, od.parent, od.parent.parent, Path.cwd()]:
            if r not in roots and r.exists():
                roots.append(r)

        # fast-path: common canonical location
        candidates = []
        for r in roots:
            p1 = r / "inputs" / "canonical" / "system_variants.parquet"
            p2 = r / "inputs" / "canonical" / "system_variants.csv"
            if p1.exists():
                candidates.append(p1)
            if p2.exists():
                candidates.append(p2)

        # fallback: search by filename (first hit wins)
        if not candidates:
            for r in roots:
                candidates.extend(list(r.rglob("system_variants.parquet")))
                candidates.extend(list(r.rglob("system_variants.csv")))
                if candidates:
                    break

        for p in candidates:
            try:
                if p.suffix.lower() == ".parquet":
                    df_gen = pd.read_parquet(p)
                else:
                    df_gen = pd.read_csv(p)
                sys_source = str(p)
                break
            except Exception:
                continue

    # --- standardize generated df if found
    if df_gen is not None and not df_gen.empty:
        df_gen = df_gen.copy()

        # find id column in generated df
        if id_col not in df_gen.columns:
            for alt in ("variant", "variant_id", "system_variant_id", "id", "system"):
                if alt in df_gen.columns:
                    df_gen = df_gen.rename(columns={alt: id_col})
                    break

        if id_col not in df_gen.columns:
            logger.warning("[heatmap] system_variants file found but no id col; skipping Option A heatmap. source=%s", sys_source)
            df_gen = None
        else:
            df_gen[id_col] = df_gen[id_col].astype(str).str.strip()

            # span + total load (generated)
            span_col_gen = "max_span" if "max_span" in df_gen.columns else ("span" if "span" in df_gen.columns else None)
            load_col_gen = _infer_total_load_col(df_gen)

            if span_col_gen is None or load_col_gen is None:
                logger.warning("[heatmap] system_variants missing span/load; skipping Option A heatmap. source=%s", sys_source)
                df_gen = None
            else:
                df_gen["_span"] = pd.to_numeric(df_gen[span_col_gen], errors="coerce")
                df_gen["_total_load"] = pd.to_numeric(df_gen[load_col_gen], errors="coerce")

                # Option A definition: end-to-end yield on generated catalogue
                df_gen["pass_overall"] = df_gen[id_col].isin(passing_set)

    # --- dataset-specific binning (fixes empty grids + 1x1 pivots)
    def _fmt_interval(iv) -> str:
        import pandas as _pd
        if isinstance(iv, _pd.Interval):
            # shorter labels
            return f"{iv.left:.1f}–{iv.right:.1f}"
        return str(iv)

    def _pivot_quantile(D: pd.DataFrame, n_span_bins: int, n_load_bins: int) -> pd.DataFrame:
        X = D.copy()
        X = X.dropna(subset=["_span", "_total_load", "pass_overall"])
        if X.empty:
            return pd.DataFrame()

        # If very few unique values, reduce bins automatically
        # Auto-cap bins so we don't over-fragment:
        # - never more than unique values
        # - never more than ~sqrt(N) (keeps average ~1 sample per 2D cell from exploding)
        N = len(X)
        cap = max(2, int(np.sqrt(N)))  # heuristic; adjust if you want
        n_span = max(1, min(int(n_span_bins), int(pd.Series(X["_span"]).nunique()), cap))
        n_load = max(1, min(int(n_load_bins), int(pd.Series(X["_total_load"]).nunique()), cap))

        # If a variable is essentially constant, you will *still* get 1 bin — that's correct.
        if n_span == 1:
            span_bin = pd.cut(X["_span"], bins=1, include_lowest=True)
        else:
            span_bin = pd.qcut(X["_span"], q=n_span, duplicates="drop")

        if n_load == 1:
            load_bin = pd.cut(X["_total_load"], bins=1, include_lowest=True)
        else:
            load_bin = pd.qcut(X["_total_load"], q=n_load, duplicates="drop")

        X["span_bin"] = span_bin
        X["load_bin"] = load_bin

        # Pivot with full category grids (so you see consistent axes)
        piv = X.pivot_table(
            index="span_bin",
            columns="load_bin",
            values="pass_overall",
            aggfunc="mean",
        )

        # Reindex to include all bins even if some combos are empty
        if hasattr(span_bin, "cat"):
            piv = piv.reindex(index=span_bin.cat.categories)
        if hasattr(load_bin, "cat"):
            piv = piv.reindex(columns=load_bin.cat.categories)

        return piv

    def _plot(pivot: pd.DataFrame, out_path: Path, title: str, cbar_label: str) -> Optional[Path]:
        if pivot is None or pivot.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)

        k = 5  # show every 5th label
        # Show ~12 labels max per axis (adaptive)
        max_labels = 12
        x_step = max(1, int(np.ceil(len(pivot.columns) / max_labels)))
        y_step = max(1, int(np.ceil(len(pivot.index) / max_labels)))

        ax.set_xticks(list(range(0, len(pivot.columns), x_step)))
        ax.set_xticklabels(
            [_fmt_interval(pivot.columns[i]) for i in range(0, len(pivot.columns), x_step)],
            rotation=45, ha="right"
        )

        ax.set_yticks(list(range(0, len(pivot.index), y_step)))
        ax.set_yticklabels(
            [_fmt_interval(pivot.index[i]) for i in range(0, len(pivot.index), y_step)]
        )

        ax.set_xlabel("Total load bin")
        ax.set_ylabel("Span bin")
        ax.set_title(title)

        fig.colorbar(im, ax=ax, label=cbar_label)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    paths: Dict[str, Path] = {}

    # Option B (evaluated denom)
    # Option B (evaluated denom) — use MORE bins if you want finer resolution
    # 5–10× finer than 6 bins:
    piv_eval = _pivot_quantile(df_eval, n_span_bins=20, n_load_bins=20)

    p = _plot(
        piv_eval,
        fig_dir / "feasibility_heatmap_optionB_post_systems.png",
        title="Feasibility heatmap — pass rate of evaluated variants (post systems.py)",
        cbar_label="Pass rate (evaluated denom)",
    )
    if p:
        paths["feasibility_heatmap_optionB_post_systems"] = p

    # Option A (generated denom) — often benefits from a few more bins
    if df_gen is not None and not df_gen.empty:
        piv_gen = _pivot_quantile(df_gen, n_span_bins=20, n_load_bins=20)
        p = _plot(
            piv_gen,
            fig_dir / "feasibility_heatmap_optionA_generated.png",
            title="Feasibility heatmap — pass rate of generated variants (end-to-end yield)",
            cbar_label="Pass rate (generated denom)",
        )
        if p:
            paths["feasibility_heatmap_optionA_generated"] = p
    else:
        logger.info("[heatmap] Skipping Option A heatmap (system_variants not found).")

    return paths

def plot_failure_breakdown(df_all: pd.DataFrame,
                           fig_dir: Path) -> Dict[str, Path]:

    failure_cols = [c for c in df_all.columns if c.endswith("_util")]

    if not failure_cols:
        return {}

    failures = {}

    for col in failure_cols:
        failures[col] = (df_all[col] > 1.0).sum()

    s = pd.Series(failures).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    s.plot(kind="bar", ax=ax)

    ax.set_ylabel("Count of failures")

    out_path = fig_dir / "failure_breakdown.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"failure_breakdown": out_path}

def plot_carbon_breakdown(df_all: pd.DataFrame,
                          materials_df: pd.DataFrame,
                          fig_dir: Path,
                          top_n: int = 5,
                          carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:

    required_cols = [
        "material_concrete_id",
        "material_steel_id",
        "material_timber_id",
        "concrete_volume",
        "steel_volume",
        "timber_volume",
    ]

    if not all(c in df_all.columns for c in required_cols):
        return {}

    if "ec_kgco2_per_unit" not in materials_df.columns:
        return {}

    # Build material intensity lookup
    intensity_lookup = (
        materials_df
        .set_index("material_id")["ec_kgco2_per_unit"]
        .to_dict()
    )

    df = df_all.nsmallest(top_n, carbon_col).copy()

    def compute_component(row, mat_id_col, vol_col):
        mat_id = row[mat_id_col]
        intensity = intensity_lookup.get(mat_id, 0.0)
        return row[vol_col] * intensity

    df["carbon_concrete"] = df.apply(
        lambda r: compute_component(r, "material_concrete_id", "concrete_volume"),
        axis=1,
    )

    df["carbon_steel"] = df.apply(
        lambda r: compute_component(r, "material_steel_id", "steel_volume"),
        axis=1,
    )

    df["carbon_timber"] = df.apply(
        lambda r: compute_component(r, "material_timber_id", "timber_volume"),
        axis=1,
    )

    breakdown = df[
        ["carbon_concrete", "carbon_steel", "carbon_timber"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    breakdown.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Embodied Carbon (kgCO₂/m²)")
    ax.set_title("Carbon breakdown – top systems")
    ax.legend(["Concrete", "Steel", "Timber"])

    out_path = fig_dir / "carbon_breakdown_top.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"carbon_breakdown_top": out_path}

def plot_span_vs_total_load_global(df, out_fp: Path):
    span = "max_span" if "max_span" in df.columns else None
    total_col = None
    for candidate in ["total_load", "total_capacity", "sdl_total", "sdl_total", "sdl", "ll", "total_load_synth"]:
        if candidate in df.columns:
            total_col = candidate
            break
    # if missing, try to synthesize
    if total_col is None and "sdl_total" in df.columns and "ll" in df.columns:
        df = df.copy()
        df["total_load_synth"] = pd.to_numeric(df["sdl_total"], errors="coerce") + pd.to_numeric(df["ll"], errors="coerce")
        total_col = "total_load_synth"
    if span is None or total_col is None:
        return None
    df2 = df.copy()
    df2["_span"] = pd.to_numeric(df2[span], errors="coerce")
    df2["_load"] = pd.to_numeric(df2[total_col], errors="coerce")
    df2 = df2.dropna(subset=["_span","_load"])
    if df2.empty:
        return None
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df2["_span"], df2["_load"], s=18, alpha=0.8, label="variants")
    ax.legend(loc="best", frameon=False)
    ax.set_xlabel("Maximum Grid Span (m)"); ax.set_ylabel("Total load (kN/m²)")
    ax.set_title("Maximum Grid Span vs Total load (all variants)")
    ax.grid(True, linestyle=":", alpha=0.4)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out_fp)




def plot_lowest_family_per_typology(
    df_all: pd.DataFrame,
    out_dir: Union[str, Path],
    *,
    success_only: bool = False,
    metric: str = "carbon_per_m2",
) -> Optional[str]:
    """For each typology, pick the lowest-carbon system_family and plot its curve(s).

    Output is a single figure: span vs carbon for the chosen families (one line per typology:family).
    """
    if df_all is None or df_all.empty:
        return None

    out_dir = Path(out_dir)
    out_fp = out_dir / "lowest_family_per_typology.png"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    D = _normalize_columns(df_all)
    if "system_family" not in D.columns or "typology" not in D.columns:
        _write_placeholder_figure(out_fp, "Missing 'system_family' or 'typology' columns.")
        return str(out_fp)

    try:
        D = _ensure_carbon(D)
    except KeyError:
        _write_placeholder_figure(out_fp, "No carbon column found.")
        return str(out_fp)

    span_col = _get_span_col(D)
    if span_col is None:
        _write_placeholder_figure(out_fp, "No span column found (expected 'max_span').")
        return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_carbon"] = pd.to_numeric(D["carbon_per_m2"], errors="coerce")
    D = D.dropna(subset=["_span", "_carbon"])

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    fam_mins = (
        D.groupby(["typology", "system_family"])["_carbon"]
        .min()
        .reset_index()
    )
    if fam_mins.empty:
        _write_placeholder_figure(out_fp, "No typology/family minima.")
        return str(out_fp)

    chosen_idx = fam_mins.groupby("typology")["_carbon"].idxmin()
    chosen = fam_mins.loc[chosen_idx].reset_index(drop=True)
    if chosen.empty:
        _write_placeholder_figure(out_fp, "No chosen families.")
        return str(out_fp)

    chosen_pairs = pd.MultiIndex.from_frame(chosen[["typology", "system_family"]])
    row_pairs = pd.MultiIndex.from_frame(D[["typology", "system_family"]])
    plot_df = D.loc[row_pairs.isin(chosen_pairs)].copy()
    if plot_df.empty:
        _write_placeholder_figure(out_fp, "No rows for chosen families.")
        return str(out_fp)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for (typ, fam), g in plot_df.groupby(["typology", "system_family"]):
        g = g.sort_values("_span")
        ax.plot(g["_span"], g["_carbon"], marker="o", linewidth=1.6, alpha=0.9, label=f"{typ}: {fam}")

    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title("Lowest-carbon family per typology — span vs carbon")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)

def plot_lowest_family_per_type(
    df_all: pd.DataFrame,
    out_dir: Union[str, Path],
    *,
    success_only: bool = False,
    metric: str = "carbon_per_m2",
) -> Optional[str]:
    """For each type, pick the lowest-carbon system_family and plot its curve(s).

    Output is a single figure: span vs carbon for the chosen families (one line per type:family).
    """
    if df_all is None or df_all.empty:
        return None

    out_dir = Path(out_dir)
    out_fp = out_dir / "lowest_family_per_type.png"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    D = _normalize_columns(df_all)
    if "system_family" not in D.columns or "type" not in D.columns:
        _write_placeholder_figure(out_fp, "Missing 'system_family' or 'type' columns.")
        return str(out_fp)

    try:
        D = _ensure_carbon(D)
    except KeyError:
        _write_placeholder_figure(out_fp, "No carbon column found.")
        return str(out_fp)

    span_col = _get_span_col(D)
    if span_col is None:
        _write_placeholder_figure(out_fp, "No span column found (expected 'max_span').")
        return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_carbon"] = pd.to_numeric(D["carbon_per_m2"], errors="coerce")
    D = D.dropna(subset=["_span", "_carbon"])

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    fam_mins = (
        D.groupby(["type", "system_family"])["_carbon"]
        .min()
        .reset_index()
    )
    if fam_mins.empty:
        _write_placeholder_figure(out_fp, "No type/family minima.")
        return str(out_fp)

    chosen_idx = fam_mins.groupby("type")["_carbon"].idxmin()
    chosen = fam_mins.loc[chosen_idx].reset_index(drop=True)
    if chosen.empty:
        _write_placeholder_figure(out_fp, "No chosen families.")
        return str(out_fp)

    chosen_pairs = pd.MultiIndex.from_frame(chosen[["type", "system_family"]])
    row_pairs = pd.MultiIndex.from_frame(D[["type", "system_family"]])
    plot_df = D.loc[row_pairs.isin(chosen_pairs)].copy()
    if plot_df.empty:
        _write_placeholder_figure(out_fp, "No rows for chosen families.")
        return str(out_fp)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for (typ, fam), g in plot_df.groupby(["type", "system_family"]):
        g = g.sort_values("_span")
        ax.plot(g["_span"], g["_carbon"], marker="o", linewidth=1.6, alpha=0.9, label=f"{typ}: {fam}")

    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title("Lowest-carbon family per type — span vs carbon")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)


def _lowest_variant_per_group(
    df: pd.DataFrame,
    *,
    group_col: str = "system_family",
    metric: str = "carbon_per_m2",
    success_only: bool = False,
) -> pd.DataFrame:
    """Return one row per group: the lowest-metric variant in that group."""
    if df is None or df.empty:
        return pd.DataFrame()

    D = df.copy()
    try:
        D = _ensure_carbon(D)
    except KeyError:
        return pd.DataFrame()

    if group_col not in D.columns:
        return pd.DataFrame()

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    D["_metric"] = pd.to_numeric(D.get(metric), errors="coerce")
    D = D.dropna(subset=["_metric"])
    if D.empty:
        return pd.DataFrame()

    idx = D.groupby(group_col)["_metric"].idxmin().dropna().astype(int).tolist()
    if not idx:
        return pd.DataFrame()

    return D.loc[idx].copy()


def plot_span_vs_total_and_carbon_for_lowest(
    df: pd.DataFrame,
    group_col: str,
    figures_dir: Union[str, Path],
    *,
    success_only: bool = False,
) -> Dict[str, str]:
    """Two scatters for the lowest-carbon variant per group: span vs total_load and span vs carbon."""
    out: Dict[str, str] = {}
    if df is None or df.empty:
        return out

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    lowest = _lowest_variant_per_group(df, group_col=group_col, success_only=success_only)
    if lowest.empty:
        return out

    span_col = _get_span_col(lowest)
    if not span_col:
        return out

    lowest = lowest.copy()
    lowest["_span"] = pd.to_numeric(lowest[span_col], errors="coerce")

    total_col = _infer_total_load_col(lowest)
    if total_col:
        lowest["_total"] = pd.to_numeric(lowest[total_col], errors="coerce")
        sub = lowest.dropna(subset=["_span", "_total"])
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.scatter(sub["_span"], sub["_total"], s=34, alpha=0.9, edgecolors="none", label="lowest per group")
            ax.legend(loc="best", frameon=False)
            ax.set_xlabel("Maximum Grid Span (m)")
            ax.set_ylabel("Total load")
            ax.set_title(f"Lowest-carbon variant per {group_col}: span vs total load")
            ax.grid(True, linestyle=":", alpha=0.4)
            fp = figures_dir / f"lowest_per_{group_col}_span_vs_total_load.png"
            fig.savefig(fp, bbox_inches="tight", dpi=200)
            plt.close(fig)
            out[f"lowest_per_{group_col}_span_vs_total_load"] = str(fp)

    lowest["_carbon"] = pd.to_numeric(lowest.get("carbon_per_m2"), errors="coerce")
    sub = lowest.dropna(subset=["_span", "_carbon"])
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(sub["_span"], sub["_carbon"], s=34, alpha=0.9, edgecolors="none", label="lowest per group")
        ax.legend(loc="best", frameon=False)
        ax.set_xlabel("Maximum Grid Span (m)")
        ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
        ax.set_title(f"Lowest-carbon variant per {group_col}: span vs carbon")
        ax.grid(True, linestyle=":", alpha=0.4)
        fp = figures_dir / f"lowest_per_{group_col}_span_vs_carbon.png"
        fig.savefig(fp, bbox_inches="tight", dpi=200)
        plt.close(fig)
        out[f"lowest_per_{group_col}_span_vs_carbon"] = str(fp)

    return out


def plot_span_vs_load_curves_by_family_grid(
    df: pd.DataFrame,
    out_fp: Union[str, Path],
    *,
    max_families: int = 12,
    ncols: int = 3,
    success_only: bool = True,
    highlight_variants: Optional[Union[set[str], List[str]]] = None,
    passing_variants: Optional[Union[set[str], List[str]]] = None,
) -> Optional[str]:
    """
    Small-multiples grid: span vs total_load curves, one panel per family.

    If `passing_variants` is provided, **all** variants in `df` are plotted:
      - variants in `passing_variants` are drawn normally (typed color)
      - variants NOT in `passing_variants` are greyed out
    """
    if df is None or df.empty:
        return None

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    span_col = _get_span_col(df)
    if not span_col:
        _write_placeholder_figure(out_fp, "No span column found.")
        return str(out_fp)

    D = df.copy()
    D = add_typology(D)

    total_col = _infer_total_load_col(D)
    if total_col is None:
        _write_placeholder_figure(out_fp, "No total load column found.")
        return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_load"] = pd.to_numeric(D[total_col], errors="coerce")
    D = D.dropna(subset=["_span", "_load"])

    pass_set: Optional[set[str]] = None
    if passing_variants is not None:
        pass_set = set(str(x) for x in passing_variants)

    # Legacy behavior: if no pass_set, optionally filter to success-only
    if pass_set is None and success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    family_col = "system_family" if "system_family" in D.columns else ("manufacturer" if "manufacturer" in D.columns else None)
    if family_col is None:
        _write_placeholder_figure(out_fp, "No family column found.")
        return str(out_fp)

    variant_col = "system_variant" if "system_variant" in D.columns else family_col
    hv: set[str] = set(str(x) for x in (highlight_variants or []))

    # family ordering: highlight first, then most common
    families: List[str] = []
    if hv and "system_variant" in D.columns:
        fams_h = (
            D[D["system_variant"].astype(str).isin(hv)][family_col]
            .dropna().astype(str).unique().tolist()
        )
        families.extend(fams_h)

    counts = D[family_col].astype(str).value_counts()
    for fam in counts.index.tolist():
        if fam not in families:
            families.append(fam)

    families = families[:max_families]
    if not families:
        _write_placeholder_figure(out_fp, "No families to plot.")
        return str(out_fp)

    n = len(families)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig_w = 4.2 * ncols
    fig_h = 2.8 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    typ_cmap = get_color_map_for("typology", ["steel", "concrete", "timber", "other"])
    has_pass_col = "pass_overall" in D.columns

    for i, fam in enumerate(families):
        ax = axes[i]
        sub_f = D[D[family_col].astype(str) == str(fam)].copy()
        if sub_f.empty:
            ax.set_axis_off()
            continue

        for vname, sub_v in sub_f.groupby(variant_col):
            sub_v = sub_v.sort_values("_span")
            if sub_v.empty:
                continue

            vname_s = str(vname)
            is_h = vname_s in hv if hv else False
            is_passing = (vname_s in pass_set) if pass_set is not None else True

            typ = str(sub_v["typology"].iloc[0]) if "typology" in sub_v.columns else "other"
            line_color = "0.55" if not is_passing else typ_cmap.get(typ, typ_cmap.get("other"))

            base_alpha = 0.90 if is_h else (0.40 if is_passing else 0.16)
            lw = 2.6 if is_h else (1.25 if is_passing else 0.95)

            ax.plot(sub_v["_span"], sub_v["_load"], linewidth=lw, alpha=base_alpha, color=line_color)

            if has_pass_col:
                pm = _coerce_bool_series(sub_v["pass_overall"]).fillna(False)
                sub_pass = sub_v[pm]
                sub_fail = sub_v[~pm]
                if not sub_pass.empty:
                    ax.scatter(
                        sub_pass["_span"], sub_pass["_load"],
                        s=28 if is_h else (14 if is_passing else 10),
                        alpha=0.95 if is_h else (0.35 if is_passing else 0.16),
                        edgecolors="none",
                        color=line_color,
                    )
                if not sub_fail.empty:
                    ax.scatter(
                        sub_fail["_span"], sub_fail["_load"],
                        s=28 if is_h else (14 if is_passing else 10),
                        alpha=0.95 if is_h else (0.35 if is_passing else 0.16),
                        marker="x",
                        color=line_color,
                    )
            else:
                ax.scatter(
                    sub_v["_span"], sub_v["_load"],
                    s=28 if is_h else (14 if is_passing else 10),
                    alpha=0.95 if is_h else (0.35 if is_passing else 0.16),
                    edgecolors="none",
                    color=line_color,
                )

        ax.set_title(str(fam), fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    fig.supxlabel("Maximum Grid Span (m)")
    fig.supylabel("Total load / capacity")
    fig.suptitle("Span vs Load curves by family (small multiples)", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)


def plot_span_vs_load_curves_by_family_highlight(
    df: pd.DataFrame,
    out_fp: Union[str, Path],
    *,
    success_only: bool = True,
    highlight_variants: Optional[Union[set[str], List[str]]] = None,
    passing_variants: Optional[Union[set[str], List[str]]] = None,
    base_alpha: float = 0.12,
) -> Optional[str]:
    """Span vs total-load curves (all variants) with optional passing/selected emphasis.

    - Always draws the full curve context (all variants) from df.
    - If `passing_variants` is provided, curves NOT in that set are greyed out.
    - If `highlight_variants` is provided, those variants are emphasized on top.
    """
    if df is None or df.empty:
        return None

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    span_col = _get_span_col(df)
    if not span_col:
        _write_placeholder_figure(out_fp, "No span column found.")
        return str(out_fp)

    D = df.copy()
    total_col = _infer_total_load_col(D)
    if total_col is None:
        _write_placeholder_figure(out_fp, "No total load column found.")
        return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_load"] = pd.to_numeric(D[total_col], errors="coerce")
    D = D.dropna(subset=["_span", "_load"])

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    family_col = "system_family" if "system_family" in D.columns else ("manufacturer" if "manufacturer" in D.columns else None)
    variant_col = "system_variant" if "system_variant" in D.columns else ("variant_id" if "variant_id" in D.columns else family_col)
    if family_col is None or variant_col is None:
        _write_placeholder_figure(out_fp, "No family/variant column found.")
        return str(out_fp)

    hv: set[str] = set(str(x) for x in (highlight_variants or []))
    pv: Optional[set[str]] = set(str(x) for x in passing_variants) if passing_variants is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Base curves: either grey (non-passing) or lightly colored (passing)
    for vname, sub_v in D.groupby(variant_col):
        sub_v = sub_v.sort_values("_span")
        if sub_v.empty:
            continue

        vkey = str(vname)
        is_pass = True if pv is None else (vkey in pv)
        is_h = vkey in hv if hv else False

        if not is_pass:
            color = "0.7"
            lw = 1.0
            alpha = base_alpha
        else:
            # passing: color by material family (typology) if available, else fallback
            tlabel = None
            for col in ["typology", "type", "system_type", "system_typology"]:
                if col in sub_v.columns:
                    tlabel = str(sub_v.iloc[0][col])
                    break
            fam = _infer_material_family(tlabel or "")
            color = _GLOBAL_COLOR_MAPS.get("material_family", {}).get(fam, None)
            if color is None:
                # fallback to variant color map
                color = _GLOBAL_COLOR_MAPS.get("variant", {}).get(vkey, None)
            if color is None:
                color = "0.4"
            lw = 2.8 if is_h else 1.4
            alpha = 0.98 if is_h else 0.20

        ax.plot(sub_v["_span"], sub_v["_load"], linewidth=lw, alpha=alpha, color=color)

    # Points (optional): show pass/fail markers if pass_overall exists
    has_pass_flag = "pass_overall" in D.columns
    if has_pass_flag:
        for vname, sub_v in D.groupby(variant_col):
            sub_v = sub_v.sort_values("_span")
            if sub_v.empty:
                continue
            vkey = str(vname)
            is_pass = True if pv is None else (vkey in pv)
            is_h = vkey in hv if hv else False
            if not is_pass and not is_h:
                # keep non-passing background light
                continue

            pm = _coerce_bool_series(sub_v["pass_overall"]).fillna(False)
            sub_pass = sub_v[pm]
            sub_fail = sub_v[~pm]
            if not sub_pass.empty:
                ax.scatter(
                    sub_pass["_span"], sub_pass["_load"],
                    s=34 if is_h else 10, alpha=0.98 if is_h else 0.25,
                    edgecolors="none"
                )
            if not sub_fail.empty:
                ax.scatter(
                    sub_fail["_span"], sub_fail["_load"],
                    s=34 if is_h else 10, alpha=0.98 if is_h else 0.25,
                    marker="x"
                )

    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Total load")
    ax.grid(True, linestyle=":", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)

    D = df.copy()
    total_col = _infer_total_load_col(D)
    if total_col is None:
        _write_placeholder_figure(out_fp, "No total load column found.")
        return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_load"] = pd.to_numeric(D[total_col], errors="coerce")
    D = D.dropna(subset=["_span", "_load"])

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    family_col = "system_family" if "system_family" in D.columns else ("manufacturer" if "manufacturer" in D.columns else None)
    variant_col = "system_variant" if "system_variant" in D.columns else family_col
    if family_col is None or variant_col is None:
        _write_placeholder_figure(out_fp, "No family/variant column found.")
        return str(out_fp)

    hv: set[str] = set(str(x) for x in (highlight_variants or []))

    fig, ax = plt.subplots(figsize=(10, 6))

    for vname, sub_v in D.groupby(variant_col):
        sub_v = sub_v.sort_values("_span")
        ax.plot(sub_v["_span"], sub_v["_load"], linewidth=1.0, alpha=base_alpha, color="0.5")

    if hv:
        D_h = D[D[variant_col].astype(str).isin(hv)].copy()
        if not D_h.empty:
            for fam, sub_f in D_h.groupby(family_col):
                    has_pass = "pass_overall" in D.columns

    for vname, sub_v in D.groupby(variant_col):
        sub_v = sub_v.sort_values("_span")
        if sub_v.empty:
            continue
        is_h = str(vname) in hv if hv else False

        ax.plot(
            sub_v["_span"],
            sub_v["_load"],
            linewidth=2.8 if is_h else 1.2,
            alpha=0.98 if is_h else 0.18,
        )

        if has_pass:
            pm = _coerce_bool_series(sub_v["pass_overall"]).fillna(False)
            sub_pass = sub_v[pm]
            sub_fail = sub_v[~pm]
            if not sub_pass.empty:
                ax.scatter(sub_pass["_span"], sub_pass["_load"],
                           s=34 if is_h else 10, alpha=0.98 if is_h else 0.25,
                           edgecolors="none")
            if not sub_fail.empty:
                ax.scatter(sub_fail["_span"], sub_fail["_load"],
                           s=34 if is_h else 10, alpha=0.98 if is_h else 0.25,
                           marker="x")
        else:
            ax.scatter(sub_v["_span"], sub_v["_load"],
                       s=34 if is_h else 10, alpha=0.98 if is_h else 0.25,
                       edgecolors="none")
    ax.grid(True, linestyle=":", alpha=0.35)

    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)

def scatter_successful_span_vs_carbon_by_type(
    df_all: pd.DataFrame,
    fig_dir: Path,
    carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:
    """
    Scatter of max_span vs carbon for passing variants,
    color-coded by type.
    """

    # Ensure required columns exist
    if carbon_col not in df_all.columns:
        return {}

    span_col = "max_span" if "max_span" in df_all.columns else (
        "span" if "span" in df_all.columns else None
    )

    if span_col is None:
        return {}

    if "type" not in df_all.columns:
        return {}

    # Filter to passing if available
    if "pass_overall" in df_all.columns:
        df = df_all[df_all["pass_overall"] == True].copy()
        if df.empty:
            print("[INFO] No passing rows found — plotting all instead.")
            df = df_all.copy()
    else:
        df = df_all.copy()

    if df.empty:
        print("[INFO] No passing rows — plotting all candidates instead.")
        df = df_all.copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot one scatter per type
    for t, sub in df.groupby("type"):
        ax.scatter(
            sub[span_col],
            sub[carbon_col],
            alpha=0.7,
            label=str(t)
        )

    ax.set_xlabel("Maximum grid span (m)")
    ax.set_ylabel("Embodied Carbon (kgCO₂/m²)")
    ax.set_title("Successful variants: Maximum span vs Embodied Carbon by Type")

    ax.legend(title="Type", fontsize=8)

    out_path = fig_dir / "scatter_successful_span_vs_carbon_by_type.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"scatter_successful_span_vs_carbon_by_type": out_path}

def plot_span_vs_load_curves_by_family(
    df: pd.DataFrame,
    out_fp: Union[str, Path],
    *,
    group_by: str = "system_family",
    show_legend: bool = True,
    success_only: bool = True,
    title: str = "Span vs Load curves",
) -> Optional[str]:
    """Single plot: span vs total_load curves grouped by family/manufacturer/system_family."""
    if df is None or df.empty:
        return None

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    span_col = _get_span_col(df)
    if not span_col:
        _write_placeholder_figure(out_fp, "No span column found.")
        return str(out_fp)

    D = df.copy()
    total_col = _infer_total_load_col(D)
    if total_col is None:
        _write_placeholder_figure(out_fp, "No total load column found.")
        return str(out_fp)

    if group_by not in D.columns:
        group_by = "manufacturer" if "manufacturer" in D.columns else ("system_family" if "system_family" in D.columns else group_by)
        if group_by not in D.columns:
            _write_placeholder_figure(out_fp, f"Missing grouping column: {group_by}")
            return str(out_fp)

    D["_span"] = pd.to_numeric(D[span_col], errors="coerce")
    D["_load"] = pd.to_numeric(D[total_col], errors="coerce")
    D = D.dropna(subset=["_span", "_load"])

    if success_only:
        D = D[_ensure_success_mask(D).fillna(False)].copy()

    if D.empty:
        _write_placeholder_figure(out_fp, "No rows after filtering.")
        return str(out_fp)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = sorted(D[group_by].astype(str).unique().tolist())
    colmap = get_color_map_for(group_by, names)

    for name, g in D.groupby(group_by):
        g = g.sort_values("_span")
        ax.plot(g["_span"], g["_load"], marker="o", linewidth=1.2, alpha=0.75,
                label=str(name), color=colmap.get(str(name)))

    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Total load / capacity")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)

    if show_legend:
        ax.legend(fontsize=7, ncol=2, frameon=False)

    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)

def plot_lowest_per_group_aggregate(
    df: pd.DataFrame,
    group_col: str,
    out_dir: Path,
    *,
    color_by: Optional[str] = None,
    show_legend: bool = False,
) -> Dict[str, str]:
    """Pick the lowest-carbon variant per group and plot aggregate scatters.

    Saves:
      - lowest_per_<group>_span_vs_carbon_aggregate.png
      - lowest_per_<group>_span_vs_total_load_aggregate.png (if total load available)
    """
    out: Dict[str, str] = {}
    if df is None or df.empty:
        return out

    D = _ensure_carbon(df)
    span_col = "max_span" if "max_span" in D.columns else None
    if span_col is None or group_col not in D.columns:
        return out

    D = D.copy()
    D["_carb"] = pd.to_numeric(D["carbon_per_m2"], errors="coerce")
    D = D.dropna(subset=["_carb", span_col])
    if D.empty:
        return out

    idx = D.groupby(group_col)["_carb"].idxmin().dropna().astype(int).tolist()
    if not idx:
        return out
    lowest = D.loc[idx].copy()

    # Choose color grouping
    if color_by and color_by in lowest.columns:
        ccol = color_by
    elif "manufacturer" in lowest.columns:
        ccol = "manufacturer"
    else:
        ccol = group_col

    cats = lowest[ccol].astype(str).fillna("Unknown")
    colors, legend_handles = colors_for_series(cats, column_name=ccol)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # span vs carbon aggregated
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lowest[span_col], lowest["_carb"], s=34, alpha=0.9, c=colors, edgecolors="none")
    ax.set_xlabel("Maximum Grid Span (m)")
    ax.set_ylabel("Embodied Carbon (kgCO₂e / m²)")
    ax.set_title(f"Lowest-carbon variant per {group_col}: span vs carbon")
    ax.grid(True, linestyle=":", alpha=0.4)
    if show_legend:
        ax.legend(handles=legend_handles, title=ccol, loc="best", frameon=False)
    fp = out_dir / f"lowest_per_{group_col}_span_vs_carbon_aggregate.png"
    fig.savefig(fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    out["lowest_agg_carbon_" + group_col] = str(fp)

    # span vs total load aggregated (if total exists)
    total_col = _infer_total_load_col(lowest)
    if total_col and total_col in lowest.columns:
        vals = pd.to_numeric(lowest[total_col], errors="coerce")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(lowest[span_col], vals, s=34, alpha=0.9, c=colors, edgecolors="none")
        ax.set_xlabel("Maximum Grid Span (m)")
        ax.set_ylabel("Total load")
        ax.set_title(f"Lowest-carbon variant per {group_col}: span vs total load")
        ax.grid(True, linestyle=":", alpha=0.4)
        if show_legend:
            ax.legend(handles=legend_handles, title=ccol, loc="best", frameon=False)
        fp2 = out_dir / f"lowest_per_{group_col}_span_vs_total_load_aggregate.png"
        fig.savefig(fp2, bbox_inches="tight", dpi=200)
        plt.close(fig)
        out["lowest_agg_load_" + group_col] = str(fp2)

    return out

def plot_lowest_family_variants(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    """Backwards-compatible wrapper used by older report pipelines.

    Generates 'lowest per system_family' and 'lowest per manufacturer' aggregate scatters.
    Returns a dict of figure paths.
    """
    out: Dict[str, str] = {}
    if df is None or len(df) == 0:
        return out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have a usable total load column for total-load plots
    D = df.copy()
    if _infer_total_load_col(D) is None:
        # common case in this project: sdl + ll
        if "sdl" in D.columns and "ll" in D.columns:
            D["total_load"] = pd.to_numeric(D["sdl"], errors="coerce") + pd.to_numeric(D["ll"], errors="coerce")

    # Lowest per system_family
    if "system_family" in D.columns:
        out.update(plot_lowest_per_group_aggregate(
            D, "system_family", out_dir,
            color_by="manufacturer" if "manufacturer" in D.columns else None,
            show_legend=False,
        ))

    # Lowest per manufacturer
    if "manufacturer" in D.columns:
        out.update(plot_lowest_per_group_aggregate(
            D, "manufacturer", out_dir,
            color_by="manufacturer",
            show_legend=False,
        ))

    return out


# ------------------------
# Public API
# ------------------------
def write_edca_reports(
    candidates_input: Any,
    *,
    summary_df: pd.DataFrame,
    out_dir: Union[str, Path],
    floor_assignments: Optional[Dict[int, str]] = None,
    floor_area_lookup: Optional[Dict[int, float]] = None,
    verbose: bool = False,
    metric: str = "carbon_per_m2") -> ReportArtifacts:
    """
    Write report tables/figures to disk.

    Normal outputs:
      - ideal_per_category table (priority #1)
      - best_per_type table + bar chart (priority #4)
      - best_per_typology table + bar chart (priority #5)
      - code_checks_selected table (priority #11, selected variants)

    Verbose additionally writes:
      - next_best_type_per_category table (priority #2)
      - families_ranked_by_carbon table (priority #3)
      - scatter plots (success + all) span vs carbon (priorities #6/#7)
      - lowest-carbon family plots (priorities #8/#9)
      - lowest-carbon family per typology plot (priority #10)
      - code_checks_all table (priority #11, all)
    """

    df_all = candidates_input.copy()

    if "system_variant" not in df_all.columns:
        raise ValueError("system_variant column required.")

    # Initialize consistent color mappings once for the whole report run
    init_global_color_maps(df_all)

    # Load full catalogue points (success + failure) for span-vs-load curve plots, if available
    df_curve_points = load_system_curve_points(out_dir) if out_dir is not None else None
    if df_curve_points is None or df_curve_points.empty:
        df_curve_points = df_all

    # -------------------------------------------------
    # Handle summary_df being either dict or DataFrame
    # -------------------------------------------------

    if isinstance(summary_df, dict):
        # Accept a few common keys used elsewhere in the pipeline.
        for k in ("summary_ranked_all", "ranked_all", "ranked_union", "ranked"):
            v = summary_df.get(k)
            if isinstance(v, pd.DataFrame):
                summary_ranked = v
                break
        else:
            raise ValueError(
                "summary_df is a dict but does not contain a ranked DataFrame under "
                "one of: summary_ranked_all, ranked_all, ranked_union, ranked."
             )

    elif isinstance(summary_df, pd.DataFrame):

        summary_ranked = summary_df

    else:
        raise TypeError(
            f"summary_df must be DataFrame or dict, "
            f"got {type(summary_df)}"
        )
    
    # --- Determine ID column automatically ---

    possible_id_cols = [
        "system_variant",
        "variant",
        "variant_id",
        "system_variant_id",
        "id"
    ]

    summary_id_col = next(
        (c for c in possible_id_cols if c in summary_ranked.columns),
        None
    )

    # Variants that passed the main filtering (used to grey-out curves that were filtered away)
    passing_variants_all: set[str] = set(summary_ranked[summary_id_col].astype(str).dropna()) if summary_id_col else set()


    catalogue_id_col = next(
        (c for c in possible_id_cols if c in candidates_input.columns),
        None
    )

    if summary_id_col is None or catalogue_id_col is None:
        raise ValueError(
            f"Could not find matching variant ID column.\n"
            f"Summary columns: {summary_ranked.columns.tolist()}\n"
            f"Catalogue columns: {candidates_input.columns.tolist()}"
        )

    df_all = candidates_input.copy()

    # -------------------------------------------------
    # Determine pass_overall correctly
    # Prefer an existing pass_overall on candidates_input;
    # otherwise infer a passing set from summary_ranked using pass_overall / success mask.
    # -------------------------------------------------
    if "pass_overall" in df_all.columns:
        df_all["pass_overall"] = _coerce_bool_series(df_all["pass_overall"])
    else:
        # If summary_ranked has a pass column, use only the passing subset.
        if "pass_overall" in summary_ranked.columns:
            mask = _coerce_bool_series(summary_ranked["pass_overall"])
        else:
            mask = _success_mask(summary_ranked)

        # If mask is empty/unusable, fall back to treating summary_ranked as "passing" (legacy behavior)
        if mask is None or len(mask) == 0:
            mask = pd.Series([True] * len(summary_ranked), index=summary_ranked.index)

        passing_variants = set(
            summary_ranked.loc[mask, summary_id_col]
            .astype(str).str.strip()
            .unique()
            .tolist()
        )

        df_all["pass_overall"] = (
            df_all[catalogue_id_col].astype(str).str.strip().isin(passing_variants)
        )

    # Log the evaluated/post-systems pass rate (what df_all currently represents)
    try:
        _id = catalogue_id_col if catalogue_id_col in df_all.columns else "system_variant"
        n_eval = int(df_all[_id].astype(str).str.strip().nunique())
        n_pass = int(df_all.loc[_coerce_bool_series(df_all["pass_overall"]), _id].astype(str).str.strip().nunique())
        logger.info("[reporting] Pass rate (evaluated/post-systems): %d / %d", n_pass, n_eval)
    except Exception:
        logger.debug("[reporting] Could not compute evaluated pass rate for logging", exc_info=True)

    if "carbon_per_m2" not in df_all.columns and "carbon_total_kgCO2" in df_all.columns:
        try:
            # copy and coerce to numeric
            df_all = df_all.copy()
            df_all["carbon_per_m2"] = pd.to_numeric(df_all["carbon_total_kgCO2"], errors="coerce")
            logger.info("[reporting] Created 'carbon_per_m2' from 'carbon_total_kgCO2' (assumed per m²).")
        except Exception:
            logger.exception("[reporting] Failed to create carbon_per_m2 from carbon_total_kgCO2; tables may be empty.")

    df_all = standardize_schema(df_all)
    df_all = _join_system_families_metadata(df_all)

    # IMPORTANT:
    # run_edca.py now owns outputs/summary_ranked_all.csv (CONDENSED wide merge).
    # Do NOT overwrite it here. If you want an enriched long-form file for reporting,
    # write it to summary_ranked_all_long.csv instead.
    try:
        write_enriched_summary_ranked_all(df_all, out_dir, filename="summary_ranked_all_long.csv")
    except Exception:
        logger.exception("[reporting] failed to write enriched summary_ranked_all_long.csv")

    # --- Ensure manufacturer and total_load are present for plots ---
    # manufacturer will be filled by _join_system_families_metadata above if available.
    if "manufacturer" not in df_all.columns:
        # fallback: set manufacturer = system_family (temporary)
        df_all["manufacturer"] = df_all.get("system_family")
        logger.info("[reporting] manufacturer column missing; using system_family as manufacturer fallback")

    # total_load: prefer existing column, otherwise synthesize from sdl_total or sdl + ll
    if "total_load" not in df_all.columns:
        # try a few canonical names
        if "total_capacity" in df_all.columns:
            df_all["total_load"] = pd.to_numeric(df_all["total_capacity"], errors="coerce")
        elif "sdl_total" in df_all.columns and "ll" in df_all.columns:
            df_all["total_load"] = pd.to_numeric(df_all["sdl_total"], errors="coerce") + pd.to_numeric(df_all["ll"], errors="coerce")
        elif "sdl" in df_all.columns and "ll" in df_all.columns:
            df_all["total_load"] = pd.to_numeric(df_all["sdl"], errors="coerce") + pd.to_numeric(df_all["ll"], errors="coerce")
        else:
            # If none available, create NaN column for uniformity
            df_all["total_load"] = pd.Series([pd.NA] * len(df_all))
        logger.info("[reporting] ensured total_load column (synthesized if required)")

    dirs = _ensure_report_dirs(out_dir)

    tables: Dict[str, pd.DataFrame] = {}

    # -------------------------
    # Pass-rate funnel table (both denominators)
    #   - pass_rate_of_generated  = passing / total system_variants
    #   - pass_rate_of_evaluated  = passing / variants that made it past systems.py (i.e., candidates_input)
    # -------------------------
    try:
        # Use the ID column that actually exists in df_all after standardization
        possible_id_cols = ["system_variant", "variant", "variant_id", "system_variant_id", "id"]
        id_col = next((c for c in possible_id_cols if c in df_all.columns), catalogue_id_col)

        # evaluated/post-systems counts come from df_all (what write_edca_reports receives)
        n_after_systems = int(df_all[id_col].astype(str).str.strip().nunique())
        n_passing = int(
            df_all.loc[_coerce_bool_series(df_all["pass_overall"]), id_col]
            .astype(str).str.strip().nunique()
        )

        # Try to locate a system_variants catalogue near out_dir so we can compute "of generated"
        n_generated = None
        system_variants_source = None

        search_roots = [Path(out_dir), Path(out_dir).parent]
        candidates = []
        for root in search_roots:
            if root.exists():
                candidates += list(root.rglob("system_variants.parquet"))
                candidates += list(root.rglob("system_variants.csv"))
                candidates += list(root.rglob("*system_variants*.parquet"))
                candidates += list(root.rglob("*system_variants*.csv"))

        # Prefer exact filenames + short paths
        candidates = sorted(
            set(candidates),
            key=lambda p: (0 if p.name in ("system_variants.parquet", "system_variants.csv") else 1, len(str(p)))
        )

        for p in candidates:
            try:
                if p.suffix.lower() == ".parquet":
                    sv = pd.read_parquet(p)
                else:
                    sv = pd.read_csv(p)

                sv_id = next((c for c in possible_id_cols if c in sv.columns), None)
                if sv_id:
                    n_generated = int(sv[sv_id].astype(str).str.strip().nunique())
                else:
                    n_generated = int(len(sv))

                system_variants_source = str(p)
                break
            except Exception:
                continue

        # If we can't find system_variants on disk, still write the table (but mark the "generated" denom as fallback)
        if not n_generated or n_generated <= 0:
            n_generated = n_after_systems
            system_variants_source = "(fallback: candidates_input)"
            logger.warning(
                "[reporting] Could not locate system_variants file near out_dir; "
                "pass_rate_of_generated will equal pass_rate_of_evaluated."
            )

        case_col = "floor_load_category" if "floor_load_category" in df_all.columns else ("case" if "case" in df_all.columns else None)

        rows = [{
            "group": "ALL",
            "n_generated_system_variants": n_generated,
            "n_after_systems_filter": n_after_systems,
            "n_passing": n_passing,
            "pass_rate_of_generated": (n_passing / n_generated) if n_generated else np.nan,
            "pass_rate_of_evaluated": (n_passing / n_after_systems) if n_after_systems else np.nan,
            "system_variants_source": system_variants_source,
        }]

        if case_col:
            for gname, gdf in df_all.groupby(case_col, dropna=False):
                n_after = int(gdf[id_col].astype(str).str.strip().nunique())
                n_pass = int(
                    gdf.loc[_coerce_bool_series(gdf["pass_overall"]), id_col]
                    .astype(str).str.strip().nunique()
                )
                rows.append({
                    "group": str(gname),
                    "n_generated_system_variants": n_generated,
                    "n_after_systems_filter": n_after,
                    "n_passing": n_pass,
                    "pass_rate_of_generated": (n_pass / n_generated) if n_generated else np.nan,
                    "pass_rate_of_evaluated": (n_pass / n_after) if n_after else np.nan,
                    "system_variants_source": system_variants_source,
                })

        tables["pass_rate_funnel"] = pd.DataFrame(rows)

    except Exception:
        logger.exception("[reporting] Failed to build pass_rate_funnel table")

    if floor_assignments:
        tables["ideal_per_category"] = table_ideal_per_floor_category(
            df_all, floor_assignments, floor_area_lookup or {}, metric=metric
        )
        if verbose:
            tables["next_best_type_per_category"] = table_next_best_type_per_category(
                df_all, floor_assignments, metric=metric
            )
    else:
        tables["ideal_per_category"] = pd.DataFrame()

    if verbose:
        tables["families_ranked_by_carbon"] = table_families_ranked_by_carbon(df_all, metric=metric)

    tables["best_per_type"] = table_best_variant_per_group(df_all, "type", metric=metric)
    tables["best_per_typology"] = table_best_variant_per_group(df_all, "typology", metric=metric)

    # selected variants for normal output
    selected_variants: set[str] = set()
    if not tables["ideal_per_category"].empty and "best_variant" in tables["ideal_per_category"].columns:
        selected_variants |= set(tables["ideal_per_category"]["best_variant"].dropna().astype(str).tolist())
    if not tables["best_per_type"].empty and "system_variant" in tables["best_per_type"].columns:
        selected_variants |= set(tables["best_per_type"]["system_variant"].dropna().astype(str).tolist())
    if not tables["best_per_typology"].empty and "system_variant" in tables["best_per_typology"].columns:
        selected_variants |= set(tables["best_per_typology"]["system_variant"].dropna().astype(str).tolist())

    # df_all is your ranked/enriched summary dataframe
    sort_col = "carbon_total_kgCO2" if "carbon_total_kgCO2" in df_all.columns else metric
    if sort_col not in df_all.columns:
        df_all = _ensure_carbon(df_all)
        sort_col = "carbon_per_m2"

    group_col = None
    if "floor_load_category" in df_all.columns:
        group_col = "floor_load_category"
    elif "case" in df_all.columns:
        group_col = "case"

    if group_col is None:
        df_winners = df_all.sort_values(sort_col).head(1).copy()
    else:
        df_winners = (
            df_all.sort_values(sort_col)
                .groupby([group_col], as_index=False, dropna=False)
                .head(1)
                .copy()
        )

    # --------------------------
    # Optional: run code checks for winners and merge safely
    # --------------------------
    try:
        # Resolve the materials_per_floor_expanded.csv path robustly:
        # - in your logs it is written to: outputs/edca_run/materials_per_floor_expanded.csv
        # - but out_dir here is usually: outputs/edca_run/reporting
        materials_csv = Path(out_dir) / "materials_per_floor_expanded.csv"
        if not materials_csv.exists():
            materials_csv = Path(out_dir).parent / "materials_per_floor_expanded.csv"
        if not materials_csv.exists():
            materials_csv = None  # allow downstream fallback

        df_checks = run_code_checks_if_requested(
            candidates_df=df_winners,
            out_dir=Path(out_dir),   # or whatever your reporting output dir variable is
            run_flag=True, # boolean
            ) 

        if df_checks is not None and not df_checks.empty:
            # If df_all already contains old pandas merge suffix columns, clean them up first.
            # This prevents errors like "duplicate columns {'codecheck_family_x', ...} is not allowed"
            cols = list(df_all.columns)
            for c in cols:
                if c.endswith("_x") or c.endswith("_y"):
                    base = c[:-2]
                    # If the base column exists too, the suffixed one is redundant -> drop it
                    if base in df_all.columns:
                        df_all = df_all.drop(columns=[c], errors="ignore")

            # Now drop any direct overlaps with incoming df_checks (except the key)
            overlap = [c for c in df_checks.columns if c != "system_variant" and c in df_all.columns]
            if overlap:
                df_all = df_all.drop(columns=overlap, errors="ignore")

            df_all = df_all.merge(df_checks, on="system_variant", how="left")

            # --- END DROP-IN REPLACEMENT ---


            logger.info("[reporting] merged code check results into df_all for %d winners", len(df_checks))

    except Exception:
        logger.exception("[reporting] failed running code checks for winners")

    # code checks
    code_all = table_code_checks(df_all)
    if verbose:
        tables["code_checks_all"] = code_all

    if not code_all.empty and "system_variant" in code_all.columns and selected_variants:
        tables["code_checks_selected"] = code_all[code_all["system_variant"].astype(str).isin(selected_variants)].copy()
    else:
        tables["code_checks_selected"] = pd.DataFrame()
    
    # --- NEW: carbon per material_id table (from carbon_by_material_id_json) ---
    try:
        mats = None
        mats_path = Path("inputs/canonical/materials.parquet")
        if mats_path.exists():
            mats = pd.read_parquet(mats_path)
        tables["carbon_per_material_id"] = build_carbon_per_material_table(df_all, mats)
    except Exception as e:
        logger.warning("[reporting] failed to build carbon_per_material_id table: %s", e)
        tables["carbon_per_material_id"] = pd.DataFrame()

    # save tables
    table_paths: Dict[str, str] = {}
    for name, tdf in tables.items():
        if tdf is None:
            continue
        fp = dirs["tables"] / f"{name}.csv"
        try:
            tdf.to_csv(fp, index=False)
            table_paths[name] = str(fp)
        except Exception as e:
            logger.warning("[reporting] failed to save table %s: %s", name, e)
    
    print("Reporting columns:", df_all.columns.tolist())
    
    print(df_all["pass_overall"].value_counts(dropna=False))
    
    # figures
    figure_paths: Dict[str, str] = {}
    p = plot_bar_best_by_group(
        tables["best_per_type"], "type", "carbon_per_m2",
        dirs["figures"] / "bar_best_per_type.png",
        title="Best (lowest-carbon) variant per type",
        color_by="type",
        show_legend=False,
    )
    if p:
        figure_paths["bar_best_per_type"] = p

    p = plot_bar_best_by_group(
        tables["best_per_typology"], "typology", "carbon_per_m2",
        dirs["figures"] / "bar_best_per_typology.png",
        title="Best (lowest-carbon) variant per typology",
    )
    if p:
        figure_paths["bar_best_per_typology"] = p

    p = plot_span_vs_carbon_colored(
        df_all,
        dirs["figures"] / "scatter_successful_span_vs_carbon_by_type.png",
        success_only=True,
        color_by="type",
        title="Successful system_variants: span vs embodied carbon (colored by type)",
    )
    if p:
        figure_paths["scatter_successful_span_vs_carbon_by_type"] = p

    p = plot_span_vs_carbon_colored(
        df_all,
        dirs["figures"] / "scatter_all_span_vs_carbon_by_type.png",
        success_only=False,
        color_by="type",
        title="All system_variants: span vs embodied carbon (colored by type)",
    )
    if p:
        figure_paths["scatter_all_span_vs_carbon_by_type"] = p

    figure_paths.update(plot_lowest_family_variants(df_all, dirs["figures"]))

    p = plot_span_vs_carbon_by_type(df_all, dirs["figures"] / "span_vs_carbon_by_type.png", label_prefix="")
    if p:
        figure_paths["span_vs_carbon_by_type"] = p

    # all variants
    p = plot_span_vs_carbon_by_type(df_all, dirs["figures"] / "span_vs_carbon_all.png", label_prefix="")
    if p:
        figure_paths["span_vs_carbon_all"] = p
    # successful only
    p = plot_span_vs_carbon_by_type(df_all, dirs["figures"] / "span_vs_carbon_successful.png", success_only=True)
    if p:
        figure_paths["span_vs_carbon_successful"] = p

    out = plot_span_vs_total_and_carbon_for_lowest(df_all, "system_family", dirs["figures"])
    figure_paths.update(out)
    if "manufacturer" in df_all.columns:
        out = plot_span_vs_total_and_carbon_for_lowest(df_all, "manufacturer", dirs["figures"])
        figure_paths.update(out)

    p = plot_lowest_family_per_typology(df_all, dirs["figures"])
    if p:
        figure_paths["lowest_family_per_typology"] = p

    p = plot_lowest_family_per_type(df_all, dirs["figures"])
    if p:
        figure_paths["lowest_family_per_type"] = p

    p = plot_span_vs_carbon_global(df_all, dirs["figures"] / "span_vs_carbon.png")
    if p:
        figure_paths["span_vs_carbon"] = p

    p = plot_span_vs_total_load_global(df_all, dirs["figures"] / "span_vs_total_load.png")
    if p: figure_paths["span_vs_total_load"] = p

    figure_paths.update(plot_lowest_per_group_aggregate(df_all, "system_family", dirs["figures"], color_by="type" if "type" in df_all.columns else ("typology" if "typology" in df_all.columns else ("manufacturer" if "manufacturer" in df_all.columns else "system_family")), show_legend=True))
    if "manufacturer" in df_all.columns:
        figure_paths.update(plot_lowest_per_group_aggregate(df_all, "manufacturer", dirs["figures"], color_by="manufacturer", show_legend=True))
    # Diagnostics plot (kept as-is)
    p = plot_span_vs_load_curves_by_family(
        df_curve_points,
        dirs["figures"] / "span_vs_load_curves_by_family.png",
        group_by="system_family",
        show_legend=False,
        success_only=False,
    )
    if p:
        figure_paths["span_vs_load_curves_by_family"] = p

    # Additional requested copies:
    p = plot_span_vs_load_curves_by_family(
        df_curve_points,
        dirs["figures"] / "span_vs_load_curves_by_family_no_legend.png",
        group_by="system_family",
        show_legend=False,
        success_only=False,
        title="Span vs Load curves by family (no legend)",
    )
    if p:
        figure_paths["span_vs_load_curves_by_family_no_legend"] = p

    '''
    p = plot_span_vs_load_curves_by_family(
        df_curve_points,
        dirs["figures"] / "span_vs_load_curves_by_manufacturer_no_legend.png",
        group_by="manufacturer",
        show_legend=False,
        success_only=False,
        title="Span vs Load curves by manufacturer (no legend)",
    )
    if p:
        figure_paths["span_vs_load_curves_by_manufacturer_no_legend"] = p
    '''

    figure_paths.update(plot_span_vs_carbon_pareto(df_all, dirs["figures"]))
    figure_paths.update(plot_depth_vs_carbon(df_all, dirs["figures"]))
    figure_paths.update(plot_carbon_distribution_by_type(df_all, dirs["figures"]))
    figure_paths.update(plot_feasibility_heatmap(df_all, dirs["figures"], out_dir=out_dir))
    figure_paths.update(plot_failure_breakdown(df_all, dirs["figures"]))

    # load materials
    materials_df = pd.read_parquet("inputs/canonical/materials.parquet")
    figure_paths.update(plot_carbon_breakdown(df_all, materials_df, dirs["figures"]))

        # --- NEW: stacked carbon composition + share plots (lowest-carbon per type/typology) ---
    try:
        p = plot_carbon_composition_stacked(
            df_all,
            dirs["figures"] / "carbon_composition_stacked.png",
            carbon_total_col="carbon_per_m2",
            success_only=True,
        )
        if p:
            figure_paths["carbon_composition_stacked"] = str(p)

        p = plot_carbon_composition_share(
            df_all,
            dirs["figures"] / "carbon_composition_share.png",
            carbon_total_col="carbon_per_m2",
            success_only=True,
        )
        if p:
            figure_paths["carbon_composition_share"] = str(p)

    except Exception as e:
        logger.warning("[reporting] failed composition plots: %s", e)

    # ------------------------------------------------------------
    # Duplicate ALL figures with a non-negative carbon constraint
    # ------------------------------------------------------------
    try:
        df_nn = _ensure_carbon(df_all)
        df_nn = df_nn.copy()
        df_nn["carbon_per_m2"] = pd.to_numeric(df_nn["carbon_per_m2"], errors="coerce")
        df_nn = df_nn[df_nn["carbon_per_m2"] > 0].copy()

        # If nothing remains, still emit placeholder figures so the directory isn't empty
        if df_nn.empty:
            nn_root = Path(out_dir) / "non-negative-carbon"
            nn_dirs = _ensure_report_dirs(nn_root)
            msg = "No rows with carbon_per_m2 > 0 after filtering"
            for fn in [
                "scatter_successful_span_vs_carbon_by_type.png",
                "scatter_all_span_vs_carbon_by_type.png",
                "lowest_per_system_family_span_vs_carbon_aggregate.png",
                "lowest_per_system_family_span_vs_total_load_aggregate.png",
            ]:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis("off")
                ax.text(0.02, 0.6, msg, fontsize=12)
                fig.savefig(nn_dirs["figures"] / fn, bbox_inches="tight", dpi=200)
                plt.close(fig)
            # continue without raising

        nn_root = Path(out_dir) / "non-negative-carbon"
        nn_dirs = _ensure_report_dirs(nn_root)

        # Recompute best tables under the constraint (so bar charts reflect positive-only winners)
        best_type_nn = table_best_variant_per_group(df_nn, "type", metric=metric)
        best_typology_nn = table_best_variant_per_group(df_nn, "typology", metric=metric)

        # Bar charts (positive-only)
        plot_bar_best_by_group(
            best_type_nn, "type", "carbon_per_m2",
            nn_dirs["figures"] / "bar_best_per_type.png",
            title="Best (lowest-carbon) variant per type (positive only)",
            orientation="h",
            add_value_labels=True,
            color_by="type" if "type" in best_type_nn.columns else None,
            show_legend=False,
        )
        plot_bar_best_by_group(
            best_typology_nn, "typology", "carbon_per_m2",
            nn_dirs["figures"] / "bar_best_per_typology.png",
            title="Best (lowest-carbon) variant per typology (positive only)",
            orientation="v",
            add_value_labels=True,
            color_by="type" if "type" in best_typology_nn.columns else None,
            show_legend=False,
        )

        # Scatter: successful + all + highlights
        plot_span_vs_carbon_colored(
            df_nn,
            nn_dirs["figures"] / "scatter_successful_span_vs_carbon_by_type.png",
            success_only=False,
            color_by="type",
            positive_only=True,
            title="Successful system_variants (positive only): span vs embodied carbon (colored by type)",
        )
        plot_span_vs_carbon_colored(
            df_nn,
            nn_dirs["figures"] / "scatter_all_span_vs_carbon_by_type.png",
            success_only=False,
            color_by="type",
            positive_only=True,
            title="All system_variants (positive only): span vs embodied carbon (colored by type)",
        )
        if selected_variants:
            plot_span_vs_carbon_colored(
                df_nn,
                nn_dirs["figures"] / "scatter_all_span_vs_carbon_by_type_selected_highlight.png",
                success_only=False,
                color_by="type",
                positive_only=True,
                highlight_variants=selected_variants,
                title="All system_variants (positive only): selected highlighted (others grey)",
            )

        # Aggregate "lowest per group" plots (positive-only)
        plot_lowest_per_group_aggregate(
            df_nn, "manufacturer", nn_dirs["figures"],
            color_by="manufacturer", show_legend=True,
        )
        plot_lowest_per_group_aggregate(
            df_nn, "system_family", nn_dirs["figures"],
            color_by="manufacturer" if "manufacturer" in df_nn.columns else "system_family",
            show_legend=False,
        )

        if "type" in df_nn.columns:
            _filtered_types = ["solid_plank", "hollowcore", "double_tee", "composite_deck"]
            _present = [t for t in _filtered_types if (df_nn["type"].astype(str) == t).any()]
            if _present:
                df_f = df_nn[df_nn["type"].astype(str).isin(_present)].copy()
                plot_span_vs_carbon_colored(
                    df_f,
                    nn_dirs["figures"] / "scatter_filtered_span_vs_carbon_by_type_selected_highlight.png",
                    success_only=False,
                    color_by="type",
                    positive_only=True,
                    highlight_variants=selected_variants,
                    title="Filtered types (positive only): selected highlighted (others grey)",
                )

        # Lowest-family / aggregate plots
        plot_lowest_family_variants(df_nn, nn_dirs["figures"])
        plot_span_vs_carbon_by_type(df_nn, nn_dirs["figures"] / "span_vs_carbon_by_type.png", label_prefix="")
        plot_span_vs_carbon_by_type(df_nn, nn_dirs["figures"] / "span_vs_carbon_all.png", label_prefix="")
        plot_span_vs_carbon_by_type(df_nn, nn_dirs["figures"] / "span_vs_carbon_successful.png", success_only=True)

        plot_span_vs_total_and_carbon_for_lowest(df_nn, "system_family", nn_dirs["figures"])
        if "manufacturer" in df_nn.columns:
            plot_span_vs_total_and_carbon_for_lowest(df_nn, "manufacturer", nn_dirs["figures"])

        plot_lowest_family_per_typology(df_nn, nn_dirs["figures"])
        plot_span_vs_carbon_global(df_nn, nn_dirs["figures"] / "span_vs_carbon.png")
        plot_span_vs_total_load_global(df_nn, nn_dirs["figures"] / "span_vs_total_load.png")
        # Carbon-span frontier (positive-only directory)
        plot_span_vs_carbon_pareto(df_nn, nn_dirs["figures"])


        plot_lowest_per_group_aggregate(df_nn, "system_family", nn_dirs["figures"], color_by="type" if "type" in df_nn.columns else ("typology" if "typology" in df_nn.columns else "system_family"), show_legend=True)
        if "manufacturer" in df_nn.columns:
            plot_lowest_per_group_aggregate(df_nn, "manufacturer", nn_dirs["figures"])

        # Span vs load curves (same set, computed on positive-only df for consistency)
        plot_span_vs_load_curves_by_family_grid(
            df_curve_points,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_grid.png",
            max_families=12,
            ncols=3,
            success_only=False,
            passing_variants=passing_variants_all,
            highlight_variants=selected_variants if selected_variants else None,
        )
        plot_span_vs_load_curves_by_family_highlight(
            df_curve_points,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_highlight.png",
            success_only=False,
            passing_variants=passing_variants_all,
            highlight_variants=selected_variants if selected_variants else None,
        )
        plot_span_vs_load_curves_by_family(
            df_curve_points,
            nn_dirs["figures"] / "span_vs_load_curves_by_family.png",
            group_by="system_family",
            show_legend=False,
            success_only=False,
        )
        plot_span_vs_load_curves_by_family(
            df_curve_points,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_no_legend.png",
            group_by="system_family",
            show_legend=False,
            success_only=False,
            title="Span vs Load curves by family (no legend)",
        )
        
        '''
        plot_span_vs_load_curves_by_family(
            df_curve_points,
            nn_dirs["figures"] / "span_vs_load_curves_by_manufacturer_no_legend.png",
            group_by="manufacturer",
            show_legend=False,
            success_only=False,
            title="Span vs Load curves by manufacturer (no legend)",
        )
        '''
        # If your non-negative section re-runs plotting rather than copying, rerun these too:
        try:
            p = plot_carbon_composition_stacked(
                df_nn,
                nn_dirs["figures"] / "carbon_composition_stacked.png",
                carbon_total_col="carbon_per_m2",
                success_only=True,
            )
            p = plot_carbon_composition_share(
                df_nn,
                nn_dirs["figures"] / "carbon_composition_share.png",
                carbon_total_col="carbon_per_m2",
                success_only=True,
            )
        except Exception as e:
            logger.warning("[reporting] failed non-negative composition plots: %s", e)


        # Mirror the remaining diagnostics/summary figures in the non-negative folder
        try:
            plot_lowest_family_per_type(df_nn, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting lowest_family_per_type")

        try:
            plot_span_vs_carbon_pareto(df_nn, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting span_vs_carbon_pareto")

        try:
            plot_depth_vs_carbon(df_nn, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting depth_vs_carbon")

        try:
            plot_carbon_distribution_by_type(df_nn, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting carbon_distribution_by_type")

        try:
            plot_feasibility_heatmap(df_nn, nn_dirs["figures"], out_dir=out_dir)
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting feasibility_heatmap")

        try:
            plot_failure_breakdown(df_nn, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting failure_breakdown")

        try:
            materials_df = pd.read_parquet("inputs/canonical/materials.parquet")
            plot_carbon_breakdown(df_nn, materials_df, nn_dirs["figures"])
        except Exception:
            logger.exception("[reporting] (nn) Failed plotting carbon_breakdown")

        figure_paths["non_negative_carbon_dir"] = str(nn_root)
        logger.info("[reporting] Wrote non-negative-carbon figure duplicates to %s", nn_root)

    except Exception:
        logger.exception("[reporting] failed to write non-negative-carbon duplicate figures")

    return ReportArtifacts(tables=tables, table_paths=table_paths, figure_paths=figure_paths)

# Backwards compatible wrapper (keep your old runner working)
def generate_report(
    candidates: Any,
    floor_assignments: Optional[Dict[int, str]] = None,
    floor_area_lookup: Optional[Dict[int, float]] = None,
    metric: str = "carbon_per_m2",
    verbose: bool = False,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Back-compat API: returns {'tables': {name: DataFrame}, 'figures': {name: path}}.
    If save_dir is None, saves to './reporting' under current working directory.
    """
    out_dir = Path(save_dir) if save_dir else Path("reporting")
    art = write_edca_reports(
        candidates_input=candidates,
        out_dir=out_dir,
        floor_assignments=floor_assignments,
        floor_area_lookup=floor_area_lookup,
        verbose=verbose,
        metric=metric,
    )
    if show:
        logger.warning("[reporting] show=True requested, but figures are saved and closed; open PNGs from %s.", out_dir)
    return {"tables": art.tables, "figures": art.figure_paths, "table_paths": art.table_paths}