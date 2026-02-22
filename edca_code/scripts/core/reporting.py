# edca_code/scripts/core/reporting.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


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
    Load the canonical 'summary_ranked_all.csv' from the run output directory.
    Tries root first, then falls back to searching case folders.
    """
    out_dir = Path(out_dir)

    # Most common location (root)
    p0 = out_dir / "summary_ranked_all.csv"
    if p0.exists():
        return pd.read_csv(p0)

    # Fallback: search in immediate subfolders (systems_* etc.)
    for p in out_dir.rglob("summary_ranked_all.csv"):
        try:
            return pd.read_csv(p)
        except Exception:
            continue

    raise FileNotFoundError(f"Could not find summary_ranked_all.csv under {out_dir}")

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

def write_edca_reports_from_summary(
    *,
    out_dir: str | Path,
    floor_assignments: dict[int, str] | None = None,
    floor_area_lookup: dict[int, float] | None = None,
    verbose: bool = False,
    metric: str = "carbon_per_m2") -> ReportArtifacts:
    """
    Same as write_edca_reports, but uses summary_ranked_all.csv files found under `out_dir`
    as the canonical input. This function will:

      1. Search recursively under `out_dir` for all files named "summary_ranked_all.csv".
      2. Create a union (outer) table -> total_summary_all.csv (concat + drop exact duplicates).
      3. Create an intersection (inner) table -> compatible_summary_all.csv
         (keeps system_variant values present in every found file).
      4. Write both tables under: <out_dir>/reporting/tables/
      5. Use the union table as the canonical `candidates_input` to call write_edca_reports(...)

    Returns the same ReportArtifacts/datatype as write_edca_reports.
    """
    log = logging.getLogger(__name__)
    out_dir = Path(out_dir)
    out_dir_resolved = out_dir.resolve()
    logging.debug("[reporting] combining summary_ranked_all files under %s", out_dir_resolved)

    # find all summary_ranked_all.csv under the given out_dir
    found = list(out_dir.rglob("summary_ranked_all.csv"))

    if not found:
        # existing behavior: load_summary_ranked_all would raise; keep similar behavior but clearer message
        raise FileNotFoundError(f"No summary_ranked_all.csv files found under {out_dir_resolved}")

    # read and tag each found file
    dfs = []
    for p in found:
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as e:
            log.warning("[reporting] failed to read %s: %s", p, e)
            continue
        # normalize column names (strip whitespace)
        d.columns = [c.strip() for c in d.columns]
        d["_source_path"] = str(p)
        d["_source_dir"] = str(p.parent.name)
        dfs.append(d)

    if not dfs:
        raise ValueError(f"Found summary files at {len(found)}, but none were readable.")

    # Prepare output folders: <out_dir>/reporting/tables and figures
    reporting_root = out_dir / "reporting"
    tables_dir = reporting_root / "tables"
    figures_dir = reporting_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) total (outer) union: concat and drop exact duplicates
        # ---------------------------
    # Build union and intersection
    # ---------------------------

    # create raw union first (keep provenance columns)
    total_df_raw = pd.concat(dfs, ignore_index=True, sort=False)
    total_df_raw = total_df_raw.drop_duplicates()

    log.info("[reporting] Combined raw union rows (before grouping): %d", len(total_df_raw))

        # --- JOIN canonical typology/type from inputs/canonical/system_families ---
    try:
        sf_path_parquet = Path("inputs") / "canonical" / "system_families.parquet"
        sf_path_csv = Path("inputs") / "canonical" / "system_families.csv"
        system_families_df = None

        if sf_path_parquet.exists():
            try:
                system_families_df = pd.read_parquet(sf_path_parquet)
                log.info("[reporting] Loaded canonical system_families from %s", sf_path_parquet)
            except Exception as e_parq:
                log.warning("[reporting] Failed to read parquet %s: %s (will try CSV)", sf_path_parquet, e_parq)

        if system_families_df is None and sf_path_csv.exists():
            try:
                system_families_df = pd.read_csv(sf_path_csv)
                log.info("[reporting] Loaded canonical system_families from %s", sf_path_csv)
            except Exception as e_csv:
                log.warning("[reporting] Failed to read CSV %s: %s", sf_path_csv, e_csv)

        if system_families_df is not None:
            # normalize columns lower-case and strip whitespace
            system_families_df.columns = [c.strip() for c in system_families_df.columns]
            # For safety, canonical file may call the category column something else; check common names
            # We expect canonical to have a column called 'system_family' and one called 'category' (typology) and/or 'type'
            cols_lower = {c.lower(): c for c in system_families_df.columns}
            # find canonical columns
            sysfam_col = None
            if "system_family" in cols_lower:
                sysfam_col = cols_lower["system_family"]
            else:
                # try alternatives
                for alt in ("family", "systemfamily", "system-family"):
                    if alt in cols_lower:
                        sysfam_col = cols_lower[alt]
                        break

            cat_col = None
            if "category" in cols_lower:
                cat_col = cols_lower["category"]
            elif "typology" in cols_lower:
                cat_col = cols_lower["typology"]

            type_col = None
            if "type" in cols_lower:
                type_col = cols_lower["type"]

            if sysfam_col is None:
                log.warning("[reporting] canonical system_families found but no 'system_family' column; skipping join.")
            else:
                # Prepare for join: standardize join key names to 'system_family'
                sfdf = system_families_df.copy()
                sfdf = sfdf.rename(columns={sysfam_col: "system_family"})
                # pick typology and type columns if present and rename them
                if cat_col:
                    sfdf = sfdf.rename(columns={cat_col: "typology"})
                if type_col:
                    sfdf = sfdf.rename(columns={type_col: "type"})

                # coerce system_family in both frames to str and strip for robust matches
                total_df_raw["system_family"] = total_df_raw.get("system_family", "").astype(str).str.strip()
                sfdf["system_family"] = sfdf["system_family"].astype(str).str.strip()

                # perform left join; do not overwrite existing typology/type if present in total_df_raw
                merged = total_df_raw.merge(sfdf[["system_family"] + [c for c in ("typology", "type") if c in sfdf.columns]],
                                            on="system_family", how="left", suffixes=("", "_canon"))

                # if main DF is missing typology/type, fill from canonical; otherwise keep existing
                if "typology" not in merged.columns or merged["typology"].isnull().all():
                    if "typology_canon" in merged.columns:
                        merged["typology"] = merged["typology_canon"]
                else:
                    # fill only missing values
                    if "typology_canon" in merged.columns:
                        merged["typology"] = merged["typology"].fillna(merged.get("typology_canon"))

                if "type" not in merged.columns or merged["type"].isnull().all():
                    if "type_canon" in merged.columns:
                        merged["type"] = merged["type_canon"]
                else:
                    if "type_canon" in merged.columns:
                        merged["type"] = merged["type"].fillna(merged.get("type_canon"))

                # drop the *_canon helper cols
                for col in ("typology_canon", "type_canon"):
                    if col in merged.columns:
                        merged = merged.drop(columns=[col])

                total_df_raw = merged
                log.info("[reporting] Joined canonical system_families (typology/type) onto combined summary (matches=%d/%d)",
                         total_df_raw["typology"].notna().sum(), len(total_df_raw))
        else:
            log.info("[reporting] No canonical system_families file found at inputs/canonical; typology/type will be inferred or left as-is.")
    except Exception:
        log.exception("[reporting] Unexpected failure while attempting to join canonical system_families; continuing without join.")

    # Determine key for compatibility (same logic as before)
    key_col = None
    preferred_key = "system_variant"
    if preferred_key in total_df_raw.columns:
        key_col = [preferred_key]
    else:
        common_cols = set(dfs[0].columns)
        for d in dfs[1:]:
            common_cols &= set(d.columns)
        common_cols = [c for c in common_cols if not c.startswith("_source")]
        for alt in ("system", "variant", "system_variant_id"):
            if alt in common_cols:
                key_col = [alt]
                break
        if key_col is None:
            if common_cols:
                key_col = sorted(common_cols)
            else:
                log.warning("[reporting] No sensible key found for intersection; compatible == total (fallback).")
                # group total now and write both as identical grouped tables
                total_df_grouped = _group_by_system_variant(total_df_raw)
                total_fp = tables_dir / "total_summary_all.csv"
                total_df_grouped.to_csv(total_fp, index=False)
                log.info("[reporting] Wrote combined (outer union, grouped) -> %s (rows=%d)", total_fp, len(total_df_grouped))

                compat_df_grouped = total_df_grouped.copy()
                compat_fp = tables_dir / "compatible_summary_all.csv"
                compat_df_grouped.to_csv(compat_fp, index=False)
                log.info("[reporting] Wrote compatible (fallback == total, grouped) -> %s (rows=%d)", compat_fp, len(compat_df_grouped))

                # write canonical and call downstream writer using grouped total
                canonical_fp = reporting_root / "summary_ranked_all.csv"
                total_df_grouped.to_csv(canonical_fp, index=False)
                return write_edca_reports(
                    candidates_input=total_df_grouped,
                    out_dir=reporting_root,
                    floor_assignments=floor_assignments,
                    floor_area_lookup=floor_area_lookup,
                    verbose=verbose,
                    metric=metric,
                )

    # --- NORMALISE: treat carbon_total_kgCO2 as carbon_per_m2 if needed ---
    # (Your CSV labels this as total but it is actually already per m².)
    if "carbon_per_m2" not in total_df_raw.columns and "carbon_total_kgCO2" in total_df_raw.columns:
        try:
            total_df_raw["carbon_per_m2"] = pd.to_numeric(total_df_raw["carbon_total_kgCO2"], errors="coerce")
            log.info("[reporting] Note: created column 'carbon_per_m2' from 'carbon_total_kgCO2' (assumed per m²).")
        except Exception:
            log.exception("[reporting] Failed to coerce carbon_total_kgCO2 -> carbon_per_m2; proceeding without conversion.")


    # At this point key_col is set (single or composite). Compute compatible intersection using the RAW union
    if isinstance(key_col, (list, tuple)) and len(key_col) == 1:
        key_col_single = key_col[0]
    else:
        key_col_single = key_col

    # Compute compatible intersection
    if isinstance(key_col_single, str):
        counts = total_df_raw.groupby(key_col_single)["_source_dir"].nunique().reset_index().rename(columns={"_source_dir": "n_files"})
        n_files = len(dfs)
        present_in_all = counts[counts["n_files"] == n_files][key_col_single].tolist()
        compat_df_raw = total_df_raw[total_df_raw[key_col_single].isin(present_in_all)].copy()
    else:
        # composite key list
        tuple_sets = []
        for d in dfs:
            subset = d.dropna(subset=key_col)
            tuples = set(tuple(row[c] for c in key_col) for _, row in subset.iterrows())
            tuple_sets.append(tuples)
        common_tuples = set.intersection(*tuple_sets) if tuple_sets else set()
        if not common_tuples:
            compat_df_raw = total_df_raw.iloc[0:0].copy()
        else:
            def in_common(row):
                return tuple(row[c] for c in key_col) in common_tuples
            compat_df_raw = total_df_raw[total_df_raw.apply(in_common, axis=1)].copy()

    # Now group both raw tables by system_variant (collapse to one row per variant)
    total_df = _group_by_system_variant(total_df_raw)
    compat_df = _group_by_system_variant(compat_df_raw)

    # Write grouped outputs
    total_fp = tables_dir / "total_summary_all.csv"
    total_df.to_csv(total_fp, index=False)
    log.info("[reporting] Wrote combined (outer union, grouped) total_summary_all -> %s (rows=%d)", total_fp, len(total_df))

    compat_fp = tables_dir / "compatible_summary_all.csv"
    compat_df.to_csv(compat_fp, index=False)
    log.info("[reporting] Wrote compatible (inner intersection, grouped) compatible_summary_all -> %s (rows=%d)", compat_fp, len(compat_df))

    # (Optional) also write a "canonical" summary_ranked_all.csv under the reporting root
    canonical_fp = reporting_root / "summary_ranked_all.csv"
    try:
        # By default make the canonical file the total (outer union) so reporting uses the union
        total_df.to_csv(canonical_fp, index=False)
        log.info("[reporting] Wrote canonical summary_ranked_all -> %s", canonical_fp)
    except Exception:
        log.warning("[reporting] Could not write canonical summary_ranked_all at %s", canonical_fp)

    # Finally call the original writer with the total_df as the candidates input
    return write_edca_reports(
        candidates_input=total_df,
        out_dir=reporting_root,
        floor_assignments=floor_assignments,
        floor_area_lookup=floor_area_lookup,
        verbose=verbose,
        metric=metric,
    )

def to_numeric_safe(series: Any) -> pd.Series:
    """Convert a Series-like to numeric, coercing errors to NaN."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * len(series)) if hasattr(series, "__len__") else pd.Series([np.nan])

def add_typology(df: pd.DataFrame, out_col: str = "typology") -> pd.DataFrame:
    """
    Add a coarse typology label (concrete/composite/steel/timber) if missing.
    Uses 'system_family'/'type'/'category' text heuristics.
    """
    if df is None or df.empty:
        return df
    if out_col in df.columns:
        return df

    D = df.copy()
    text_cols = [c for c in ("typology", "category", "type", "system_family") if c in D.columns]
    if not text_cols:
        D[out_col] = None
        return D

    def infer(row: pd.Series) -> str:
        s = " ".join(str(row.get(c, "")) for c in text_cols).lower()
        if "timber" in s or "clt" in s or "glulam" in s:
            return "timber"
        if "composite" in s or "steel-concrete" in s or "slimdek" in s:
            return "composite"
        if "steel" in s:
            return "steel"
        if "concrete" in s or "precast" in s or "pt" in s or "post-tension" in s:
            return "concrete"
        return "other"

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

def compute_pareto_frontier(df: pd.DataFrame,
                            x_col: str,
                            y_col: str,
                            minimize_y: bool = True) -> pd.DataFrame:
    """
    Compute 2D Pareto frontier.
    Assumes larger x is 'more' (e.g. span),
    and lower y is better (e.g. carbon).
    """
    df = df[[x_col, y_col]].dropna().sort_values(x_col)

    frontier = []
    best_y = float("inf")

    for _, row in df.iterrows():
        y = row[y_col]
        if y < best_y:
            frontier.append(row)
            best_y = y

    return pd.DataFrame(frontier)

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
        return None

    color_by = "type" if "type" in df.columns else "system_family" if "system_family" in df.columns else None
    if color_by is None:
        color_by = "_const_type"
        df[color_by] = "all"

    groups = list(df.groupby(color_by))
    fig, ax = plt.subplots(figsize=(8,5))
    for name, g in groups:
        ax.scatter(g["_span"], g["_carbon"], label=str(name), s=28, alpha=0.8, edgecolors="none")
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(f"{label_prefix}Span vs Carbon (colored by {color_by})")
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
        uniq = sorted(cats.unique())
        cmap = plt.get_cmap("tab20")
        colmap = {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}
        colors = [colmap[v] for v in cats]
        # proxy handles for legend
        legend_handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, color=colmap[u], label=u)
                          for u in uniq]

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
      - highlight_variants: grey all points, then overlay the highlighted variants
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

    # Build stable category order
    cats = []
    if color_by:
        cats = sorted(D[color_by].dropna().astype(str).unique())

    fig, ax = plt.subplots(figsize=(9, 6))

    if has_highlight:
        # Context (all points, grey)
        ax.scatter(D["_span"], D["_carbon"], s=18, alpha=0.15, edgecolors="none")
        H = D[D[id_col].astype(str).isin(highlight_set)].copy()
        if H.empty:
            # Still save a useful figure
            ax.set_title(title or "Span vs embodied carbon")
            ax.set_xlabel("Span")
            ax.set_ylabel("Embodied carbon")
            fig.savefig(out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return str(out_path)

        if color_by:
            for gv in cats:
                sub = H[H[color_by].astype(str) == gv]
                if sub.empty:
                    continue
                ax.scatter(sub["_span"], sub["_carbon"], s=55, alpha=1.0, label=str(gv), edgecolors="none")
            ax.legend(title=color_by, loc="best", frameon=True)
        else:
            ax.scatter(H["_span"], H["_carbon"], s=55, alpha=1.0, edgecolors="none", label="highlighted")
            ax.legend(loc="best", frameon=True)
    else:
        # Regular colored scatter
        if color_by:
            for gv in cats:
                sub = D[D[color_by].astype(str) == gv]
                ax.scatter(sub["_span"], sub["_carbon"], s=22, alpha=0.85, label=str(gv), edgecolors="none")
            ax.legend(title=color_by, loc="best", frameon=True)
        else:
            ax.scatter(D["_span"], D["_carbon"], s=22, alpha=0.85, edgecolors="none", label="variants")
            ax.legend(loc="best", frameon=True)

    ax.set_title(title or "Span vs embodied carbon")
    ax.set_xlabel("Span")
    ax.set_ylabel("Embodied carbon")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(out_path)

def plot_span_vs_carbon_pareto(df_all: pd.DataFrame,
                               fig_dir: Path,
                               carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:

    span_col = "span" if "span" in df_all.columns else (
        "max_span" if "max_span" in df_all.columns else None
    )

    if span_col is None or carbon_col not in df_all.columns:
        return {}

    df = df_all[[span_col, carbon_col]].copy()

    has_pass = "pass_overall" in df_all.columns

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all faint
    ax.scatter(df[span_col], df[carbon_col],
               alpha=0.2, label="All candidates")

    if has_pass:
        df_pass = df_all[df_all["pass_overall"] == True][[span_col, carbon_col]].copy()

        ax.scatter(df_pass[span_col], df_pass[carbon_col],
                   alpha=0.8, label="Passing")

        frontier = compute_pareto_frontier(df_pass, span_col, carbon_col)
    else:
        frontier = compute_pareto_frontier(df, span_col, carbon_col)

    if not frontier.empty:
        ax.plot(frontier[span_col], frontier[carbon_col],
                linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Carbon (kgCO₂/m²)")
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
        ax.set_xlabel("Span (m)")
        ax.set_ylabel("Carbon (kgCO₂e / m²)")
        ax.set_title("Span vs Carbon (all variants)")
        ax.grid(True, linestyle=":", alpha=0.4)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fp, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return str(out_fp)
    except Exception:
        logger.exception("plot_span_vs_carbon_global failed")
        return None
    
def plot_depth_vs_carbon(df_all: pd.DataFrame,
                         fig_dir: Path,
                         carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:

    depth_cols = ["slab_depth", "beam_depth", "steel_depth", "screed_depth"]
    depth_cols = [c for c in depth_cols if c in df_all.columns]

    if not depth_cols:
        return {}

    df = df_all.copy()
    df["total_depth"] = df[depth_cols].sum(axis=1)

    span_col = "span" if "span" in df.columns else "max_span"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["total_depth"], df[carbon_col], alpha=0.3)

    ax.set_xlabel("Total structural depth (m)")
    ax.set_ylabel("Carbon (kgCO₂/m²)")

    out_path = fig_dir / "depth_vs_carbon.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"depth_vs_carbon": out_path}

def plot_carbon_distribution_by_type(df_all: pd.DataFrame,
                                     fig_dir: Path,
                                     carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:

    if "type" not in df_all.columns:
        return {}

    fig, ax = plt.subplots(figsize=(10, 6))

    df_all.boxplot(column=carbon_col, by="type", ax=ax)

    ax.set_ylabel("Carbon (kgCO₂/m²)")
    ax.set_title("Carbon distribution by type")
    plt.suptitle("")

    out_path = fig_dir / "carbon_distribution_by_type.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"carbon_distribution_by_type": out_path}

def plot_feasibility_heatmap(df_all: pd.DataFrame,
                             fig_dir: Path,
                             carbon_col: str = "carbon_per_m2") -> Dict[str, Path]:

    if "pass_overall" not in df_all.columns:
        return {}

    span_col = "span" if "span" in df_all.columns else "max_span"

    df = df_all.copy()
    df["span_bin"] = pd.cut(df[span_col], bins=8)
    df["load_bin"] = pd.cut(df["sdl"] + df["ll"], bins=8)

    pivot = df.pivot_table(index="span_bin",
                           columns="load_bin",
                           values="pass_overall",
                           aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Pass rate heatmap")
    fig.colorbar(im)

    out_path = fig_dir / "feasibility_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"feasibility_heatmap": out_path}

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

    ax.set_ylabel("Carbon (kgCO₂/m²)")
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
    ax.set_xlabel("Span (m)"); ax.set_ylabel("Total load")
    ax.set_title("Span vs Total load (all variants)")
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

    ax.set_xlabel("Span (m)")
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

    ax.set_xlabel("Span (m)")
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
            ax.set_xlabel("Span (m)")
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
        ax.set_xlabel("Span (m)")
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
) -> Optional[str]:
    """Small-multiples grid: span vs total_load curves, one panel per family."""
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
    if family_col is None:
        _write_placeholder_figure(out_fp, "No family column found.")
        return str(out_fp)

    variant_col = "system_variant" if "system_variant" in D.columns else family_col

    hv: set[str] = set(str(x) for x in (highlight_variants or []))

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
            is_h = str(vname) in hv if hv else False
            ax.plot(
                sub_v["_span"],
                sub_v["_load"],
                marker="o" if is_h else None,
                linewidth=2.2 if is_h else 1.0,
                alpha=0.95 if is_h else 0.25,
            )

        ax.set_title(str(fam), fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    fig.supxlabel("Span (m)")
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
    base_alpha: float = 0.12,
) -> Optional[str]:
    """Highlight + context plot: all curves grey, highlighted curves emphasized."""
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
                for vname, sub_v in sub_f.groupby(variant_col):
                    sub_v = sub_v.sort_values("_span")
                    ax.plot(sub_v["_span"], sub_v["_load"], marker="o", linewidth=2.2, alpha=0.95, label=str(fam))

            handles, labels = ax.get_legend_handles_labels()
            uniq: Dict[str, Any] = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), title=family_col, fontsize=8, loc="best", frameon=False)

    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Total load / capacity")
    ax.set_title("Span vs Load curves (highlight + context)")
    ax.grid(True, linestyle=":", alpha=0.35)

    plt.tight_layout()
    fig.savefig(out_fp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return str(out_fp)


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
    for name, g in D.groupby(group_by):
        g = g.sort_values("_span")
        ax.plot(g["_span"], g["_load"], marker="o", linewidth=1.2, alpha=0.75, label=str(name))

    ax.set_xlabel("Span (m)")
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
    uniq = sorted(cats.unique())
    cmap = plt.get_cmap("tab20")
    colmap = {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}
    colors = [colmap[v] for v in cats]
    legend_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=colmap[u], label=u)
                      for u in uniq]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # span vs carbon aggregated (no labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lowest[span_col], lowest["_carb"], s=34, alpha=0.9, c=colors, edgecolors="none")
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Carbon (kgCO₂e / m²)")
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
        ax.scatter(lowest[span_col], vals, s=34, alpha=0.9, c=colors, edgecolors="none", label="lowest per group")
        ax.legend(loc="best", frameon=False)
        ax.set_xlabel("Span (m)")
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


    df_all = concat_candidates_input(candidates_input)

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

    # overwrite/regenerate summary_ranked_all.csv right before reporting outputs
    try:
        write_enriched_summary_ranked_all(df_all, out_dir, filename="summary_ranked_all.csv")
    except Exception:
        logger.exception("[reporting] failed to overwrite enriched summary_ranked_all.csv")

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
    df_winners = (
        df_all.sort_values("carbon_total_kgCO2")
            .groupby(["floor_load_category"], as_index=False)
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

    figure_paths.update(plot_lowest_per_group_aggregate(df_all, "system_family", dirs["figures"], color_by="manufacturer" if "manufacturer" in df_all.columns else "system_family", show_legend=True))
    if "manufacturer" in df_all.columns:
        figure_paths.update(plot_lowest_per_group_aggregate(df_all, "manufacturer", dirs["figures"], color_by="manufacturer", show_legend=True))
    # Diagnostics plot (kept as-is)
    p = plot_span_vs_load_curves_by_family(
        df_all,
        dirs["figures"] / "span_vs_load_curves_by_family.png",
        group_by="system_family",
        show_legend=True,
        success_only=True,
    )
    if p:
        figure_paths["span_vs_load_curves_by_family"] = p

    # Additional requested copies:
    p = plot_span_vs_load_curves_by_family(
        df_all,
        dirs["figures"] / "span_vs_load_curves_by_family_no_legend.png",
        group_by="system_family",
        show_legend=False,
        success_only=True,
        title="Span vs Load curves by family (no legend)",
    )
    if p:
        figure_paths["span_vs_load_curves_by_family_no_legend"] = p

    '''
    p = plot_span_vs_load_curves_by_family(
        df_all,
        dirs["figures"] / "span_vs_load_curves_by_manufacturer_no_legend.png",
        group_by="manufacturer",
        show_legend=False,
        success_only=True,
        title="Span vs Load curves by manufacturer (no legend)",
    )
    if p:
        figure_paths["span_vs_load_curves_by_manufacturer_no_legend"] = p
    '''

    figure_paths.update(plot_span_vs_carbon_pareto(df_all, dirs["figures"]))
    figure_paths.update(plot_depth_vs_carbon(df_all, dirs["figures"]))
    figure_paths.update(plot_carbon_distribution_by_type(df_all, dirs["figures"]))
    figure_paths.update(plot_feasibility_heatmap(df_all, dirs["figures"]))
    figure_paths.update(plot_failure_breakdown(df_all, dirs["figures"]))

    # load materials
    materials_df = pd.read_parquet("inputs/canonical/materials.parquet")
    figure_paths.update(plot_carbon_breakdown(df_all, materials_df, dirs["figures"]))

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
            success_only=True,
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
            show_legend=True,
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

        plot_lowest_per_group_aggregate(df_nn, "system_family", nn_dirs["figures"])
        if "manufacturer" in df_nn.columns:
            plot_lowest_per_group_aggregate(df_nn, "manufacturer", nn_dirs["figures"])

        # Span vs load curves (same set, computed on positive-only df for consistency)
        plot_span_vs_load_curves_by_family_grid(
            df_nn,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_grid.png",
            max_families=12,
            ncols=3,
            success_only=True,
            highlight_variants=selected_variants if selected_variants else None,
        )
        plot_span_vs_load_curves_by_family_highlight(
            df_nn,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_highlight.png",
            success_only=True,
            highlight_variants=selected_variants if selected_variants else None,
        )
        plot_span_vs_load_curves_by_family(
            df_nn,
            nn_dirs["figures"] / "span_vs_load_curves_by_family.png",
            group_by="system_family",
            show_legend=True,
            success_only=True,
        )
        plot_span_vs_load_curves_by_family(
            df_nn,
            nn_dirs["figures"] / "span_vs_load_curves_by_family_no_legend.png",
            group_by="system_family",
            show_legend=False,
            success_only=True,
            title="Span vs Load curves by family (no legend)",
        )
        
        '''
        plot_span_vs_load_curves_by_family(
            df_nn,
            nn_dirs["figures"] / "span_vs_load_curves_by_manufacturer_no_legend.png",
            group_by="manufacturer",
            show_legend=False,
            success_only=True,
            title="Span vs Load curves by manufacturer (no legend)",
        )
        '''

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