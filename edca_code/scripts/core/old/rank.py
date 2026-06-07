# edca_code/scripts/core/rank.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import logging
from pathlib import Path
from edca_code.scripts.core.old.utils import reorder_output_columns, infer_type

logger = logging.getLogger("rank")

# -------------------------
# Debug helpers (drop-in)
# -------------------------
import os
from typing import Iterable

def _dbg_enabled(explicit: bool | None = None) -> bool:
    """
    Debug is enabled if:
      - explicit=True passed by caller, OR
      - EDCA_DEBUG=1 environment variable, OR
      - logger level is DEBUG.
    """
    if explicit is True:
        return True
    if explicit is False:
        return False
    if str(os.getenv("EDCA_DEBUG", "")).strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    return bool(getattr(logger, "isEnabledFor", lambda *_: False)(logging.DEBUG))

def _dbg_kv(name: str, d: dict, *, explicit: bool | None = None, level: int = logging.DEBUG) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        items = ", ".join([f"{k}={d[k]!r}" for k in sorted(d.keys())])
    except Exception:
        items = str(d)
    logger.log(level, "[debug] %s: %s", name, items)

def _dbg_df(
    name: str,
    df,
    *,
    explicit: bool | None = None,
    max_rows: int = 15,
    cols: list[str] | None = None,
    level: int = logging.DEBUG,
) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        import pandas as pd
        if df is None:
            logger.log(level, "[debug] %s: df=None", name)
            return
        if not isinstance(df, pd.DataFrame):
            logger.log(level, "[debug] %s: not a DataFrame (%s)", name, type(df).__name__)
            return
        logger.log(level, "[debug] %s: shape=%s", name, df.shape)
        logger.log(level, "[debug] %s: cols=%s", name, list(df.columns))
        sub = df
        if cols:
            keep = [c for c in cols if c in sub.columns]
            sub = sub[keep] if keep else sub
        logger.log(level, "[debug] %s head(%d):\n%s", name, max_rows, sub.head(max_rows).to_string(index=False))
    except Exception:
        logger.exception("[debug] Failed dumping df %s", name)

def _to_dataframe(maybe_df_or_dict: Union[pd.DataFrame, Dict[int, pd.DataFrame]]) -> pd.DataFrame:
    """
    Accept either a DataFrame or a dict floor->DataFrame and return a single DataFrame.
    If input is a dict, concat values and preserve a '_for_floor' column if present.
    """
    if isinstance(maybe_df_or_dict, pd.DataFrame):
        return maybe_df_or_dict.copy()
    if isinstance(maybe_df_or_dict, dict):
        parts = []
        for floor, df in maybe_df_or_dict.items():
            d = df.copy()
            # ensure _for_floor column exists and matches floor (if not present)
            if "_for_floor" not in d.columns:
                d["_for_floor"] = int(floor)
            parts.append(d)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)
    raise TypeError("Input must be a pandas.DataFrame or a dict[int, DataFrame]")

def rank_candidates_by_carbon(candidates_df: pd.DataFrame,
                              carbon_total_col: str = "carbon_total_kgCO2",
                              group_by: Optional[List[str]] = None,
                              take_lowest_per_group: bool = True) -> pd.DataFrame:
    """
    Rank candidate rows by carbon_total_kgCO2 (ascending). Optionally group and take the lowest
    carbon per group, then return the ranked set.
    """
    df = candidates_df.copy()
    if carbon_total_col not in df.columns:
        raise KeyError(f"carbon total column '{carbon_total_col}' not found in candidates_df")

    df = df.sort_values(by=carbon_total_col, ascending=True).reset_index(drop=True)

    if group_by:
        # take the first (lowest carbon) row in each group (preserves order by carbon)
        grouped = df.groupby(group_by, as_index=False).idxmin()
        return grouped.sort_values(by=carbon_total_col, ascending=True).reset_index(drop=True)

    return df

def find_lowest_by_type_and_brand(
    candidates_df: pd.DataFrame,
    type_col: str = "system_family",
    brand_col: str = "system_variant",
    carbon_col: str = "carbon_total_kgCO2",
) -> pd.DataFrame:
    """
    For each (type_col, brand_col) group return the row with lowest carbon_col.
    Groups where carbon_col is all NaN are reported and skipped.
    """
    if candidates_df.empty:
        logger.warning("find_lowest_by_type_and_brand: empty input DataFrame")
        return pd.DataFrame(columns=candidates_df.columns)

    grp = candidates_df.groupby([type_col, brand_col])

    # Series of row-labels (index of candidates_df) that are argmin per group
    idx = grp[carbon_col].idxmin()

    missing_groups = idx[idx.isna()].index.tolist()
    if missing_groups:
        logger.warning(
            "Found %d groups where %s is all-NaN. Examples: %s",
            len(missing_groups),
            carbon_col,
            missing_groups[:10],
        )

    idx_nonan = idx.dropna()
    if idx_nonan.empty:
        logger.warning("No groups with valid %s found. Returning empty DataFrame.", carbon_col)
        return pd.DataFrame(columns=candidates_df.columns)

    out = candidates_df.loc[idx_nonan.values].reset_index(drop=True)
    out.attrs["missing_groups_all_nan"] = missing_groups
    return out


# ---------- NEW helpers for material grouping ----------
def infer_material_column(df: pd.DataFrame, candidates: List[str] = None) -> Optional[str]:
    """
    Try to infer a column that encodes a material category (concrete/steel/timber).
    Look for common names. Returns column name or None.
    """
    if candidates is None:
        candidates = ["primary_material", "dominant_material", "material_type", "main_material", "construction_material"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try to infer from system_family values (if present) by checking for substrings
    if "system_family" in df.columns:
        # create a synthetic column by mapping some families -> 'concrete','steel','timber' heuristically
        return None
    return None


def lowest_per_material_category(
    candidates_df: pd.DataFrame,
    carbon_col: str = "carbon_total_kgCO2",
    material_col: Optional[str] = None,
    material_map: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Compute the lowest-carbon candidate per material category.

    - material_col: column in candidates_df that explicitly lists a material/category (e.g., 'primary_material').
      If omitted, caller should provide material_map or function will attempt to use common column names.
    - material_map: optional mapping {category_name: [list_of_strings_to_match_in_row]} used to infer category from row values.
      Example: {'concrete': ['concrete','slab','block'], 'steel': ['steel','composite'], 'timber': ['timber','wood']}
    """
    df = candidates_df.copy()
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    if material_col and material_col in df.columns:
        # filter out rows where material is missing
        df = df[df[material_col].notna()]
        # group by material_col and take argmin
        grp = df.groupby(material_col)
        idx = grp[carbon_col].idxmin()
        idx = idx.dropna()
        if idx.empty:
            return pd.DataFrame(columns=df.columns)
        return df.loc[idx.values].reset_index(drop=True)

    # If no explicit column, try to infer using material_map
    if material_map:
        # create a synthetic column '_material_category' by matching strings
        def match_row_to_category(row_val: str):
            if not isinstance(row_val, str):
                return None
            rv = row_val.lower()
            for cat, tokens in material_map.items():
                for t in tokens:
                    if t.lower() in rv:
                        return cat
            return None

        # prefer columns to search in
        search_cols = ["system_family", "system_variant", "description", "notes"]
        df["_material_category"] = None
        for col in search_cols:
            if col in df.columns:
                df["_material_category"] = df["_material_category"].fillna(df[col].astype(str).apply(match_row_to_category))
        # drop NA
        df2 = df[df["_material_category"].notna()]
        if df2.empty:
            logger.warning("lowest_per_material_category: no rows matched material_map tokens.")
            return pd.DataFrame(columns=df.columns)
        grp = df2.groupby("_material_category")
        idx = grp[carbon_col].idxmin().dropna()
        return df2.loc[idx.values].reset_index(drop=True)

    # unable to compute
    logger.warning("lowest_per_material_category: neither material_col provided nor material_map supplied; returning empty DataFrame.")
    return pd.DataFrame(columns=df.columns)


# ---------- Main exporter ----------
def rank_and_export_summary(
    candidates_input: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
    out_dir: Optional[Path] = None,
    file_prefix: str = "summary",
    carbon_col: str = "carbon_total_kgCO2",
    type_col: str = "system_family",
    brand_col: str = "system_variant",
    material_col: Optional[str] = None,
    material_map: Optional[Dict[str, List[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:

    """
    Produce summary tables:
      - 'ranked_all': all candidates sorted by carbon (ascending)
      - 'lowest_overall': single-row DataFrame with lowest-carbon candidate overall
      - 'lowest_per_brand': lowest carbon row per (type, brand)
      - 'lowest_per_type': lowest carbon row per type (across brands)
      - 'lowest_per_material': lowest carbon row per material category (concrete/steel/timber)
    Input may be a DataFrame or dict floor->DataFrame (they will be concatenated).
    """
    log = logger or globals().get("logger") or logging.getLogger("rank")
    df_all = _to_dataframe(candidates_input)
    if df_all.empty:
        logger.warning("rank_and_export_summary: input candidates empty; returning empty summaries.")
        empty = pd.DataFrame(columns=[])
        return {
            "ranked_all": empty,
            "lowest_overall": empty,
            "lowest_per_brand": empty,
            "lowest_per_type": empty,
            "lowest_per_material": empty,
        }

    # ensure carbon column present
    if carbon_col not in df_all.columns:
        raise KeyError(f"rank_and_export_summary: carbon_col '{carbon_col}' not in input DataFrame")

    # ranked all
    ranked_all = rank_candidates_by_carbon(df_all, carbon_total_col=carbon_col, group_by=None)

    if _dbg_enabled(None):
        _dbg_kv("rank.ranked_all", {
            "n_rows": int(len(ranked_all)),
            "carbon_col": carbon_col,
            "min_carbon": float(pd.to_numeric(ranked_all[carbon_col], errors="coerce").min()) if carbon_col in ranked_all.columns else None,
        }, explicit=True, level=logging.INFO)

    # lowest overall: first row in ranked_all
    lowest_overall = ranked_all.head(1).reset_index(drop=True)

    # lowest per (type,brand)
    lowest_per_brand = find_lowest_by_type_and_brand(df_all, type_col=type_col, brand_col=brand_col, carbon_col=carbon_col)

    # lowest per type
    if not lowest_per_brand.empty:
        idx_type = lowest_per_brand.groupby(type_col)[carbon_col].idxmin()
        lowest_per_type = lowest_per_brand.loc[idx_type].reset_index(drop=True)
    else:
        lowest_per_type = pd.DataFrame(columns=df_all.columns)

    # lowest per material category
    lowest_per_material = pd.DataFrame(columns=df_all.columns)
    if material_col or material_map:
        lowest_per_material = lowest_per_material_category(df_all, carbon_col=carbon_col, material_col=material_col, material_map=material_map)
    else:
        # try to infer candidate column
        inferred = infer_material_column(df_all)
        if inferred:
            lowest_per_material = lowest_per_material_category(df_all, carbon_col=carbon_col, material_col=inferred)
        else:
            # last resort: try a basic map from family names
            basic_map = {
                "concrete": ["concrete", "block", "slab"],
                "steel": ["steel", "composite"],
                "timber": ["timber", "wood"],
            }
            lowest_per_material = lowest_per_material_category(df_all, carbon_col=carbon_col, material_map=basic_map)

    # ---------- optional export ----------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _write(name: str, df: pd.DataFrame) -> None:
            fp = out_dir / f"{file_prefix}_{name}.csv"
            try:
                df_out = df.copy()
                # keep column order consistent with rest of pipeline when possible
                try:
                    df_out = reorder_output_columns(df_out)
                except Exception:
                    pass
                df_out.to_csv(fp, index=False)
                log.info("[rank] wrote %s (%d rows) -> %s", name, len(df_out), fp)
            except Exception:
                log.exception("[rank] failed writing %s -> %s", name, fp)

        _write("ranked_all", ranked_all)
        _write("lowest_overall", lowest_overall)
        _write("lowest_per_brand", lowest_per_brand)
        _write("lowest_per_type", lowest_per_type)
        _write("lowest_per_material", lowest_per_material)

    return {
        "ranked_all": ranked_all,
        "lowest_overall": lowest_overall,
        "lowest_per_brand": lowest_per_brand,
        "lowest_per_type": lowest_per_type,
        "lowest_per_material": lowest_per_material,
    }

def assemble_candidates_all_from_cases(*, out_dir, systems_df):
    log = logging.getLogger("rank")
    log.info("[run_edca] Building global candidates_all_spans.csv from per-case winners")
    out_dir = Path(out_dir)  # ensure Path type if not already
    systems_case_dirs = sorted([p for p in out_dir.glob("systems_*") if p.is_dir()])

    logger.info("[run_edca] Building global candidate set from each case's summary_lowest_overall.csv (one winner per case)")

    selected_rows = []     # will hold full candidate rows (with BOM columns) for each case
    missing_cases = []

    for case_dir in systems_case_dirs:
        case_name = case_dir.name.replace("systems_", "")
        summary_fp = case_dir / f"summary_lowest_overall.csv"
        candidates_fp = case_dir / f"candidates_{case_name}_EC_spans.csv"

        # NOTE: if your actual file is candidates_{case_name}_spans.csv, fix here (not a wrapper issue).
        if not summary_fp.exists():
            logger.warning("[run_edca] summary_lowest_overall.csv missing for case %s (expected at %s). Skipping case.", case_name, summary_fp)
            missing_cases.append(case_name)
            continue

        try:
            summary_df = pd.read_csv(summary_fp)
        except Exception:
            logger.exception("[run_edca] Failed to read %s — skipping case %s", summary_fp, case_name)
            missing_cases.append(case_name)
            continue

        variant_col = None
        for c in ("system_variant", "variant", "variant_id", "system_id"):
            if c in summary_df.columns:
                variant_col = c
                break
        if variant_col is None:
            logger.warning("[run_edca] Could not find variant id column in %s (columns: %s). Skipping.", summary_fp, list(summary_df.columns))
            missing_cases.append(case_name)
            continue

        chosen_variants = summary_df[variant_col].astype(str).unique().tolist()
        if not chosen_variants:
            logger.warning("[run_edca] No variant found in %s for case %s; skipping.", summary_fp, case_name)
            missing_cases.append(case_name)
            continue

        candidates_df = None
        if candidates_fp.exists():
            try:
                candidates_df = pd.read_csv(candidates_fp)
            except Exception:
                logger.exception("[run_edca] Error reading candidates file %s for case %s — will fallback to summary only", candidates_fp, case_name)
                candidates_df = None

        for v in chosen_variants:
            chosen_row = None

            if candidates_df is not None:
                cand_variant_col = None
                for c in ("system_variant","variant","variant_id","system_id"):
                    if c in candidates_df.columns:
                        cand_variant_col = c
                        break
                if cand_variant_col:
                    matches = candidates_df[candidates_df[cand_variant_col].astype(str) == str(v)]
                    if not matches.empty:
                        if "carbon_total_kgCO2" in matches.columns or "carbon" in matches.columns:
                            carb_col = "carbon_total_kgCO2" if "carbon_total_kgCO2" in matches.columns else "carbon"
                            matches_sorted = matches.sort_values(by=carb_col, na_position="last")
                            chosen_row = matches_sorted.iloc[0:1]
                        else:
                            chosen_row = matches.iloc[0:1]

            if chosen_row is None:
                rows = summary_df[summary_df[variant_col].astype(str) == str(v)]
                if not rows.empty:
                    chosen_row = rows.iloc[0:1]
                    logger.debug("[run_edca] Using summary row for variant %s in case %s (no full candidate row found)", v, case_name)
                else:
                    logger.warning("[run_edca] Couldn't find full or summary row for variant %s in case %s; skipping", v, case_name)
                    continue

            selected_rows.append(chosen_row.copy().assign(_source_case=case_name))

    if selected_rows:
        candidates_all = pd.concat(selected_rows, ignore_index=True, sort=False)
        logger.info("[run_edca] Built candidates_all from %d case winner rows (cases found: %d). Shape: %s",
                    len(selected_rows), len(systems_case_dirs) - len(missing_cases), candidates_all.shape)
    else:
        candidates_all = pd.DataFrame()
        logger.warning("[run_edca] No per-case winners could be assembled into candidates_all. candidates_all is empty.")

    return candidates_all, missing_cases

def finalize_and_write_candidates_all(
    *,
    candidates_all: pd.DataFrame,
    out_dir: Path,
    systems_df: pd.DataFrame,
    file_name: str = "candidates_all_spans.csv",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    log = logger or globals().get("logger") or logging.getLogger("rank")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If empty, still write an empty CSV with the expected schema
    if candidates_all is None or candidates_all.empty:
        log.info("[systems] No candidates found across any load cases. Writing empty %s.", file_name)
        try:
            empty_fp = out_dir / file_name
            pd.DataFrame(columns=systems_df.columns).to_csv(empty_fp, index=False)
            log.info("[systems] Wrote empty combined candidates file -> %s", empty_fp)
        except Exception:
            log.exception("[systems] Failed to write empty combined candidates file")
        return pd.DataFrame()

    # Ensure expected columns
    if "carbon_total_kgCO2" not in candidates_all.columns:
        candidates_all["carbon_total_kgCO2"] = candidates_all.get("carbon_per_m2", 0.0)
    if "system_family" not in candidates_all.columns and "system_variant" in candidates_all.columns:
        candidates_all["system_family"] = candidates_all["system_variant"]
    if "typology" not in candidates_all.columns:
        candidates_all["typology"] = candidates_all.apply(infer_type, axis=1)

    candidates_all_out = reorder_output_columns(candidates_all)

    combined_fp = out_dir / file_name
    try:
        candidates_all_out.to_csv(combined_fp, index=False)
        log.info("[rank] Wrote %d combined candidate rows -> %s", len(candidates_all_out), combined_fp)
    except Exception:
        log.exception("[rank] Failed to write combined candidates file %s", combined_fp)

    return candidates_all_out
