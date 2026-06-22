#!/usr/bin/env python3
"""edca_code/scripts/run_edca.py

Native orchestrator for EDCA.

Runs with:
python -m edca_code.scripts.run_edca \
  --control setup/control_files/control_file.yaml \
  --systems inputs/canonical/system_variants.parquet \
  --materials inputs/source/presets/materials/materials.csv \
  --occupancies inputs/source/presets/loads/occupancies.csv \
  --out outputs/edca_run \
  --run-codechecks

Fixes included:
- Code checks: calls run_code_checks_if_requested(candidates_df, out_dir, run_flag, **kwargs) with required args.
- Reporting: ensures a 'floor_load_category' column exists in summary outputs (set to load case name), so reporting.py
  can group without KeyError.

This file assumes the rest of your package code is under edca_code.scripts.core.* and edca_code.scripts.code_checks.*.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edca_code.scripts.core.parse import ControlFile, build_floor_area_lookup, parameters_from_control_file
from edca_code.scripts.core.old import systems as systems_mod
from edca_code.scripts.core.old import carbon as carbon_mod
from edca_code.scripts.core.old import loads as loads_mod
from edca_code.scripts.core import spans as spans_mod
from edca_code.scripts.core.old import rank as rank_mod
from edca_code.scripts.core.old import reporting as reporting_mod
from edca_code.scripts.core.old import takeoff as takeoff_mod
from edca_code.scripts.core.old import utils as utils_mod

from edca_code.scripts.code_checks.code_runner import run_code_checks_if_requested


logger = logging.getLogger(__name__)

# -------------------------
# Debug helpers (drop-in)
# -------------------------
import os
from typing import Iterable

_BEAM_SIMPLE_MOMENT_COEFF = 1.0 / 8.0
_BEAM_INTERIOR_SUPPORT_MOMENT_COEFF = 0.106
_BEAM_INTERIOR_SHEAR_COEFF = 0.63

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

def _dbg_snapshot_df(df: pd.DataFrame, name: str, out_dir: Path, *, enabled: bool, max_rows: int = 25) -> None:
    if not enabled:
        return
    try:
        ddir = utils_mod.ensure_dir(Path(out_dir) / "debug")
        fp = ddir / f"{name}.csv"
        df.to_csv(fp, index=False)
        logger.debug("[debug] snapshot wrote %s (%s)", fp, df.shape)
        _dbg_df(f"snapshot::{name}", df, explicit=True, max_rows=max_rows)
    except Exception:
        logger.exception("[debug] failed writing snapshot %s", name)

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    logger.setLevel(level)


def default_code_standard_if_missing(cf: Any) -> None:
    """Add cf.code_standard if not present; helps loads combos pick Eurocode when metric."""
    if getattr(cf, "code_standard", None):
        return
    unit_lower = (getattr(cf, "unit", "") or "").strip().lower()
    if unit_lower in ("metric", "si", "eu", "euro", "european"):
        setattr(cf, "code_standard", "Eurocode")
    else:
        setattr(cf, "code_standard", "Default")


def collapse_floor_loads_to_cases(loads_df_floor: pd.DataFrame, code_standard: str) -> pd.DataFrame:
    """Collapse per-floor loads_df_floor into per-occupancy/load_case governing rows."""
    if loads_df_floor is None or loads_df_floor.empty:
        return pd.DataFrame(columns=["load_case", "raw_sdl", "raw_ll", "factored_total", "unit"])

    df = loads_df_floor.copy()

    if "occupancy" not in df.columns:
        for alt in ("load_case", "case", "space_type", "program", "use", "occ"):
            if alt in df.columns:
                df = df.rename(columns={alt: "occupancy"})
                break
    if "occupancy" not in df.columns:
        raise ValueError(f"[loads] loads_df_floor missing occupancy column. Columns={list(df.columns)}")

    if "SDL" not in df.columns or "LL" not in df.columns:
        raise ValueError(f"[loads] loads_df_floor missing SDL/LL columns. Columns={list(df.columns)}")

    out: List[Dict[str, Any]] = []
    for occ, g in df.groupby("occupancy", dropna=False):
        load_case = str(occ)
        raw_sdl = float(pd.to_numeric(g["SDL"], errors="coerce").max() or 0.0)
        raw_ll = float(pd.to_numeric(g["LL"], errors="coerce").max() or 0.0)

        ft = pd.to_numeric(g.get("factored_total", pd.Series([], dtype=float)), errors="coerce")
        if len(ft) == 0 or ft.isna().all():
            factored_total = float(raw_sdl + raw_ll)
            best_row = g.iloc[0]
        else:
            idx_max = int(ft.fillna(-1e30).idxmax())
            best_row = g.loc[idx_max]
            factored_total = float(best_row.get("factored_total", raw_sdl + raw_ll) or (raw_sdl + raw_ll))

        unit_val = best_row.get("unit", getattr(best_row, "unit", None))

        out.append({
            "load_case": load_case,
            "raw_sdl": raw_sdl,
            "raw_ll": raw_ll,
            "factored_total": factored_total,
            "unit": unit_val,
        })

    return pd.DataFrame(out)


def select_winners_from_ranked_all(df_ranked_all: pd.DataFrame) -> pd.DataFrame:
    """Winners = first row per floor_load_category (or case)."""
    if df_ranked_all is None or df_ranked_all.empty:
        return pd.DataFrame()

    d = df_ranked_all.copy()
    if "floor_load_category" in d.columns:
        group_col = "floor_load_category"
    elif "case" in d.columns:
        group_col = "case"
    else:
        # single global winner
        return d.sort_values("carbon_total_kgCO2").head(1).copy()

    return (
        d.sort_values("carbon_total_kgCO2")
         .groupby(group_col, as_index=False, dropna=False)
         .head(1)
         .copy()
    )

def select_lowest_carbon_per_type(df_ranked_all: pd.DataFrame) -> pd.DataFrame:
    """
    Select one row per type (fallback typology/system_family), choosing the lowest-carbon row.
    Prefer passing rows first if a pass column exists.
    """
    if df_ranked_all is None or df_ranked_all.empty:
        return pd.DataFrame()

    d = df_ranked_all.copy()

    # normalize id-ish columns
    if "system_variant" in d.columns:
        d["system_variant"] = d["system_variant"].astype(str).str.strip()

    # pick carbon column
    sort_col = "carbon_total_kgCO2" if "carbon_total_kgCO2" in d.columns else (
        "carbon_per_m2" if "carbon_per_m2" in d.columns else None
    )
    if sort_col is None:
        return pd.DataFrame()

    d[sort_col] = pd.to_numeric(d[sort_col], errors="coerce")
    d = d.dropna(subset=[sort_col])
    if d.empty:
        return pd.DataFrame()

    # pick grouping column
    group_col = "type" if "type" in d.columns else (
        "typology" if "typology" in d.columns else (
            "system_family" if "system_family" in d.columns else None
        )
    )
    if group_col is None:
        return d.sort_values(sort_col).head(1).copy()

    # prefer passing if available
    pass_col = "pass_overall" if "pass_overall" in d.columns else None
    if pass_col:
        # robust bool coercion without importing a helper
        def _to_bool(x):
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s in ("true", "1", "yes", "y", "t")
        d["_pass_pref"] = d[pass_col].map(_to_bool).fillna(False).astype(int)
        d = d.sort_values(by=["_pass_pref", sort_col], ascending=[False, True], kind="mergesort")
    else:
        d = d.sort_values(sort_col, ascending=True, kind="mergesort")

    out = (
        d.dropna(subset=[group_col])
         .groupby(group_col, as_index=False, dropna=False)
         .head(1)
         .copy()
    )

    return out.drop(columns=["_pass_pref"], errors="ignore")

def collapse_summary_ranked_all_by_system_variant(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse rows so each system_variant appears exactly once.

    Typical cause of duplicates: span sweep creates multiple rows with same system_variant but slightly different span.
    We keep ONE representative row per system_variant (prefer PASSING + lowest carbon),
    and we attach span_min/span_max/span_n for traceability.

    This is intended for SUMMARY outputs (CSV tables), not for the full candidate set used for code checks.
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame()

    d = df_in.copy()

    # normalize system id column
    if "system_variant" not in d.columns:
        for alt in ("system_id", "system_variant_id", "variant_id", "system"):
            if alt in d.columns:
                d = d.rename(columns={alt: "system_variant"})
                break
    if "system_variant" not in d.columns:
        raise ValueError(f"[collapse] Expected system_variant column. Columns={list(d.columns)}")

    d["system_variant"] = d["system_variant"].astype(str).str.strip()

    # detect a "pass" column if present
    pass_col = None
    preferred = [
        "pass_overall", "pass_all", "pass", "passes", "is_passing", "overall_pass",
        "pass_code", "pass_checks", "all_pass"
    ]
    for c in preferred:
        if c in d.columns:
            pass_col = c
            break
    if pass_col is None:
        # heuristic: any column containing "pass" that looks boolean-ish
        for c in d.columns:
            if "pass" in c.lower():
                vals = d[c].dropna().astype(str).str.lower().unique().tolist()
                if set(vals).issubset({"true", "false", "1", "0", "yes", "no", "t", "f"}):
                    pass_col = c
                    break

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("true", "1", "yes", "y", "t")

    if pass_col is not None:
        d[pass_col] = d[pass_col].map(_to_bool)

    # Choose representative row per system_variant:
    # sort so "best" row comes first: PASSING first, then lowest carbon, then best rank.
    sort_cols = []
    ascending = []

    if pass_col is not None:
        sort_cols.append(pass_col)
        ascending.append(False)  # True first

    for c in ("carbon_total_kgCO2", "rank_overall", "rank_carbon", "rank", "cost_total"):
        if c in d.columns:
            sort_cols.append(c)
            ascending.append(True)

    if sort_cols:
        d = d.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")

    best = (
        d.groupby("system_variant", dropna=False, as_index=False)
         .head(1)
         .copy()
    )

    # Span stats (keep representative span column as-is for compatibility, but also add min/max/count)
    if "span" in d.columns:
        span_num = pd.to_numeric(d["span"], errors="coerce")
        span_stats = (
            d.assign(_span=span_num)
             .groupby("system_variant", dropna=False)["_span"]
             .agg(span_min="min", span_max="max", span_n="count")
             .reset_index()
        )
        best = best.merge(span_stats, on="system_variant", how="left")

    # Guarantee 1 row per variant
    if best["system_variant"].duplicated().any():
        raise ValueError("[collapse] Still duplicated system_variant after collapse (unexpected).")

    return best

def finalize_root_summary_ranked_all(out_dir: Path, logger: logging.Logger) -> None:
    """
    Build root-level summaries from per-case subfolder summary_ranked_all.csv files.

    Writes:
      - <out_dir>/summary_ranked_all_long.csv : long form (case × system_variant), filtered to intersection across cases
      - <out_dir>/summary_ranked_all.csv      : wide inner-join across cases, 1 row per system_variant
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        logger.warning("[summary] Output directory %s does not exist; skipping root summary build.", out_dir)
        return
    case_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("systems_")])
    if not case_dirs:
        logger.warning("[summary] No systems_* case folders found under %s; skipping root summary build.", out_dir)
        return

    # Load per-case summaries
    case_tables = []
    for d in case_dirs:
        p = d / "summary_ranked_all.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            logger.exception("[summary] Failed reading %s", p)
            continue

        if "system_variant" not in df.columns:
            logger.warning("[summary] %s missing system_variant; skipping.", p)
            continue

        case_name = d.name[len("systems_"):] or d.name
        df = df.copy()
        df["case"] = case_name
        df["system_variant"] = df["system_variant"].astype(str).str.strip()
        case_tables.append((case_name, df))

    if not case_tables:
        logger.warning("[summary] No per-case summary_ranked_all.csv files found; skipping.")
        return

    # Inner-join key set: variants present in every case
    common = None
    for case_name, df in case_tables:
        keys = set(df["system_variant"].dropna().astype(str).str.strip().unique().tolist())
        common = keys if common is None else (common & keys)

    if not common:
        logger.warning("[summary] No common system_variants across cases; writing empty root summaries.")
        pd.DataFrame().to_csv(out_dir / "summary_ranked_all_long.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "summary_ranked_all.csv", index=False)
        return

    common = sorted(common)

    # -------------------------
    # (1) Long file: concat all cases restricted to common variants
    # -------------------------
    long_parts = []
    for case_name, df in case_tables:
        long_parts.append(df[df["system_variant"].isin(common)].copy())
    long_df = pd.concat(long_parts, ignore_index=True, sort=False)
    long_df.to_csv(out_dir / "summary_ranked_all_long.csv", index=False)

    # -------------------------
    # (2) Wide file: collapse duplicates within each case to 1 row/variant, then inner-join merge across cases
    # -------------------------
    import re as _re

    def _safe_suffix(x: str) -> str:
        x = _re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")
        return x or "case"

    def _collapse_one_case(df_case: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse multiple rows per system_variant within a case.
        If duplicates exist because of span (or other small param changes), aggregate them.
        """
        D = df_case.copy()
        D["system_variant"] = D["system_variant"].astype(str).str.strip()

        # Handle span specially (if present): keep min/max, drop raw span
        if "span" in D.columns:
            span_num = pd.to_numeric(D["span"], errors="coerce")
            D["_span_num"] = span_num
            D["_span_str"] = D["span"].astype(str)

        # Build aggregation rules
        agg = {}
        for c in D.columns:
            if c in ("system_variant", "case"):
                continue
            if c in ("system_family",):
                agg[c] = "first"
                continue

            s = c.lower()

            # span handled later
            if c in ("span", "_span_num", "_span_str"):
                continue

            if pd.api.types.is_bool_dtype(D[c]):
                agg[c] = "all" if ("pass" in s or s.startswith("ok")) else "first"
                continue

            if pd.api.types.is_numeric_dtype(D[c]):
                if "rank" in s:
                    agg[c] = "min"  # best
                elif any(k in s for k in ("carbon", "co2", "kgco2", "cost", "price", "usd", "gbp")):
                    agg[c] = "min"  # best
                elif any(k in s for k in ("depth", "util", "unity", "dcr", "demand")):
                    agg[c] = "max"  # worst-case
                else:
                    agg[c] = "first"
            else:
                agg[c] = "first"

        collapsed = D.groupby("system_variant", dropna=False, as_index=False).agg(agg)

        # add span_min/span_max if span existed
        if "_span_num" in D.columns:
            span_stats = (
                D.groupby("system_variant", dropna=False)["_span_num"]
                .agg(span_min="min", span_max="max")
                .reset_index()
            )
            collapsed = collapsed.merge(span_stats, on="system_variant", how="left")
        elif "_span_str" in D.columns:
            span_stats = (
                D.groupby("system_variant", dropna=False)["_span_str"]
                .agg(span_min="min", span_max="max")
                .reset_index()
            )
            collapsed = collapsed.merge(span_stats, on="system_variant", how="left")

        return collapsed

    wide_tables = []
    for idx, (case_name, df_case) in enumerate(case_tables):
        suffix = _safe_suffix(case_name)

        df_case = df_case[df_case["system_variant"].isin(common)].copy()
        df_case = _collapse_one_case(df_case)

        # Keep system_family only once (unsuffixed) if present
        if idx > 0 and "system_family" in df_case.columns:
            df_case = df_case.drop(columns=["system_family"])

        # Drop case column before widening
        if "case" in df_case.columns:
            df_case = df_case.drop(columns=["case"])

        # Suffix all columns except system_variant
        ren = {}
        for c in df_case.columns:
            if c == "system_variant":
                continue
            ren[c] = f"{c}__{suffix}"
        df_case = df_case.rename(columns=ren)

        wide_tables.append(df_case)

    merged = wide_tables[0]
    for t in wide_tables[1:]:
        merged = merged.merge(t, on="system_variant", how="inner")

    # Guarantee uniqueness
    if merged["system_variant"].duplicated().any():
        dups = merged.loc[merged["system_variant"].duplicated(), "system_variant"].head(10).tolist()
        raise ValueError(f"[summary] Root wide merge still has duplicate system_variant rows (e.g., {dups}).")

    merged.to_csv(out_dir / "summary_ranked_all.csv", index=False)
    logger.info("[summary] Wrote %s (wide) and %s (long).", out_dir / "summary_ranked_all.csv", out_dir / "summary_ranked_all_long.csv")

def run_codechecks_on_winners(
    *,
    df_winners: pd.DataFrame,
    out_dir: Path,
    run_flag: bool,
    material_csv_path: Optional[str] = None,
    load_combos_yaml: Optional[str] = None,
    load_values_yaml: Optional[str] = None,
    debug_inputs: bool = False,
    debug_only_on_fail: bool = True,
    debug_max_rows: int = 50,
) -> pd.DataFrame:
    if not run_flag:
        return pd.DataFrame()
    if df_winners is None or df_winners.empty:
        logger.warning("[codechecks] No winners to check; skipping.")
        return pd.DataFrame()

    code_out_dir = utils_mod.ensure_dir(Path(out_dir) / "code_checks")

    df_in = df_winners.copy()
    if "system_variant" not in df_in.columns:
        for alt in ("system_id", "system_variant_id", "variant_id", "system"):
            if alt in df_in.columns:
                df_in = df_in.rename(columns={alt: "system_variant"})
                break
    if "system_variant" in df_in.columns:
        df_in["system_variant"] = df_in["system_variant"].astype(str)

    try:
        return run_code_checks_if_requested(
            df_in,
            code_out_dir,
            True,
            material_csv_path=material_csv_path,
            load_combos_yaml=load_combos_yaml,
            load_values_yaml=load_values_yaml,
            debug_inputs=debug_inputs,
            debug_only_on_fail=debug_only_on_fail,
            debug_max_rows=debug_max_rows,
        )
    except Exception:
        logger.exception("[codechecks] Code checks failed (df_winners only).")
        return pd.DataFrame()


COMPONENT_TABLES: Dict[str, Dict[str, str]] = {
    "floor": {
        "variant_file": "floor_variants",
        "family_file": "floor_families",
        "variant_id": "floor_variant_id",
        "family_id": "floor_family_id",
        "category_col": "floor_category",
        "type_col": "floor_type",
        "basis": "per_m2",
    },
    "beam": {
        "variant_file": "beam_variants",
        "family_file": "beam_families",
        "variant_id": "beam_variant_id",
        "family_id": "beam_family_id",
        "category_col": "beam_category",
        "type_col": "beam_type",
        "basis": "per_linear_m",
    },
    "column": {
        "variant_file": "column_variants",
        "family_file": "column_families",
        "variant_id": "column_variant_id",
        "family_id": "column_family_id",
        "category_col": "column_category",
        "type_col": "column_type",
        "basis": "per_linear_m",
    },
    "lateral": {
        "variant_file": "lateral_variants",
        "family_file": "lateral_families",
        "variant_id": "lateral_variant_id",
        "family_id": "lateral_family_id",
        "category_col": "lateral_category",
        "type_col": "lateral_type",
        "basis": "per_catalog_unit",
    },
}


def _read_component_table(canonical_dir: Path, name: str) -> Optional[pd.DataFrame]:
    for ext in (".parquet", ".pq", ".csv"):
        p = canonical_dir / f"{name}{ext}"
        if not p.exists():
            continue
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        return pd.read_csv(p, low_memory=False)
    return None


def _load_component_catalog(
    *,
    canonical_dir: Path,
    component: str,
    unit_filter: Optional[str],
) -> pd.DataFrame:
    cfg = COMPONENT_TABLES[component]
    variants = _read_component_table(canonical_dir, cfg["variant_file"])
    families = _read_component_table(canonical_dir, cfg["family_file"])

    if variants is None or variants.empty:
        return pd.DataFrame()

    d = variants.copy()
    if families is not None and not families.empty and cfg["family_id"] in d.columns and cfg["family_id"] in families.columns:
        fam = families.copy()
        d = d.merge(fam, on=cfg["family_id"], how="left", suffixes=("", "_family"))

        # Variant rows often carry empty material ids while family rows carry defaults.
        for c in [
            "material_concrete_id",
            "material_screed_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
        ]:
            fc = f"{c}_family"
            if c in d.columns and fc in d.columns:
                left = d[c].replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
                d[c] = left.fillna(d[fc])

        # Some steel beam/column catalog rows are mislabeled with their section area
        # stored under concrete_volume instead of steel_volume. Repair only rows that
        # are clearly steel members and have no other concrete signal.
        if component in {"beam", "column"}:
            material_family = d.get("material_family")
            category = d.get("category")
            type_name = d.get("type")
            steel_volume = pd.to_numeric(d.get("steel_volume"), errors="coerce").fillna(0.0)
            concrete_volume = pd.to_numeric(d.get("concrete_volume"), errors="coerce").fillna(0.0)
            rebar_volume = pd.to_numeric(d.get("rebar_volume"), errors="coerce").fillna(0.0)
            pt_volume = pd.to_numeric(d.get("pt_volume"), errors="coerce").fillna(0.0)
            timber_volume = pd.to_numeric(d.get("timber_volume"), errors="coerce").fillna(0.0)
            steel_id = d.get("material_steel_id", pd.Series(index=d.index, dtype="object")).fillna("").astype(str).str.strip()
            conc_id = d.get("material_concrete_id", pd.Series(index=d.index, dtype="object")).fillna("").astype(str).str.strip()

            steel_like = (
                material_family.fillna("").astype(str).str.lower().eq("steel")
                if material_family is not None else pd.Series(False, index=d.index)
            )
            if category is not None:
                steel_like = steel_like | category.fillna("").astype(str).str.lower().str.contains("steel", regex=False)
            if type_name is not None:
                steel_like = steel_like | type_name.fillna("").astype(str).str.lower().str.contains(
                    r"\b(hfchs|cfchs|shs|rhs|hss|uc|ub|ipe|hea|heb|hem|w\d)",
                    regex=True,
                )

            mislabeled_steel = (
                steel_like
                & steel_volume.le(0.0)
                & concrete_volume.gt(0.0)
                & rebar_volume.le(0.0)
                & pt_volume.le(0.0)
                & timber_volume.le(0.0)
                & steel_id.eq("")
                & conc_id.isin(["", "concrete"])
            )
            if mislabeled_steel.any():
                d.loc[mislabeled_steel, "steel_volume"] = concrete_volume[mislabeled_steel]
                d.loc[mislabeled_steel, "concrete_volume"] = 0.0
                d.loc[mislabeled_steel, "material_steel_id"] = "steel"
                d.loc[mislabeled_steel, "material_concrete_id"] = pd.NA

    if cfg["variant_id"] in d.columns:
        d["system_variant"] = d[cfg["variant_id"]].astype(str).str.strip()
    if cfg["family_id"] in d.columns:
        d["system_family"] = d[cfg["family_id"]].astype(str).str.strip()

    d["component"] = component
    d["component_type"] = component
    d["quantity_basis"] = cfg["basis"]

    category_col = cfg["category_col"]
    type_col = cfg["type_col"]
    if category_col in d.columns:
        d["category"] = d[category_col]
        if "typology" not in d.columns:
            d["typology"] = d[category_col]
    if type_col in d.columns:
        d["type"] = d[type_col]

    if unit_filter and "unit" in d.columns:
        def _unit_family(value: Any) -> str:
            s = str(value or "").strip().lower()
            if s in {"", "none", "nan", "na", "n/a"}:
                return "unitless"
            if s in {"metric", "m", "meter", "metre", "meters", "metres", "si", "eu"}:
                return "metric"
            if s in {"imperial", "ft", "foot", "feet", "us", "usa"}:
                return "imperial"
            return s

        requested = _unit_family(unit_filter)
        allowed = {requested, "unitless"}
        d = d[d["unit"].map(_unit_family).isin(allowed)].copy()

    for material_col, quantity_col, default_material in [
        ("material_steel_id", "steel_volume", "steel"),
        ("material_rebar_id", "rebar_volume", "rebar"),
        ("material_pt_id", "pt_volume", "pt"),
        ("material_concrete_id", "concrete_volume", "concrete"),
        ("material_timber_id", "timber_volume", "timber"),
    ]:
        if material_col not in d.columns or quantity_col not in d.columns:
            continue
        missing = d[material_col].isna() | d[material_col].astype(str).str.strip().str.lower().isin({"", "none", "nan"})
        positive_quantity = pd.to_numeric(d[quantity_col], errors="coerce").fillna(0.0).gt(0.0)
        d.loc[missing & positive_quantity, material_col] = default_material

    return d


def _interior_beam_demands(load_kpa: float, span_m: float, trib_width_m: float) -> tuple[float, float, float]:
    """Return line load, moment, and shear for an interior continuous beam line."""
    line_load = load_kpa * trib_width_m
    moment = max(
        _BEAM_SIMPLE_MOMENT_COEFF * line_load * span_m ** 2,
        _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * line_load * span_m ** 2,
    )
    shear = _BEAM_INTERIOR_SHEAR_COEFF * line_load * span_m
    return line_load, moment, shear


def _interior_column_moment_demand(load_kpa: float, span_x_m: float, span_y_m: float) -> float:
    """Approximate two-sided beam end-moment demand on an interior column."""
    if load_kpa <= 0 or span_x_m <= 0 or span_y_m <= 0:
        return 0.0
    m_x = _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * (load_kpa * span_y_m) * span_x_m ** 2
    m_y = _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * (load_kpa * span_x_m) * span_y_m ** 2
    return max(m_x, m_y)


def _rc_beam_code_check_rows(
    d: pd.DataFrame,
    *,
    span_m: float,
    line_load_kn_m: float,
    prefix: str = "",
) -> dict[str, pd.Series]:
    """Run the local EC2 RC-beam check for concrete beam rows.

    Many generated RC beam catalog rows do not carry moment_capacity values.
    Treating those blanks as a pass lets undersized beams leak through the full
    engine, so this computes a real section check from the beam geometry.
    """
    index = d.index
    out = {
        f"{prefix}rc_beam_checked": pd.Series(False, index=index),
        f"{prefix}pass_rc_beam_strength": pd.Series(True, index=index),
        f"{prefix}pass_rc_beam_deflection": pd.Series(True, index=index),
        f"{prefix}rc_beam_M_support_kNm": pd.Series(pd.NA, index=index, dtype="Float64"),
        f"{prefix}rc_beam_V_int_kN": pd.Series(pd.NA, index=index, dtype="Float64"),
        f"{prefix}rc_beam_L_over_d": pd.Series(pd.NA, index=index, dtype="Float64"),
    }
    if span_m <= 0 or line_load_kn_m <= 0 or not {"beam_width", "beam_depth"}.issubset(d.columns):
        return out

    material = (
        d["material_family"].fillna("").astype(str).str.strip().str.lower()
        if "material_family" in d.columns
        else pd.Series("", index=index)
    )
    conc_id = (
        d["material_concrete_id"].fillna("").astype(str).str.strip().str.lower()
        if "material_concrete_id" in d.columns
        else pd.Series("", index=index)
    )
    is_rc = material.eq("concrete") | conc_id.ne("") & ~conc_id.isin({"nan", "none"})
    if not is_rc.any():
        return out

    from edca_code.scripts.code_checks.rc_beam import check_rc_beam

    unit_col = d["unit"].fillna("metric").str.strip().str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)

    for idx, row in d[is_rc].iterrows():
        try:
            bw = pd.to_numeric(pd.Series([row.get("beam_width")]), errors="coerce").iloc[0]
            bd = pd.to_numeric(pd.Series([row.get("beam_depth")]), errors="coerce").iloc[0]
            if pd.isna(bw) or pd.isna(bd) or float(bw) <= 0 or float(bd) <= 0:
                continue
            _unit = unit_col.loc[idx] if idx in unit_col.index else "metric"
            if _unit == "imperial":
                b_m = float(bw) * 0.0254  # inches → metres
                h_m = float(bd) * 0.0254
            else:
                b_m = float(bw) / 1000.0 if float(bw) > 10.0 else float(bw)
                h_m = float(bd) / 1000.0 if float(bd) > 10.0 else float(bd)
            result = check_rc_beam(h_m=h_m, b_m=b_m, L_m=span_m, n_ULS=line_load_kn_m)
        except Exception:
            logger.debug("[components] RC beam code check failed for row %s", idx, exc_info=True)
            continue

        checked = bool(result)
        success = bool(result.get("success")) if checked else False
        defl = bool(result.get("deflection_pass", True)) if checked else False
        out[f"{prefix}rc_beam_checked"].loc[idx] = checked
        out[f"{prefix}pass_rc_beam_strength"].loc[idx] = success
        out[f"{prefix}pass_rc_beam_deflection"].loc[idx] = defl
        out[f"{prefix}rc_beam_M_support_kNm"].loc[idx] = result.get("M_support_kNm")
        out[f"{prefix}rc_beam_V_int_kN"].loc[idx] = result.get("V_int_kN")
        out[f"{prefix}rc_beam_L_over_d"].loc[idx] = result.get("L_over_d")
    return out


def _component_pass_flags(
    df: pd.DataFrame,
    component: str,
    cf: Any,
    span_values: List[float],
    required_loads: Dict[str, float],
    *,
    trib_width_m: float = 0.0,
    bay_area_m2: float = 0.0,
    beam_span_m: float = 0.0,
    bay_span_x_m: float = 0.0,
    moment_frame_columns: bool = False,
    steel_beam_sls_checks: bool = True,
    steel_beam_max_span_depth_ratio: float = 16.0,
    steel_secondary_beam_max_span_depth_ratio: float = 16.0,
    rc_beam_span_depth_checks: bool = True,
    rc_beam_max_span_depth_ratio: float = 12.0,
    rc_secondary_beam_max_span_depth_ratio: float = 12.0,
    fire_resistance_minima: bool = True,
    timber_beam_moment_capacity_factor: float = 0.75,
) -> pd.DataFrame:
    """Apply structural demand filters to a component catalog.

    Pass flags added per component
    ───────────────────────────────
    floor  : pass_span  — max_span >= required span (half-span allowed with secondary beam)
             pass_sdl   — catalog sdl  >= project governing SDL  (kPa)
             pass_ll    — catalog ll   >= project governing LL   (kPa)

    beam   : pass_span  — max_span >= required span
             pass_moment — moment_capacity (kNm) >= interior continuous-beam envelope
             pass_shear  — shear_capacity  (kN)  >= interior support reaction envelope
             (legacy pass_load_capacity kept only when moment/shear columns absent)

    column : pass_axial  — axial_capacity (kN) >= w × bay_area × column storeys
             pass_moment_column — moment_capacity (kNm) >= two-sided interior-frame moment
             pass_story_count — fallback when axial_capacity not populated

    lateral: pass_story_count — maximum_story_count >= storeys
    """
    d = df.copy()
    span_req = max([float(s) for s in span_values], default=0.0)
    storeys_req = int(getattr(cf, "num_floors", 0) or 0)
    column_storeys_req = max(storeys_req, 1)  # base column must support all num_floors stacks
    # factored ULS area load (kN/m²); fall back to unfactored total if absent
    load_req = float(required_loads.get("max_factored_total", required_loads.get("max_total", 0.0)) or 0.0)
    sdl_req = float(required_loads.get("max_sdl", 0.0) or 0.0)
    ll_req = float(required_loads.get("max_ll", 0.0) or 0.0)
    # tributary width for beam line-load: explicit arg or fall back to square bay
    trib = trib_width_m if trib_width_m > 0 else span_req
    # bay area for column axial demand: explicit arg or square bay
    bay = bay_area_m2 if bay_area_m2 > 0 else (span_req ** 2 if span_req > 0 else 0.0)
    M_demand = V_demand = M_sec = V_sec = None

    # ── span check (floors and beams) ──────────────────────────────────────
    if "max_span" in d.columns and span_req > 0 and component in {"floor", "beam"}:
        max_span = pd.to_numeric(d["max_span"], errors="coerce")
        if component == "floor" and beam_span_m > span_req and "beam_requirements" in d.columns:
            # Two-way beamless floors must span the full bay in both directions.
            # When the bay is rectangular (beam_span > slab_span), use the longer
            # dimension as the effective span requirement for variants that carry
            # load in both directions without support beams (beam_requirements="none").
            beamless = d["beam_requirements"].fillna("").str.strip().str.lower() == "none"
            eff_span = pd.Series(span_req, index=d.index)
            eff_span[beamless] = beam_span_m
            passes_full = max_span.isna() | (max_span >= eff_span)
            passes_half = ~passes_full & (max_span >= eff_span / 2.0)
        else:
            passes_full = max_span.isna() | max_span.ge(span_req)
            passes_half = ~passes_full & max_span.ge(span_req / 2.0) if component == "floor" else pd.Series(False, index=d.index)
        if component == "floor":
            d["pass_span"] = passes_full | passes_half
            d["needs_secondary_beam"] = passes_half.fillna(False)
            # Composite deck variants with beam_requirements="secondary" are always span-adequate:
            # secondary beams are placed at max_span intervals so the deck panel never exceeds
            # max_span, regardless of bay size.  Bypass the full/half-span check for these,
            # but leave needs_secondary_beam unchanged so full-span variants (e.g. ComFlor100
            # with max_span >= bay span) remain in the preferred pool.
            if "beam_requirements" in d.columns:
                is_sec_deck = d["beam_requirements"].fillna("").str.strip().str.lower() == "secondary"
                d.loc[is_sec_deck & max_span.gt(0), "pass_span"] = True
        else:
            d["pass_span"] = passes_full
    else:
        d["pass_span"] = True
    if "needs_secondary_beam" not in d.columns:
        d["needs_secondary_beam"] = False

    # ── floor: total load capacity (SDL + LL combined) ────────────────────
    # The catalog sdl/ll fields are the design loads under which max_span was
    # derived (i.e. span-table parameters), not independent capacity limits.
    # What matters is that the floor's total design load >= project total demand,
    # so a floor with lower SDL but higher LL allowance (or vice versa) can still
    # be valid as long as the combined total is sufficient.
    if component == "floor":
        # Use the governing total from any single load case, not the chimera of
        # independently maximised SDL and LL (which can overstate demand by ~15%).
        total_req = float(required_loads.get("max_total") or (sdl_req + ll_req))
        if "sdl" in d.columns and "ll" in d.columns and total_req > 0:
            cap_sdl = pd.to_numeric(d["sdl"], errors="coerce").fillna(0.0)
            cap_ll  = pd.to_numeric(d["ll"],  errors="coerce").fillna(0.0)
            cap_total = cap_sdl + cap_ll
            # NaN in both → assume acceptable
            both_nan = d["sdl"].isna() & d["ll"].isna()
            d["pass_total_load"] = both_nan | cap_total.ge(total_req)
        elif "ll" in d.columns and ll_req > 0:
            # Only LL column present — check LL alone
            cap = pd.to_numeric(d["ll"], errors="coerce")
            d["pass_total_load"] = cap.isna() | cap.ge(ll_req)

        # Note: no load bypass for secondary-beam-supported floors — slab panels
        # must still satisfy SDL+LL capacity even though their span check is
        # bypassed (the panel spans secondary beam spacing, not the full bay).

    # ── beam: moment and shear demand ──────────────────────────────────────
    if component == "beam" and load_req > 0 and span_req > 0 and trib > 0:
        line_load, M_demand, V_demand = _interior_beam_demands(load_req, span_req, trib)
        rc_primary = _rc_beam_code_check_rows(d, span_m=span_req, line_load_kn_m=line_load)
        for col, values in rc_primary.items():
            d[col] = values

        # Moment and shear capacities must BOTH be in kN / kNm to compare
        # against these demands.  Only apply the shear check alongside the moment
        # check — when shear_capacity exists without moment_capacity it is
        # typically stored in MPa (concrete shear stress), not kN, and would give
        # a spuriously failing comparison.
        moment_col_ok = "moment_capacity" in d.columns
        shear_col_ok  = "shear_capacity"  in d.columns and moment_col_ok

        if moment_col_ok:
            cap = pd.to_numeric(d["moment_capacity"], errors="coerce")
            d["pass_moment"] = cap.isna() | cap.ge(M_demand)
        if shear_col_ok:
            cap = pd.to_numeric(d["shear_capacity"], errors="coerce")
            d["pass_shear"] = cap.isna() | cap.ge(V_demand)
        if "rc_beam_checked" in d.columns:
            checked = d["rc_beam_checked"].fillna(False).astype(bool)
            d.loc[checked, "pass_moment"] = d.loc[checked, "pass_rc_beam_strength"].astype(bool)
            d.loc[checked, "pass_shear"] = d.loc[checked, "pass_rc_beam_strength"].astype(bool)
        if rc_beam_span_depth_checks and "beam_depth" in d.columns:
            material = (
                d["material_family"].fillna("").astype(str).str.strip().str.lower()
                if "material_family" in d.columns
                else pd.Series("", index=d.index)
            )
            depth = pd.to_numeric(d["beam_depth"], errors="coerce")
            unit_col = d["unit"].fillna("metric").str.strip().str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)
            depth_m = pd.Series(index=d.index, dtype=float)
            imp_mask = unit_col.isin(["imperial", "ft"])
            depth_m[imp_mask]  = depth[imp_mask] * 0.0254          # inches → m
            depth_m[~imp_mask] = depth[~imp_mask].where(depth[~imp_mask].le(10.0), depth[~imp_mask] / 1000.0)
            min_depth = span_req / max(float(rc_beam_max_span_depth_ratio or 12.0), 1.0)
            d["pass_rc_beam_span_depth"] = ~material.eq("concrete") | depth_m.isna() | depth_m.ge(min_depth)
        if steel_beam_sls_checks and "beam_depth" in d.columns:
            material = (
                d["material_family"].fillna("").astype(str).str.strip().str.lower()
                if "material_family" in d.columns
                else pd.Series("", index=d.index)
            )
            depth = pd.to_numeric(d["beam_depth"], errors="coerce")
            unit_col = d["unit"].fillna("metric").str.strip().str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)
            depth_m = pd.Series(index=d.index, dtype=float)
            imp_mask = unit_col.isin(["imperial", "ft"])
            depth_m[imp_mask]  = depth[imp_mask] * 0.0254
            depth_m[~imp_mask] = depth[~imp_mask].where(depth[~imp_mask].le(10.0), depth[~imp_mask] / 1000.0)
            min_depth = span_req / max(float(steel_beam_max_span_depth_ratio or 16.0), 1.0)
            d["pass_steel_beam_sls"] = ~material.eq("steel") | depth_m.isna() | depth_m.ge(min_depth)

        # Legacy path: load_capacity is a line-load capacity in kN/m.
        # Use it only when no moment/shear capacity is available.
        if "pass_moment" not in d.columns and "pass_shear" not in d.columns:
            if "load_capacity" in d.columns:
                cap = pd.to_numeric(d["load_capacity"], errors="coerce")
                # load_capacity (kN/m) must exceed the beam's actual line load demand
                d["pass_load_capacity"] = cap.isna() | cap.ge(line_load)

        # ── secondary-beam demand ────────────────────────────────────────
        # This generic component screen does not know the eventual floor system's
        # support spacing, so retain a half-bay prefilter. The assembly stage
        # re-checks the chosen secondary beam against the selected floor row's
        # actual secondary_beam_spacing_m/max_span support module.
        half_trib = trib / 2.0
        line_load_sec, M_sec, V_sec = _interior_beam_demands(load_req, span_req, half_trib)
        rc_secondary = _rc_beam_code_check_rows(
            d,
            span_m=span_req,
            line_load_kn_m=line_load_sec,
            prefix="secondary_",
        )
        for col, values in rc_secondary.items():
            d[col] = values

        if moment_col_ok:
            cap = pd.to_numeric(d["moment_capacity"], errors="coerce")
            # Apply timber capacity factor (kmod/γM) so the secondary-beam pre-filter
            # is consistent with the assembly-level _filter_beams_sw check.
            if timber_beam_moment_capacity_factor > 0 and "material_family" in d.columns:
                is_timber_beam = d["material_family"].fillna("").astype(str).str.strip().str.lower().eq("timber")
                eff_cap = cap.where(~is_timber_beam, cap * timber_beam_moment_capacity_factor)
            else:
                eff_cap = cap
            d["pass_moment_secondary"] = eff_cap.isna() | eff_cap.ge(M_sec)
        if shear_col_ok:
            cap = pd.to_numeric(d["shear_capacity"], errors="coerce")
            d["pass_shear_secondary"] = cap.isna() | cap.ge(V_sec)
        if "secondary_rc_beam_checked" in d.columns:
            checked = d["secondary_rc_beam_checked"].fillna(False).astype(bool)
            d.loc[checked, "pass_moment_secondary"] = d.loc[checked, "secondary_pass_rc_beam_strength"].astype(bool)
            d.loc[checked, "pass_shear_secondary"] = d.loc[checked, "secondary_pass_rc_beam_strength"].astype(bool)
        if rc_beam_span_depth_checks and "beam_depth" in d.columns:
            material = (
                d["material_family"].fillna("").astype(str).str.strip().str.lower()
                if "material_family" in d.columns
                else pd.Series("", index=d.index)
            )
            depth = pd.to_numeric(d["beam_depth"], errors="coerce")
            unit_col = d["unit"].fillna("metric").str.strip().str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)
            depth_m = pd.Series(index=d.index, dtype=float)
            imp_mask = unit_col.isin(["imperial", "ft"])
            depth_m[imp_mask]  = depth[imp_mask] * 0.0254
            depth_m[~imp_mask] = depth[~imp_mask].where(depth[~imp_mask].le(10.0), depth[~imp_mask] / 1000.0)
            min_depth = span_req / max(float(rc_secondary_beam_max_span_depth_ratio or 12.0), 1.0)
            d["pass_rc_beam_span_depth_secondary"] = ~material.eq("concrete") | depth_m.isna() | depth_m.ge(min_depth)
        if steel_beam_sls_checks and "beam_depth" in d.columns:
            material = (
                d["material_family"].fillna("").astype(str).str.strip().str.lower()
                if "material_family" in d.columns
                else pd.Series("", index=d.index)
            )
            depth = pd.to_numeric(d["beam_depth"], errors="coerce")
            unit_col = d["unit"].fillna("metric").str.strip().str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)
            depth_m = pd.Series(index=d.index, dtype=float)
            imp_mask = unit_col.isin(["imperial", "ft"])
            depth_m[imp_mask]  = depth[imp_mask] * 0.0254
            depth_m[~imp_mask] = depth[~imp_mask].where(depth[~imp_mask].le(10.0), depth[~imp_mask] / 1000.0)
            min_depth = span_req / max(float(steel_secondary_beam_max_span_depth_ratio or 16.0), 1.0)
            d["pass_steel_beam_sls_secondary"] = ~material.eq("steel") | depth_m.isna() | depth_m.ge(min_depth)
        if "pass_load_capacity" in d.columns:
            cap = pd.to_numeric(d.get("load_capacity"), errors="coerce")
            d["pass_load_capacity_secondary"] = cap.isna() | cap.ge(line_load_sec)

        sec_flag_cols = [c for c in [
            "pass_span", "pass_moment_secondary", "pass_shear_secondary",
            "secondary_pass_rc_beam_deflection", "pass_load_capacity_secondary",
            "pass_steel_beam_sls_secondary", "pass_rc_beam_span_depth_secondary",
        ] if c in d.columns]
        if sec_flag_cols:
            d["pass_overall_secondary"] = d[sec_flag_cols].all(axis=1)

    # ── column: axial demand ───────────────────────────────────────────────
    if component == "column":
        if "axial_capacity" in d.columns and load_req > 0 and bay > 0 and column_storeys_req > 0:
            N_demand = load_req * bay * column_storeys_req          # kN
            cap = pd.to_numeric(d["axial_capacity"], errors="coerce")

            # For rows with no recorded capacity (catalog gap), estimate from section
            # geometry using a simplified Eurocode interaction:
            #   Nrd = 0.8 × (fcd × Ac + fyd × As)
            # where fcd = 20 MPa (C30/37, γc=1.5) and fyd = 435 MPa (B500, γs=1.15).
            # concrete_volume (m³/lm) = Ac (m²); rebar_volume (m³/lm) = As (m²).
            if "concrete_volume" in d.columns:
                fcd_MPa, fyd_MPa = 20.0, 435.0   # EC2 design strengths
                Ac_m2 = pd.to_numeric(d["concrete_volume"], errors="coerce").fillna(0.0)
                As_m2 = pd.to_numeric(d.get("rebar_volume", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0.0)
                Nrd_kN = 0.8 * (fcd_MPa * Ac_m2 * 1e6 + fyd_MPa * As_m2 * 1e6) / 1e3
                cap = pd.concat([cap, Nrd_kN.where(Nrd_kN > 0)], axis=1).max(axis=1, skipna=True)

            d["pass_axial"] = cap.isna() | cap.ge(N_demand)

        M_col_demand = _interior_column_moment_demand(load_req, bay_span_x_m or span_req, trib) if moment_frame_columns else 0.0
        if moment_frame_columns and "moment_capacity" in d.columns and M_col_demand > 0:
            cap_m = pd.to_numeric(d["moment_capacity"], errors="coerce")
            d["pass_moment_column"] = cap_m.isna() | cap_m.ge(M_col_demand)

        # Fallback: storey-count proxy when axial_capacity is not populated
        if "pass_axial" not in d.columns and "maximum_story_count" in d.columns and column_storeys_req > 0:
            stories = pd.to_numeric(d["maximum_story_count"], errors="coerce")
            d["pass_story_count"] = stories.isna() | stories.ge(column_storeys_req)

    # ── lateral: storey count ──────────────────────────────────────────────
    if component == "lateral":
        if "maximum_story_count" in d.columns and storeys_req > 0:
            stories = pd.to_numeric(d["maximum_story_count"], errors="coerce")
            d["pass_story_count"] = stories.isna() | stories.ge(storeys_req)

    # ── floor: fire resistance rating ─────────────────────────────────────
    if fire_resistance_minima and component == "floor" and "floor_type" in d.columns:
        fire_period = int(getattr(cf, "fire_resistance_period", 60) or 60)
        # Minimum depth (m) by fire resistance period (min) per floor category.
        # composite: concrete above ribs (EN 1994-1-2 Table D.4)
        # rc:        overall concrete depth (EN 1992-1-2 Tables 5.8/5.9)
        # hollowcore: overall depth (EN 1992-1-2 Table 5.12)
        # timber:    residual structural depth after charring at 0.8 mm/min (EN 1995-1-2)
        _FIRE = {
            "composite": {30: 0.060, 60: 0.080, 90: 0.100, 120: 0.120, 180: 0.150, 240: 0.170},
            "rc":        {30: 0.060, 60: 0.080, 90: 0.100, 120: 0.120, 180: 0.150, 240: 0.175},
            "hollowcore":{30: 0.100, 60: 0.150, 90: 0.200, 120: 0.260, 180: 0.320, 240: 0.400},
            "timber":    {30: 0.054, 60: 0.079, 90: 0.104, 120: 0.128, 180: 0.177, 240: 0.226},
        }
        ft_col = d["floor_type"]
        unit_col = d["unit"].str.lower() if "unit" in d.columns else pd.Series("metric", index=d.index)
        pass_fire = pd.Series(True, index=d.index)

        # composite_deck: concrete above ribs stored in slab_depth (after rib_depth normalisation)
        is_comp = ft_col == "composite_deck"
        if is_comp.any() and "rib_depth" in d.columns and "slab_depth" in d.columns:
            rib = pd.to_numeric(d["rib_depth"], errors="coerce")
            above_ribs = pd.to_numeric(d["slab_depth"], errors="coerce")
            has_data = is_comp & rib.notna() & (rib > 0) & above_ribs.notna()
            thr = _FIRE["composite"].get(fire_period, 0.0)
            pass_fire = pass_fire.where(~has_data, above_ribs.ge(thr))

        # RC / precast concrete floors — depth stored in mm (metric RC slabs),
        # inches (imperial RC slabs), feet (imperial double_tee), or metres (beam_block).
        _RC_SLABS = {"flat_slab", "flat_slab_drop_panel", "pt_slab", "two_way_slab",
                     "solid_slab", "waffle_slab", "joist_slab"}
        _RC_ALL   = _RC_SLABS | {"beam_block", "double_tee"}
        is_rc = ft_col.isin(_RC_ALL)
        if is_rc.any() and "overall_depth" in d.columns:
            raw = pd.to_numeric(d["overall_depth"], errors="coerce")
            dm = raw.copy().astype(float)
            # RC slab types: metric values >2 are in mm; imperial are in inches
            slab_m = ft_col.isin(_RC_SLABS)
            dm = dm.where(~(slab_m & (unit_col == "metric") & (raw > 2.0)), raw / 1000.0)
            dm = dm.where(~(slab_m & (unit_col == "imperial")), raw * 0.0254)
            # double_tee imperial values are in feet
            dm = dm.where(~((ft_col == "double_tee") & (unit_col == "imperial")), raw * 0.3048)
            thr = _FIRE["rc"].get(fire_period, 0.0)
            pass_fire = pass_fire.where(~is_rc, dm.isna() | dm.ge(thr))

        # hollowcore: metric=metres, imperial=feet
        is_hc = ft_col == "hollowcore"
        if is_hc.any() and "overall_depth" in d.columns:
            raw = pd.to_numeric(d["overall_depth"], errors="coerce")
            dm = raw.copy().astype(float)
            dm = dm.where(~(is_hc & (unit_col == "imperial")), raw * 0.3048)
            thr = _FIRE["hollowcore"].get(fire_period, 0.0)
            pass_fire = pass_fire.where(~is_hc, dm.isna() | dm.ge(thr))

        # timber floors (CLT, LVL, solid plank): overall_depth in metres
        _TIMBER = {"clt_floor", "lvl_panel", "solid_plank", "thermal_floor"}
        is_timber = ft_col.isin(_TIMBER)
        if is_timber.any() and "overall_depth" in d.columns:
            raw = pd.to_numeric(d["overall_depth"], errors="coerce")
            thr = _FIRE["timber"].get(fire_period, 0.0)
            pass_fire = pass_fire.where(~is_timber, raw.isna() | raw.ge(thr))

        d["pass_fire"] = pass_fire

    # ── combine ────────────────────────────────────────────────────────────
    _pass_flag_cols = [
        "pass_span", "pass_total_load",
        "pass_moment", "pass_shear", "pass_load_capacity", "pass_steel_beam_sls",
        "pass_rc_beam_deflection", "pass_rc_beam_span_depth",
        "pass_axial", "pass_moment_column", "pass_story_count", "pass_fire",
    ]
    pass_cols = [c for c in _pass_flag_cols if c in d.columns]
    d["pass_overall"] = d[pass_cols].all(axis=1) if pass_cols else True

    # Traceability columns
    # For column components in irregular-bay mode, span_req comes from the generic
    # sweep list and does not reflect the actual beam span. Use bay_span_x_m when
    # provided so demand_span_m shows the real beam span rather than the sweep value.
    d["demand_span_m"] = bay_span_x_m if (component == "column" and bay_span_x_m > 0) else span_req
    d["demand_trib_width_m"] = trib
    d["demand_factored_load_kpa"] = load_req
    if component == "floor":
        d["demand_sdl_kpa"] = sdl_req
        d["demand_ll_kpa"] = ll_req
        d["demand_total_unfactored_kpa"] = sdl_req + ll_req
    if component == "beam" and span_req > 0 and trib > 0:
        _ll = load_req * trib
        d["demand_line_load_kNm"]      = _ll
        d["demand_M_kNm"]              = M_demand if M_demand is not None else _ll * span_req ** 2 / 8.0
        d["demand_V_kN"]               = V_demand if V_demand is not None else _ll * span_req / 2.0
        d["demand_M_secondary_kNm"]    = M_sec if M_sec is not None else (_ll / 2.0) * span_req ** 2 / 8.0
        d["demand_V_secondary_kN"]     = V_sec if V_sec is not None else (_ll / 2.0) * span_req / 2.0
    if component == "column" and bay > 0:
        d["demand_N_kN"] = load_req * bay * column_storeys_req
        d["demand_M_column_kNm"] = (
            _interior_column_moment_demand(load_req, bay_span_x_m or span_req, trib)
            if moment_frame_columns
            else 0.0
        )
    d["demand_storeys"] = column_storeys_req if component == "column" else storeys_req
    return d


_LINEAR_M_VOLUME_COLS = [
    "concrete_volume", "steel_volume", "rebar_volume", "pt_volume", "timber_volume",
]
_LINEAR_M_MASS_COLS = [
    "steel_mass", "steel_kg", "rebar_mass", "rebar_kg", "pt_mass", "pt_kg",
]


def _scale_row_to_per_m2(row: pd.Series, span_x_m: float, span_y_m: float, storey_height_m: float) -> pd.Series:
    """Return a copy of row with per-linear-m quantities scaled to per-m² floor area.

    For a rectangular bay of span_x_m × span_y_m:
      Primary beam (spans x): one beam per bay → volume/m² = volume/lm × span_x / (span_x × span_y)
                                                             = volume/lm / span_y
      Column:  one per bay → volume/m² = volume/lm × storey_height / (span_x × span_y)

    For square bays (span_x == span_y) this is identical to the old formula.
    The basis is stored in the 'quantity_basis' field of each row.

    Imperial note:
      Imperial catalog volumes are in ft³/ft (per linear foot).  span_x_m/span_y_m and
      storey_height_m are always in metres (after parse.py converts the control file).
      To keep the result in ft³/ft² (depth-equivalent) — which takeoff.py then multiplies
      by _FT2_PER_M2 to produce ft³/m² for carbon.py — the scale factor must use feet, not
      metres.  We apply a 0.3048 correction:
        column beam scale_ft = scale_m × 0.3048   (since ft/ft² = m/m² × 0.3048)
    """
    basis = str(row.get("quantity_basis", "")).strip()
    if basis != "per_linear_m" or span_x_m <= 0 or span_y_m <= 0:
        return row

    bay_area = span_x_m * span_y_m
    component = str(row.get("component", "")).strip()
    _row_unit = str(row.get("unit", "metric") or "metric").strip().lower()
    _is_imperial = _row_unit in ("imperial", "ft")

    if component == "beam":
        scale = 1.0 / span_y_m
        if _is_imperial:
            scale *= 0.3048   # 1/span_y_ft = 0.3048/span_y_m
    elif component == "column":
        scale = storey_height_m / bay_area
        if _is_imperial:
            scale *= 0.3048   # h_ft/area_ft2 = h_m×0.3048 / area_m2
    else:
        return row

    scaled = row.copy()
    for col in _LINEAR_M_VOLUME_COLS + _LINEAR_M_MASS_COLS:
        if col in scaled.index:
            v = scaled[col]
            try:
                scaled[col] = float(v) * scale if v and not pd.isna(v) else v
            except (TypeError, ValueError):
                pass
    return scaled


_REBAR_DETAILING_MULT_BEAM  = 1.30   # uplift for laps, continuity bars, anchorage tails
_REBAR_DETAILING_MULT_COL   = 1.60   # uplift for column laps, splice zones, bar rounding,
                                      # beam-column joint bars (anchor tails 40φ), and
                                      # storey-height lap-zone doubling (raised from 1.35).
                                      # EC2 structural+links × ~1.33 detailing factor.
_COLUMN_LINK_MULT           = 0.20   # links ≈ 20 % of longitudinal (4-leg stirrups)
_COLUMN_PRACTICAL_MIN_RHO   = 0.01   # 1 % ρ — constructability / minimum bar-count floor
_COLUMN_MIN_BARS_MM2        = 804.0  # 4 × T16 — absolute minimum 4-bar arrangement
_CONCRETE_UNIT_WEIGHT_KN_M3 = 25.0


def _secondary_support_spacing_from_row(row: pd.Series, span_x_m: float) -> tuple[float | None, str | None]:
    """Return structural support spacing for floor systems that need secondary beams."""
    def _float_field(name: str) -> float:
        try:
            val = row.get(name)
            if val is None or pd.isna(val):
                return 0.0
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    explicit = _float_field("secondary_beam_spacing_m")
    if explicit > 0:
        return explicit, "secondary_beam_spacing_m"

    max_span = _float_field("max_span")
    if max_span > 0:
        return min(max_span, 4.5), "max_span_capped_4.5m"

    if span_x_m > 0:
        return span_x_m, "bay_span_fallback"
    return None, None


def _evaluate_component_carbon(
    df: pd.DataFrame,
    materials_df: pd.DataFrame,
    span_x_m: float = 0.0,
    span_y_m: float = 0.0,
    storey_height_m: float = 0.0,
    required_loads: Optional[Dict[str, float]] = None,
    trib_width_m: float = 0.0,
    storey_count: int = 1,
    moment_frame_columns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from edca_code.scripts.code_checks.rc_beam   import check_rc_beam
    from edca_code.scripts.code_checks.rc_column import check_rc_column

    _sdl      = float((required_loads or {}).get("max_sdl",           0.0) or 0.0)
    _ll       = float((required_loads or {}).get("max_ll",            0.0) or 0.0)
    _factored = float((required_loads or {}).get("max_factored_total", 0.0) or 0.0)
    _trib     = trib_width_m if trib_width_m > 0 else span_y_m
    _bay_area = span_x_m * span_y_m

    def _get_material_props(mat_conc_id: str, mat_rebar_id: str):
        f_ck, f_yk = 30.0, 500.0
        try:
            f_ck = float(materials_df.loc[mat_conc_id].get("concrete_f_ck", 30.0) or 30.0)
        except Exception:
            pass
        try:
            f_yk = float(materials_df.loc[mat_rebar_id].get("steel_fy", 500.0) or 500.0)
        except Exception:
            pass
        return f_ck, f_yk

    rows: List[Dict[str, Any]] = []
    takeoff_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        out = row.to_dict()
        row = row.copy()
        _component = str(row.get("component", "")).strip()

        # ── RC beam code-check rebar override ──────────────────────────────────
        # Compute EC2 structurally-required rebar (longitudinal + links), apply a
        # detailing multiplier for laps/anchorage, then keep the larger of the
        # catalog and code-check quantities.
        if (
            _component == "beam"
            and str(row.get("material_concrete_id", "") or "").strip()
            and span_x_m > 0 and _trib > 0 and (_sdl > 0 or _ll > 0)
        ):
            try:
                _bw = row.get("beam_width")
                _bd = row.get("beam_depth")
                _row_unit = str(row.get("unit", "metric") or "metric").strip().lower()
                if _row_unit == "imperial":
                    _b_m = float(_bw) * 0.0254 if _bw and not pd.isna(_bw) else None
                    _h_m = float(_bd) * 0.0254 if _bd and not pd.isna(_bd) else None
                else:
                    _b_m = float(_bw) / 1000.0 if _bw and not pd.isna(_bw) else None
                    _h_m = float(_bd) / 1000.0 if _bd and not pd.isna(_bd) else None
                if _b_m and _h_m and _b_m > 0 and _h_m > 0:
                    _f_ck, _f_yk = _get_material_props(
                        str(row.get("material_concrete_id", "") or "").strip(),
                        str(row.get("material_rebar_id",   "") or "").strip(),
                    )
                    _cc = check_rc_beam(
                        h_m=_h_m, b_m=_b_m, L_m=span_x_m,
                        g_k=(_sdl * _trib) + (_b_m * _h_m * _CONCRETE_UNIT_WEIGHT_KN_M3),
                        q_k=_ll * _trib,
                        f_ck_MPa=_f_ck, f_yk_MPa=_f_yk,
                    )
                    if _cc.get("success"):
                        _cc_rebar = (
                            float(_cc.get("rebar_volume_m3_per_m", 0.0) or 0.0)
                            + float(_cc.get("link_volume_m3_per_m",  0.0) or 0.0)
                        ) * _REBAR_DETAILING_MULT_BEAM
                        _cat = float(row.get("rebar_volume", 0.0) or 0.0)
                        if _cc_rebar > 0:
                            row["rebar_volume"] = max(_cat, _cc_rebar) if _cat > 0 else _cc_rebar
            except Exception:
                pass  # fall back to catalog rebar unchanged

        # ── RC column code-check rebar override ────────────────────────────────
        # Cumulative axial demand = factored area load × bay area × number of storeys
        # above (worst case = ground-floor column carrying all floors).
        elif (
            _component == "column"
            and str(row.get("material_concrete_id", "") or "").strip()
            and _bay_area > 0 and _factored > 0 and storey_count > 0
        ):
            try:
                _cw = row.get("column_width")
                _cd = row.get("column_depth")
                _row_unit = str(row.get("unit", "metric") or "metric").strip().lower()
                if _row_unit == "imperial":
                    _b_m = float(_cw) * 0.0254 if _cw and not pd.isna(_cw) else None
                    _h_m = float(_cd) * 0.0254 if _cd and not pd.isna(_cd) else None
                else:
                    _b_m = float(_cw) / 1000.0 if _cw and not pd.isna(_cw) else None
                    _h_m = float(_cd) / 1000.0 if _cd and not pd.isna(_cd) else None
                if _b_m and _h_m and _b_m > 0 and _h_m > 0:
                    _f_ck, _f_yk = _get_material_props(
                        str(row.get("material_concrete_id", "") or "").strip(),
                        str(row.get("material_rebar_id",   "") or "").strip(),
                    )
                    _N_Ed = _factored * _bay_area * storey_count  # kN cumulative
                    _M_02 = (
                        _interior_column_moment_demand(_factored, span_x_m, span_y_m)
                        if moment_frame_columns
                        else 0.0
                    )
                    _clear_h = storey_height_m if storey_height_m > 0 else 3.5
                    _cc = check_rc_column(
                        h_m=_h_m, b_m=_b_m,
                        clear_height_m=_clear_h,
                        N_Ed_kN=_N_Ed,
                        M_02_kNm=_M_02,
                        f_ck_MPa=_f_ck, f_yk_MPa=_f_yk,
                    )
                    if _cc.get("success"):
                        _cc_long = float(_cc.get("rebar_volume_m3_per_m", 0.0) or 0.0)
                        # Practical constructability floor: 1 % ρ or 4-bar minimum,
                        # whichever is larger. EC2 code minimum is often below what
                        # any real engineer would detail on site.
                        _Ac_mm2 = (_h_m * 1000) * (_b_m * 1000)
                        _As_prac_min = max(
                            _COLUMN_PRACTICAL_MIN_RHO * _Ac_mm2,
                            _COLUMN_MIN_BARS_MM2,
                        ) / 1e6   # m³/m
                        _cc_long = max(_cc_long, _As_prac_min)
                        # Use the explicit link/tie steel from the column
                        # check when it exceeds the fallback percentage.
                        _cc_links = max(
                            float(_cc.get("link_rebar_volume_m3_per_m", 0.0) or 0.0),
                            _cc_long * _COLUMN_LINK_MULT,
                        )
                        _cc_rebar = (_cc_long + _cc_links) * _REBAR_DETAILING_MULT_COL
                        _cat = float(row.get("rebar_volume", 0.0) or 0.0)
                        if _cc_rebar > 0:
                            row["rebar_volume"] = max(_cat, _cc_rebar) if _cat > 0 else _cc_rebar
            except Exception:
                pass  # fall back to catalog rebar unchanged

        scaled_row = _scale_row_to_per_m2(row, span_x_m, span_y_m, storey_height_m)
        bom = takeoff_mod.bom_per_m2_from_system_row(scaled_row)
        carbon = carbon_mod.compute_assembly_carbon_from_bom(bom, materials_df)
        totals = carbon.get("totals", {})
        by_cat = carbon.get("totals_by_category", {})

        # Store evaluation spans and secondary beam count for use in assembly_summary.
        # n_secondary_beams is the number of intermediate support lines inside a bay.
        out["eval_span_x_m"] = span_x_m
        out["eval_span_y_m"] = span_y_m
        if row.get("component") == "floor" and span_x_m > 0:
            import math as _math
            _spacing, _spacing_source = _secondary_support_spacing_from_row(row, span_x_m)
            _beam_req = str(row.get("beam_requirements", "") or "").strip().lower()
            _needs_secondary = _beam_req == "secondary" or bool(row.get("needs_secondary_beam", False))
            if _needs_secondary and _spacing and _spacing > 0:
                out["n_secondary_beams"] = max(0, _math.ceil(span_x_m / _spacing) - 1)
            else:
                out["n_secondary_beams"] = 0
            out["secondary_beam_spacing_used_m"] = _spacing
            out["secondary_beam_spacing_source"] = _spacing_source

        out["carbon_total_kgCO2"] = float(totals.get("total", 0.0) or 0.0)
        out["carbon_total_per_m2"] = out["carbon_total_kgCO2"]
        out["carbon_per_m2"] = out["carbon_total_kgCO2"]
        out["carbon_concrete_per_m2"] = float(by_cat.get("concrete", 0.0) or 0.0)
        out["carbon_steel_per_m2"] = float(by_cat.get("steel", 0.0) or 0.0)
        out["carbon_structural_steel_per_m2"] = float(by_cat.get("structural_steel", 0.0) or 0.0)
        out["carbon_rebar_per_m2"] = float(by_cat.get("rebar", 0.0) or 0.0)
        out["carbon_pt_per_m2"] = float(by_cat.get("pt", 0.0) or 0.0)
        out["carbon_timber_per_m2"] = float(by_cat.get("timber", 0.0) or 0.0)
        out["carbon_screed_per_m2"] = float(by_cat.get("screed", 0.0) or 0.0)

        # Physical quantities from BOM — volume in m³/m² GFA, mass in kg/m² GFA.
        # Parses each BOM entry so that mass-based entries (e.g. composite steel deck stored
        # as steel_kg) give the correct volume via volume = mass / density, and volumetric
        # entries (e.g. concrete_volume) give mass via mass = volume × density.
        #
        # Unit handling:
        #   Imperial catalog rows store concrete/rebar/steel volumes in ft³/ft² (floors)
        #   or ft³/ft (columns).  takeoff._FT2_PER_M2 scaling produces ft³/m² in the BOM.
        #   We convert to m³/m² using _FT3_TO_M3 for those categories.
        #   PT and screed are EXEMPT — PT vol is stored as m³/m² (vol_scale=1.0 in takeoff),
        #   and screed depth is always in metres.  Timber is also metric in this catalog.
        #   Material densities for imperial materials are in lb/ft³; convert to kg/m³
        #   via _LB_FT3_TO_KG_M3 so that m³/m² × kg/m³ = kg/m².
        _FT3_TO_M3       = 0.028317   # 1 ft³ = 0.028317 m³
        _LB_FT3_TO_KG_M3 = 16.0185   # 1 lb/ft³ = 16.0185 kg/m³
        # Row unit drives volume conversion; "ft" catalogs (AISC sections) are imperial.
        _row_is_imperial = str(row.get("unit", "metric") or "metric").strip().lower() in ("imperial", "ft")
        # Only these categories received the _FT2_PER_M2 upscale in takeoff.py — they
        # need the inverse conversion here.  pt/screed/timber are always metric m³/m².
        _IMPERIAL_VOL_CATS = {"concrete", "structural_steel", "rebar"}

        _BOM_CAT_MAP = {
            "concrete": "concrete", "steel": "structural_steel",
            "rebar": "rebar", "pt": "pt", "timber": "timber", "screed": "screed",
        }
        for _c in ["concrete", "structural_steel", "rebar", "pt", "timber", "screed"]:
            out[f"{_c}_volume_per_m2"] = 0.0
        for _c in ["structural_steel", "rebar", "pt", "timber"]:
            out[f"{_c}_mass_per_m2"] = 0.0

        for _key, _qty_raw in bom.items():
            if ":" not in str(_key):
                continue
            _raw_cat, _mat_id = str(_key).split(":", 1)
            _qty = float(_qty_raw or 0.0)
            if _qty <= 0.0:
                continue
            if _raw_cat.endswith("_kg"):
                _cat_base, _unit = _raw_cat[:-3], "kg"
            elif _raw_cat.endswith("_m3"):
                _cat_base, _unit = _raw_cat[:-3], "m3"
            else:
                _cat_base, _unit = _raw_cat, "m3"
            _out_cat = _BOM_CAT_MAP.get(_cat_base, _cat_base)
            _vol_col  = f"{_out_cat}_volume_per_m2"
            _mass_col = f"{_out_cat}_mass_per_m2"
            _density_raw, _mat_unit_str = 0.0, "metric"
            try:
                _mat_row      = materials_df.loc[_mat_id]
                _density_raw  = float(_mat_row.get("density", 0.0) or 0.0)
                _mat_unit_str = str(_mat_row.get("unit", "metric") or "metric").strip().lower()
            except Exception:
                pass
            # Determine density in kg/m³ based on the material's declared unit.
            # lb/ft³ → kg/m³; kg/m³ stays as-is.  (No kN→kg conversion: densities
            # in the materials CSV are already in the units declared by the "unit" column.)
            _density_kg_m3 = (
                _density_raw * _LB_FT3_TO_KG_M3
                if _mat_unit_str == "imperial" and _density_raw > 0
                else _density_raw
            )
            if _unit == "kg":
                # Mass-based BOM entry: quantity already in kg (m³ path not needed).
                if _mass_col in out:
                    out[_mass_col] += _qty
                if _density_kg_m3 > 0 and _vol_col in out:
                    out[_vol_col] += _qty / _density_kg_m3  # m³/m²
            else:  # volumetric
                # Imperial rows: concrete/rebar/steel BOM carries ft³/m²; convert to m³/m².
                # pt, screed, timber are exempt — they are always stored in m³/m².
                _needs_vol_conv = _row_is_imperial and _out_cat in _IMPERIAL_VOL_CATS
                _qty_vol = _qty * _FT3_TO_M3 if _needs_vol_conv else _qty
                if _vol_col in out:
                    out[_vol_col] += _qty_vol
                if _density_kg_m3 > 0 and _mass_col in out:
                    out[_mass_col] += _qty_vol * _density_kg_m3

        rows.append(out)

        for item in carbon.get("per_material", []):
            takeoff_rows.append({
                "component": out.get("component"),
                "component_type": out.get("component_type"),
                "system_variant": out.get("system_variant"),
                "system_family": out.get("system_family"),
                "quantity_basis": out.get("quantity_basis"),
                **item,
            })

    return pd.DataFrame(rows), pd.DataFrame(takeoff_rows)


def evaluate_and_write_component_catalogs(
    *,
    canonical_dir: Path,
    out_dir: Path,
    unit_filter: Optional[str],
    cf: Any,
    span_values: List[float],
    required_loads: Dict[str, float],
    materials_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    component_root = utils_mod.ensure_dir(Path(out_dir) / "components")
    evaluated_parts: List[pd.DataFrame] = []
    takeoff_parts: List[pd.DataFrame] = []

    _irregular = getattr(cf, "one_way_irregular", False)
    _beam_span_override = float(getattr(cf, "one_way_beam_min_span", None) or 0.0) or None
    _slab_span_override = float(getattr(cf, "one_way_slab_min_span", None) or 0.0) or None

    for component in COMPONENT_TABLES:
        catalog = _load_component_catalog(
            canonical_dir=canonical_dir,
            component=component,
            unit_filter=unit_filter,
        )
        comp_dir = utils_mod.ensure_dir(component_root / component)
        if catalog.empty:
            logger.warning("[components] %s: no canonical variants found.", component)
            pd.DataFrame().to_csv(comp_dir / f"candidates_{component}.csv", index=False)
            continue

        if _irregular and component == "beam" and _beam_span_override:
            comp_span_values = [_beam_span_override]
        elif _irregular and component == "floor" and _slab_span_override:
            comp_span_values = [_slab_span_override]
        else:
            comp_span_values = span_values

        storey_height_m = float(getattr(cf, "floor_to_floor_height", 0.0) or 0.0)

        # Determine the two bay dimensions for scaling per-m² quantities and
        # for structural demand checks (beam tributary width, column bay area).
        # For a rectangular (one-way irregular) bay:
        #   beam spans beam_span (x);  perpendicular/tributary is slab_span (y)
        #   column tributary area = beam_span × slab_span
        # For a square bay both are the same sweep value.
        _sq = max([float(s) for s in comp_span_values], default=0.0)
        if _irregular and _beam_span_override and _slab_span_override:
            if component == "beam":
                span_x_m, span_y_m = float(_beam_span_override), float(_slab_span_override)
            elif component == "floor":
                span_x_m = span_y_m = float(_slab_span_override)
            else:
                span_x_m, span_y_m = float(_beam_span_override), float(_slab_span_override)
        else:
            span_x_m = span_y_m = _sq

        catalog = _component_pass_flags(
            catalog, component, cf, comp_span_values, required_loads,
            trib_width_m=span_y_m,              # perpendicular span → beam tributary width
            bay_area_m2=span_x_m * span_y_m,   # full bay area → column axial demand
            beam_span_m=float(_beam_span_override or 0.0),  # for two-way beamless floor span check
            bay_span_x_m=span_x_m,             # actual beam span for column traceability
            moment_frame_columns=bool(getattr(cf, "moment_frame_columns", False)),
            steel_beam_sls_checks=bool(getattr(cf, "steel_beam_sls_checks", True)),
            steel_beam_max_span_depth_ratio=float(getattr(cf, "steel_beam_max_span_depth_ratio", 16.0) or 16.0),
            steel_secondary_beam_max_span_depth_ratio=float(getattr(cf, "steel_secondary_beam_max_span_depth_ratio", 16.0) or 16.0),
            rc_beam_span_depth_checks=bool(getattr(cf, "rc_beam_span_depth_checks", True)),
            rc_beam_max_span_depth_ratio=float(getattr(cf, "rc_beam_max_span_depth_ratio", 12.0) or 12.0),
            rc_secondary_beam_max_span_depth_ratio=float(getattr(cf, "rc_secondary_beam_max_span_depth_ratio", 12.0) or 12.0),
            fire_resistance_minima=bool(getattr(cf, "fire_resistance_minima", True)),
            timber_beam_moment_capacity_factor=float(getattr(cf, "timber_beam_moment_capacity_factor", 0.75) or 0.75),
        )

        evaluated, takeoffs = _evaluate_component_carbon(
            catalog, materials_df,
            span_x_m=span_x_m, span_y_m=span_y_m, storey_height_m=storey_height_m,
            required_loads=required_loads, trib_width_m=span_y_m,
            storey_count=(
                max(int(getattr(cf, "num_floors", 1) or 1), 1)
                if component == "column"
                else int(getattr(cf, "num_floors", 1) or 1)
            ),
            moment_frame_columns=bool(getattr(cf, "moment_frame_columns", False)),
        )
        evaluated.to_csv(comp_dir / f"candidates_{component}.csv", index=False)

        if not takeoffs.empty:
            takeoffs.to_csv(comp_dir / f"takeoff_{component}_per_variant.csv", index=False)
            takeoff_parts.append(takeoffs)

        if component == "lateral":
            wall_cols = [c for c in ["type", "category", "lateral_type_detail"] if c in evaluated.columns]
            if wall_cols:
                wall_mask = (
                    evaluated[wall_cols]
                    .astype(str)
                    .apply(lambda s: s.str.contains("wall", case=False, na=False))
                    .any(axis=1)
                )
                walls = evaluated.loc[wall_mask].copy()
                wall_dir = utils_mod.ensure_dir(component_root / "wall")
                walls.to_csv(wall_dir / "candidates_wall.csv", index=False)
                if not takeoffs.empty and "system_variant" in takeoffs.columns:
                    takeoffs[takeoffs["system_variant"].isin(walls["system_variant"])].to_csv(
                        wall_dir / "takeoff_wall_per_variant.csv",
                        index=False,
                    )
                logger.info("[components] wall: wrote %d lateral wall variants -> %s", len(walls), wall_dir / "candidates_wall.csv")

        try:
            rank_mod.rank_and_export_summary(
                evaluated,
                out_dir=comp_dir,
                file_prefix="summary",
                carbon_col="carbon_total_kgCO2",
                type_col="system_family",
                brand_col="system_variant",
                material_col=("material_family" if "material_family" in evaluated.columns else None),
                logger=logger,
            )
        except Exception:
            logger.exception("[components] %s: failed writing ranked summaries", component)

        evaluated_parts.append(evaluated.dropna(axis=1, how="all"))
        logger.info("[components] %s: evaluated %d variants -> %s", component, len(evaluated), comp_dir / f"candidates_{component}.csv")

    if evaluated_parts:
        all_components = pd.concat(evaluated_parts, ignore_index=True, sort=False)
    else:
        all_components = pd.DataFrame()
    all_components.to_csv(component_root / "component_candidates_all.csv", index=False)

    if takeoff_parts:
        pd.concat(takeoff_parts, ignore_index=True, sort=False).to_csv(component_root / "component_takeoffs_all.csv", index=False)

    return all_components


def _merge_variant_files(paths: list, out_dir: Path, *, unit: str | None = None) -> str:
    """Merge multiple per-component variant parquets into one combined file.

    Each per-component file may use a component-specific ID column
    (e.g. ``floor_variant_id``, ``beam_variant_id``).  These are all
    normalised to ``system_variant`` / ``system_family`` so the rest of
    the pipeline sees a single consistent schema.

    If ``unit`` is provided and the merged file has no ``unit`` column (or it
    is all-NaN), the column is backfilled with that value so downstream
    ``filter_systems`` calls don't eliminate every row.
    """
    import pandas as pd

    _ID_ALIASES = [
        "floor_variant_id", "beam_variant_id", "column_variant_id", "lateral_variant_id",
        "floor_family_id",  "beam_family_id",  "column_family_id",  "lateral_family_id",
    ]
    _FAMILY_ALIASES = [
        "floor_family_id", "beam_family_id", "column_family_id", "lateral_family_id",
        "floor_family",    "beam_family",    "column_family",    "lateral_family",
    ]

    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        # Normalise variant ID column → system_variant
        if "system_variant" not in df.columns:
            for alias in _ID_ALIASES:
                if alias in df.columns:
                    df = df.rename(columns={alias: "system_variant"})
                    break

        # Normalise family ID column → system_family
        if "system_family" not in df.columns:
            for alias in _FAMILY_ALIASES:
                if alias in df.columns and alias != "system_variant":
                    df = df.rename(columns={alias: "system_family"})
                    break

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Fill unit column so filter_systems doesn't eliminate every row.
    # Per-component parquets typically don't carry a 'unit' column; we inject
    # it from the control file so downstream filtering still works.
    if unit is not None:
        if "unit" not in merged.columns or merged["unit"].isna().all():
            merged["unit"] = unit
        else:
            merged["unit"] = merged["unit"].fillna(unit)

    # Write as CSV — avoids pyarrow type-inference errors that arise when the
    # same column is stored as string in one component file and float in another
    # (e.g. timber_volume stored as "0" in a beam parquet vs. float in floor).
    out_path = out_dir / "_merged_system_variants.csv"
    merged.to_csv(out_path, index=False)
    return str(out_path)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="run_edca", description="Run EDCA pipeline.")
    p.add_argument("--control", "-c", required=True, help="Path to control file YAML")
    p.add_argument("--systems", "-s", required=True, nargs="+",
                   help="Path(s) to variant parquet/csv files. Pass one combined file or all four "
                        "component files: floor_variants, beam_variants, column_variants, lateral_variants.")
    p.add_argument("--materials", "-m", required=True, help="Path to materials CSV")
    p.add_argument("--occupancies", "-f", required=True, help="Path to occupancies CSV")
    p.add_argument("--out", "-o", default="edca_outputs", help="Output directory")

    # span sweep options used by spans.resolve_span_values
    p.add_argument("--span-step", type=float, default=0.5, help="Span sweep step (m)")
    p.add_argument("--span-min", type=float, default=None, help="Override minimum span (m)")
    p.add_argument("--span-max", type=float, default=None, help="Override maximum span (m)")
    p.add_argument("--no-sweep", action="store_true", help="Do not perform span sweep")

    p.add_argument("--depth-limit-mm", type=float, default=None, help="Optional depth limit (mm)")
    p.add_argument("--run-codechecks", action="store_true", help="Run code checks on winners only")
    p.add_argument("--codechecks-debug-inputs", action="store_true",
                   help="Log and attach the exact inputs/material properties used by code checks.")
    p.add_argument("--codechecks-debug-all", action="store_true",
                   help="If set, print codecheck inputs for ALL rows (otherwise only failures).")
    p.add_argument("--codechecks-debug-max-rows", type=int, default=50,
                   help="Cap how many debug rows are logged/attached (default 50).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # --- NEW debug flags ---
    p.add_argument("--debug", action="store_true", help="Enable comprehensive debug logging (also EDCA_DEBUG=1).")
    p.add_argument("--debug-max-rows", type=int, default=25, help="Max rows to print when dumping debug tables.")
    p.add_argument("--debug-save-snapshots", action="store_true",
                   help="Write debug snapshots (CSV) into <out>/debug/.")

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    out_dir = utils_mod.ensure_dir(Path(args.out))

    cf = ControlFile.from_path(args.control)
    default_code_standard_if_missing(cf)

    # Resolve --systems: may be one combined file or multiple per-component files.
    # Normalise to a single merged DataFrame path stored in a temp attribute so
    # the rest of the pipeline sees a single path string.
    systems_paths = [Path(s) for s in args.systems]
    if len(systems_paths) == 1:
        _systems_resolved_path = str(systems_paths[0])
        _canonical_dir = systems_paths[0].parent
    else:
        _systems_resolved_path = _merge_variant_files(
            systems_paths, out_dir, unit=getattr(cf, "unit", None)
        )
        _canonical_dir = systems_paths[0].parent  # all files share the same directory

    logger.info("[parse] Starting EDCA run. Outputs -> %s", out_dir)
    logger.info("[parse] Loaded control file: project=%s, unit=%s, code_standard=%s",
                getattr(cf, "project_name", None), getattr(cf, "unit", None), getattr(cf, "code_standard", None))

    try:
        _params = parameters_from_control_file(args.control)
        logger.debug("[parse] control file keys: %s", sorted(list(_params.keys())))
    except Exception:
        logger.debug("[parse] parameters_from_control_file failed; continuing", exc_info=True)

    dbg = bool(getattr(args, "debug", False))
    dbg_rows = int(getattr(args, "debug_max_rows", 25) or 25)
    dbg_snap = bool(getattr(args, "debug_save_snapshots", False))

    _dbg_kv("parse.controlfile.highlevel", {
        "project": getattr(cf, "project_name", None),
        "unit": getattr(cf, "unit", None),
        "code_standard": getattr(cf, "code_standard", None),
        "num_floors": getattr(cf, "num_floors", None),
        "area_per_floor": getattr(cf, "area_per_floor", None),
        "depth_limit_enabled": getattr(cf, "depth_limit_enabled", None),
        "depth_limit": getattr(cf, "depth_limit", None),
        "spans": getattr(cf, "spans", None),
    }, explicit=dbg, level=logging.INFO)

    # Load systems + materials
    systems_df, _families_df, _variants_df = systems_mod.load_systems_catalog(
        _systems_resolved_path,
        unit_filter=getattr(cf, "unit", None),
    )
    materials_df = carbon_mod.load_materials_table(args.materials)

    _dbg_df("systems_df.loaded", systems_df, explicit=dbg, max_rows=dbg_rows,
            cols=["system_variant","system_family","unit","max_span","sdl","sdl_partition","ll","slab_depth","screed_depth"])
    _dbg_df("materials_df.loaded", materials_df.reset_index(drop=True) if hasattr(materials_df, "reset_index") else materials_df,
            explicit=dbg, max_rows=dbg_rows)
    if dbg_snap:
        _dbg_snapshot_df(systems_df, "systems_df_loaded", out_dir, enabled=True, max_rows=dbg_rows)

    # Spans — resolved early so the floor SW lookup below can use the design span.
    span_values = spans_mod.resolve_span_values(cf, args, logger=logger)
    logger.info("[span] Evaluating spans: %s", span_values)

    # Loads context (auto-finds load_values.yaml / load_combinations.yaml next to occupancies.csv)
    # Structural self-weight (SW) is handled at the assembly stage: build_assembly_rankings()
    # re-filters beam and column candidates using each floor's own swt (kN/m²) read from the
    # catalog row, giving a per-floor-type SW-adjusted ULS demand rather than a single global
    # approximation.  No SW injection here.
    required_loads_global, floors_by_case, loads_df_floor = loads_mod.build_load_context(
        cf,
        args.occupancies,
        load_values_yaml=None,
        load_combinations_yaml=None,
        debug=dbg,
    )

    _dbg_kv("loads.required_loads_global", required_loads_global, explicit=dbg, level=logging.INFO)
    _dbg_kv("loads.floors_by_case", {k: v for k, v in floors_by_case.items()}, explicit=dbg, level=logging.INFO)
    _dbg_df("loads.loads_df_floor", loads_df_floor, explicit=dbg, max_rows=dbg_rows)

    if dbg_snap:
        _dbg_snapshot_df(loads_df_floor, "loads_df_floor", out_dir, enabled=True, max_rows=dbg_rows)

    # Cases from occupancy categories
    loads_df_cases = collapse_floor_loads_to_cases(loads_df_floor, str(getattr(cf, "code_standard", "Default")))
    if loads_df_cases is None or loads_df_cases.empty:
        logger.warning("[loads] No case loads derived; using single 'global' case.")
        loads_df_cases = pd.DataFrame([{
            "load_case": "global",
            "raw_sdl": float(required_loads_global.get("max_sdl", 0.0) or 0.0),
            "raw_ll": float(required_loads_global.get("max_ll", 0.0) or 0.0),
            "factored_total": float(required_loads_global.get("max_factored_total", 0.0) or 0.0),
            "unit": getattr(cf, "unit", None),
        }])

    # Floor areas for reporting/expansion
    floor_area_lookup, _ = build_floor_area_lookup(
        getattr(cf, "area_per_floor", 0.0) or 0.0,
        floors_by_case=floors_by_case,
    )

    _dbg_kv("span.span_values", {"n": len(span_values), "values": span_values[:50]}, explicit=dbg, level=logging.INFO)

    # Component-aware catalog outputs. The legacy span sweep below remains the
    # floor-specific path, but this writes evaluated floor/beam/column/lateral
    # candidate, summary, and takeoff outputs from the canonical component catalogs.
    try:
        evaluate_and_write_component_catalogs(
            canonical_dir=_canonical_dir,
            out_dir=out_dir,
            unit_filter=getattr(cf, "unit", None),
            cf=cf,
            span_values=span_values,
            required_loads=required_loads_global,
            materials_df=materials_df,
            logger=logger,
        )
    except Exception:
        logger.exception("[components] Failed evaluating component catalogs")

    # -------------------------
    # Per-case sweep + rank exports
    # -------------------------
    all_ranked_all: List[pd.DataFrame] = []
    all_evaluated: List[pd.DataFrame] = []   # <-- NEW: post-systems evaluated candidates
    ra = pd.DataFrame()  # initialised here so it's always bound even if all cases are skipped

    for _, row in loads_df_cases.iterrows():
        case_name = str(row.get("load_case", "case"))
        case_out_dir = utils_mod.ensure_dir(out_dir / f"systems_{case_name}")

        use_factored = bool(getattr(cf, "factored_loads", True))

        required_loads_case = {
            "max_sdl": float(row.get("raw_sdl", 0.0) or 0.0),
            "max_ll": float(row.get("raw_ll", 0.0) or 0.0),
            "max_total": float((row.get("raw_sdl", 0.0) or 0.0) + (row.get("raw_ll", 0.0) or 0.0)),
        }

        if use_factored:
            required_loads_case["max_factored_total"] = float(row.get("factored_total", 0.0) or 0.0)

        # Ensure per-case output directory exists
        case_dir = utils_mod.ensure_dir(Path(out_dir) / f"case_{case_name}")

        candidates_case_all = spans_mod.run_span_sweep(
            load_case_name=case_name,
            out_dir=out_dir,
            systems_df=systems_df,
            materials_df=materials_df,
            span_values=span_values,
            required_loads_case=required_loads_case,
            cf_unit=cf.unit,
            depth_limit_mm=getattr(args, "depth_limit_mm", None),
            logger=logger,
            debug=dbg,
            loads_df_floor=loads_df_floor,
        )


        _dbg_df(f"{case_name}.candidates_case_all", candidates_case_all, explicit=dbg, max_rows=dbg_rows,
        cols=["system_variant","system_family","span","max_span","sdl_total","ll","total_capacity","carbon_total_kgCO2","carbon_per_m2","pass_overall"])
        if dbg_snap:
            _dbg_snapshot_df(candidates_case_all, f"{case_name}__candidates_case_all", out_dir, enabled=True, max_rows=dbg_rows)

        # -------------------------
        # Save the "evaluated/post-systems" population for Option B
        # (this is the correct denominator for evaluated pass rate)
        # -------------------------
        ce = candidates_case_all.copy()
        if "system_variant" in ce.columns:
            ce["system_variant"] = ce["system_variant"].astype(str).str.strip()
        if "system_family" in ce.columns:
            ce["system_family"] = ce["system_family"].astype(str).str.strip()
        all_evaluated.append(ce)

        if candidates_case_all is None or candidates_case_all.empty:
            logger.warning("[run] Case %s: no candidates.", case_name)
            continue

        # >>> CRITICAL FIX for reporting.py:
        # reporting groups by 'floor_load_category'. We set it to the load case name.
        if "floor_load_category" not in candidates_case_all.columns:
            candidates_case_all = candidates_case_all.copy()
            candidates_case_all["floor_load_category"] = case_name
        if "case" not in candidates_case_all.columns:
            candidates_case_all["case"] = case_name

        summaries_case = rank_mod.rank_and_export_summary(
            candidates_case_all,
            out_dir=case_out_dir,
            file_prefix="summary",
            carbon_col="carbon_total_kgCO2",
            type_col="system_family",
            brand_col="system_variant",
            logger=logger,
        )

        # -------------------------
        # Immediately collapse per-case summary_ranked_all.csv to 1 row per system_variant
        # (THIS is where duplicates from span sweep must be removed)
        # -------------------------
        try:
            case_summary_path = case_out_dir / "summary_ranked_all.csv"

            # Prefer the file written by rank_and_export_summary (authoritative)
            if case_summary_path.exists():
                df_case_summary = pd.read_csv(case_summary_path)
            else:
                # fallback to whatever the rank module returned
                df_case_summary = summaries_case.get("summary_ranked_all", pd.DataFrame())
                if df_case_summary is None or df_case_summary.empty:
                    df_case_summary = summaries_case.get("ranked_all", pd.DataFrame())

            if df_case_summary is not None and not df_case_summary.empty:
                df_case_collapsed = collapse_summary_ranked_all_by_system_variant(df_case_summary)

                # preserve case label for downstream merging
                if "case" not in df_case_collapsed.columns:
                    df_case_collapsed["case"] = case_name

                # overwrite the per-case summary so it is unique-by-variant going forward
                df_case_collapsed.to_csv(case_summary_path, index=False)
                logger.info("[run] Case %s: collapsed summary_ranked_all.csv %d -> %d rows",
                            case_name, len(df_case_summary), len(df_case_collapsed))
            else:
                logger.warning("[run] Case %s: no summary_ranked_all found to collapse.", case_name)

        except Exception:
            logger.exception("[run] Case %s: failed to collapse per-case summary_ranked_all.csv", case_name)

        ra = summaries_case.get("ranked_all", pd.DataFrame())
    if isinstance(ra, pd.DataFrame) and not ra.empty:
        ra = ra.copy()

        # ensure case tags exist
        if "floor_load_category" not in ra.columns:
            ra["floor_load_category"] = case_name
        if "case" not in ra.columns:
            ra["case"] = case_name

        # normalize ids
        if "system_variant" in ra.columns:
            ra["system_variant"] = ra["system_variant"].astype(str).str.strip()
        if "system_family" in ra.columns:
            ra["system_family"] = ra["system_family"].astype(str).str.strip()

        # coerce span numeric so it aggregates correctly
        if "span" in ra.columns:
            ra["span"] = pd.to_numeric(ra["span"], errors="coerce")

        # ---- THIS is the critical dedupe/aggregation: one row per (case, family, variant) ----
        group_cols = [c for c in ["case", "system_family", "system_variant"] if c in ra.columns]
        if "system_variant" not in group_cols:
            raise ValueError("[run] ranked_all is missing system_variant; cannot collapse duplicates")

        # if span sweep created multiple rows per variant, collapse them NOW
        if ra.duplicated(subset=group_cols, keep=False).any():
            agg = {}
            for c in ra.columns:
                if c in group_cols:
                    continue

                cl = c.lower()

                # booleans
                if pd.api.types.is_bool_dtype(ra[c]):
                    agg[c] = "all" if "pass" in cl else "first"
                    continue

                # numerics
                if pd.api.types.is_numeric_dtype(ra[c]):
                    if "rank" in cl:
                        agg[c] = "min"          # best rank
                    elif "span" in cl:
                        agg[c] = "max"          # merge span sweep -> keep max span
                    elif any(s in cl for s in ["depth", "util", "unity", "dcr", "demand", "mass", "carbon", "co2", "kgco2", "cost"]):
                        agg[c] = "max"          # conservative merge for sizing/constraints/cost/carbon
                    else:
                        agg[c] = "first"
                else:
                    agg[c] = "first"

            ra = (
                ra.groupby(group_cols, dropna=False, as_index=False)
                .agg(agg)
            )

        all_ranked_all.append(ra)

        logger.info("[run] Case %s: done (candidates=%d)", case_name, len(candidates_case_all))

    # -------------------------
    # Build ROOT summary files EARLY from the (now-collapsed) per-case summaries
    # -------------------------
    try:
        # Load collapsed per-case summaries
        case_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("systems_")])
        case_tables = []
        for d in case_dirs:
            p = d / "summary_ranked_all.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if df is None or df.empty or "system_variant" not in df.columns:
                continue
            case_name = d.name[len("systems_"):] or d.name
            df = df.copy()
            df["case"] = case_name
            df["system_variant"] = df["system_variant"].astype(str).str.strip()
            case_tables.append((case_name, df))

        if case_tables:
            # intersection of variants across cases (inner join semantics)
            common = None
            for case_name, df in case_tables:
                keys = set(df["system_variant"].dropna().unique().tolist())
                common = keys if common is None else (common & keys)
            common = sorted(common) if common else []

            if not common:
                pd.DataFrame().to_csv(out_dir / "summary_ranked_all_long.csv", index=False)
                pd.DataFrame().to_csv(out_dir / "summary_ranked_all.csv", index=False)
            else:
                # long
                long_df = pd.concat(
                    [df[df["system_variant"].isin(common)].copy() for _, df in case_tables],
                    ignore_index=True,
                    sort=False,
                )
                long_df.to_csv(out_dir / "summary_ranked_all_long.csv", index=False)

                # wide (1 row per system_variant)
                import re as _re
                def _safe_suffix(x: str) -> str:
                    x = _re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")
                    return x or "case"

                wide_tables = []
                for i, (case_name, df) in enumerate(case_tables):
                    suffix = _safe_suffix(case_name)
                    t = df[df["system_variant"].isin(common)].copy()

                    # keep system_family only once if present
                    if i > 0 and "system_family" in t.columns:
                        t = t.drop(columns=["system_family"])

                    # drop case col before suffixing
                    if "case" in t.columns:
                        t = t.drop(columns=["case"])

                    ren = {c: f"{c}__{suffix}" for c in t.columns if c != "system_variant"}
                    t = t.rename(columns=ren)
                    wide_tables.append(t)

                merged = wide_tables[0]
                for t in wide_tables[1:]:
                    merged = merged.merge(t, on="system_variant", how="inner")

                if merged["system_variant"].duplicated().any():
                    raise ValueError("[root summary] duplicate system_variant rows still present (should be impossible after collapse).")

                merged.to_csv(out_dir / "summary_ranked_all.csv", index=False)

        else:
            logger.warning("[root summary] No per-case collapsed summaries found to build root summaries.")

    except Exception:
        logger.exception("[root summary] Failed to build root summary_ranked_all_long.csv and summary_ranked_all.csv")

    # -------------------------
    # Expand winners to per-floor materials (optional convenience)
    # -------------------------
    try:
        # Build winners using per-case ranked_all outputs (best per category)
        ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()
        df_winners = select_lowest_carbon_per_type(ranked_union)

        if df_winners is not None and not df_winners.empty:
            spans_mod.expand_winners_and_write_materials_per_floor(
                candidates_all=df_winners,
                floors_by_case=floors_by_case,
                floor_area_lookup=floor_area_lookup,
                materials_df=materials_df,
                out_dir=out_dir,
                logger=logger,
            )
    except Exception:
        logger.exception("[takeoff] Failed global expanded materials write")

    # -------------------------
    # Code checks (winners only)
    # -------------------------
    try:
        ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()
        df_winners = select_winners_from_ranked_all(ranked_union)
        df_checks = run_codechecks_on_winners(
            df_winners=df_winners,
            out_dir=out_dir,
            run_flag=bool(args.run_codechecks),
            material_csv_path=str(args.materials),
            debug_inputs=bool(getattr(args, "codechecks_debug_inputs", False)),
            debug_only_on_fail=not bool(getattr(args, "codechecks_debug_all", False)),
            debug_max_rows=int(getattr(args, "codechecks_debug_max_rows", 50) or 50),
        )
        if isinstance(df_checks, pd.DataFrame) and not df_checks.empty and df_winners is not None and not df_winners.empty:
            merged = df_winners.merge(df_checks, on="system_variant", how="left", suffixes=("", "_codecheck"))
            merged.to_csv(out_dir / "winners_codechecked.csv", index=False)
    except Exception:
        logger.exception("[codechecks] unexpected failure in codechecks block")

    # -------------------------
    # Reporting (global)
    # -------------------------
    try:
        ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()
        if ranked_union is None or ranked_union.empty:
            logger.warning('[reporting] No ranked results to report; skipping.')
        else:
            evaluated_union = pd.concat(all_evaluated, ignore_index=True, sort=False) if all_evaluated else pd.DataFrame()
            ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()
            if evaluated_union is None or evaluated_union.empty:
                logger.warning("[reporting] No evaluated results (post systems.py) to report; skipping.")
            else:
                # Use ranked_union ONLY as the source of "which variants are passing"
                # (reporting.py will infer the passing set via pass_overall/success mask)
                reporting_mod.write_edca_reports(
                    candidates_input=evaluated_union,  # <-- Option B denom (post systems.py)
                    summary_df=ranked_union,           # <-- pass-set source (NOT the denom)
                    out_dir=out_dir,
                    metric="carbon_per_m2",
                    materials_properties_path=str(args.materials)
                )
            logger.info('[reporting] Wrote reports -> %s', Path(out_dir) / 'reporting')
    except Exception:
        logger.exception('[reporting] Failed to write reports')

     # -------------------------
    # Finalize root summaries LAST (prevents later steps from leaving a long-form summary_ranked_all.csv)
    # -------------------------
    try:
        finalize_root_summary_ranked_all(out_dir=out_dir, logger=logger)
    except Exception:
        logger.exception("[summary] Failed to finalize root summary_ranked_all.csv")

    # -------------------------
    # Assembly-level ranking — combine per-component summaries into full buildings
    # -------------------------
    try:
        from edca_code.scripts.core.assembly_summary import build_assembly_rankings
        from edca_code.scripts.core.visualize import (
            plot_structural_class_comparison,
            write_assembly_sensitivity_figure_bundle,
            write_component_figure_bundle,
            write_selected_typology_comparison_charts,
        )

        comp_dir = out_dir / "components"

        def _load_comp(name: str) -> pd.DataFrame:
            p = comp_dir / name / "summary_ranked_all.csv"
            if p.exists():
                return pd.read_csv(p)
            return pd.DataFrame()

        _total_gfa_m2 = (
            float(getattr(cf, "area_per_floor", 0.0) or 0.0)
            * int(getattr(cf, "num_floors", 1) or 1)
        )

        df_assemblies = build_assembly_rankings(
            floors_df     = _load_comp("floor"),
            beams_df      = _load_comp("beam"),
            columns_df    = _load_comp("column"),
            laterals_df   = _load_comp("lateral"),
            total_gfa_m2  = _total_gfa_m2,
            storey_height_m = float(getattr(cf, "floor_to_floor_height", 0.0) or 0.0),
            columns_by_floor = bool(getattr(cf, "columns_by_floor", True)),
            moment_frame_columns = bool(getattr(cf, "moment_frame_columns", False)),
            steel_beam_sls_checks = bool(getattr(cf, "steel_beam_sls_checks", True)),
            steel_beam_max_span_depth_ratio = float(getattr(cf, "steel_beam_max_span_depth_ratio", 16.0) or 16.0),
            steel_secondary_beam_max_span_depth_ratio = float(getattr(cf, "steel_secondary_beam_max_span_depth_ratio", 16.0) or 16.0),
            steel_primary_beam_moment_capacity_factor = float(getattr(cf, "steel_primary_beam_moment_capacity_factor", 1.0) or 1.0),
            steel_secondary_beam_moment_capacity_factor = float(getattr(cf, "steel_secondary_beam_moment_capacity_factor", 1.0) or 1.0),
            steel_beam_include_self_weight = bool(getattr(cf, "steel_beam_include_self_weight", True)),
            timber_beam_moment_capacity_factor = float(getattr(cf, "timber_beam_moment_capacity_factor", 0.75) or 0.75),
            rc_beam_span_depth_checks = bool(getattr(cf, "rc_beam_span_depth_checks", True)),
            rc_beam_max_span_depth_ratio = float(getattr(cf, "rc_beam_max_span_depth_ratio", 12.0) or 12.0),
            rc_secondary_beam_max_span_depth_ratio = float(getattr(cf, "rc_secondary_beam_max_span_depth_ratio", 12.0) or 12.0),
        )

        if not df_assemblies.empty:
            asm_path = out_dir / "summary_assemblies_ranked.csv"
            df_assemblies.to_csv(asm_path, index=False)
            logger.info("[assembly] Wrote summary_assemblies_ranked.csv (%d rows)", len(df_assemblies))

            try:
                _write_diagnostic_txt(cf, span_values, loads_df_cases, df_assemblies, out_dir)
            except Exception:
                logger.exception("[diagnostic] Failed to write run_diagnostic.txt")

            from edca_code.scripts.core.visualize import plot_material_breakdown_comparison

            _SUBTITLE_BASE = "Best assembly per structural class | per-component pass criteria applied"

            for _excl, _suffix in [(False, ""), (True, "_no_lateral")]:
                try:
                    plot_structural_class_comparison(
                        df_assemblies,
                        title="Superstructure Options — Embodied Carbon (kgCO₂e/m² GFA)",
                        subtitle=_SUBTITLE_BASE + (" | excl. lateral system" if _excl else ""),
                        out_path=out_dir / f"comparison_chart{_suffix}.png",
                        exclude_lateral=_excl,
                    )
                    logger.info("[assembly] Wrote comparison_chart%s.png", _suffix)
                except Exception:
                    logger.exception("[assembly] Failed to write comparison_chart%s.png", _suffix)

                try:
                    plot_material_breakdown_comparison(
                        df_assemblies,
                        title="Superstructure Options — Embodied Carbon by Material (kgCO₂e/m² GFA)",
                        subtitle=_SUBTITLE_BASE + (" | excl. lateral system" if _excl else ""),
                        out_path=out_dir / f"material_breakdown_chart{_suffix}.png",
                        exclude_lateral=_excl,
                    )
                    logger.info("[assembly] Wrote material_breakdown_chart%s.png", _suffix)
                except Exception:
                    logger.exception("[assembly] Failed to write material_breakdown_chart%s.png", _suffix)

            try:
                _paths = write_selected_typology_comparison_charts(df_assemblies, out_dir)
                logger.info("[assembly] Wrote %d selected-typology comparison charts", len(_paths))
            except Exception:
                logger.exception("[assembly] Failed to write selected-typology comparison charts")

            try:
                component_fig_root = out_dir / "figures" / "components"
                component_sources = {
                    "floor": _load_comp("floor"),
                    "beam": _load_comp("beam"),
                    "column": _load_comp("column"),
                    "lateral": _load_comp("lateral"),
                    "wall": _load_comp("wall"),
                    "foundation": _load_comp("foundation"),
                    "cladding": _load_comp("cladding"),
                }
                for _component, _df_component in component_sources.items():
                    if _df_component.empty:
                        continue
                    _paths = write_component_figure_bundle(
                        _df_component,
                        component_fig_root / _component,
                        component=_component,
                    )
                    logger.info(
                        "[figures] Wrote %d %s component figures -> %s",
                        len(_paths),
                        _component,
                        component_fig_root / _component,
                    )
            except Exception:
                logger.exception("[figures] Failed to write component figure bundle")

            try:
                comp_all_path = comp_dir / "component_candidates_all.csv"
                comp_all_df = pd.read_csv(comp_all_path, low_memory=False) if comp_all_path.exists() else pd.DataFrame()
                _paths = write_assembly_sensitivity_figure_bundle(
                    df_assemblies,
                    out_dir / "figures" / "sensitivity",
                    component_candidates_df=comp_all_df,
                )
                logger.info("[figures] Wrote %d sensitivity figures -> %s", len(_paths), out_dir / "figures" / "sensitivity")
            except Exception:
                logger.exception("[figures] Failed to write sensitivity figure bundle")
        else:
            logger.warning("[assembly] No assembly rows generated — skipping chart")
    except Exception:
        logger.exception("[assembly] Failed to build assembly summary")

    logger.info("[run_edca] Done.")


def _write_diagnostic_txt(
    cf: Any,
    span_values: list,
    loads_df_cases: "pd.DataFrame",
    df_assemblies: "pd.DataFrame",
    out_dir: "Path",
) -> None:
    """Write a human-readable diagnostic summary alongside the run outputs."""
    import datetime

    W = 80
    sep_h = "=" * W
    sep_l = "-" * W

    def _cf(attr: str, default: str = "—") -> str:
        v = getattr(cf, attr, None)
        return str(v) if v is not None and str(v).strip() not in ("", "None") else default

    lines: list[str] = []
    lines += [sep_h, "EDCA RUN DIAGNOSTIC SUMMARY", sep_h]
    lines += [
        f"Project:        {_cf('project_name')}",
        f"Location:       {_cf('location')}",
        f"Run date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output dir:     {out_dir}",
    ]

    # --- Grid configuration ---
    lines += ["", sep_l, "GRID CONFIGURATION", sep_l]
    irregular = getattr(cf, "one_way_irregular", False)
    if irregular:
        beam_span = getattr(cf, "one_way_beam_min_span", None)
        slab_span = getattr(cf, "one_way_slab_min_span", None)
        orient = _cf("one_way_orientation")
        lines += [
            "One-way irregular:   Yes",
            f"  Beam span (x):     {beam_span} m",
            f"  Slab span (y):     {slab_span} m",
            f"  Orientation:       {orient}",
        ]
        if beam_span is not None and slab_span is not None:
            lines.append(f"Bay area:            {float(beam_span) * float(slab_span):.2f} m²")
    else:
        n = len(span_values)
        if n == 1:
            lines.append(f"Grid span:           {span_values[0]} m (square bay)")
        elif n > 1:
            lines.append(f"Grid span sweep:     {span_values[0]} – {span_values[-1]} m  ({n} values)")
        else:
            lines.append("Grid span:           —")

    # --- Building parameters ---
    lines += ["", sep_l, "BUILDING PARAMETERS", sep_l]
    area = getattr(cf, "area_per_floor", None)
    n_floors = getattr(cf, "num_floors", None)
    gfa = (float(area) * int(n_floors)) if (area and n_floors) else None
    depth_lim = getattr(cf, "depth_limit_enabled", False)
    lines += [
        f"Storeys:             {_cf('num_floors')}",
        f"Floor-to-floor:      {_cf('floor_to_floor_height')} m",
        f"Floor plate area:    {float(area):.0f} m²/floor" if area else "Floor plate area:    —",
        f"Total GFA:           {gfa:.0f} m²" if gfa else "Total GFA:           —",
        f"Typology:            {_cf('typology_id')}",
        f"Code standard:       {_cf('code_standard')}",
        f"Unit system:         {_cf('unit')}",
        f"Depth limit:         " + (
            f"{_cf('depth_limit')} m" if depth_lim else "Disabled"
        ),
    ]

    # --- Loads ---
    lines += ["", sep_l, "LOADS (per occupancy/load case)", sep_l]
    if loads_df_cases is not None and not loads_df_cases.empty:
        hdr = f"{'Load case':<25}  {'SDL (kPa)':>10}  {'LL (kPa)':>10}  {'Factored (kPa)':>14}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for _, row in loads_df_cases.iterrows():
            case = str(row.get("load_case", "—"))
            sdl = row.get("raw_sdl", None)
            ll = row.get("raw_ll", None)
            ft = row.get("factored_total", None)
            sdl_s = f"{float(sdl):.2f}" if sdl is not None else "—"
            ll_s = f"{float(ll):.2f}" if ll is not None else "—"
            ft_s = f"{float(ft):.2f}" if ft is not None else "—"
            lines.append(f"{case:<25}  {sdl_s:>10}  {ll_s:>10}  {ft_s:>14}")
    else:
        lines.append("(no load case data available)")

    # --- Span sweep ---
    lines += ["", sep_l, "SPAN SWEEP", sep_l]
    if span_values:
        vals_str = ", ".join(str(s) for s in span_values[:20])
        suffix = f"  ... (+{len(span_values)-20} more)" if len(span_values) > 20 else ""
        lines.append(f"Spans evaluated:     [{vals_str}]{suffix}  ({len(span_values)} value(s))")
        if irregular:
            lines.append(
                "  Note: ONE_WAY_IRREGULAR=true — beam/slab spans above override this list"
            )
    else:
        lines.append("Spans evaluated:     (none)")

    # --- Top 5 assemblies ---
    lines += ["", sep_l, "TOP 5 ASSEMBLIES BY EMBODIED CARBON (kgCO₂e/m² GFA)", sep_l]
    if df_assemblies is not None and not df_assemblies.empty:
        carbon_col = "total_embodied_carbon_per_m2"
        df_top = (
            df_assemblies.copy()
            if carbon_col not in df_assemblies.columns
            else df_assemblies.sort_values(carbon_col, ascending=True).head(5)
        )
        for i, (_, row) in enumerate(df_top.iterrows(), start=1):
            sc = str(row.get("structural_class", "—"))
            ec = row.get(carbon_col, None)
            ec_s = f"{float(ec):.2f}" if ec is not None else "—"

            def _comp_s(fam_col: str, var_col: str) -> str:
                var = str(row.get(var_col, "") or "").strip()
                fam = str(row.get(fam_col, "") or "").strip()
                return var if var and var != "None" else (fam if fam and fam != "None" else "—")

            floor_s = _comp_s("floor_family", "floor_variant")
            beam_s = _comp_s("beam_family", "beam_variant")
            secb_s = _comp_s("secondary_beam_family", "secondary_beam_variant")
            col_s = _comp_s("column_family", "column_variant")
            lat_s = _comp_s("lateral_family", "lateral_variant")

            def _ec(col: str) -> str:
                v = row.get(col, None)
                return f"{float(v):.1f}" if v is not None else "—"

            lines += [
                "",
                f"  #{i}  {sc}",
                f"       Total carbon: {ec_s} kgCO₂e/m²",
                f"       Floor:        {floor_s}",
                f"       Beam:         {beam_s}",
                f"       Sec. beam:    {secb_s}",
                f"       Column:       {col_s}",
                f"       Lateral:      {lat_s}",
                f"       Carbon split (kgCO₂e/m²): "
                f"Floor {_ec('floor_carbon_per_m2')} | "
                f"Beam {_ec('beam_carbon_per_m2')} | "
                f"Sec.beam {_ec('secondary_beam_carbon_per_m2')} | "
                f"Column {_ec('column_carbon_per_m2')} | "
                f"Lateral {_ec('lateral_carbon_per_m2')}",
            ]
    else:
        lines.append("(no assembly results available)")

    lines += ["", sep_h]

    txt_path = out_dir / "run_diagnostic.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("[diagnostic] Wrote run_diagnostic.txt")


if __name__ == "__main__":
    main()
