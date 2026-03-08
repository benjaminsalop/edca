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
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edca_code.scripts.core.parse import ControlFile, build_floor_area_lookup, parameters_from_control_file
from edca_code.scripts.core import systems as systems_mod
from edca_code.scripts.core import carbon as carbon_mod
from edca_code.scripts.core import loads as loads_mod
from edca_code.scripts.core import spans as spans_mod
from edca_code.scripts.core import rank as rank_mod
from edca_code.scripts.core import reporting as reporting_mod
from edca_code.scripts.core import utils as utils_mod

from edca_code.scripts.code_checks.code_runner import run_code_checks_if_requested


logger = logging.getLogger(__name__)


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

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="run_edca", description="Run EDCA pipeline.")
    p.add_argument("--control", "-c", required=True, help="Path to control file YAML")
    p.add_argument("--systems", "-s", required=True, help="Path to systems_variants parquet/csv")
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

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    out_dir = utils_mod.ensure_dir(Path(args.out))

    cf = ControlFile.from_path(args.control)
    default_code_standard_if_missing(cf)

    logger.info("[parse] Starting EDCA run. Outputs -> %s", out_dir)
    logger.info("[parse] Loaded control file: project=%s, unit=%s, code_standard=%s",
                getattr(cf, "project_name", None), getattr(cf, "unit", None), getattr(cf, "code_standard", None))

    try:
        _params = parameters_from_control_file(args.control)
        logger.debug("[parse] control file keys: %s", sorted(list(_params.keys())))
    except Exception:
        logger.debug("[parse] parameters_from_control_file failed; continuing", exc_info=True)

    # Load systems + materials
    systems_df, _families_df, _variants_df = systems_mod.load_systems_catalog(
        args.systems,
        unit_filter=getattr(cf, "unit", None),
    )
    materials_df = carbon_mod.load_materials_table(args.materials)

    # Loads context (auto-finds load_values.yaml / load_combinations.yaml next to occupancies.csv)
    required_loads_global, floors_by_case, loads_df_floor = loads_mod.build_load_context(
        cf,
        args.occupancies,
        load_values_yaml=None,
        load_combinations_yaml=None,
    )

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

    # Spans
    span_values = spans_mod.resolve_span_values(cf, args, logger=logger)
    logger.info("[span] Evaluating spans: %s", span_values)

    # -------------------------
    # Per-case sweep + rank exports
    # -------------------------
    all_ranked_all: List[pd.DataFrame] = []
    all_evaluated: List[pd.DataFrame] = []   # <-- NEW: post-systems evaluated candidates

    for _, row in loads_df_cases.iterrows():
        case_name = str(row.get("load_case", "case"))
        case_out_dir = utils_mod.ensure_dir(out_dir / f"systems_{case_name}")

        required_loads_case = {
            "max_sdl": float(row.get("raw_sdl", 0.0) or 0.0),
            "max_ll": float(row.get("raw_ll", 0.0) or 0.0),
            "max_total": float((row.get("raw_sdl", 0.0) or 0.0) + (row.get("raw_ll", 0.0) or 0.0)),
        }

        candidates_case_all = spans_mod.run_span_sweep(
            load_case_name=case_name,
            out_dir=case_out_dir,
            systems_df=systems_df,
            materials_df=materials_df,
            span_values=span_values,
            required_loads_case=required_loads_case,
            cf_unit=str(getattr(cf, "unit", "")) or None,
            depth_limit_mm=args.depth_limit_mm,
            logger=logger,
        )

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

    logger.info("[run_edca] Done.")


if __name__ == "__main__":
    main()
