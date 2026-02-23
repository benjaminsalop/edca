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
        if "_source_case" not in candidates_case_all.columns:
            candidates_case_all["_source_case"] = case_name

        summaries_case = rank_mod.rank_and_export_summary(
            candidates_case_all,
            out_dir=case_out_dir,
            file_prefix="summary",
            carbon_col="carbon_total_kgCO2",
            type_col="system_family",
            brand_col="system_variant",
            logger=logger,
        )

        ra = summaries_case.get("ranked_all", pd.DataFrame())
        if isinstance(ra, pd.DataFrame) and not ra.empty:
            # also ensure the *ranked* output keeps the column even if rank reorders
            if "floor_load_category" not in ra.columns:
                ra = ra.copy()
                ra["floor_load_category"] = case_name
            if "case" not in ra.columns:
                ra["case"] = case_name
            all_ranked_all.append(ra)

        logger.info("[run] Case %s: done (candidates=%d)", case_name, len(candidates_case_all))

    # -------------------------
    # Expand winners to per-floor materials (optional convenience)
    # -------------------------
    try:
        # Build winners using per-case ranked_all outputs (best per category)
        ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()
        df_winners = select_winners_from_ranked_all(ranked_union)

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
        # Build GLOBAL summary outputs:
        #   - summary_ranked_all_long.csv : what EDCA previously wrote to summary_ranked_all.csv
        #       (collapsed/aggregated across cases over the *intersection* of systems)
        #   - summary_ranked_all.csv      : NEW wide inner-join merge across case subfolders
        #       (exactly one row per system_variant)
        # -------------------------
        ranked_union = pd.concat(all_ranked_all, ignore_index=True, sort=False) if all_ranked_all else pd.DataFrame()

        if not ranked_union.empty:
            if "system_variant" not in ranked_union.columns:
                raise ValueError(
                    "Cannot build global summary: no system identifier column found "
                    "(expected system_variant)."
                )

            # --- normalize join key to avoid whitespace mismatches
            def _norm_variant(s):
                return str(s).strip()

            # Compute intersection of system_variants present in EVERY case
            common_variants = None
            for df_case in all_ranked_all:
                if df_case is None or df_case.empty or "system_variant" not in df_case.columns:
                    continue
                keys = df_case[["system_variant"]].copy()
                keys["system_variant"] = keys["system_variant"].map(_norm_variant)
                keys = keys.drop_duplicates()
                common_variants = keys if common_variants is None else common_variants.merge(
                    keys, on=["system_variant"], how="inner"
                )

            if common_variants is None or common_variants.empty:
                # nothing common across cases — write empty files so downstream doesn't crash mysteriously
                pd.DataFrame().to_csv(Path(out_dir) / "summary_ranked_all_long.csv", index=False)
                pd.DataFrame().to_csv(Path(out_dir) / "summary_ranked_all.csv", index=False)
            else:
                # Filter the union to only common systems
                ranked_common = ranked_union.copy()
                ranked_common["system_variant"] = ranked_common["system_variant"].map(_norm_variant)
                ranked_common = ranked_common.merge(common_variants, on=["system_variant"], how="inner")

                # -------------------------
                # (A) Long file = what you USED to write to summary_ranked_all.csv
                #     i.e., collapsed across cases using groupby+aggregation.
                # -------------------------
                join_cols = []
                if all(c in ranked_common.columns for c in ["system_family", "system_variant"]):
                    join_cols = ["system_family", "system_variant"]
                else:
                    join_cols = ["system_variant"]

                def _agg_rule(col: str) -> str:
                    # worst-case across cases for sizing/constraints-ish metrics:
                    if any(s in col for s in ["depth", "util", "unity", "dcr", "demand", "mass", "carbon", "cost"]):
                        return "max"
                    return "first"

                agg = {}
                for c in ranked_common.columns:
                    if c in join_cols:
                        continue
                    if pd.api.types.is_numeric_dtype(ranked_common[c]):
                        agg[c] = _agg_rule(c)
                    else:
                        agg[c] = "first"

                summary_ranked_all_old_behavior = (
                    ranked_common
                    .groupby(join_cols, dropna=False, as_index=False)
                    .agg(agg)
                )

                # <-- this is what summary_ranked_all.csv USED to be
                summary_ranked_all_old_behavior.to_csv(
                    Path(out_dir) / "summary_ranked_all_long.csv",
                    index=False
                )

                # -------------------------
                # (B) Condensed file = TRUE inner-join MERGE across case tables
                #     One row per system_variant; columns suffixed by case name.
                # -------------------------
                import re as _re

                def _safe_suffix(x: str) -> str:
                    x = _re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")
                    return x or "case"

                def _best_per_variant(df: pd.DataFrame) -> pd.DataFrame:
                    """Ensure at most one row per system_variant within a case."""
                    D = df.copy()
                    D["system_variant"] = D["system_variant"].map(_norm_variant)

                    # Prefer best by rank columns if available
                    sort_cols = [c for c in ["rank_overall", "rank_carbon", "rank", "carbon_total_kgCO2", "cost_total"] if c in D.columns]
                    if sort_cols:
                        D = D.sort_values(by=sort_cols, ascending=True, kind="mergesort")

                    return D.drop_duplicates(subset=["system_variant"], keep="first")

                wide_tables = []
                for i, df_case in enumerate(all_ranked_all):
                    if df_case is None or df_case.empty or "system_variant" not in df_case.columns:
                        continue

                    # pick a readable suffix for this case
                    if "case" in df_case.columns and df_case["case"].notna().any():
                        case_name = str(df_case["case"].dropna().iloc[0])
                    elif "occupancy" in df_case.columns and df_case["occupancy"].notna().any():
                        case_name = str(df_case["occupancy"].dropna().iloc[0])
                    else:
                        case_name = f"case_{i+1}"

                    suffix = _safe_suffix(case_name)

                    D = _best_per_variant(df_case)
                    D = D.merge(common_variants, on=["system_variant"], how="inner")

                    # Keep system_family once (unsuffixed) if present; drop it from subsequent cases
                    if i > 0 and "system_family" in D.columns:
                        D = D.drop(columns=["system_family"])

                    # Rename all non-key columns with suffix to avoid collisions
                    ren = {}
                    for c in D.columns:
                        if c == "system_variant":
                            continue
                        ren[c] = f"{c}__{suffix}"
                    D = D.rename(columns=ren)

                    wide_tables.append(D)

                if not wide_tables:
                    pd.DataFrame().to_csv(Path(out_dir) / "summary_ranked_all.csv", index=False)
                else:
                    merged = wide_tables[0]
                    for t in wide_tables[1:]:
                        merged = merged.merge(t, on=["system_variant"], how="inner")

                    # Guarantee uniqueness: one row per system_variant
                    if merged["system_variant"].duplicated().any():
                        raise ValueError(
                            "Condensed summary merge produced duplicate system_variant rows. "
                            "Check your per-case inputs."
                        )

                    merged.to_csv(Path(out_dir) / "summary_ranked_all.csv", index=False)

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
            reporting_mod.write_edca_reports(
                candidates_input=ranked_union,
                summary_df=ranked_union,
                out_dir=out_dir,
                metric='carbon_per_m2',
            )
            logger.info('[reporting] Wrote reports -> %s', Path(out_dir) / 'reporting')
    except Exception:
        logger.exception('[reporting] Failed to write reports')

    logger.info("[run_edca] Done.")


if __name__ == "__main__":
    main()
