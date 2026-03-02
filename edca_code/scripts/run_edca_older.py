#!/usr/bin/env python3
"""
run_edca.py (ORCHESTRATOR)

Path A:
- Parse control file
- Load systems + materials
- Build load context (program + occupancies -> per-floor loads + floors_by_case)
- Collapse to per-load-case governing loads
- For each load_case:
    span sweep -> takeoff + carbon -> rank summaries
- Assemble global winners across cases (optional convenience)
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from edca_code.scripts.core.parse import ControlFile, parameters_from_control_file, build_floor_area_lookup
from edca_code.scripts.core import utils as utils_mod
from edca_code.scripts.core import systems as systems_mod
from edca_code.scripts.core import loads as loads_mod
from edca_code.scripts.core import spans as spans_mod
from edca_code.scripts.core import carbon as carbon_mod
from edca_code.scripts.core import takeoff as takeoff_mod
from edca_code.scripts.core import rank as rank_mod
from edca_code.scripts.core import reporting as reporting_mod
from edca_code.scripts.code_checks import code_runner as code_runner_mod
from edca_code.scripts.code_checks.code_runner import run_code_checks_if_requested

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------
# Small wrappers (safe + repeatable)
# ------------------------
def configure_logging(verbose: bool) -> None:
    if verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)


def default_code_standard_if_missing(cf: ControlFile) -> None:
    # Keep your previous behavior: default Eurocode when metric-ish
    if getattr(cf, "code_standard", None):
        return
    unit_lower = (getattr(cf, "unit", "") or "").strip().lower()
    if unit_lower in ("metric", "si", "eu", "european", "euro"):
        cf.code_standard = "Eurocode"
    else:
        cf.code_standard = "Default"


def collapse_floor_loads_to_cases(
    loads_df: pd.DataFrame,
    code_standard: str,) -> pd.DataFrame:
    """
    build_load_context() returns per-floor loads with columns like:
      occupancy, SDL, LL, EN1990_ULS, ASCE7_LRFD, unit ...

    Your span-sweep loop expects per-case rows with:
      load_case, raw_sdl, raw_ll, factored_sdl, factored_ll, factored_total, combo, gamma_g, gamma_q, unit

    This collapses floors -> one governing row per load_case (occupancy).
    """
    if loads_df is None or loads_df.empty:
        return pd.DataFrame(columns=[
            "load_case", "raw_sdl", "raw_ll",
            "factored_sdl", "factored_ll", "factored_total",
            "combo", "gamma_g", "gamma_q", "unit",
        ])

    df = loads_df.copy()

    # -------------------------
    # normalize column names (be permissive)
    # -------------------------

    # 1) occupancy / case key
    if "occupancy" not in df.columns:
        for alt in ["load_case", "case", "space_type", "program", "occ", "occupancy_type", "use"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "occupancy"})
                break

    if "occupancy" not in df.columns:
        raise ValueError(
            "collapse_floor_loads_to_cases needs an occupancy/case column. "
            f"Expected 'occupancy' (or one of load_case/case/space_type/program/occ/use). "
            f"Got columns: {list(df.columns)}"
        )

    # 2) Choose SDL/LL sources if SDL/LL are not already present
    #    (your current loads_df has raw_sdl/raw_ll and factored_sdl/factored_ll)
    if "SDL" not in df.columns or "LL" not in df.columns:
        # Prefer RAW for raw_sdl/raw_ll (we factor later using gammas)
        if "raw_sdl" in df.columns and "raw_ll" in df.columns:
            df = df.rename(columns={"raw_sdl": "SDL", "raw_ll": "LL"})
        # Fallback: if only factored present, treat those as SDL/LL (least ideal, but unblocks)
        elif "factored_sdl" in df.columns and "factored_ll" in df.columns:
            df = df.rename(columns={"factored_sdl": "SDL", "factored_ll": "LL"})
        else:
            # Try common aliases
            if "SDL" not in df.columns:
                for alt in ["sdl", "dead_load", "DL", "gk", "Gk", "sdl_kpa", "sdl_kn_m2"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "SDL"})
                        break
            if "LL" not in df.columns:
                for alt in ["ll", "live_load", "LLk", "Qk", "qk", "imposed", "imposed_load", "ll_kpa", "ll_kn_m2"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "LL"})
                        break

    # 3) Now enforce SDL/LL
    if "SDL" not in df.columns or "LL" not in df.columns:
        raise ValueError(
            "collapse_floor_loads_to_cases needs SDL and LL columns (or raw_sdl/raw_ll, factored_sdl/factored_ll). "
            f"Got columns: {list(df.columns)}"
        )

    combo_col = "EN1990_ULS" if "euro" in (code_standard or "").lower() else "ASCE7_LRFD"

    out_rows: List[Dict[str, Any]] = []
    for occ, g in df.groupby("occupancy", dropna=False):
        load_case = str(occ)

        raw_sdl = float(pd.to_numeric(g["SDL"], errors="coerce").max() or 0.0)
        raw_ll = float(pd.to_numeric(g["LL"], errors="coerce").max() or 0.0)

        # pick governing factored combo across all floors in this load_case
        best_combo_name = None
        best_total = None
        best_factors: Dict[str, float] = {}

        if combo_col in g.columns:
            for _, r in g.iterrows():
                combos = r.get(combo_col)
                if isinstance(combos, list):
                    for item in combos:
                        try:
                            tot = float(item.get("total", 0.0))
                        except Exception:
                            continue
                        if best_total is None or tot > best_total:
                            best_total = tot
                            best_combo_name = item.get("name")
                            best_factors = item.get("factors") or {}

        # infer gammas
        gamma_g = None
        gamma_q = None
        if best_factors:
            # Eurocode uses DL/LL keys; ASCE uses D/L keys (per loads.py)
            if "DL" in best_factors or "LL" in best_factors:
                gamma_g = best_factors.get("DL")
                gamma_q = best_factors.get("LL")
            elif "D" in best_factors or "L" in best_factors:
                gamma_g = best_factors.get("D")
                gamma_q = best_factors.get("L")

        # conservative fallback if combo not available
        if best_total is None:
            best_total = (raw_sdl + raw_ll)
        if gamma_g is None:
            gamma_g = 1.0
        if gamma_q is None:
            gamma_q = 1.0

        factored_sdl = float(gamma_g) * raw_sdl
        factored_ll = float(gamma_q) * raw_ll
        factored_total = float(best_total)

        unit_val = None
        if "unit" in g.columns and not g["unit"].isna().all():
            unit_val = g["unit"].dropna().iloc[0]

        out_rows.append({
            "load_case": load_case,
            "raw_sdl": raw_sdl,
            "raw_ll": raw_ll,
            "factored_sdl": factored_sdl,
            "factored_ll": factored_ll,
            "factored_total": factored_total,
            "combo": best_combo_name,
            "gamma_g": gamma_g,
            "gamma_q": gamma_q,
            "unit": unit_val,
        })

    return pd.DataFrame(out_rows)

def run_codechecks_on_winners(
    *,
    df_winners: pd.DataFrame,
    out_dir: Path,
    cf: ControlFile,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    Run code checks on winners only.

    Returns a DataFrame of codecheck results (or empty df if skipped).
    Writes a CSV to out_dir / "code_checks" by default (safe).
    """
    if df_winners is None or df_winners.empty:
        logger.warning("[codechecks] No winners to check; skipping.")
        return pd.DataFrame()

    if not bool(getattr(args, "run_codechecks", False)):
        logger.info("[codechecks] --run-codechecks not set; skipping.")
        return pd.DataFrame()

    # make a dedicated output folder
    code_out_dir = utils_mod.ensure_dir(out_dir / "code_checks")

    # IMPORTANT: normalize expected ID column for code runner
    df_in = df_winners.copy()

    # Prefer system_variant; otherwise try common aliases
    if "system_variant" not in df_in.columns:
        for alt in ("system_id", "system_variant_id", "variant_id", "system"):
            if alt in df_in.columns:
                df_in = df_in.rename(columns={alt: "system_variant"})
                break

    if "system_variant" in df_in.columns:
        df_in["system_variant"] = df_in["system_variant"].astype(str)

    # You imported BOTH:
    #  - code_runner_mod (module)
    #  - run_code_checks_if_requested (function)
    #
    # Use the function you already import (cleaner).
    try:
        # This function should do the right thing based on args/run_codechecks flag.
        # If it expects more params in your repo, adjust *here* only.
        results = run_code_checks_if_requested(
            df_in,
            out_dir=code_out_dir,
            cf=cf,
            args=args,
        )

    except TypeError:
        # Fallback if your signature differs (common in refactors):
        # try a simpler call pattern.
        results = run_code_checks_if_requested(df_in, out_dir=code_out_dir)

    # Normalize return → DataFrame
    if isinstance(results, pd.DataFrame):
        df_checks = results
    elif isinstance(results, dict):
        # pick first DF-like value
        df_checks = next((v for v in results.values() if isinstance(v, pd.DataFrame)), pd.DataFrame())
    else:
        df_checks = pd.DataFrame()

    # Always write a CSV if we got anything
    if not df_checks.empty:
        df_checks.to_csv(code_out_dir / "codecheck_results.csv", index=False)
        logger.info("[codechecks] Wrote results -> %s", code_out_dir / "codecheck_results.csv")
    else:
        logger.warning("[codechecks] No codecheck output produced (empty).")

    return df_checks


def run_one_load_case(
    *,
    case_name: str,
    out_dir: Path,
    systems_df: pd.DataFrame,
    required_max_loads: Dict[str, Any],
    cf: ControlFile,
    spans_override: Optional[List[float]],
    material_csv_path: Optional[str],
    material_id: Optional[str],
    use_codechecks: bool,
    load_values_yaml: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one load case and return:
      (1) candidates_case_all: all passing candidates for this case (possibly span-swept)
      (2) summary_case_df: ranked summary dataframe for this case (for downstream reporting)
    """
    logger.info(f"[run] Case {case_name}: processing...")

    def _extract_summary_df(obj: Any) -> pd.DataFrame:
        """Best-effort extraction of a DataFrame from rank_mod.rank_and_export_summary output."""
        if obj is None:
            return pd.DataFrame()

        # If rank_mod returns a dataframe directly
        if isinstance(obj, pd.DataFrame):
            return obj

        # If it returns a dict: pick the first DataFrame-looking value, prefer common keys
        if isinstance(obj, dict):
            preferred_keys = [
                "summary_ranked_all",
                "summary_ranked",
                "summary",
                "df_summary",
                "df",
            ]
            for k in preferred_keys:
                v = obj.get(k, None)
                if isinstance(v, pd.DataFrame):
                    return v

            # otherwise first dataframe value in dict
            for v in obj.values():
                if isinstance(v, pd.DataFrame):
                    return v

        # If it returns a tuple/list: first DataFrame element
        if isinstance(obj, (list, tuple)):
            for v in obj:
                if isinstance(v, pd.DataFrame):
                    return v

        return pd.DataFrame()

    # ---------------------------------------------
    # Filter compatible candidates for this load case
    # ---------------------------------------------
    candidates_case = systems_df.copy()

    # Always initialize this so later checks can't reference an undefined variable
    candidates_case_all = candidates_case.copy()


    # Sanity: ranking requires a carbon column; if it's missing, the pipeline order is wrong
    if "carbon_total_kgCO2" not in candidates_case_all.columns:
        raise RuntimeError(
            f"[run_one_load_case] '{case_name}' has no carbon results yet. "
            "You are ranking raw systems_df. Run span sweep + takeoff + carbon before rank."
        )

    # ---------------------------------------------
    # Span sweep + takeoff + carbon  (MUST happen before rank)
    # ---------------------------------------------
    # You need candidates_case_all to include carbon_total_kgCO2.
    # Implement this using your existing spans/takeoff/carbon pipeline.

    case_out_dir = utils_mod.ensure_dir(out_dir / f"systems_{case_name}")

    # Choose spans to evaluate (prefer explicit override, else control/CLI resolver)
    span_values = spans_override
    if not span_values:
        # NOTE: if you already resolved span_values in main(), pass them into run_one_load_case instead
        span_values = None

    # >>> REPLACE THIS with the actual function(s) in your repo that:
    # (a) evaluate spans, (b) compute takeoff, (c) compute carbon totals per candidate
    candidates_case_all = spans_mod.compute_candidates_with_takeoff_and_carbon(
        candidates_case=candidates_case,
        span_values=span_values,
        case_name=case_name,
        case_loads=required_max_loads,   # contains raw_sdl/raw_ll/gammas/etc
        materials_csv_path=material_csv_path,
        cf=cf,
        out_dir=case_out_dir,
        include_a4_a5=bool(getattr(cf, "include_a4_a5", False)),
    )

    # If your pipeline uses a different carbon column name, normalize it here
    if "carbon_total_kgCO2" not in candidates_case_all.columns:
        for alt in ("carbon_total", "carbon_total_kgCO2e", "carbon_total_kg", "kgCO2_total"):
            if alt in candidates_case_all.columns:
                candidates_case_all = candidates_case_all.rename(columns={alt: "carbon_total_kgCO2"})
                break

    # ---------------------------------------------
    # Rank + export summary (per-case)
    # ---------------------------------------------
    summaries_case = rank_mod.rank_and_export_summary(
    candidates_case_all,
    out_dir=out_dir / f"systems_{case_name}",
    carbon_col="carbon_total_kgCO2",
    )


    summary_case_df = _extract_summary_df(summaries_case)

    if isinstance(summaries_case, dict) and "summary_ranked_all" in summaries_case:
        df_sum = summaries_case["summary_ranked_all"].copy()
    else:
        raise RuntimeError("rank_and_export_summary did not return summary_ranked_all")

    # Make sure the case name is carried through (helps later unions)
    if not summary_case_df.empty and "case" not in summary_case_df.columns:
        summary_case_df = summary_case_df.copy()
        summary_case_df["case"] = case_name

    logger.info(f"[run] Case {case_name}: done (candidates={len(candidates_case_all)}, summary_rows={len(summary_case_df)})")

    return candidates_case_all, summary_case_df

# ------------------------
# Main
# ------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run_edca",
        description="Run EDCA pipeline (systems -> takeoff -> carbon -> rank), looping load cases."
    )
    p.add_argument("--control", "-c", required=True, help="Path to control file YAML")
    p.add_argument("--systems", "-s", required=True, help="Path to systems_variants parquet/csv")
    p.add_argument("--materials", "-m", required=True, help="Path to materials CSV")
    p.add_argument("--out", "-o", default="edca_outputs", help="Output directory")
    p.add_argument("--occupancies", "-f", required=True, help="Path to occupancies CSV")

    # span sweep options
    p.add_argument("--span-step", type=float, default=0.5, help="Span sweep step in metres")
    p.add_argument("--span-min", type=float, default=None, help="Override minimum span (m)")
    p.add_argument("--span-max", type=float, default=None, help="Override maximum span (m)")
    p.add_argument("--no-sweep", action="store_true", help="Do not perform span sweep; evaluate only min span")
    p.add_argument("--spans", default=None, help="Override spans as comma-separated list, e.g. '6,7.5,9'")

    p.add_argument("--depth-limit-mm", type=float, default=None, help="Optional depth limit (mm) for filtering systems")
    p.add_argument("--run-codechecks", action="store_true", help="Run code checks on selected candidates (optional)")
    p.add_argument("--include-a4-a5", action="store_true", help="Include A4/A5 in carbon totals where possible")
    p.add_argument("--units", default=None, help="Units hint: 'metric' or 'imperial'")
    p.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    cf = ControlFile.from_path(args.control)
    if args.units:
        cf.unit = args.units
    default_code_standard_if_missing(cf)

    # --- spans_override: CLI beats control file ---
    spans_override = args.spans if getattr(args, "spans", None) is not None else getattr(cf, "spans", None)

    # Accept a comma-separated string or a list
    if isinstance(spans_override, str):
        spans_override = [float(x.strip()) for x in spans_override.split(",") if x.strip()]
    elif spans_override is not None:
        spans_override = [float(x) for x in spans_override]

    out_dir = utils_mod.ensure_dir(Path(args.out))
    logger.info("[parse] Starting EDCA run. Outputs -> %s", out_dir)

    # quick control-file params dump (your existing behavior)
    try:
        parameters = parameters_from_control_file(args.control)
        logger.debug(parameters)
    except Exception:
        logger.debug("[parse] parameters_from_control_file failed; continuing", exc_info=True)

    logger.info("[parse] Loaded control file: project=%s, unit=%s, code_standard=%s",
                getattr(cf, "project_name", None), getattr(cf, "unit", None), getattr(cf, "code_standard", None))

    # load systems + materials once
    systems_df, systems_families_df, systems_variants_df = systems_mod.load_systems_catalog(
        args.systems,
        unit_filter=getattr(cf, "unit", None),
    )
    materials_df = carbon_mod.load_materials_table(args.materials)

    # build load context once (program + occupancies)
    required_loads, floors_by_case, loads_df_floor = loads_mod.build_load_context(
        cf,
        args.occupancies,
    )

    # collapse per-floor loads to per-load-case loads_df
    loads_df = collapse_floor_loads_to_cases(loads_df_floor, getattr(cf, "code_standard", "Default"))

    # fallback if empty
    if loads_df is None or loads_df.empty:
        logger.warning("[loads] loads_df empty; falling back to a single global load case.")
        loads_df = pd.DataFrame([{
            "load_case": "global",
            "raw_sdl": float(required_loads.get("max_sdl", 0.0) or 0.0),
            "raw_ll": float(required_loads.get("max_ll", 0.0) or 0.0),
            "factored_sdl": float(required_loads.get("max_sdl", 0.0) or 0.0),
            "factored_ll": float(required_loads.get("max_ll", 0.0) or 0.0),
            "factored_total": float((required_loads.get("max_sdl", 0.0) or 0.0) + (required_loads.get("max_ll", 0.0) or 0.0)),
            "combo": None,
            "gamma_g": 1.0,
            "gamma_q": 1.0,
            "unit": getattr(cf, "unit", None),
        }])

    # build floor_area_lookup AFTER floors_by_case is known
    floor_area_lookup, area_per_floor_val = build_floor_area_lookup(
        getattr(cf, "area_per_floor", None),
        floors_by_case=floors_by_case,
    )

    # span sweep values (delegated to spans.py)
    span_values = spans_mod.resolve_span_values(cf, args, logger=logger)
    logger.info("[span] Evaluating spans: %s", span_values)

    candidates_all = []
    summary_frames = []

    # We'll loop real load cases from loads_df (NOT keys like max_sdl/max_ll)
    for _, lc in loads_df.iterrows():
        case_name = str(lc["load_case"])

        # package the governing values for this case (handy to pass downstream)
        case_loads = {
            "raw_sdl": float(lc.get("raw_sdl", 0.0) or 0.0),
            "raw_ll": float(lc.get("raw_ll", 0.0) or 0.0),
            "factored_total": float(lc.get("factored_total", 0.0) or 0.0),
            "gamma_g": float(lc.get("gamma_g", 1.0) or 1.0),
            "gamma_q": float(lc.get("gamma_q", 1.0) or 1.0),
            "combo": lc.get("combo", None),
            "unit": lc.get("unit", getattr(cf, "unit", None)),
        }

        candidates_case_all, summary_case_df = run_one_load_case(
            case_name=case_name,
            out_dir=out_dir,
            systems_df=systems_df,
            required_max_loads=case_loads,   # keep param name for now; it’s really “case_loads”
            cf=cf,
            spans_override=spans_override,
            material_csv_path=args.materials,
            material_id=None,
            load_values_yaml=None,
            use_codechecks=False)

        candidates_all.append(candidates_case_all)
        if isinstance(summary_case_df, pd.DataFrame) and not summary_case_df.empty:
            summary_frames.append(summary_case_df)

    candidates_all_out = pd.concat(candidates_all, ignore_index=True) if candidates_all else pd.DataFrame()

    # This is what reporting expects
    df_sum = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()


    # optional: global expanded materials per floor from winners (convenience)
    try:
        # build a minimal winners set with _source_case so expansion maps correctly
        if candidates_all_out is not None and not candidates_all_out.empty:
            # candidates_all_out already includes _source_case from assemble_candidates_all_from_cases
            winners_expanded = spans_mod.expand_winners_and_write_materials_per_floor(
                candidates_all=candidates_all_out,
                floors_by_case=floors_by_case,
                floor_area_lookup=floor_area_lookup,
                materials_df=materials_df,
                out_dir=out_dir,
                logger=logger,
            )
    except Exception:
        logger.exception("[takeoff] Failed global expanded materials write")

    # -------------------------
    # code checks (optional)
    # -------------------------
    # Run code checks (optional) — WINNERS ONLY
    # Optional code checks on winners (only)
    if df_sum is None or not isinstance(df_sum, pd.DataFrame) or df_sum.empty:
        sum_path = out_dir / "summary_ranked_all.csv"
        if sum_path.exists():
            df_sum = pd.read_csv(sum_path, low_memory=False)
        else:
            # last-resort fallback (lets pipeline continue)
            df_sum = candidates_all_out.copy()

    # 2) Select winners robustly
    df_winners = df_sum.copy()
    if "rank" in df_winners.columns:
        df_winners["rank"] = pd.to_numeric(df_winners["rank"], errors="coerce")
        df_winners = df_winners[df_winners["rank"] == 1]
    elif "rank_within_case" in df_winners.columns:
        df_winners["rank_within_case"] = pd.to_numeric(df_winners["rank_within_case"], errors="coerce")
        df_winners = df_winners[df_winners["rank_within_case"] == 1]
    else:
        # fallback: one per (typology,type,total_load) by min carbon
        carbon_col=next((c for c in ("carbon_total_kgCO2", "carbon_total", "carbon_total_kgCO2e") if c in candidates_case_all.columns), "carbon_total_kgCO2"),

        group_cols = [c for c in ["typology", "type", "total_load"] if c in df_winners.columns]
        if carbon_col and group_cols:
            df_winners[carbon_col] = pd.to_numeric(df_winners[carbon_col], errors="coerce")
            idx = df_winners.groupby(group_cols, dropna=False)[carbon_col].idxmin()
            df_winners = df_winners.loc[idx]

        # ensure key exists/consistent
        if "system_variant" in df_winners.columns:
            df_winners["system_variant"] = df_winners["system_variant"].astype(str)

        # NOTE: end of winner-selection branches

    # -------------------------
    # code checks (optional) — WINNERS ONLY
    # -------------------------
    try:
        df_codechecks = run_codechecks_on_winners(
            df_winners=df_winners,
            out_dir=out_dir,
            cf=cf,
            args=args,
        )
    except Exception:
        logger.exception("[codechecks] Failed running code checks (winners-only).")
        df_codechecks = pd.DataFrame()

    # -------------------------
    # reporting (tables + charts)
    # -------------------------
    try:
        # floor assignments are your program map: floor -> category (occupancy)
        floor_assignments = dict(getattr(cf, "program", {}) or {})
        report_out_dir = out_dir / "reporting"

        reporting_mod.write_edca_reports_from_summary(
            out_dir=out_dir,
            floor_assignments=dict(getattr(cf, "program", {}) or {}),
            floor_area_lookup=floor_area_lookup,
            verbose=bool(getattr(args, "report_verbose", False) or getattr(args, "verbose", False)),
            metric="carbon_per_m2",
        )


        logger.info("[reporting] Wrote reports -> %s", report_out_dir)

    except Exception:
        logger.exception("[reporting] Failed to write reports")

    missing_cases = []

    logger.info("[run_edca] Done. Missing cases (if any): %s", missing_cases)

if __name__ == "__main__":
    main()