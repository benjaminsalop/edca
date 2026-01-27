#!/usr/bin/env python3
"""
run_edca.py

Top-level runner to execute the EDCA pipeline:
  - parse control file
  - load systems catalog & materials
  - (optionally) span sweep
  - compute per-candidate BOM -> carbon (per m²) and cost (per m²)
  - rank and summarise
  - write outputs & plots
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import math
import pandas as pd
import logging
from pathlib import Path

# import your modules (adjust paths if you moved files)
from edca_code.scripts.core.parse import ControlFile
from edca_code.scripts.core import systems as systems_mod
from edca_code.scripts.core import takeoff as takeoff_mod
from edca_code.scripts.core import carbon as carbon_mod
from edca_code.scripts.core import rank as rank_mod
from edca_code.scripts.core import reporting as reporting_mod

# optional code checks module (user provided example)
# Prefer an adapter at edca_code.scripts.code_checks, otherwise try likely continuouslab locations.
code_checks_mod = None
try:
    # preferred adapter (project-level)
    from edca_code.scripts import code_checks as code_checks_mod  # type: ignore
except Exception:
    code_checks_mod = None

# if no adapter, try known continuouslab locations (try the code_checks subpackage first)
if code_checks_mod is None:
    try:
        # your file lives at edca_code/scripts/code_checks/continuouslab.py
        from edca_code.scripts.code_checks import continuouslab as continuouslab_mod  # type: ignore
        code_checks_mod = continuouslab_mod
    except Exception:
        try:
            # fallback: some layouts put it under edca_code.scripts.core
            from edca_code.scripts.core import continuouslab as continuouslab_mod2  # type: ignore
            code_checks_mod = continuouslab_mod2
        except Exception:
            # last-ditch: try top-level package path
            try:
                from edca_code import continuouslab as continuouslab_mod3  # type: ignore
                code_checks_mod = continuouslab_mod3
            except Exception:
                code_checks_mod = None

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_candidates_for_span(systems_df: pd.DataFrame,
                                materials_df: pd.DataFrame,
                                span_value: float,
                                required_loads: dict,
                                depth_enabled: bool,
                                depth_limit_mm: float | None) -> pd.DataFrame:
    """
    Given systems_df (full catalog) and materials table, filter for span_value
    and compute carbon_per_m2 and cost_per_m2 for each surviving candidate.
    Returns a DataFrame with candidate rows plus appended carbon/cost columns.
    """
    # filter systems by span + loads + depth
    filtered = systems_mod.filter_systems(
        systems_df,
        min_span_required=span_value,
        required_loads=required_loads,
        depth_limit_enabled=depth_enabled,
        depth_limit_mm=depth_limit_mm,
    )

    if filtered.empty:
        return filtered

    results = []
    for _, row in filtered.iterrows():
        # basic identity columns (preserve commonly used names)
        idx_row = row.to_dict()
        sys_id = idx_row.get("system_variant") or idx_row.get("system_id") or idx_row.get("system_family", "")
        idx_row["system_variant"] = sys_id

        # compute BOM per m2
        bom_m2 = takeoff_mod.bom_per_m2_from_system_row(row)
        # expand to 1 m² to get per-m² totals
        floor_bom = takeoff_mod.expand_bom_to_floor(bom_m2, floor_area_m2=1.0, assemblies=1)

        # carbon & cost for 1 m²
        carbon_res = carbon_mod.compute_assembly_carbon_from_bom(floor_bom, materials_df, include_a4_a5=True)
        totals = carbon_res.get("totals", {}) if isinstance(carbon_res, dict) else {}

        # defensive numeric extraction for total carbon
        _raw_total = totals.get("total", None)
        try:
            # handle numeric strings, Decimal, etc.
            total_carbon = float(_raw_total) if _raw_total is not None else float("nan")
        except Exception:
            # try coercion via pandas if not plain float-able
            try:
                import pandas as _pd
                total_carbon = float(_pd.to_numeric(_raw_total, errors="coerce"))
            except Exception:
                total_carbon = float("nan")
                # helpful debug logging
                print(f"[WARN] Could not parse total carbon for system {sys_id!r}: raw_total={repr(_raw_total)}; carbon_res keys={list(carbon_res.keys())}")

        # same for cost
        _raw_cost = totals.get("total_cost", None)
        try:
            total_cost = float(_raw_cost) if _raw_cost is not None else float("nan")
        except Exception:
            try:
                import pandas as _pd
                total_cost = float(_pd.to_numeric(_raw_cost, errors="coerce"))
            except Exception:
                total_cost = float("nan")

        # DEBUG: inspect carbon_res / totals for problematic candidates
        if totals is None or totals.get("total", None) is None:
            print("[DEBUG] Missing totals['total'] for system:", sys_id)
            print("  carbon_res keys:", list(carbon_res.keys()))
            print("  totals:", repr(totals))
            # show a small sample of per_material breakdown if present
            pm = carbon_res.get("per_material")
            if pm is not None:
                print("  per_material (sample):", pm if isinstance(pm, (list, dict)) else repr(pm)[:200])

        idx_row["span_evaluated_m"] = float(span_value)
        idx_row["carbon_total_kgCO2"] = total_carbon
        idx_row["carbon_per_m2"] = total_carbon
        idx_row["cost_total"] = total_cost
        idx_row["cost_per_m2"] = total_cost
        # attach breakdown for debugging (optional heavy field)
        idx_row["_carbon_breakdown"] = carbon_res["per_material"]
        results.append(idx_row)

    out_df = pd.DataFrame(results)
    # coerce numeric columns
    for c in ("carbon_total_kgCO2", "carbon_per_m2", "cost_total", "cost_per_m2", "span_evaluated_m"):
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    return out_df


# -------------------------
# Main runner
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="run_edca", description="Run the EDCA pipeline (systems -> takeoff -> carbon -> rank -> report)")
    p.add_argument("--control", "-c", required=True, help="Path to control file YAML")
    p.add_argument("--systems", "-s", required=True, help="Path to systems_variants parquet/csv")
    p.add_argument("--materials", "-m", required=True, help="Path to materials CSV")
    p.add_argument("--out", "-o", default="edca_outputs", help="Output directory")
    p.add_argument("--span-step", type=float, default=0.5, help="Span sweep step in metres")
    p.add_argument("--span-min", type=float, default=None, help="Override minimum span (m)")
    p.add_argument("--span-max", type=float, default=None, help="Override maximum span (m)")
    p.add_argument("--no-sweep", action="store_true", help="Do not perform span sweep; evaluate only min span")
    p.add_argument("--run-codechecks", action="store_true", help="Run code checks on best candidates (may need code_checks module)")
    args = p.parse_args(argv)

    out_dir = ensure_dir(Path(args.out))

    # 1) load control file
    cf = ControlFile.from_path(args.control)
    print(f"[run_edca] Loaded control file: project={cf.project_name}, unit={cf.unit}")

    # 2) load systems catalog + materials
    systems_df = systems_mod.load_systems_catalog(args.systems)
    materials_df = carbon_mod.load_materials_table(args.materials)
    print(f"[run_edca] Systems rows: {len(systems_df):,}; Materials rows: {len(materials_df):,}")

    # 3) compute required loads from occupancies (use control.unit as filter)
    occ_path = Path(cf.data_dir) / "occupancies.csv"
    required_loads = systems_mod.compute_required_max_loads_from_occupancies(str(occ_path), unit_filter=getattr(cf, "unit", None))
    print(f"[run_edca] Required loads (from occupancies): LL={required_loads['max_ll']}, SDL={required_loads['max_sdl']}")

    # 4) choose span sweep values
    spans_setting = getattr(cf, "spans", None)
    # prefer CLI overrides
    if args.span_min is not None and args.span_max is not None:
        span_min = float(args.span_min)
        span_max = float(args.span_max)
    elif spans_setting and isinstance(spans_setting, (list, tuple)) and len(spans_setting) >= 2:
        span_min = float(min(spans_setting))
        span_max = float(max(spans_setting))
    else:
        # fallback to single-value or zero
        if spans_setting and isinstance(spans_setting, (list, tuple)) and len(spans_setting) == 1:
            span_min = float(spans_setting[0])
            span_max = span_min
        else:
            span_min = float(args.span_min) if args.span_min is not None else 0.0
            span_max = float(args.span_max) if args.span_max is not None else span_min

    if args.no_sweep or span_min == span_max:
        span_values = [span_min]
    else:
        step = float(args.span_step)
        # inclusive range
        nsteps = max(1, int(math.floor((span_max - span_min) / step)) + 1)
        span_values = [round(span_min + i * step, 6) for i in range(nsteps + 1) if (span_min + i * step) <= span_max + 1e-9]
        if not span_values:
            span_values = [span_min]

    print(f"[run_edca] Evaluating spans: {span_values}")

    # depth limits from control
    depth_enabled = bool(getattr(cf, "depth_limit_enabled", False))
    depth_limit_val = getattr(cf, "depth_limit", None) if depth_enabled else None

    # 5) For each span, compute candidate carbon per m2
    all_span_results = []
    for span in span_values:
        print(f"[run_edca] Evaluating span = {span} m ...")
        span_candidates = compute_candidates_for_span(
            systems_df=systems_df,
            materials_df=materials_df,
            span_value=span,
            required_loads=required_loads,
            depth_enabled=depth_enabled,
            depth_limit_mm=depth_limit_val,
        )

        if span_candidates is None or span_candidates.empty:
            print(f"[run_edca] No candidates for span {span}")
            continue

        # add span column already present; ensure have key identity columns
        span_candidates["evaluated_span_m"] = span

        all_span_results.append(span_candidates)

        # save per-span CSV
        span_csv = out_dir / f"candidates_span_{span:.2f}m.csv"
        span_candidates.to_csv(span_csv, index=False)
        print(f"[run_edca] wrote {len(span_candidates)} candidate rows to {span_csv}")

    if not all_span_results:
        print("[run_edca] No candidates found for any span — exiting")
        sys.exit(1)

    # concat across spans
    candidates_all = pd.concat(all_span_results, ignore_index=True, sort=False)

    # load the families table (use the same file the rest of your pipeline has)
    # Replace this path variable with whatever you pass as --system_families or have already loaded.
    sf_path = Path(args.system_families) if hasattr(args, "system_families") else Path("inputs/canonical/system_families.parquet")
    if sf_path.exists():
        sf_df = pd.read_parquet(str(sf_path))
        # ensure the expected key exists (system_family) and the columns type/category exist in sf_df
        if "system_family" not in sf_df.columns:
            # try index
            if sf_df.index.name == "system_family":
                sf_df = sf_df.reset_index()
        # pick columns to merge (if missing fallback to empty)
        expected_cols = []
        if "type" in sf_df.columns:
            expected_cols.append("type")
        if "category" in sf_df.columns:
            expected_cols.append("category")
        # always include system_family for merge
        if "system_family" not in sf_df.columns:
            raise RuntimeError("system_families table missing 'system_family' column")
        merge_cols = ["system_family"] + expected_cols
        sf_small = sf_df[merge_cols].drop_duplicates(subset=["system_family"]).copy()
        # merge into candidates_all (left join)
        candidates_all = candidates_all.merge(sf_small, on="system_family", how="left")
    else:
        # if families file not found, create placeholder columns to avoid later KeyErrors
        candidates_all["type"] = candidates_all["system_family"]
        candidates_all["category"] = candidates_all.get("category", "")

    # Now call ranking, but group lowest_per_type by the "type" column we just merged in
    summaries = rank_mod.rank_and_export_summary(
        candidates_all,
        group_by_type=True,
        type_col="type",           # <- use 'type' (from system_families)
        brand_col="system_variant",
        carbon_col="carbon_total_kgCO2",
    )

    # After summaries are written, produce the requested PNGs.
    # Assume rank_and_export_summary returned (or wrote) dataframes; if it returns them use them,
    # otherwise read the CSVs it wrote.
    # Prefer using returned DataFrames if your rank function returns them:
    ranked_all = summaries.get("ranked_all") if isinstance(summaries, dict) else None
    lowest_per_type = summaries.get("lowest_per_type") if isinstance(summaries, dict) else None
    lowest_per_brand = summaries.get("lowest_per_brand") if isinstance(summaries, dict) else None

    # fallback: read the files written to outputs if returned None
    if lowest_per_type is None:
        try:
            lowest_per_type = pd.read_csv(str(out_dir / "lowest_per_type.csv"))
        except Exception:
            lowest_per_type = pd.read_csv(str(out_dir / "lowest_per_type.csv"), index_col=None)

    # Make sure the merged type/category columns are present (if they came from the candidates)
    if "type" not in lowest_per_type.columns and "system_family" in lowest_per_type.columns and sf_path.exists():
        # join to fetch type/category
        lowest_per_type = lowest_per_type.merge(sf_small, on="system_family", how="left")

    # plotting helpers in reporting_mod: (they will be used below)
    # produce type.png (grouping by 'type')
    try:
        type_png = out_dir / "type.png"
        reporting_mod.plot_lowest_rows_by_group(lowest_per_type, group_col="type", out_path=type_png)
    except Exception:
        logging.exception("could not create type.png")

    # produce typology.png (grouping by 'category')
    try:
        typ_png = out_dir / "typology.png"
        reporting_mod.plot_lowest_rows_by_group(lowest_per_type, group_col="category", out_path=typ_png)
    except Exception:
        logging.exception("could not create typology.png")


    # 6) ranking: basic ranking & group summaries
    # ensure required ranking column presence
    if "carbon_total_kgCO2" not in candidates_all.columns:
        candidates_all["carbon_total_kgCO2"] = candidates_all.get("carbon_per_m2", 0.0)

    # Add convenience columns if missing
    if "system_family" not in candidates_all.columns and "system_variant" in candidates_all.columns:
        candidates_all["system_family"] = candidates_all["system_variant"]

    # produce ranked tables: all, lowest per (type, system_family) etc.
    summaries = rank_mod.rank_and_export_summary(candidates_all, group_by_type=True, type_col="system_family", brand_col="system_variant", carbon_col="carbon_total_kgCO2")

    # write outputs
    candidates_all.to_csv(out_dir / "candidates_all_spans.csv", index=False)
    (out_dir / "candidates_all_spans.json").write_text(candidates_all.to_json(orient="records"))

    # write summaries (CSV)
    for name, dfsum in summaries.items():
        path = out_dir / f"{name}.csv"
        dfsum.to_csv(path, index=False)
        print(f"[run_edca] wrote summary {name} -> {path}")

    # --- compute a numeric analysed area (m²) from control file FLOOR_PLATE ---
    def _compute_area_m2_from_floorplate(fp: object, unit_flag: str, n_floors: int) -> float | None:
        """
        fp: FLOOR_PLATE value from control file (could be dict or numeric)
        unit_flag: cf.unit (expected 'metric' or 'imperial')
        n_floors: cf.num_floors
        Returns total analysed area in m² (area_per_floor * n_floors) or None if unknown.
        """
        if fp is None:
            return None

        # If user already provided a numeric area (legacy), accept it as area per floor
        if isinstance(fp, (int, float)):
            area_per_floor = float(fp)
        elif isinstance(fp, dict):
            mode = str(fp.get("mode", "dims")).strip().lower()
            area_per_floor = None
            if mode == "area":
                # prefer explicit key 'area_per_floor' or fallback 'area'
                area_per_floor = fp.get("area_per_floor") or fp.get("area")
            else:
                # dims mode (length & width expected)
                length = fp.get("length")
                width = fp.get("width")
                if length is not None and width is not None:
                    try:
                        area_per_floor = float(length) * float(width)
                    except Exception:
                        area_per_floor = None
        else:
            # unknown type
            return None

        if area_per_floor is None:
            return None

        # convert to metric (m²) if unit is imperial (inputs likely ft or ft²)
        if isinstance(unit_flag, str) and unit_flag.strip().lower() != "metric":
            # if area_per_floor was given as ft² (mode 'area' or computed from ft dims),
            # convert ft² -> m²: 1 ft² = 0.09290304 m²
            area_per_floor_m2 = float(area_per_floor) * 0.09290304
        else:
            area_per_floor_m2 = float(area_per_floor)

        # total analysed area (all floors)
        try:
            total_area = area_per_floor_m2 * float(n_floors)
        except Exception:
            total_area = area_per_floor_m2

        return total_area

    # compute once and pass numeric area_m2 into reporting
    _area_m2 = _compute_area_m2_from_floorplate(getattr(cf, "floor_plate", None), getattr(cf, "unit", None), getattr(cf, "num_floors", 1))

    # call reporting with numeric area (or None if unknown)
    reporting_mod.print_summary(candidates_all, metric="carbon_per_m2", area_m2=_area_m2)
    reporting_mod.generate_simple_report(candidates_all, area_m2=_area_m2, save_dir=str(out_dir), show=False)

    # 8) Optional code checks on best candidates per type
        # 8) Optional code checks on best candidates per type
        # 8) Optional code checks on best candidates per type
        # 8) Optional code checks on best candidates per type (verbose CLI + text file + json)
        # 8) Optional code checks on best candidates per type (DIRECT import of known submodule)
    if args.run_codechecks:
        print("[run_edca] Running code checks on best per type...")

        # load lowest_per_type (prefer in-memory summaries, else on-disk CSV)
        best_by_type = None
        if isinstance(summaries, dict) and "lowest_per_type" in summaries:
            best_by_type = summaries["lowest_per_type"]
        else:
            try:
                best_by_type = pd.read_csv(out_dir / "lowest_per_type.csv")
            except Exception:
                best_by_type = None

        if best_by_type is None or best_by_type.empty:
            print("[run_edca] No lowest_per_type table found; skipping code checks.")
        else:
            # results containers
            codecheck_results = []
            verbose_lines = []
            num_success = 0
            num_fail = 0
            total_runs = 0

            # Try explicit imports of the module you reported exists
            preferred_module = None
            tried_names = []
            import importlib
            explicit_names = [
                "edca_code.scripts.code_checks.continuouslab",
                "edca_code.scripts.code_checks",
                "edca_code.scripts.core.continuouslab",
                "edca_code.continuouslab",
            ]
            for nm in explicit_names:
                tried_names.append(nm)
                try:
                    m = importlib.import_module(nm)
                    # prefer actual module that has function
                    if hasattr(m, "run_code_check_for_typology"):
                        preferred_module = m
                        break
                    # if package itself has no function but submodule exists under same package, try import submodule explicitly
                    if nm.endswith("code_checks"):
                        try:
                            sub = importlib.import_module(nm + ".continuouslab")
                            if hasattr(sub, "run_code_check_for_typology"):
                                preferred_module = sub
                                break
                        except Exception:
                            pass
                except Exception:
                    # continue trying other names
                    continue

            if preferred_module is None:
                # nothing found; report and still write empty verbose report
                print("[run_edca] Could not import any code_checks / continuouslab module. Tried:", tried_names)
                verbose_lines.append("No runnable code-check entrypoint found. Tried: " + ", ".join(tried_names))
            else:
                # get function
                fn = getattr(preferred_module, "run_code_check_for_typology", None)
                if fn is None:
                    print(f"[run_edca] Module {preferred_module.__name__} imported but run_code_check_for_typology not found.")
                    verbose_lines.append(f"Module {preferred_module.__name__} imported but no run_code_check_for_typology.")
                else:
                    fn_name = f"{preferred_module.__name__}.run_code_check_for_typology"
                    print(f"[run_edca] Using code-check function: {fn_name}")

                    # defensive invoker that tries plausible signatures
                    def _invoke(fn, selector):
                        try:
                            # try: fn(systems_catalog, typology_selector=..., material_csv_path=..., ...)
                            try:
                                return {"ok": True, "result": fn(systems_df, typology_selector=selector, material_csv_path=args.materials, load_combos_yaml=None, load_values_yaml=None, program_df=None)}
                            except TypeError:
                                pass
                            # try: fn(systems_df, selector)
                            try:
                                return {"ok": True, "result": fn(systems_df, selector)}
                            except TypeError:
                                pass
                            # try: fn(selector)
                            try:
                                return {"ok": True, "result": fn(selector)}
                            except TypeError:
                                pass
                            # try no-arg
                            try:
                                return {"ok": True, "result": fn()}
                            except TypeError:
                                pass
                            return {"ok": False, "error": "no supported signature matched"}
                        except Exception as e:
                            return {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

                    # interpreter heuristic (same as before)
                    def _interpret_pass(result):
                        if isinstance(result, bool):
                            return bool(result), ""
                        if result is None:
                            return False, "no result returned"
                        if isinstance(result, dict):
                            for key in ("pass", "passed", "success", "ok"):
                                if key in result:
                                    val = result.get(key)
                                    if isinstance(val, bool):
                                        return val, f"reported {key}={val}"
                                    try:
                                        return bool(val), f"reported {key}={val}"
                                    except Exception:
                                        continue
                            if "errors" in result and result.get("errors"):
                                return False, f"errors: {result.get('errors')}"
                            if any(k in result for k in ("diagnostics", "reinforcement", "deflection", "shear", "flex")):
                                return True, "diagnostics present (no explicit failure)"
                            return True, "dict result (treated as pass)"
                        try:
                            return bool(result), ""
                        except Exception:
                            return False, "uninterpretable result"

                    # helper to resolve a friendly selector (type / family / variant) into something
                    # continuouslab expects (prefer system_id, then system_variant, then integer index)
                    def _resolve_selector(selector):
                        # try numeric / numeric-string -> integer index if valid
                        try:
                            if isinstance(selector, int):
                                return selector
                            if isinstance(selector, str) and selector.strip().isdigit():
                                idx = int(selector.strip())
                                if 0 <= idx < len(systems_df):
                                    return idx
                        except Exception:
                            pass

                        # try matching against common columns in systems_df
                        for col in ("system_id", "system_variant", "system_family", "type"):
                            if col in systems_df.columns:
                                try:
                                    mask = systems_df[col].astype(str) == str(selector)
                                except Exception:
                                    mask = systems_df[col].astype(str).apply(lambda x: x == str(selector))
                                if mask.any():
                                    row = systems_df[mask].iloc[0]
                                    # prefer canonical system_id if present and non-null
                                    if "system_id" in systems_df.columns and pd.notna(row.get("system_id")):
                                        return row.get("system_id")
                                    # else prefer variant string
                                    if "system_variant" in row and pd.notna(row.get("system_variant")):
                                        return row.get("system_variant")
                                    # else try to return integer-like index label
                                    rn = row.name
                                    try:
                                        if isinstance(rn, int):
                                            return rn
                                        if isinstance(rn, str) and rn.isdigit():
                                            return int(rn)
                                    except Exception:
                                        pass
                                    # fallback: return the actual index label
                                    return rn

                        # last resort: return the original selector (will be passed through; errors will be captured)
                        return selector

                    # iterate rows, resolve selector, invoke the check function, and record results
                    for _, crow in best_by_type.iterrows():
                        total_runs += 1

                        # pick the best available selector from the row (same priority as before)
                        selector_type = crow.get("type") if "type" in crow.index else None
                        selector_variant = crow.get("system_variant") if "system_variant" in crow.index else None
                        selector_family = crow.get("system_family") if "system_family" in crow.index else None
                        selector = selector_type or selector_variant or selector_family or "<unknown>"

                        # resolve into something continuouslab will accept
                        resolved_selector = _resolve_selector(selector)

                        # call the function defensively using existing _invoke (which tries several signatures)
                        invoke_res = _invoke(fn, resolved_selector)

                        entry = {
                            "selector": selector,
                            "resolved_selector": resolved_selector,
                            "fn_used": fn_name,
                            "ok": bool(invoke_res.get("ok", False)),
                            "result": invoke_res.get("result") if "result" in invoke_res else None,
                            "error": invoke_res.get("error") if "error" in invoke_res else None
                        }

                        # interpret pass/fail and attach reason
                        passed, reason = _interpret_pass(entry["result"])
                        entry["passed"] = passed
                        entry["reason"] = reason
                        codecheck_results.append(entry)

                        # produce the same verbose CLI/file lines as before, now including resolved_selector
                        if entry["ok"] and entry["passed"]:
                            num_success += 1
                            line = f"PASS: selector={selector} resolved_to={resolved_selector} via {entry['fn_used']} — {reason or 'OK'}"
                            print(line)
                            verbose_lines.append(line)
                        elif entry["ok"] and not entry["passed"]:
                            num_fail += 1
                            line = f"FAIL: selector={selector} resolved_to={resolved_selector} via {entry['fn_used']} — {reason or 'FAILED (no explicit reason)'}"
                            print(line)
                            verbose_lines.append(line)
                        else:
                            num_fail += 1
                            line = f"ERROR: selector={selector} resolved_to={resolved_selector} via {entry.get('fn_used','<unknown>')} — {entry.get('error','no error message')}"
                            print(line)
                            verbose_lines.append(line)


            # write outputs (json + verbose txt)
            try:
                (out_dir / "codechecks_summary.json").write_text(json.dumps(codecheck_results, indent=2, default=str))
            except Exception:
                logging.exception("Failed to write codechecks_summary.json")

            try:
                verbose_path = out_dir / "codechecks_verbose.txt"
                with verbose_path.open("w", encoding="utf8") as vf:
                    vf.write("EDCA Code Checks - Verbose Report\n")
                    vf.write("=================================\n\n")
                    vf.write(f"Total checks attempted: {total_runs}\n")
                    vf.write(f"Passed: {num_success}\n")
                    vf.write(f"Failed/Errors: {num_fail}\n\n")
                    vf.write("\n".join(verbose_lines))
                    vf.write("\n")
                print(f"[run_edca] Wrote verbose code checks report to {verbose_path}")
            except Exception:
                logging.exception("Failed to write codechecks_verbose.txt")

            print(f"[run_edca] Code checks completed: success={num_success}, fail={num_fail}, total={len(codecheck_results)}")

    print("[run_edca] Done.")

if __name__ == "__main__":
    main()
