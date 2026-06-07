import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from typing import Any

from edca_code.scripts.core import systems as systems_mod
from edca_code.scripts.core import takeoff as takeoff_mod
from edca_code.scripts.core import carbon as carbon_mod
from edca_code.scripts.core.systems import filter_systems 
from edca_code.scripts.core.old.utils import infer_type, reorder_output_columns
from edca_code.scripts.core import rank as rank_mod
from edca_code.scripts.core.old.utils import ensure_dir

logger = logging.getLogger("spans")

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

def resolve_span_values(cf, args, logger: Optional[logging.Logger] = None) -> List[float]:
    """
    Resolve the list of span values (in metres) to evaluate.

    Priority:
      1) CLI overrides (--span-min/--span-max/--span-step/--no-sweep)
      2) Control file spans list (cf.spans) already parsed/expanded by parse.parse_spans
      3) Fallback to cf.ideal_column_spacing (single value) or 6.0 m
    """
    log = logger or globals().get("logger") or logging.getLogger("spans")

    # 1) CLI overrides take precedence if either bound is provided
    span_min = getattr(args, "span_min", None)
    span_max = getattr(args, "span_max", None)
    span_step = float(getattr(args, "span_step", 0.5) or 0.5)
    no_sweep = bool(getattr(args, "no_sweep", False))

    if span_min is not None or span_max is not None:
        # If only one bound provided, treat it as a single span
        if span_min is None:
            span_min = float(span_max)
        if span_max is None:
            span_max = float(span_min)

        mn = float(span_min)
        mx = float(span_max)
        if mx < mn:
            mn, mx = mx, mn

        if no_sweep or abs(mx - mn) < 1e-9:
            spans = [round(mn, 6)]
            log.info("[span] Span override: evaluating single span %.3f m (--no-sweep or equal bounds).", spans[0])
            return spans

        if span_step <= 0:
            span_step = 0.5
            log.warning("[span] Invalid --span-step; defaulting to %.2f m.", span_step)

        spans: List[float] = []
        x = mn
        # inclusive sweep with tolerance
        while x <= mx + 1e-9:
            spans.append(round(x, 6))
            x += span_step

        log.info("[span] Span override: sweeping %.3f..%.3f m step=%.3f (%d values).", mn, mx, span_step, len(spans))
        return spans

    # 2) Control file spans
    cf_spans = getattr(cf, "spans", None)
    if isinstance(cf_spans, (int, float, str)):
        try:
            return [float(cf_spans)]
        except Exception:
            pass
    if isinstance(cf_spans, (list, tuple)) and len(cf_spans) > 0:
        spans = [float(v) for v in cf_spans]
        if no_sweep:
            spans = [min(spans)]
            log.info("[span] Control-file spans present but --no-sweep specified; using min span %.3f m.", spans[0])
        return spans

    # 3) Fallbacks
    ics = getattr(cf, "ideal_column_spacing", None)
    if ics is not None:
        try:
            spans = [float(ics)]
            log.warning("[span] No spans specified; falling back to IDEAL_COLUMN_SPACING=%.3f m.", spans[0])
            return spans
        except Exception:
            pass

    log.warning("[span] No spans specified; falling back to default span 6.0 m.")
    return [6.0]

def compute_candidates_for_span(
    systems_df: pd.DataFrame,
    materials_df: pd.DataFrame,
    span_value: float,
    required_loads: Dict[str, float],
    unit: Optional[str] = None,
    depth_enabled: bool = False,
    depth_limit_mm: Optional[float] = None,
    require_separate_checks: bool = False,
    prefiltered: bool = False,
) -> pd.DataFrame:
    """
    If prefiltered=False, filters systems_df for span + loads.
    If prefiltered=True, assumes systems_df is already filtered (span/loads/unit/depth)
    and only computes BOM+carbon.

    Adds carbon breakdown columns (all per m²):
      - carbon_concrete_per_m2
      - carbon_steel_per_m2              (legacy combined steel bucket, backward-compatible)
      - carbon_structural_steel_per_m2   (new)
      - carbon_rebar_per_m2              (new)
      - carbon_pt_per_m2                 (new)
      - carbon_timber_per_m2
      - carbon_screed_per_m2
      - carbon_by_material_id_json
    """

    def _clean_mat_id(v) -> Optional[str]:
        """Normalize material IDs from row cells (handle NaN/blank)."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return None
        return s

    if not prefiltered:
        filtered = systems_mod.filter_systems(
            df=systems_df.copy(),
            min_span_required=span_value,
            required_loads=required_loads,
            unit=unit,
            depth_limit_enabled=depth_enabled,
            depth_limit_mm=depth_limit_mm,
            require_separate_checks=require_separate_checks,
        )
    else:
        filtered = systems_df.copy()

    if filtered is None or filtered.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in filtered.reset_index(drop=True).iterrows():
        r = row.copy()  # always define early so except block can use it
        try:
            # Compute BOM + carbon ONCE (per m²)
            bom_m2 = takeoff_mod.bom_per_m2_from_system_row(row)
            carbon_res = carbon_mod.compute_assembly_carbon_from_bom(
                bom_m2, materials_df, include_a4_a5=False
            )

            totals = carbon_res.get("totals", {}) or {}
            by_cat = carbon_res.get("totals_by_category", {}) or {}
            by_mat = carbon_res.get("totals_by_material_id", {}) or {}

            # Normalize material-id keys to strings for robust lookup
            by_mat_norm = {str(k): float(v) for k, v in by_mat.items()}

            # Total A1-A3 (per m², since BOM is per m²)
            carbon_a1a3 = float(totals.get("total_a1a3", 0.0))
            r["carbon_total_kgCO2"] = carbon_a1a3  # legacy name (actually per m²)
            r["carbon_total_per_m2"] = carbon_a1a3  # clearer alias

            # Category-level (legacy/backward-compatible)
            r["carbon_concrete_per_m2"] = float(by_cat.get("concrete", 0.0))
            r["carbon_steel_per_m2"] = float(by_cat.get("steel", 0.0))  # combined bucket
            r["carbon_timber_per_m2"] = float(by_cat.get("timber", 0.0))
            r["carbon_screed_per_m2"] = float(by_cat.get("screed", 0.0))

            # New explicit splits by material ID (robust to category taxonomy changes)
            steel_id = _clean_mat_id(row.get("material_steel_id"))
            rebar_id = _clean_mat_id(row.get("material_rebar_id"))
            pt_id = _clean_mat_id(row.get("material_pt_id"))

            r["carbon_structural_steel_per_m2"] = float(by_mat_norm.get(steel_id, 0.0)) if steel_id else 0.0
            r["carbon_rebar_per_m2"] = float(by_mat_norm.get(rebar_id, 0.0)) if rebar_id else 0.0
            r["carbon_pt_per_m2"] = float(by_mat_norm.get(pt_id, 0.0)) if pt_id else 0.0

            # Helpful trace/debug fields
            r["carbon_by_material_id_json"] = json.dumps(by_mat_norm, sort_keys=True)
            r["_bom_keys"] = ",".join(sorted(map(str, bom_m2.keys())))

            rows.append(r)

        except Exception:
            logger.exception(
                "[takeoff] Error computing BOM/carbon for candidate index %s (span %s). Falling back to NaN.",
                idx, span_value
            )

            # Safe fallback row with NaNs / empty traces
            r["carbon_total_kgCO2"] = float("nan")
            r["carbon_total_per_m2"] = float("nan")

            r["carbon_concrete_per_m2"] = float("nan")
            r["carbon_steel_per_m2"] = float("nan")
            r["carbon_structural_steel_per_m2"] = float("nan")
            r["carbon_rebar_per_m2"] = float("nan")
            r["carbon_pt_per_m2"] = float("nan")
            r["carbon_timber_per_m2"] = float("nan")
            r["carbon_screed_per_m2"] = float("nan")

            r["carbon_by_material_id_json"] = "{}"
            r["_bom_keys"] = ""

            rows.append(r)

    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_span_results(span_results: List[Tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate results for all spans into a single DataFrame, adding a 'span' column."""
    parts = []
    for span, df in span_results:
        if df is None or df.empty:
            continue
        d = df.copy()
        d["span"] = float(span)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)

def run_span_sweep(
    *,
    load_case_name: str,
    out_dir: Path,
    systems_df: pd.DataFrame,
    materials_df: pd.DataFrame,
    span_values: List[float],
    required_loads_case: Dict[str, float],
    cf_unit: Optional[str] = None,
    depth_limit_mm: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    debug: bool | None = None,
    debug_max_rows: int = 25,
    debug_save_snapshots: bool = False,
    loads_df_floor: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Span sweep with comprehensive debug + a funnel/audit CSV.

    IMPORTANT:
      - If loads_df_floor is provided and contains rows for this load_case_name,
        this function rebuilds required_loads_case from those rows so that
        max_factored_total comes from the actual governing combo results.
      - If not, it falls back to the passed required_loads_case dict.

    Writes:
      - per-span candidates CSVs
      - span_sweep_audit.csv
      - optional snapshots into out_dir/debug/
    """
    log = logger or globals().get("logger") or logging.getLogger("spans")
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    dbg = _dbg_enabled(debug)

    def _snap(df: pd.DataFrame, name: str) -> None:
        if not (dbg and debug_save_snapshots):
            return
        try:
            ddir = ensure_dir(out_dir / "debug")
            fp = ddir / f"{name}.csv"
            df.to_csv(fp, index=False)
            log.debug("[debug] wrote snapshot %s (%s)", fp, df.shape)
        except Exception:
            log.exception("[debug] snapshot failed: %s", name)

    def _pick_series(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
        for c in candidates:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return None

    def _rebuild_required_loads_case(
        load_case_name: str,
        required_loads_case: Dict[str, float],
        loads_df_floor: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Prefer rebuilding from loads_df_floor so that factored_total is based on the
        actual governing combo result for this load case.
        """
        out = dict(required_loads_case or {})

        if loads_df_floor is None or not isinstance(loads_df_floor, pd.DataFrame) or loads_df_floor.empty:
            if "max_factored_total" not in out:
                log.warning(
                    "[span] %s: loads_df_floor not provided and max_factored_total missing; "
                    "prefilter will fall back to raw max_total.",
                    load_case_name,
                )
            return out

        if "occupancy" not in loads_df_floor.columns:
            log.warning(
                "[span] %s: loads_df_floor has no 'occupancy' column; using passed required_loads_case.",
                load_case_name,
            )
            return out

        occ = loads_df_floor["occupancy"].astype(str).str.strip()
        mask = occ.eq(str(load_case_name).strip())

        # case-insensitive fallback
        if not mask.any():
            mask = occ.str.lower().eq(str(load_case_name).strip().lower())

        case_rows = loads_df_floor.loc[mask].copy()

        if case_rows.empty:
            log.warning(
                "[span] %s: no matching rows found in loads_df_floor; using passed required_loads_case=%s",
                load_case_name,
                out,
            )
            return out

        raw_sdl = _pick_series(case_rows, "SDL", "sdl")
        raw_ll = _pick_series(case_rows, "LL", "ll")
        factored_sdl = _pick_series(case_rows, "factored_sdl", "max_factored_sdl")
        factored_ll = _pick_series(case_rows, "factored_ll", "max_factored_ll")
        factored_total = _pick_series(case_rows, "factored_total", "max_factored_total")

        rebuilt: Dict[str, float] = {}

        rebuilt["max_sdl"] = float(raw_sdl.fillna(0.0).max()) if raw_sdl is not None else float(out.get("max_sdl", 0.0) or 0.0)
        rebuilt["max_ll"] = float(raw_ll.fillna(0.0).max()) if raw_ll is not None else float(out.get("max_ll", 0.0) or 0.0)

        if raw_sdl is not None and raw_ll is not None:
            rebuilt["max_total"] = float((raw_sdl.fillna(0.0) + raw_ll.fillna(0.0)).max())
        else:
            rebuilt["max_total"] = float(out.get("max_total", rebuilt["max_sdl"] + rebuilt["max_ll"]) or 0.0)

        if factored_sdl is not None:
            rebuilt["max_factored_sdl"] = float(factored_sdl.fillna(0.0).max())

        if factored_ll is not None:
            rebuilt["max_factored_ll"] = float(factored_ll.fillna(0.0).max())

        if factored_total is not None:
            rebuilt["max_factored_total"] = float(factored_total.fillna(0.0).max())
        elif ("max_factored_sdl" in rebuilt) and ("max_factored_ll" in rebuilt):
            rebuilt["max_factored_total"] = float(
                rebuilt["max_factored_sdl"] + rebuilt["max_factored_ll"]
            )
            log.warning(
                "[span] %s: factored_total column missing in loads_df_floor; "
                "using max_factored_sdl + max_factored_ll as fallback.",
                load_case_name,
            )

        return rebuilt

    unit = cf_unit

    # Rebuild demand for this case from loads_df_floor whenever possible.
    resolved_required_loads_case = _rebuild_required_loads_case(
        load_case_name=load_case_name,
        required_loads_case=required_loads_case,
        loads_df_floor=loads_df_floor,
    )

    log.info("[debug] %s required_loads_case = %s", load_case_name, resolved_required_loads_case)

    # -------------------------
    # Stage 0: load-based prefilter
    # -------------------------
    try:
        prefiltered_df = filter_systems(
            df=systems_df.copy(),
            min_span_required=None,
            required_loads=resolved_required_loads_case,
            unit=unit,
            require_separate_checks=False,
            depth_limit_enabled=False,
            depth_limit_mm=None,
            debug=dbg,
            debug_tag=f"{load_case_name}::prefilter",
        )
    except Exception:
        log.exception("[systems] %s: load-based prefilter failed; falling back to full catalog", load_case_name)
        prefiltered_df = systems_df.copy()

    log.info("[systems] %s: catalog reduced to %d rows after load pre-filter", load_case_name, len(prefiltered_df))
    if dbg:
        _dbg_df(
            f"{load_case_name}.prefiltered_df",
            prefiltered_df,
            explicit=True,
            max_rows=debug_max_rows,
            cols=["system_variant", "system_family", "unit", "max_span", "sdl", "sdl_partition", "ll", "sdl_total", "total_capacity"],
        )

    _snap(prefiltered_df, f"{load_case_name}__prefiltered_df")

    audit_rows: List[Dict[str, Any]] = []
    span_results_case: List[Tuple[float, pd.DataFrame]] = []

    for span in span_values:
        span = float(span)
        log.info("[span] %s: evaluating span = %.2f m", load_case_name, span)

        audit_rows.append({
            "case": load_case_name,
            "span": span,
            "stage": "start",
            "n_rows": int(len(prefiltered_df)),
            "max_sdl": resolved_required_loads_case.get("max_sdl"),
            "max_ll": resolved_required_loads_case.get("max_ll"),
            "max_total": resolved_required_loads_case.get("max_total"),
            "max_factored_total": resolved_required_loads_case.get("max_factored_total"),
        })

        # -------------------------
        # Stage 1: span + depth filtering
        # -------------------------
        try:
            span_filtered = filter_systems(
                df=prefiltered_df.copy(),
                min_span_required=span,
                required_loads=None,  # already applied in stage 0
                unit=unit,
                depth_limit_enabled=(depth_limit_mm is not None),
                depth_limit_mm=depth_limit_mm,
                require_separate_checks=False,
                debug=dbg,
                debug_tag=f"{load_case_name}::span={span:.2f}",
            )
        except Exception:
            log.exception("[systems] %s: per-span filter failed for span %.2f m; skipping", load_case_name, span)
            span_filtered = pd.DataFrame(columns=prefiltered_df.columns)

        audit_rows.append({
            "case": load_case_name,
            "span": span,
            "stage": "after_span_depth_filter",
            "n_rows": int(len(span_filtered)),
        })

        if dbg:
            _dbg_df(
                f"{load_case_name}.span_filtered.{span:.2f}",
                span_filtered,
                explicit=True,
                max_rows=min(debug_max_rows, 10),
                cols=["system_variant", "system_family", "max_span", "sdl_total", "ll", "total_capacity", "slab_depth", "screed_depth"],
            )

        _snap(span_filtered, f"{load_case_name}__span_filtered__{span:.2f}m")

        # -------------------------
        # Stage 2: takeoff + carbon evaluation
        # -------------------------
        try:
            candidates_for_span = compute_candidates_for_span(
                systems_df=span_filtered,
                materials_df=materials_df,
                span_value=span,
                required_loads=resolved_required_loads_case,
                unit=unit,
                depth_enabled=(depth_limit_mm is not None),
                depth_limit_mm=depth_limit_mm,
                prefiltered=True,
            )
        except Exception:
            log.exception("[systems] %s: compute_candidates_for_span failed for span %.2f m; skipping", load_case_name, span)
            candidates_for_span = pd.DataFrame(columns=systems_df.columns)

        audit_rows.append({
            "case": load_case_name,
            "span": span,
            "stage": "after_takeoff_carbon",
            "n_rows": int(len(candidates_for_span)),
            "n_variants": int(candidates_for_span["system_variant"].nunique()) if "system_variant" in candidates_for_span.columns else None,
            "min_carbon": float(pd.to_numeric(candidates_for_span.get("carbon_total_kgCO2"), errors="coerce").min())
                         if "carbon_total_kgCO2" in candidates_for_span.columns and len(candidates_for_span) else None,
        })

        if dbg:
            _dbg_df(
                f"{load_case_name}.candidates_for_span.{span:.2f}",
                candidates_for_span,
                explicit=True,
                max_rows=debug_max_rows,
                cols=["system_variant", "system_family", "span", "carbon_total_kgCO2", "carbon_total_per_m2", "carbon_concrete_per_m2", "carbon_steel_per_m2", "carbon_timber_per_m2"],
            )

        _snap(candidates_for_span, f"{load_case_name}__candidates_for_span__{span:.2f}m")

        span_fp = out_dir / f"candidates_{load_case_name}_span_{span:.2f}m.csv"
        try:
            (candidates_for_span if candidates_for_span is not None else pd.DataFrame()).to_csv(span_fp, index=False)
            log.info("[systems] %s: wrote %d candidate rows -> %s", load_case_name, len(candidates_for_span), span_fp)
        except Exception:
            log.exception("[systems] %s: failed writing per-span candidates for span %.2f m", load_case_name, span)

        span_results_case.append((span, candidates_for_span))

    try:
        audit_df = pd.DataFrame(audit_rows)
        audit_fp = out_dir / "span_sweep_audit.csv"
        audit_df.to_csv(audit_fp, index=False)
        log.info("[debug] %s: wrote span_sweep_audit.csv (%d rows)", load_case_name, len(audit_df))
    except Exception:
        log.exception("[debug] %s: failed writing span_sweep_audit.csv", load_case_name)

    candidates_case_all = aggregate_span_results(span_results_case)
    if candidates_case_all is None or candidates_case_all.empty:
        return pd.DataFrame(columns=systems_df.columns)

    return candidates_case_all


def expand_winners_and_write_materials_per_floor(
    *,
    candidates_all: pd.DataFrame,
    floors_by_case: Dict[str, List[int]],
    floor_area_lookup: Dict[int, float],
    materials_df: pd.DataFrame,
    out_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    log = logger or globals().get("logger") or logging.getLogger("spans")

    expanded_rows = []
    for _, row in candidates_all.iterrows():
        case = (row.get("governing_load_case")
                or row.get("_source_case")
                or row.get("case")
                or None)

        if case is None:
            log.warning("[span] Winner row missing case key columns; cannot expand floors.")
            continue

        case = str(case).strip()
        floors = floors_by_case.get(case)

        if not floors:
            # Helpful debug: show available keys once in a while
            log.warning("[span] No floors found for case '%s'. Known cases: %s",
                        case, sorted(list(floors_by_case.keys()))[:8])
            continue

        for f in floors:
            f_int = int(f)
            new_row = row.copy()
            new_row["floor"] = f_int
            new_row["floor_area_m2"] = float(floor_area_lookup.get(f_int, 0.0))
            expanded_rows.append(new_row)

    candidates_all_expanded = (
        pd.DataFrame(expanded_rows)
        if expanded_rows
        else pd.DataFrame(columns=list(candidates_all.columns) + ["floor", "floor_area_m2"])
    )

    takeoff_mod.materials_per_floor_csv(
        candidates_df=candidates_all_expanded,
        materials_df=materials_df,
        out_fp=Path(out_dir) / "materials_per_floor_expanded.csv",
    )
    return candidates_all_expanded
