import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from edca_code.scripts.core import systems as systems_mod
from edca_code.scripts.core import takeoff as takeoff_mod
from edca_code.scripts.core import carbon as carbon_mod
from edca_code.scripts.core.systems import filter_systems 
from edca_code.scripts.core.utils import infer_type, reorder_output_columns
from edca_code.scripts.core import rank as rank_mod
from edca_code.scripts.core.utils import ensure_dir  

logger = logging.getLogger("spans")

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
    prefiltered: bool = False) -> pd.DataFrame:
    """
    If prefiltered=False, filters systems_df for span + loads.
    If prefiltered=True, assumes systems_df is already filtered (span/loads/unit/depth) and only computes BOM+carbon.
    """
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
        try:
            bom_m2 = takeoff_mod.bom_per_m2_from_system_row(row)
            carbon_res = carbon_mod.compute_assembly_carbon_from_bom(bom_m2, materials_df, include_a4_a5=False)
            carbon_a1a3 = float(carbon_res.get("totals", {}).get("total_a1a3", 0.0))
            r = row.copy()
            r["carbon_total_kgCO2"] = carbon_a1a3
            r["_bom_keys"] = ",".join(sorted(bom_m2.keys()))
            rows.append(r)
        except Exception:
            logger.exception(
                "[takeoff] Error computing BOM/carbon for candidate index %s (span %s). Falling back to NaN.",
                idx, span_value
            )
            r = row.copy()
            r["carbon_total_kgCO2"] = float("nan")
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
) -> pd.DataFrame:
    """
    Span sweep helper used by run_edca.py (new interface).

    Responsibilities:
      - prefilter systems once by loads (required_loads_case)
      - for each span: filter by span (+ optional depth limit), compute candidates (takeoff + carbon)
      - write per-span candidates CSVs into out_dir
      - return concatenated candidates across spans
    """
    log = logger or globals().get("logger") or logging.getLogger("spans")
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    unit = cf_unit

    # Pre-filter once by loads (skip span check here)
    try:
        prefiltered_df = filter_systems(
            df=systems_df.copy(),
            min_span_required=None,
            required_loads=required_loads_case,
            unit=unit,
            require_separate_checks=False,
            depth_limit_enabled=False,
            depth_limit_mm=None,
        )
    except Exception:
        log.exception("[systems] %s: load-based prefilter failed; falling back to full catalog", load_case_name)
        prefiltered_df = systems_df.copy()

    log.info("[systems] %s: catalog reduced to %d rows after load pre-filter", load_case_name, len(prefiltered_df))

    span_results_case: List[Tuple[float, pd.DataFrame]] = []

    for span in span_values:
        log.info("[span] %s: evaluating span = %.2f m", load_case_name, float(span))

        # Apply span + (optional) depth limit starting from prefiltered_df
        try:
            span_filtered = filter_systems(
                df=prefiltered_df.copy(),
                min_span_required=float(span),
                required_loads=None,  # already applied
                unit=unit,
                depth_limit_enabled=(depth_limit_mm is not None),
                depth_limit_mm=depth_limit_mm,
                require_separate_checks=False,
            )
        except Exception:
            log.exception("[systems] %s: per-span filter failed for span %.2f m; skipping", load_case_name, float(span))
            span_filtered = pd.DataFrame(columns=prefiltered_df.columns)

        # Compute candidates (takeoff + carbon); DO NOT refilter inside compute_candidates_for_span
        try:
            candidates_for_span = compute_candidates_for_span(
                systems_df=span_filtered,
                materials_df=materials_df,
                span_value=float(span),
                required_loads=required_loads_case,
                unit=unit,
                depth_enabled=(depth_limit_mm is not None),
                depth_limit_mm=depth_limit_mm,
                prefiltered=True,
            )
        except Exception:
            log.exception("[systems] %s: compute_candidates_for_span failed for span %.2f m; skipping", load_case_name, float(span))
            candidates_for_span = pd.DataFrame(columns=systems_df.columns)

        # Per-span CSV for traceability
        span_fp = out_dir / f"candidates_{load_case_name}_span_{float(span):.2f}m.csv"
        try:
            if candidates_for_span is None or candidates_for_span.empty:
                candidates_for_span = pd.DataFrame(columns=systems_df.columns)
            candidates_for_span.to_csv(span_fp, index=False)
            log.info("[systems] %s: wrote %d candidate rows -> %s", load_case_name, len(candidates_for_span), span_fp)
        except Exception:
            log.exception("[systems] %s: failed writing per-span candidates for span %.2f m", load_case_name, float(span))

        span_results_case.append((float(span), candidates_for_span))

    # Combine across spans and return (run_edca.py will do ranking + final exports)
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

