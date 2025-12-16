# earlystruct/cli.py
from __future__ import annotations
import argparse, os, math
from pathlib import Path
import pandas as pd

from .core import control, loads, curves, systems, quantities, impacts, reporting
from .core.units import parse_spans_arg, span_to_m, area_to_m2, depth_to_m, load_to_knm2
from .core.control import parse_control
from .core.rank import carbon_first_then_cost, pareto_min_carbon_cost

# ---------------------------------
# Inline config (kept minimal)
# ---------------------------------
DEFAULT_SWEEP_MIN_FT  = 18.0
DEFAULT_SWEEP_MAX_FT  = 45.0
DEFAULT_SWEEP_STEP_FT = 1.0

EDGE_CANTILEVER_DEFAULT_M = 1.0  # max allowed cantilever at each slab edge

# ---------------------------------
# Robust number normalizer
# ---------------------------------
def _norm_num(val, *, zero_is_none=False):
    """Treat '', None, 'nan', 'NaN', 'na', 'n/a', 'null' and (optionally) 0 as missing."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "na", "n/a", "null", "none"}:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if math.isnan(x):
        return None
    if zero_is_none and abs(x) < 1e-12:
        return None 
    return x

# ---------------------------------
# Inline CSV loader (no external dependency)
# ---------------------------------
def _bool_from_ctl(ctl: dict, base_key: str, default: bool = False) -> bool:
    """
    Try to interpret a boolean from the control dict.

    Priority:
      1) <BASE_KEY>_BOOL if present
      2) raw <BASE_KEY> (Y/N, True/False, 1/0, etc.)
    """
    bool_key = f"{base_key}_BOOL"
    if bool_key in ctl:
        return bool(ctl[bool_key])

    raw = ctl.get(base_key)
    if raw is None:
        return default

    s = str(raw).strip().upper()
    return s in {"Y", "YES", "TRUE", "T", "1"}

REQUIRED = [
    "project.csv", "program.csv", "occupancies.csv",
    "materials.csv", "systems_catalog.csv", "system_curves.csv"
]
def load_all(data_dir: str):
    join = os.path.join
    missing = [f for f in REQUIRED if not os.path.exists(join(data_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing required CSVs: {', '.join(missing)}")
    project = pd.read_csv(join(data_dir, "project.csv"))
    program = pd.read_csv(join(data_dir, "program.csv"))
    occ     = pd.read_csv(join(data_dir, "occupancies.csv"))
    mats    = pd.read_csv(join(data_dir, "materials.csv"))
    cat     = pd.read_csv(join(data_dir, "systems_catalog.csv"))
    curves_ = pd.read_csv(join(data_dir, "system_curves.csv"))
    grid_options = None
    if os.path.exists(join(data_dir, "grid_options.csv")):
        grid_options = pd.read_csv(join(data_dir, "grid_options.csv"))
    # --- NEW: metric-normalized helper columns for plots ---
    from .core.units import span_to_m, depth_to_m, load_to_knm2
    def _unit(u): 
        s = str(u).strip().lower()
        return 'imperial' if s.startswith('imp') or s in {'imperial','us','us customary'} else 'metric'
    if not cat.empty:
        units = cat.get('unit', 'metric').map(_unit)
        # spans
        if 'max_span' in cat.columns:
            cat['max_span_m'] = [span_to_m(v, u) for v, u in zip(cat['max_span'], units)]
        # depths (write only if columns exist)
        for c in ('slab_depth','beam_depth','screed_depth','steel_depth'):
            if c in cat.columns:
                cat[f'{c}_m'] = [depth_to_m(v, u) for v, u in zip(cat[c], units)]
        # load allowances
        for c in ('sdl','ll'):
            if c in cat.columns:
                cat[f'{c}_knm2'] = [load_to_knm2(v, u) for v, u in zip(cat[c], units)]
    return project, program, occ, mats, cat, curves_, grid_options

# ---------------------------------
# Inline (simple) reporting
# ---------------------------------
def save_csvs(df_full: pd.DataFrame, df_pareto: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "results_full.csv")
    p2 = os.path.join(out_dir, "results_pareto.csv")
    df_full.to_csv(p1, index=False)
    df_pareto.to_csv(p2, index=False)
    return p1, p2

# ---------------------------------
# Geometry: exact interior span + limited edge cantilevers
# ---------------------------------
def derive_plate(floor_area_m2: float, L_m: float|None, W_m: float|None):
    """If L&W given, use them; else assume a square plate from area."""
    if (L_m is not None) and (W_m is not None):
        return float(L_m), float(W_m)
    side = math.sqrt(float(floor_area_m2))
    return side, side

def layout_grid_with_edge_limit(L: float, W: float, span_x: float, span_y: float, edge_limit_m: float):
    """Keep interior bay = requested span; put any leftover as equal edge cantilevers, each <= edge_limit_m."""
    def solve(S, span):
        if span <= 0:
            return 1, span, 0.0
        n_cols = max(2, math.ceil(max(0.0, (S - 2*edge_limit_m)) / span) + 1)
        n_bays = n_cols - 1
        # interior spacing is exactly 'span'; residual goes to edges
        edge = max(0.0, 0.5*(S - n_bays*span))
        edge = min(edge, edge_limit_m)
        return n_bays, span, edge
    nx, sx, ex = solve(L, span_x)
    ny, sy, ey = solve(W, span_y)
    return nx, ny, sx, sy, ex, ey

# ---------------------------------
# Span collection with sensible precedence
# ---------------------------------
def collect_spans(project_row: dict, grid_options_df, spans_arg, ctl: dict|None):
    spans = []
    # precedence: CLI/Notebook -> control SPANS -> grid_options.csv -> project ideal
    if spans_arg:
        spans.extend(spans_arg)
    elif ctl and ctl.get('SPANS'):
        spans.extend(parse_spans_arg(ctl.get('SPANS')))

    if grid_options_df is not None and not grid_options_df.empty and not spans:
        for _, r in grid_options_df.iterrows():
            spans.append((float(r['span_value']), str(r.get('unit','metric'))))

    if (not spans) and project_row:
        iv = _norm_num(project_row.get('ideal_column_spacing'), zero_is_none=True)
        if iv is not None and iv > 0:
            spans.append((iv, str(project_row.get('unit','metric'))))

    # dedupe by meter value; keep >0
    uniq, seen = [], set()
    for s in spans:
        m = round(span_to_m(s[0], s[1]), 6)
        if m not in seen and m > 0:
            uniq.append(s); seen.add(m)
    return uniq

def sweep_1ft_default():
    vals, ft = [], DEFAULT_SWEEP_MIN_FT
    while ft <= DEFAULT_SWEEP_MAX_FT + 1e-9:
        vals.append((ft, 'imperial')); ft += DEFAULT_SWEEP_STEP_FT
    return vals

def _system_max_slab_span_m(cat_df: pd.DataFrame, system_id: str) -> float:
    """
    For a given system_id, return the maximum catalog span (in meters)
    across all its rows. Used as a hard limit on slab span.
    """
    rows = cat_df[cat_df["system_id"] == system_id]
    if rows.empty:
        return 0.0

    max_m = 0.0
    for _, r in rows.iterrows():
        raw = r.get("max_span", 0.0)
        try:
            v = float(raw or 0.0)
        except Exception:
            v = 0.0
        unit = str(r.get("unit", "metric"))
        m = span_to_m(v, unit)
        if m > max_m:
            max_m = m
    return max_m

def _depth_ctl_to_m(ctl: dict, key: str, unit_flag: str, default_m: float) -> float:
    """
    Read a depth-like control value (e.g. beam width, depth) and convert to meters
    using the project unit. Falls back to default_m if missing/invalid.
    """
    raw = _norm_num(ctl.get(key), zero_is_none=True)
    if raw is None:
        return default_m
    try:
        return depth_to_m(raw, unit_flag)
    except Exception:
        return default_m


# ---------------------------------
# Main evaluation
# ---------------------------------
def evaluate(data_dir: str, spans_str: str | None, export_dir: str | None,
             control_file: str | None = None):
    # ---- parse control and load CSVs ----
    ctl = parse_control(control_file) if control_file else {}
    if ctl.get("DATA_DIR"):
        data_dir = ctl["DATA_DIR"]

    project_df, program_df, occ_df, mats_df, cat_df, curves_df, grid_df = load_all(data_dir)

    # ---- project: control overrides CSV unless USE_CSV=Y ----
    use_csv = bool(ctl.get("USE_CSV_BOOL", False))
    if use_csv or project_df.empty:
        project = project_df.iloc[0].to_dict()
    else:
        csv_proj = project_df.iloc[0].to_dict() if not project_df.empty else {}

        def pick(k, default=None):
            return ctl.get(k, csv_proj.get(k, default))

        project = {
            "project_name": pick("PROJECT_NAME", csv_proj.get("project_name", "")),
            "location": pick("LOCATION", csv_proj.get("location", "")),
            "unit": pick("UNIT", csv_proj.get("unit", "metric")),
            "floor_area_per_floor": pick(
                "FLOOR_AREA_PER_FLOOR", csv_proj.get("floor_area_per_floor", 0.0)
            ),
            "floor_to_floor": pick("FLOOR_TO_FLOOR", csv_proj.get("floor_to_floor", 0.0)),
            "num_floors": int(float(pick("NUM_FLOORS", csv_proj.get("num_floors", 1)) or 1)),
            "ideal_column_spacing": pick(
                "IDEAL_COLUMN_SPACING", csv_proj.get("ideal_column_spacing", "")
            ),
            "plate_length": pick("PLATE_LENGTH", csv_proj.get("plate_length", "")),
            "plate_width": pick("PLATE_WIDTH", csv_proj.get("plate_width", "")),
            "depth_limit_enabled": ctl.get(
                "DEPTH_LIMIT_ENABLED_BOOL",
                bool(csv_proj.get("depth_limit_enabled", False)),
            ),
            "depth_limit": pick("DEPTH_LIMIT", csv_proj.get("depth_limit", "")),
            "notes": pick("NOTES", csv_proj.get("notes", "")),
        }

    # ---- units & basic geometry inputs ----
    unit_flag = (
        project.get("UNIT")
        or project.get("unit")
        or ctl.get("UNIT")
        or ctl.get("unit")
        or "metric"
    )

    area_raw = _norm_num(project.get("floor_area_per_floor"), zero_is_none=True)
    L_raw = _norm_num(project.get("plate_length"), zero_is_none=True)
    W_raw = _norm_num(project.get("plate_width"), zero_is_none=True)

    # ---- ONE-WAY IRREGULAR SETTINGS ----
    oneway_enabled = bool(ctl.get("ONE_WAY_IRREGULAR_BOOL", False))
    unit_flag_local = (unit_flag or "metric").lower()
    is_imperial = unit_flag_local.startswith("imp") or ("ft" in unit_flag_local)

    def _span_val_to_m(val):
        v = float(val)
        return span_to_m(v, "imperial" if is_imperial else "metric")

    slab_min_m = None
    beam_min_m = None
    slab_min_raw = ctl.get("ONE_WAY_SLAB_MIN_SPAN")
    beam_min_raw = ctl.get("ONE_WAY_BEAM_MIN_SPAN")
    try:
        if slab_min_raw is not None:
            slab_min_m = _span_val_to_m(slab_min_raw)
    except Exception:
        slab_min_m = None
    try:
        if beam_min_raw is not None:
            beam_min_m = _span_val_to_m(beam_min_raw)
    except Exception:
        beam_min_m = None

    # store slab min (if any) back into ctl so span sweep can reuse it
    if slab_min_m is not None and slab_min_m > 0.0:
        ctl["_ONE_WAY_SLAB_MIN_M"] = slab_min_m

    major_factor_from_min = None
    if (
        slab_min_m is not None
        and beam_min_m is not None
        and slab_min_m > 0.0
        and beam_min_m > slab_min_m
    ):
        major_factor_from_min = beam_min_m / slab_min_m

    major_factor_raw = ctl.get("ONE_WAY_MAJOR_FACTOR")
    try:
        oneway_major_factor = float(major_factor_raw) if major_factor_raw is not None else 1.0
    except Exception:
        oneway_major_factor = 1.0

    if major_factor_from_min is not None:
        oneway_major_factor = major_factor_from_min
    if oneway_major_factor < 1.0:
        oneway_major_factor = 1.0

    major_along_longest = str(
        ctl.get("ONE_WAY_MAJOR_ALONG_LONGEST", "Y")
    ).strip().upper() in {"Y", "YES", "TRUE", "T", "1"}

    def _is_one_way_system(row_any: dict) -> bool:
        # 1) explicit flag from catalog
        val = str(row_any.get("span_behavior", "")).strip().lower()
        if val in {"one_way", "one-way", "1way"}:
            return True
        if val in {"two_way", "two-way", "2way"}:
            return False

        # 2) fallback heuristic on names/types
        txt = " ".join(
            str(row_any.get(k, "")) for k in ("type", "system_name", "category", "system_id")
        ).lower()

        keywords = [
            "one way",
            "one-way",
            "one_way",
            "1way",
            "hollowcore",
            "hollow core",
            "hollow-core",
            "solid plank",
            "solid_plank",
            "plank",
            "beam and block",
            "beam & block",
            "composite deck",
            "steel deck",
            "metal deck",
            "deck slab",
        ]
        return any(k in txt for k in keywords)

    # ---- program blocks from control (if not using CSV) ----
    if (not use_csv) and ctl.get("PROGRAM_BLOCKS"):
        program_df = pd.DataFrame(ctl["PROGRAM_BLOCKS"])

    # ---- SPANS (incl. SWEEP) ----
    spans = collect_spans(
        project,
        grid_df,
        parse_spans_arg(spans_str) if spans_str else None,
        ctl,
    )
    if not spans:
        spans = sweep_1ft_default()

    if ctl.get("SPAN_SWEEP_FROM_MIN_BOOL", False) and spans:
        unit_flag_local = (unit_flag or "metric").lower()
        is_imperial = unit_flag_local.startswith("imp") or ("ft" in unit_flag_local)

        # decide sweep start:
        # if ONE_WAY_SLAB_MIN_SPAN is set, let that drive the minimum
        slab_min_m_for_sweep = ctl.get('_ONE_WAY_SLAB_MIN_M')  # set earlier
        if slab_min_m_for_sweep is not None:
            try:
                min_span_m = float(slab_min_m_for_sweep)
            except Exception:
                # fall back to existing spans if parsing fails
                min_span_m = min(span_to_m(v, u) for (v, u) in spans)
        else:
            # no explicit slab min => use whatever spans we already have
            min_span_m = min(span_to_m(v, u) for (v, u) in spans)

        max_span_global_m = 0.0
        for _, row in cat_df.iterrows():
            try:
                m = span_to_m(
                    float(row.get("max_span", 0.0) or 0.0),
                    str(row.get("unit", "metric")),
                )
            except Exception:
                m = 0.0
            if m > max_span_global_m:
                max_span_global_m = m
        if max_span_global_m <= 0.0:
            max_span_global_m = span_to_m(60.0, "imperial")  # ~18 m

        step_raw = ctl.get("SPAN_SWEEP_STEP")
        try:
            step_val = float(step_raw) if step_raw is not None else (1.0 if is_imperial else 0.25)
        except Exception:
            step_val = 1.0 if is_imperial else 0.25

        step_m = span_to_m(step_val, "imperial" if is_imperial else "metric")
        if step_m <= 0.0:
            step_m = span_to_m(1.0, "imperial")

        new_spans = []
        s = min_span_m
        while s <= max_span_global_m + 1e-9:
            new_spans.append((s, "metric"))
            s += step_m
        spans = new_spans

    # ---- floor area & plate dims ----
    if area_raw is not None and area_raw > 0:
        floor_area_m2 = area_to_m2(area_raw, unit_flag)
    elif (L_raw is not None and L_raw > 0) and (W_raw is not None and W_raw > 0):
        L_m = span_to_m(L_raw, unit_flag)
        W_m = span_to_m(W_raw, unit_flag)
        floor_area_m2 = L_m * W_m
    else:
        raise ValueError(
            "Project needs a positive floor area OR positive plate_length & plate_width."
        )

    L_m = span_to_m(L_raw, unit_flag) if (L_raw is not None and L_raw > 0) else None
    W_m = span_to_m(W_raw, unit_flag) if (W_raw is not None and W_raw > 0) else None

    # NEW: floor-to-floor height in meters (for column volumes)
    floor_h_raw = _norm_num(project.get("floor_to_floor"), zero_is_none=True)
    if floor_h_raw is not None and floor_h_raw > 0:
        floor_height_m = depth_to_m(floor_h_raw, unit_flag)
    else:
        floor_height_m = 3.0  # fallback default

        # ---- frame (beam/column) penalties knobs ----
    frame_enabled = bool(ctl.get("FRAME_PENALTIES_ENABLED_BOOL", True))

    # Slenderness (L/d)
    try:
        frame_beam_slenderness = float(ctl.get("FRAME_BEAM_SLENDERNESS", 20.0) or 20.0)
    except Exception:
        frame_beam_slenderness = 20.0
    if frame_beam_slenderness <= 0.0:
        frame_beam_slenderness = 20.0

    # Beam & column geometry (converted from project units to meters)
    frame_beam_width_m    = _depth_ctl_to_m(ctl, "FRAME_BEAM_WIDTH",    unit_flag, 0.20)
    frame_column_width_m  = _depth_ctl_to_m(ctl, "FRAME_COLUMN_WIDTH",  unit_flag, 0.30)
    frame_column_depth_m  = _depth_ctl_to_m(ctl, "FRAME_COLUMN_DEPTH",  unit_flag, 0.30)
    frame_beam_depth_max_m= _depth_ctl_to_m(ctl, "FRAME_BEAM_DEPTH_MAX",unit_flag, 0.80)

    # ---- loads, materials, limits, systems ----
    req_df = loads.required_loads_per_block(program_df, occ_df)
    mats_by_id = {r["material_id"]: r for _, r in mats_df.iterrows()}

    depth_limit_enabled = bool(project.get("depth_limit_enabled", False))
    depth_raw = _norm_num(project.get("depth_limit"), zero_is_none=True)
    depth_limit_m = (
        depth_to_m(depth_raw, unit_flag)
        if (depth_limit_enabled and (depth_raw is not None))
        else None
    )

    edge_limit_m = (
        float(ctl.get("EDGE_CANTILEVER_MAX_M"))
        if ctl.get("EDGE_CANTILEVER_MAX_M")
        else EDGE_CANTILEVER_DEFAULT_M
    )

    # make sure system_id is uniform strings and drop missing values
    systems_list = sorted(cat_df["system_id"].dropna().astype(str).unique())

    # ---- main sweep over spans & systems ----
    out_rows = []

    for span_val, span_unit in spans:
        span_m_input = span_to_m(span_val, span_unit)  # this is the **slab** span
        L, W = derive_plate(floor_area_m2, L_m, W_m)

        for sys_id in systems_list:
            feasible_all = True
            reasons: list[str] = []

            row_any = cat_df[cat_df["system_id"] == sys_id].iloc[0].to_dict()
            system_label = row_any.get("system_name") or row_any.get("type", "") or sys_id

            is_one_way = _is_one_way_system(row_any)

            # base: symmetric spans in both directions
            span_x = span_m_input
            span_y = span_m_input

            # one-way irregular layout: beams longer than slab spans
            if oneway_enabled and is_one_way and oneway_major_factor > 1.0:
                base_span_m = span_m_input  # slab span
                if (L is not None) and (W is not None) and major_along_longest:
                    if L >= W:
                        span_x = base_span_m * oneway_major_factor  # beam dir along X
                        span_y = base_span_m                        # slab dir along Y
                    else:
                        span_x = base_span_m                        # slab dir along X
                        span_y = base_span_m * oneway_major_factor  # beam dir along Y
                else:
                    # no clear long side → just treat X as major (beam), Y as slab
                    span_x = base_span_m * oneway_major_factor
                    span_y = base_span_m

            # explicit slab vs beam spans
            if oneway_enabled and is_one_way and oneway_major_factor > 1.0:
                slab_dir_span_m = span_m_input
                beam_dir_span_m = span_m_input * oneway_major_factor
            else:
                slab_dir_span_m = span_m_input
                beam_dir_span_m = span_m_input

            span_m_for_catalog = slab_dir_span_m

            # HARD CAP: catalog max span applies to slab direction only
            sys_max_slab_m = _system_max_slab_span_m(cat_df, sys_id)
            if sys_max_slab_m > 0.0 and slab_dir_span_m > sys_max_slab_m + 1e-9:
                # this system simply can't do this slab span
                continue

            nx, ny, sx, sy, ex, ey = layout_grid_with_edge_limit(
                L, W, span_x, span_y, edge_limit_m
            )

            per_m2 = curves.get_intensities(curves_df, sys_id)
            # Exclude systems whose slab material intensities are all zero.
            def _nz(x):
                try:
                    x = float(x)
                except Exception:
                    return 0.0
                return x if x > 0.0 and x == x else 0.0  # filter nonpositive & NaN
            if (
                _nz(per_m2.get("concrete_m3_per_m2", 0.0)) == 0.0
                and _nz(per_m2.get("steel_m3_per_m2", 0.0)) == 0.0
                and _nz(per_m2.get("timber_m3_per_m2", 0.0)) == 0.0
            ):
                # Optional: uncomment to see which are skipped when running verbose
                # print(f"[skip] {sys_id}: all slab material intensities are zero in system_curves.")
                continue
            swt_sys_knm2 = per_m2.get("swt", 0.0)


            total_area = 0.0
            # UPDATED: track slab + beam + column volumes separately
            total_qty = {
                "concrete_m3": 0.0,          # overall
                "concrete_slab_m3": 0.0,
                "concrete_beams_m3": 0.0,
                "concrete_columns_m3": 0.0,
                "steel_m3": 0.0,
                "timber_m3": 0.0,
            }
            slab_depth_m = float(per_m2.get("depth", 0.0) or 0.0)
            beam_depth_m = 0.0  # will be set by frame model

            # use **slab span only** for catalog checks
            span_m_for_catalog = slab_dir_span_m

            mat_ids = {
                "material_concrete_id": row_any.get("material_concrete_id") or "",
                "material_steel_id": row_any.get("material_steel_id") or "",
                "material_timber_id": row_any.get("material_timber_id") or "",
                "material_pt_id": row_any.get("material_pt_id") or "",
            }
            has_concrete = bool(mat_ids["material_concrete_id"])

            frame_int = {
                "concrete_volume_beams_m3_per_m2": 0.0,
                "concrete_volume_columns_m3_per_m2": 0.0,
                "beam_depth_m": 0.0,
                "beam_depth_required_m": 0.0,
                "beam_ok": True,
            }

            # Only apply frame penalties if:
            #   - system actually has concrete
            #   - we know the floor height
            #   - the feature is enabled in the control file
            if frame_enabled and has_concrete and floor_height_m > 0.0:
                slab_type = "one_way" if is_one_way else "two_way"
                frame_int = quantities.frame_concrete_intensities_per_m2(
                    span_x_m=span_x,
                    span_y_m=span_y,
                    floor_height_m=floor_height_m,
                    slab_type=slab_type,
                    beam_width_m=frame_beam_width_m,
                    beam_slenderness=frame_beam_slenderness,
                    beam_depth_max_m=frame_beam_depth_max_m,
                    column_width_m=frame_column_width_m,
                    column_depth_m=frame_column_depth_m,
                )
                beam_depth_m = float(frame_int.get("beam_depth_m", 0.0) or 0.0)

                # SOFT CHECK: note that slenderness is exceeded, but don't kill feasibility.
                # This leaves you with results, and you can see which ones are "dodgy".
                if not frame_int.get("beam_ok", True):
                    reasons.append("beam_slenderness_exceeded")
            else:
                # penalties disabled or not applicable → no extra beam/column volume
                beam_depth_m = 0.0

            for _, pb in req_df.iterrows():
                floors = int(pb["end_floor"]) - int(pb["start_floor"]) + 1
                area_block = floor_area_m2 * floors
                total_area += area_block

                # LOAD CHECK MODE:
                #   - default is LOADS_ONLY (lenient, like earlier behaviour)
                #   - if control explicitly requests SPAN_PLUS_LOADS, we try that first
                raw_mode = ctl.get("LOAD_CHECK_MODE", "LOADS_ONLY")
                mode = str(raw_mode).upper()

                # First attempt: whatever mode the control asked for
                best_row, reason = systems.check_rows_for_system(
                    cat_df,
                    sys_id,
                    pb["req_sdl_non_swt_knm2"],
                    pb["req_ll_knm2"],
                    swt_sys_knm2,
                    span_m_for_catalog,  # slab direction only
                    depth_limit_m,
                    load_check_mode=mode,
                )

                # If strict SPAN_PLUS_LOADS fails, fall back to LOADS_ONLY before
                # declaring the system infeasible.
                if (best_row is None or reason) and mode == "SPAN_PLUS_LOADS":
                    best_row2, reason2 = systems.check_rows_for_system(
                        cat_df,
                        sys_id,
                        pb["req_sdl_non_swt_knm2"],
                        pb["req_ll_knm2"],
                        swt_sys_knm2,
                        span_m_for_catalog,
                        depth_limit_m,
                        load_check_mode="LOADS_ONLY",
                    )
                    if (best_row2 is not None) and (not reason2):
                        best_row, reason = best_row2, reason2
                        mode = "LOADS_ONLY"

                if best_row is None or reason:
                    feasible_all = False
                    reasons.append(reason or f"no row (mode={mode})")

                # SLAB quantities (existing behaviour)
                qty = quantities.totals_from_intensity(per_m2, area_block)

                slab_conc = float(qty.get("concrete_m3", 0.0) or 0.0)
                steel_m3  = float(qty.get("steel_m3", 0.0) or 0.0)
                timber_m3 = float(qty.get("timber_m3", 0.0) or 0.0)

                total_qty["concrete_slab_m3"] += slab_conc
                total_qty["steel_m3"]         += steel_m3
                total_qty["timber_m3"]        += timber_m3

                # BEAM & COLUMN concrete quantities (per-m² * area_block)
                if has_concrete:
                    v_beams_block = frame_int["concrete_volume_beams_m3_per_m2"] * area_block
                    v_cols_block  = frame_int["concrete_volume_columns_m3_per_m2"] * area_block
                else:
                    v_beams_block = 0.0
                    v_cols_block  = 0.0

                total_qty["concrete_beams_m3"]   += v_beams_block
                total_qty["concrete_columns_m3"] += v_cols_block

                # Keep a total concrete volume for backwards compatibility
                total_qty["concrete_m3"] = (
                    total_qty["concrete_slab_m3"]
                    + total_qty["concrete_beams_m3"]
                    + total_qty["concrete_columns_m3"]
                )

                slab_depth_m = qty.get("depth_m", slab_depth_m)

                        # Overall structural depth = max(slab, beam)
            depth_struct_m = max(slab_depth_m, beam_depth_m)

            # -------------------------------------------------
            #  SLAB / BEAMS / COLUMNS: separate impact calls
            # -------------------------------------------------
            # Assumption: our frame model only adds CONCRETE in beams/columns.
            # All steel/timber in total_qty is part of the slab system.
            qty_slab = {
                "concrete_m3": total_qty.get("concrete_slab_m3", 0.0),
                "steel_m3":    total_qty.get("steel_m3", 0.0),
                "timber_m3":   total_qty.get("timber_m3", 0.0),
            }
            qty_beams = {
                "concrete_m3": total_qty.get("concrete_beams_m3", 0.0),
            }
            qty_cols = {
                "concrete_m3": total_qty.get("concrete_columns_m3", 0.0),
            }

            # Slab impacts
            carbon_slab_kg, cost_slab, _ = impacts.calc_impacts_cost(
                qty_slab, mat_ids, mats_by_id
            )

            # Beam impacts
            carbon_beams_kg, cost_beams, _ = impacts.calc_impacts_cost(
                qty_beams, mat_ids, mats_by_id
            )

            # Column impacts
            carbon_cols_kg, cost_cols, _ = impacts.calc_impacts_cost(
                qty_cols, mat_ids, mats_by_id
            )

            # Totals
            carbon_total = carbon_slab_kg + carbon_beams_kg + carbon_cols_kg
            cost_total   = cost_slab + cost_beams + cost_cols

            carbon_per_m2 = (carbon_total / total_area) if total_area > 0 else float("nan")
            cost_per_m2   = (cost_total   / total_area) if total_area > 0 else float("nan")

            carbon_slab_per_m2   = (carbon_slab_kg  / total_area) if total_area > 0 else 0.0
            carbon_beams_per_m2  = (carbon_beams_kg / total_area) if total_area > 0 else 0.0
            carbon_cols_per_m2   = (carbon_cols_kg  / total_area) if total_area > 0 else 0.0

            cost_slab_per_m2     = (cost_slab   / total_area) if total_area > 0 else 0.0
            cost_beams_per_m2    = (cost_beams  / total_area) if total_area > 0 else 0.0
            cost_cols_per_m2     = (cost_cols   / total_area) if total_area > 0 else 0.0

            out_rows.append(
                {
                    "span_input_m": slab_dir_span_m,   # slab span
                    "span_m": beam_dir_span_m,         # controlling (beam) span if you like
                    "span_x_m": span_x,
                    "span_y_m": span_y,
                    "span_slab_dir_m": slab_dir_span_m,
                    "span_beam_dir_m": beam_dir_span_m,
                    "system_id": sys_id,
                    "system_name": system_label,
                    "category": row_any.get("category", ""),
                    "type": row_any.get("type", ""),
                    "manufacturer": row_any.get("manufacturer", ""),
                    "feasible": bool(feasible_all),
                    "reason": "; ".join([r for r in reasons if r]) if reasons else "",
                    "depth_m": depth_struct_m,
                    "slab_depth_m": slab_depth_m,
                    "beam_depth_m": beam_depth_m,
                    "area_m2": total_area,
                    # volumes
                    "concrete_m3": total_qty.get("concrete_m3", 0.0),
                    "concrete_slab_m3": total_qty.get("concrete_slab_m3", 0.0),
                    "concrete_beams_m3": total_qty.get("concrete_beams_m3", 0.0),
                    "concrete_columns_m3": total_qty.get("concrete_columns_m3", 0.0),
                    "steel_m3": total_qty.get("steel_m3", 0.0),
                    "timber_m3": total_qty.get("timber_m3", 0.0),
                    # carbon & cost
                    "carbon_total_kg": carbon_total,
                    "carbon_per_m2": carbon_per_m2,
                    "carbon_slab_kg": carbon_slab_kg,
                    "carbon_beams_kg": carbon_beams_kg,
                    "carbon_columns_kg": carbon_cols_kg,
                    "carbon_slab_per_m2": carbon_slab_per_m2,
                    "carbon_beams_per_m2": carbon_beams_per_m2,
                    "carbon_columns_per_m2": carbon_cols_per_m2,
                    "cost_total": cost_total,
                    "cost_per_m2": cost_per_m2,
                    "cost_slab": cost_slab,
                    "cost_beams": cost_beams,
                    "cost_columns": cost_cols,
                    "cost_slab_per_m2": cost_slab_per_m2,
                    "cost_beams_per_m2": cost_beams_per_m2,
                    "cost_columns_per_m2": cost_cols_per_m2,
                    "nx": nx,
                    "ny": ny,
                    "edge_canti_x_m": ex,
                    "edge_canti_y_m": ey,
                }
            )

    df = pd.DataFrame(out_rows)
    ranked = carbon_first_then_cost(df)
    pareto = pareto_min_carbon_cost(df)

    saved: dict[str, object] = {"project": project, "ctl": ctl, "spans": spans}
    if export_dir:
        csv_info = save_csvs(df, pareto, export_dir)
        saved["csv"] = csv_info

    return df, ranked, pareto, saved

def main():
    parser = argparse.ArgumentParser(description="Parametric floor design explorer")
    parser.add_argument("--control", "-c", required=True, help="Path to control file")
    parser.add_argument("--data-dir", "-d", required=True, help="Path to CSV data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose summary")
    args = parser.parse_args()

    df, ranked, pareto, saved = evaluate(
        data_dir=Path(args.data_dir),
        spans_str=None,
        export_dir=None,
        control_file=args.control,
    )

    if args.verbose:
        project = saved.get("project", {}) if isinstance(saved, dict) else {}
        ctl = saved.get("ctl", {}) if isinstance(saved, dict) else {}
        from earlystruct.core import reporting
        reporting.print_verbose_summary(df, project, ctl)

if __name__ == "__main__":
    main()
