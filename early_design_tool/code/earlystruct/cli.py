# earlystruct/cli.py
from __future__ import annotations
import argparse, os, math
import pandas as pd

from .core import control, loads, curves, systems, quantities, impacts
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

# ---------------------------------
# Main evaluation
# ---------------------------------
def evaluate(data_dir: str, spans_str: str|None, export_dir: str|None, control_file: str|None=None):
    ctl = parse_control(control_file) if control_file else {}
    if ctl.get('DATA_DIR'):  # allow override from control
        data_dir = ctl['DATA_DIR']

    project_df, program_df, occ_df, mats_df, cat_df, curves_df, grid_df = load_all(data_dir)

    # project: control takes precedence unless USE_CSV=Y
    use_csv = bool(ctl.get('USE_CSV_BOOL', False))
    if use_csv or project_df.empty:
        project = project_df.iloc[0].to_dict()
    else:
        csv_proj = project_df.iloc[0].to_dict() if not project_df.empty else {}
        def pick(k, default=None): return ctl.get(k, csv_proj.get(k, default))
        project = {
            'project_name': pick('PROJECT_NAME', csv_proj.get('project_name','')),
            'location': pick('LOCATION', csv_proj.get('location','')),
            'unit': pick('UNIT', csv_proj.get('unit','metric')),
            'floor_area_per_floor': pick('FLOOR_AREA_PER_FLOOR', csv_proj.get('floor_area_per_floor', 0.0)),
            'floor_to_floor': pick('FLOOR_TO_FLOOR', csv_proj.get('floor_to_floor', 0.0)),
            'num_floors': int(float(pick('NUM_FLOORS', csv_proj.get('num_floors', 1)) or 1)),
            'ideal_column_spacing': pick('IDEAL_COLUMN_SPACING', csv_proj.get('ideal_column_spacing', '')),
            'plate_length': pick('PLATE_LENGTH', csv_proj.get('plate_length','')),
            'plate_width': pick('PLATE_WIDTH', csv_proj.get('plate_width','')),
            'depth_limit_enabled': ctl.get('DEPTH_LIMIT_ENABLED_BOOL', bool(csv_proj.get('depth_limit_enabled', False))),
            'depth_limit': pick('DEPTH_LIMIT', csv_proj.get('depth_limit','')),
            'notes': pick('NOTES', csv_proj.get('notes','')),
        }

    # control-file program (if provided and USE_CSV != Y)
    if (not use_csv) and ctl.get('PROGRAM_BLOCKS'):
        program_df = pd.DataFrame(ctl['PROGRAM_BLOCKS'])

    # spans
    spans = collect_spans(project, grid_df, parse_spans_arg(spans_str) if spans_str else None, ctl)
    if not spans:
        spans = sweep_1ft_default()

    # geometry & loads (robust to NaN/0, area or LxW)
    unit_flag = str(project.get('unit','metric'))
    area_raw  = _norm_num(project.get('floor_area_per_floor'), zero_is_none=True)
    L_raw     = _norm_num(project.get('plate_length'),        zero_is_none=True)
    W_raw     = _norm_num(project.get('plate_width'),         zero_is_none=True)

    if area_raw is not None and area_raw > 0:
        floor_area_m2 = area_to_m2(area_raw, unit_flag)
    elif (L_raw is not None and L_raw > 0) and (W_raw is not None and W_raw > 0):
        L_m = span_to_m(L_raw, unit_flag)
        W_m = span_to_m(W_raw, unit_flag)
        floor_area_m2 = L_m * W_m
    else:
        raise ValueError("Project needs a positive floor area OR positive plate_length & plate_width.")

    L_m = span_to_m(L_raw, unit_flag) if (L_raw is not None and L_raw > 0) else None
    W_m = span_to_m(W_raw, unit_flag) if (W_raw is not None and W_raw > 0) else None

    req_df = loads.required_loads_per_block(program_df, occ_df)
    mats_by_id = {r['material_id']: r for _, r in mats_df.iterrows()}

    depth_limit_enabled = bool(project.get('depth_limit_enabled', False))
    depth_raw = _norm_num(project.get('depth_limit'), zero_is_none=True)
    depth_limit_m = depth_to_m(depth_raw, unit_flag) if (depth_limit_enabled and (depth_raw is not None)) else None

    edge_limit_m = float(ctl.get('EDGE_CANTILEVER_MAX_M')) if ctl.get('EDGE_CANTILEVER_MAX_M') else EDGE_CANTILEVER_DEFAULT_M
    systems_list = sorted(cat_df['system_id'].unique())

    out_rows = []
    for span_val, span_unit in spans:
        span_m_target = span_to_m(span_val, span_unit)
        L, W = derive_plate(floor_area_m2, L_m, W_m)
        nx, ny, sx, sy, ex, ey = layout_grid_with_edge_limit(L, W, span_m_target, span_m_target, edge_limit_m)
        span_m_used = span_m_target  # honor exact user span for feasibility

        for sys_id in systems_list:
            feasible_all = True
            reasons = []

            per_m2 = curves.get_intensities(curves_df, sys_id)
            swt_sys_knm2 = per_m2.get('swt', 0.0)

            total_area = 0.0
            total_qty = {'concrete_m3':0.0,'steel_m3':0.0,'timber_m3':0.0}
            depth_m = per_m2.get('depth', 0.0)

            for _, pb in req_df.iterrows():
                floors = int(pb['end_floor']) - int(pb['start_floor']) + 1
                area_block = floor_area_m2 * floors
                total_area += area_block

                mode = str(ctl.get('LOAD_CHECK_MODE','SPAN_PLUS_LOADS')).upper()
                best_row, reason = systems.check_rows_for_system(
                    cat_df, sys_id,
                    pb['req_sdl_non_swt_knm2'],
                    pb['req_ll_knm2'],
                    swt_sys_knm2,
                    span_m_used,
                    depth_limit_m,
                    load_check_mode=mode
                )

                if best_row is None or reason:
                    feasible_all = False
                    reasons.append(reason or 'no row')

                qty = quantities.totals_from_intensity(per_m2, area_block)
                for k in total_qty:
                    total_qty[k] += qty[k]
                depth_m = qty.get('depth_m', depth_m)

            row_any = cat_df[cat_df['system_id']==sys_id].iloc[0].to_dict()
            mat_ids = {
                'material_concrete_id': row_any.get('material_concrete_id') or '',
                'material_steel_id': row_any.get('material_steel_id') or '',
                'material_timber_id': row_any.get('material_timber_id') or '',
                'material_pt_id': row_any.get('material_pt_id') or '',
            }

            carbon_total, cost_total, _ = impacts.calc_impacts_cost(total_qty, mat_ids, mats_by_id)
            carbon_per_m2 = (carbon_total/total_area) if total_area>0 else float('nan')
            cost_per_m2   = (cost_total/total_area) if total_area>0 else float('nan')
            system_label  = row_any.get('system_name') or row_any.get('type','') or sys_id

            out_rows.append({
                "span_m": span_m_used,
                "system_id": sys_id,
                "system_name": system_label,
                "category": row_any.get('category',''),
                "type": row_any.get('type',''),
                "manufacturer": row_any.get('manufacturer',''),
                "feasible": bool(feasible_all),
                "reason": "; ".join([r for r in reasons if r]) if reasons else "",
                "depth_m": depth_m,
                "carbon_total_kg": carbon_total,
                "carbon_per_m2": carbon_per_m2,
                "cost_total": cost_total,
                "cost_per_m2": cost_per_m2,
                "area_m2": total_area,
                "nx": nx, "ny": ny,
                "edge_canti_x_m": ex, "edge_canti_y_m": ey
            })

    df = pd.DataFrame(out_rows)
    ranked = carbon_first_then_cost(df)
    pareto = pareto_min_carbon_cost(df)

    saved = (None, None)
    if export_dir:
        saved = save_csvs(df, pareto, export_dir)
    return df, ranked, pareto, saved

# ---------------------------------
# CLI entry
# ---------------------------------
def main():
    p = argparse.ArgumentParser(prog="earlystruct")
    p.add_argument("--data-dir", required=True, help="Folder with required CSVs")
    p.add_argument("--spans", help="Comma-separated spans, e.g. '9,10.5' or '28ft, 32ft'")
    p.add_argument("--export", help="Output folder to save CSV results")
    p.add_argument("--control-file", help="Path to control file (.txt). If provided, it takes precedence unless USE_CSV=Y.")
    args = p.parse_args()
    df, ranked, pareto, saved = evaluate(args.data_dir, args.spans, args.export, args.control_file)
    print("=== SUMMARY (feasible candidates, carbon-first) ===")
    if not ranked.empty:
        print(ranked[['span_m','system_id','system_name','category','carbon_per_m2','cost_per_m2','depth_m']].head(30).to_string(index=False))
    else:
        print("No feasible candidates. Showing closest failures:")
        print(df.sort_values('reason').head(10).to_string(index=False))
    if saved[0]:
        print(f"Saved CSVs to: {saved[0]} and {saved[1]}")

if __name__ == "__main__":
    main()
