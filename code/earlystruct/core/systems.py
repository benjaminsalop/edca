from __future__ import annotations
import pandas as pd
from .units import load_to_knm2, span_to_m, depth_to_m, mm_to_m

def _depth_from_row(r, unit):
    depth_m = None
    if 'slab_depth' in r and pd.notna(r['slab_depth']) and r['slab_depth'] != '':
        depth_m = depth_to_m(r['slab_depth'], unit)
    elif 'block_depth' in r and pd.notna(r['block_depth']) and r['block_depth'] != '':
        depth_m = depth_to_m(r['block_depth'], unit)
    elif 'ebc_mm' in r and pd.notna(r['ebc_mm']) and r['ebc_mm'] != '':
        depth_m = mm_to_m(r['ebc_mm'])
    return depth_m

def _row_caps(row, span_m, depth_limit_m, req_ll, req_sdl_non_swt, load_check_mode: str):
    """Compute booleans & slack for a single catalog row."""
    unit = row.get('unit', 'metric')
    max_span_m = span_to_m(row.get('max_span', 0.0), unit)
    depth_m = _depth_from_row(row, unit)
    sdl_cap = load_to_knm2(row.get('sdl', 0.0), unit)  # allowance (excl. SWT)
    ll_cap  = load_to_knm2(row.get('ll',  0.0), unit)

    span_ok  = (max_span_m + 1e-9) >= span_m  # tiny tolerance
    depth_ok = True if depth_limit_m is None else (depth_m is not None and depth_m <= depth_limit_m)

    mode = str(load_check_mode).upper()
    if mode == "SPAN_ONLY":
        load_ok = True
    else:
        load_ok = (ll_cap >= req_ll) and (sdl_cap >= req_sdl_non_swt)

    # “how much bigger than necessary” we are (prefer small positives)
    span_over = max(0.0, max_span_m - span_m)
    ll_over   = max(0.0, ll_cap - req_ll)
    sdl_over  = max(0.0, sdl_cap - req_sdl_non_swt)

    return {
        "span_ok": span_ok, "depth_ok": depth_ok, "load_ok": load_ok,
        "span_over": span_over, "ll_over": ll_over, "sdl_over": sdl_over,
        "max_span_m": max_span_m, "depth_m": depth_m, "ll_cap": ll_cap, "sdl_cap": sdl_cap
    }

def check_rows_for_system(cat_df: pd.DataFrame, system_id: str,
                          req_sdl_non_swt: float, req_ll: float,
                          swt_system: float, span_m: float,
                          depth_limit_m: float|None,
                          load_check_mode: str = "SPAN_PLUS_LOADS"):
    """
    Select ONE catalog row within this system_id that is feasible for the given span (+depth/+loads).
    Selection heuristic:
      1) Feasible rows only.
      2) Minimal (span_over), then minimal (ll_over + sdl_over).
    Returns (row, reason). reason=None → feasible.
    """
    rows = cat_df[cat_df['system_id'] == system_id]
    if rows.empty:
        return None, "no catalog rows"

    feasible = []
    best_miss = None
    best_miss_reason = "no feasible row"

    for _, r in rows.iterrows():
        caps = _row_caps(r, span_m, depth_limit_m, req_ll, req_sdl_non_swt, load_check_mode)
        if caps["span_ok"] and caps["depth_ok"] and caps["load_ok"]:
            feasible.append((caps, r))
        else:
            # track closest miss for a useful reason message
            miss_score = (
                0.0 if caps["span_ok"]  else (span_m - caps["max_span_m"]),
                0.0 if caps["depth_ok"] else 1.0,
                0.0 if caps["load_ok"]  else (max(0.0, req_ll - caps["ll_cap"]) + max(0.0, req_sdl_non_swt - caps["sdl_cap"]))
            )
            if (best_miss is None) or (miss_score < best_miss[0]):
                parts = []
                if not caps["span_ok"]:
                    parts.append(f"span {span_m:.2f} > max {caps['max_span_m']:.2f}")
                if str(load_check_mode).upper() != "SPAN_ONLY" and not caps["load_ok"]:
                    if caps["ll_cap"] < req_ll:
                        parts.append(f"LL req {req_ll:.2f} > cap {caps['ll_cap']:.2f}")
                    if caps["sdl_cap"] < req_sdl_non_swt:
                        parts.append(f"SDL req {req_sdl_non_swt:.2f} > cap {caps['sdl_cap']:.2f}")
                if not caps["depth_ok"] and depth_limit_m is not None and caps["depth_m"] is not None:
                    parts.append(f"depth {caps['depth_m']:.3f}m > limit {depth_limit_m:.3f}m")
                best_miss = (miss_score, r)
                best_miss_reason = "; ".join(parts) if parts else "no feasible row"

    if feasible:
        # choose the row with minimal oversizing (closest fit)
        feasible.sort(key=lambda t: (t[0]['span_over'], t[0]['ll_over'] + t[0]['sdl_over']))
        return feasible[0][1], None

    return None, best_miss_reason

def check_single_row(row: pd.Series,
                     req_sdl_non_swt: float, req_ll: float,
                     swt_system: float, span_m: float,
                     depth_limit_m: float|None,
                     load_check_mode: str = "SPAN_PLUS_LOADS"):
    caps = _row_caps(row, span_m, depth_limit_m, req_ll, req_sdl_non_swt, load_check_mode)
    ok = caps["span_ok"] and caps["depth_ok"] and caps["load_ok"]
    if ok:
        return True, None, caps["depth_m"]
    parts = []
    if not caps["span_ok"]: parts.append(f"span {span_m:.2f} > max {caps['max_span_m']:.2f}")
    if str(load_check_mode).upper() != "SPAN_ONLY" and not caps["load_ok"]:
        if caps["ll_cap"] < req_ll: parts.append(f"LL req {req_ll:.2f} > cap {caps['ll_cap']:.2f}")
        if caps["sdl_cap"] < req_sdl_non_swt: parts.append(f"SDL req {req_sdl_non_swt:.2f} > cap {caps['sdl_cap']:.2f}")
    return False, ("; ".join(parts) if parts else "no feasible row"), caps["depth_m"]
