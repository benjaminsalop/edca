from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from edca_code.scripts.code_checks import continuousslab
from edca_code.scripts.code_checks.code_loader import load_material_from_csv
from edca_code.scripts.code_checks.continuousslab import DesignError

from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("code_checks")


@lru_cache(maxsize=4)

def _is_missing(v: Any) -> bool:
    """True for None/NaN/empty string."""
    try:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return bool(pd.isna(v))
    except Exception:
        return v is None or (isinstance(v, str) and v.strip() == "")


def _safe_float(v: Any, default: float) -> float:
    if _is_missing(v):
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _pick_first_present(r: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Tuple[Any, Optional[str]]:
    for k in keys:
        if k in r and not _is_missing(r.get(k)):
            return r.get(k), k
    return default, None

def _coerce_material_id(r: Dict[str, Any], materials_path: Optional[str]) -> None:
    """
    Ensure r['material_id'] is set.
    Supports: material_id aliases; and mapping material_row -> material_id using the materials table.
    """
    # If already present, keep it
    if not _is_missing(r.get("material_id")):
        return

    # 1) Try aliases that might already store an ID string
    for k in (
        "material_concrete_id",
        "concrete_material_id",
        "concrete_id",
        "material",
        "material_grade",
        "concrete_grade",
        "strength_class",
        # common row-index style fields (sometimes they actually contain IDs)
        "material_row",
        "concrete_row",
        "steel_row",
        "mat_row",
    ):
        if not _is_missing(r.get(k)):
            r["material_id"] = r.get(k)
            break

    # If we got something non-empty, stop here
    if not _is_missing(r.get("material_id")):
        return

    # 2) If we only have a numeric row index, map it to material_id using the materials table
    if not materials_path:
        return

    # find a row-index key
    row_key = None
    for k in ("material_row", "concrete_row", "steel_row", "mat_row"):
        if not _is_missing(r.get(k)):
            row_key = k
            break
    if row_key is None:
        return

    # parse int
    try:
        s = str(r.get(row_key)).strip()
        if s.endswith(".0"):
            s = s[:-2]
        idx = int(s)
    except Exception:
        return

    try:
        # support CSV or Parquet for the lookup table
        if str(materials_path).lower().endswith((".parquet", ".pq")):
            dfm = pd.read_parquet(materials_path)
        else:
            dfm = pd.read_csv(materials_path, dtype=str)
    except Exception:
        return

    # normalize column name
    cols_low = {c.lower(): c for c in dfm.columns}
    mid_col = cols_low.get("material_id")
    if not mid_col:
        return

    # try 0-based then 1-based indexing
    if 0 <= idx < len(dfm):
        r["material_id"] = str(dfm.iloc[idx][mid_col])
    elif 1 <= idx <= len(dfm):
        r["material_id"] = str(dfm.iloc[idx - 1][mid_col])
        
def _material_debug(material_csv_path: Optional[str], material_id: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"material_id": material_id}
    if not material_csv_path or _is_missing(material_id):
        out.update({"loaded": False, "error": "material_csv_path or material_id missing"})
        return out

    try:
        mat = load_material_from_csv(material_csv_path, str(material_id))
        raw = mat.raw or {}

        # Interpreted properties actually used by checks
        out.update(
            {
                "loaded": True,
                "f_ck_MPa": mat.f_ck_MPa,
                "f_yk_MPa": mat.f_yk_MPa,
                "gamma_c": mat.gamma_c,
                "gamma_s": mat.gamma_s,
                "density_kN_m3": mat.density_kN_m3,
                "original_units": mat.original_units,
            }
        )

        # Raw CSV fields likely to be mismatched / worth verifying
        possible_fy_cols = {k: raw.get(k) for k in raw.keys() if ("fy" in k.lower() or "f_y" in k.lower())}
        possible_fy_cols = dict(list(possible_fy_cols.items())[:25])

        out["raw_subset"] = {
            "material_id": raw.get("material_id"),
            "unit": raw.get("unit")
            or raw.get("units")
            or raw.get("unit_system")
            or raw.get("unitsystem"),
            "concrete_f_ck": raw.get("concrete_f_ck"),
            "steel_fy": raw.get("steel_fy"),
            "density": raw.get("density"),
            "gamma_c": raw.get("gamma_c"),
            "gamma_s": raw.get("gamma_s"),
            "possible_fy_columns": possible_fy_cols,
        }
        return out
    except Exception as e:
        out.update({"loaded": False, "error": f"{type(e).__name__}: {e}"})
        return out


def _build_inputs_snapshot(r: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic snapshot of what the checker *will* use (keys + resolved floats)."""
    snap: Dict[str, Any] = {}

    slab_depth_raw, slab_depth_key = _pick_first_present(r, ("slab_depth", "depth"), None)
    span_raw, span_key = _pick_first_present(r, ("span", "max_span"), None)
    cover_raw, cover_key = _pick_first_present(r, ("cover_m", "nominal_cover_m"), None)
    dev_raw, dev_key = _pick_first_present(r, ("deviation_allowance_m", "deviation_m"), None)

    snap["material_id"] = r.get("material_id")
    snap["slab_depth"] = {"value": slab_depth_raw, "key": slab_depth_key}
    snap["slab_width"] = {"value": r.get("slab_width"), "key": "slab_width" if "slab_width" in r else None}
    snap["span"] = {"value": span_raw, "key": span_key}
    snap["screed_depth"] = {"value": r.get("screed_depth"), "key": "screed_depth" if "screed_depth" in r else None}
    snap["live_load_kN_m2"] = {
        "value": r.get("live_load_kN_m2", r.get("ll")),
        "key": "live_load_kN_m2" if "live_load_kN_m2" in r else ("ll" if "ll" in r else None),
    }
    snap["partition_load_kN_m2"] = {
        "value": r.get("partition_load_kN_m2"),
        "key": "partition_load_kN_m2" if "partition_load_kN_m2" in r else None,
    }
    snap["cover_m"] = {"value": cover_raw, "key": cover_key}
    snap["deviation_allowance_m"] = {"value": dev_raw, "key": dev_key}
    snap["wall_thickness_m"] = {"value": r.get("wall_thickness_m"), "key": "wall_thickness_m" if "wall_thickness_m" in r else None}
    snap["d_bar_m"] = {"value": r.get("d_bar_m"), "key": "d_bar_m" if "d_bar_m" in r else None}

    for k in ("slab_code_loading", "screed_code_loading", "finish_code_loading", "service_code_loading"):
        snap[k] = {"value": r.get(k), "key": k if k in r else None}

    # Context keys useful for debugging merges / load cases
    for k in ("case", "load_case", "_source_case", "floor_load_category", "unit"):
        if k in r:
            snap[k] = r.get(k)

    snap["resolved"] = {
        "slab_depth_m": _safe_float(slab_depth_raw, 0.175),
        "slab_width_m": _safe_float(r.get("slab_width"), 1.0),
        "span_m": _safe_float(span_raw, continuousslab.LOCAL_DEFAULTS.get("fallback_span_m", 6.0)),
        "screed_depth_m": _safe_float(r.get("screed_depth"), 0.0),
        "live_load_kN_m2": _safe_float(r.get("live_load_kN_m2", r.get("ll")), 2.0),
        "partition_load_kN_m2": _safe_float(r.get("partition_load_kN_m2"), 0.0),
        "cover_m": _safe_float(cover_raw, 0.015),
        "deviation_allowance_m": _safe_float(dev_raw, 0.01),
        "wall_thickness_m": _safe_float(r.get("wall_thickness_m"), 0.2),
        "d_bar_m": _safe_float(r.get("d_bar_m"), continuousslab.LOCAL_DEFAULTS.get("d_bar_m", 0.012)),
        "slab_code_loading": _safe_float(r.get("slab_code_loading"), 0.0),
        "screed_code_loading": _safe_float(r.get("screed_code_loading"), 0.0),
        "finish_code_loading": _safe_float(r.get("finish_code_loading"), 0.0),
        "service_code_loading": _safe_float(r.get("service_code_loading"), 0.0),
    }

    # Presence check (not validity) — helps you confirm inputs are at least present.
    required = ["system_variant", "material_id"]
    required_any = [("slab_depth", "depth"), ("span", "max_span"), ("live_load_kN_m2", "ll")]
    missing: List[str] = []
    for k in required:
        if _is_missing(r.get(k)):
            missing.append(k)
    for a, b in required_any:
        if _is_missing(r.get(a)) and _is_missing(r.get(b)):
            missing.append(f"{a}|{b}")
    snap["missing_required"] = missing

    return snap

def _log_debug_blob(sv: str, blob: Dict[str, Any]) -> None:
    try:
        logger.info("[codechecks][inputs] %s", json.dumps({"system_variant": sv, **blob}, default=str))
    except Exception:
        logger.info("[codechecks][inputs] %s %s", sv, str(blob))

def _load_materials_table_cached(path: str) -> pd.DataFrame:
    """Load materials file once; supports CSV or Parquet; returns all strings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
    elif p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df = df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str)
        df = df.fillna("")
    else:
        raise ValueError(f"Unsupported materials file type: {p.suffix}")

    # ensure material_id exists (case-insensitive)
    cols_low = {c.lower(): c for c in df.columns}
    if "material_id" not in df.columns and "material_id" in cols_low:
        df = df.rename(columns={cols_low["material_id"]: "material_id"})
    return df

def _parse_intish(v: Any) -> Optional[int]:
    if _is_missing(v):
        return None
    try:
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return int(s)
    except Exception:
        return None

def _maybe_set_material_id_from_row_index(r: Dict[str, Any], materials_df: Optional[pd.DataFrame]) -> None:
    """If r has material_row (or similar) and materials_df has material_id, map by row index."""
    if materials_df is None or materials_df.empty or "material_id" not in materials_df.columns:
        return
    if not _is_missing(r.get("material_id")):
        return

    row_key = None
    for k in ("material_row", "concrete_row", "steel_row", "mat_row"):
        if not _is_missing(r.get(k)):
            row_key = k
            break
    if row_key is None:
        return

    idx = _parse_intish(r.get(row_key))
    if idx is None:
        # if they stuffed an ID string into material_row, accept it
        candidate = str(r.get(row_key)).strip()
        if candidate:
            r["material_id"] = candidate
        return

    # Try 0-based first, then 1-based (common in spreadsheets)
    if 0 <= idx < len(materials_df):
        r["material_id"] = materials_df.iloc[idx]["material_id"]
        return
    if 1 <= idx <= len(materials_df):
        r["material_id"] = materials_df.iloc[idx - 1]["material_id"]
        return

def _maybe_set_material_id_from_unambiguous_props(r: Dict[str, Any], materials_df: Optional[pd.DataFrame]) -> None:
    """Conservative: if exactly one match by family + grade/strength, fill material_id."""
    if materials_df is None or materials_df.empty or "material_id" not in materials_df.columns:
        return
    if not _is_missing(r.get("material_id")):
        return

    # family helps narrow (if present)
    fam = r.get("family")
    if _is_missing(fam) and "material_family" in r:
        fam = r.get("material_family")
    fam = None if _is_missing(fam) else str(fam).strip().lower()

    df = materials_df
    if fam and "family" in df.columns:
        df = df[df["family"].astype(str).str.strip().str.lower() == fam]

    # Try exact matches on common numeric fields (string compare is fine since we cast to str)
    for key in ("standard_grade", "concrete_f_ck", "steel_fy"):
        if key in r and not _is_missing(r.get(key)) and key in df.columns:
            val = str(r.get(key)).strip()
            cand = df[df[key].astype(str).str.strip() == val]
            if len(cand) == 1:
                r["material_id"] = cand.iloc[0]["material_id"]
                return
            
def run_code_checks_on_candidates(
    candidates_df: pd.DataFrame,
    *,
    material_csv_path: Optional[str] = None,
    load_combos_yaml: Optional[str] = None,
    load_values_yaml: Optional[str] = None,
    debug_inputs: bool = False,
    debug_only_on_fail: bool = True,
    debug_max_rows: int = 50,
) -> List[Dict[str, Any]]:
    """Run code checks on a candidates table.

    If debug_inputs=True, this will attach an `inputs_debug` blob (and log it) so you can verify
    depths/material_id/loads/etc are present and see the parsed material properties.
    """

    results: List[Dict[str, Any]] = []
    if candidates_df is None or candidates_df.empty:
        return results
    
    debug_printed = 0

    materials_df = None
    if material_csv_path:
        try:
            materials_df = _load_materials_table_cached(str(material_csv_path))
        except Exception:
            # Don't hard-fail codechecks if materials table can't be loaded; checks will fallback.
            materials_df = None

    for _, row in candidates_df.iterrows():
        sv = str(row.get("system_variant", "") or "")
        debug_blob: Dict[str, Any] = {}
        try:
            r = row.to_dict()

            # map material_id expected by continuousslab
            if "material_id" not in r or _is_missing(r.get("material_id")):
                for k in (
                    "material_concrete_id",
                    "concrete_material_id",
                    "concrete_id",
                    "material",
                    "material_grade",
                    "concrete_grade",
                    "strength_class",):
                    if not _is_missing(r.get(k)):
                        r["material_id"] = r.get(k)
                        break
            # If still missing, try mapping material_row -> material_id using the materials table
            _maybe_set_material_id_from_row_index(r, materials_df)

            # Last resort (conservative): if exactly one match by properties, fill it
            _maybe_set_material_id_from_unambiguous_props(r, materials_df)

            # map load keys
            if "live_load_kN_m2" not in r and not _is_missing(r.get("ll")):
                r["live_load_kN_m2"] = r["ll"]

            # Build debug blob early (so you can see inputs even if the checker crashes)
            inputs_snapshot = _build_inputs_snapshot(r)
            material_snapshot = _material_debug(material_csv_path, r.get("material_id"))
            debug_blob = {
                "paths": {
                    "material_csv_path": material_csv_path,
                    "load_combos_yaml": load_combos_yaml,
                    "load_values_yaml": load_values_yaml,
                },
                "inputs": inputs_snapshot,
                "material": material_snapshot,
            }

            if debug_inputs and (not debug_only_on_fail) and debug_printed < int(debug_max_rows):
                _log_debug_blob(sv, debug_blob)
                debug_printed += 1

            out = continuousslab.check_slab_row_preserve_math(
                r,
                material_csv_path=material_csv_path,
                load_combos_yaml=load_combos_yaml,
                load_values_yaml=load_values_yaml,
            )

            row_out: Dict[str, Any] = {
                "system_variant": sv,
                "success": True,
                "codecheck_family": "continuousslab",
                "ULS_kN_m2": out.get("ULS_kN_m2"),
                "ULS_combo_name": out.get("ULS_combo_name"),
                "code_outputs": out,
            }
            if debug_inputs and (not debug_only_on_fail):
                row_out["inputs_debug"] = debug_blob
            results.append(row_out)

        except DesignError as e:
            logger.warning("[codechecks] design fail for %s: %s", sv, e)

            row_out = {
                "system_variant": sv,
                "success": False,
                "error": str(e),
                "codecheck_family": "continuousslab",
            }
            if debug_inputs:
                # log/attach on fail
                if debug_printed < int(debug_max_rows):
                    _log_debug_blob(sv, debug_blob)
                    debug_printed += 1
                row_out["inputs_debug"] = debug_blob
            results.append(row_out)

        except Exception as e:
            logger.exception("[codechecks] unexpected error for %s", sv)

            row_out = {
                "system_variant": sv,
                "success": False,
                "error": str(e),
                "codecheck_family": "continuousslab",
            }
            if debug_inputs:
                if debug_printed < int(debug_max_rows):
                    _log_debug_blob(sv, debug_blob)
                    debug_printed += 1
                row_out["inputs_debug"] = debug_blob
            results.append(row_out)

    return results
