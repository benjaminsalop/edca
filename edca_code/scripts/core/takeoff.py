# edca_code/scripts/core/takeoff.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger("takeoff")

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(default)

def bom_per_m2_from_system_row(system_row: pd.Series) -> Dict[str, float]:
    """
    Compute a basic BOM (per m2) for a single system variant row.
    Returns a dict mapping material_key -> quantity per m2.

    BOM key convention:
      - volumetric: "<category>:<material_id>"     (e.g., concrete, timber, screed)
      - mass-based: "<category>_kg:<material_id>"  (e.g., steel_kg, rebar_kg, pt_kg)
      - volumetric steel-like: "<category>_m3:<material_id>" (fallback if only volume exists)
    """
    bom: Dict[str, float] = {}

    def _clean_mat_id(v: Any, default: str | None = None) -> str | None:
        if v is None:
            return default
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return default
        return s

    def _first_positive(cols: List[str]) -> tuple[float, str | None]:
        """Return (value, source_col) for first positive numeric column found."""
        for c in cols:
            if c in system_row.index:
                v = safe_float(system_row.get(c), 0.0)
                if v > 0.0:
                    return float(v), c
        return 0.0, None

    def _add_steel_like(
        *,
        category_base: str,          # "steel" | "rebar" | "pt"
        material_id_col: str,        # e.g. "material_rebar_id"
        default_mat_id: str,         # fallback id string if missing
        mass_cols: List[str],        # preferred
        vol_cols: List[str],         # fallback
    ) -> None:
        mat_id = _clean_mat_id(system_row.get(material_id_col), default=default_mat_id)

        # Prefer mass if present (avoids density assumptions upstream and avoids double-counting)
        mass_val, _ = _first_positive(mass_cols)
        if mass_val > 0.0 and mat_id:
            bom[f"{category_base}_kg:{mat_id}"] = mass_val
            return

        vol_val, _ = _first_positive(vol_cols)
        if vol_val > 0.0 and mat_id:
            bom[f"{category_base}_m3:{mat_id}"] = vol_val
            return

    # --- concrete volume (explicit)
    conc_m3_per_m2 = safe_float(system_row.get("concrete_volume", 0.0))
    mat_conc = _clean_mat_id(system_row.get("material_concrete_id"), default="concrete")
    if conc_m3_per_m2 > 0.0 and mat_conc:
        bom[f"concrete:{mat_conc}"] = conc_m3_per_m2

    # --- screed / topping: screed_depth may be stored in mm or m
    screed_depth = system_row.get("screed_depth", 0.0)
    screed_m = safe_float(screed_depth, 0.0)
    # heuristic: if screed depth > 5, assume mm and convert to metres
    if screed_m > 5:
        screed_m = screed_m / 1000.0
    if screed_m > 0.0:
        # IMPORTANT: do NOT use PT material for screed
        screed_mat = (
            _clean_mat_id(system_row.get("material_screed_id"))
            or _clean_mat_id(system_row.get("material_topping_id"))
            or "screed"
        )
        # Treat screed as a concrete-like volumetric addition (m3/m2)
        bom[f"concrete:{screed_mat}"] = bom.get(f"concrete:{screed_mat}", 0.0) + screed_m

    # --- structural steel (legacy "steel")
    _add_steel_like(
        category_base="steel",
        material_id_col="material_steel_id",
        default_mat_id="steel",
        mass_cols=[
            "steel_mass_per_m2",
            "steel_kg_per_m2",
            "steel_mass",
            "steel_kg",
        ],
        vol_cols=[
            "steel_volume",
            "steel_m3_per_m2",
            "steel_m3",
        ],
    )

    # --- rebar (NEW)
    _add_steel_like(
        category_base="rebar",
        material_id_col="material_rebar_id",
        default_mat_id="rebar",
        mass_cols=[
            "rebar_mass_per_m2",
            "rebar_kg_per_m2",
            "rebar_mass",
            "rebar_kg",
        ],
        vol_cols=[
            "rebar_volume",
            "rebar_m3_per_m2",
            "rebar_m3",
        ],
    )

    # --- PT steel / tendons (NEW)
    _add_steel_like(
        category_base="pt",
        material_id_col="material_pt_id",
        default_mat_id="pt",
        mass_cols=[
            "pt_mass_per_m2",
            "pt_kg_per_m2",
            "pt_mass",
            "pt_kg",
        ],
        vol_cols=[
            "pt_volume",
            "pt_m3_per_m2",
            "pt_m3",
        ],
    )

    # NOTE: do not include record-only fields (e.g. swt) in BOM; keep BOM strictly material quantities

    # --- timber
    timber_m3 = safe_float(system_row.get("timber_volume", 0.0))
    timber_mat = _clean_mat_id(system_row.get("material_timber_id"), default="timber")
    if timber_m3 > 0.0 and timber_mat:
        bom[f"timber:{timber_mat}"] = timber_m3

    return bom

def expand_bom_to_floor(bom_per_m2: Dict[str, float], floor_area_m2: float, assemblies: int = 1) -> Dict[str, float]:
    """
    Multiply per-m2 BOM to floor-level quantities.
    - floor_area_m2: total gross floor area covered by the assembly
    - assemblies: number of identical assemblies per floor (default 1)
    Returns mapping material_key -> total quantity for the floor.
    """
    factor = float(floor_area_m2) * int(assemblies)
    out: Dict[str, float] = {}
    for k, v in bom_per_m2.items():
        out[k] = float(v) * factor
    return out

def combined_floor_takeoffs(systems_df: pd.DataFrame,
                            floor_area_lookup: Dict[int, float],
                            mapping_system_to_floor: Dict[int, str]) -> Dict[int, Dict[str, float]]:
    """
    For each floor (keyed by floor number) compute the combined floor-level material totals.
    """
    results: Dict[int, Dict[str, float]] = {}

    # ensure we have an index we can query by variant name
    if "system_variant" in systems_df.columns:
        sys_index = systems_df.set_index("system_variant", drop=False)
    else:
        # if no variant column, create an index from the DataFrame index but warn the user
        sys_index = systems_df.copy()
        logger.warning("combined_floor_takeoffs: systems_df has no 'system_variant' column; using integer index. Prefer 'system_variant' for reliable lookup.")

    for floor, system_variant in mapping_system_to_floor.items():
        if system_variant not in sys_index.index:
            raise KeyError(f"System variant '{system_variant}' for floor {floor} not found in systems catalog (checked index).")
        row = sys_index.loc[system_variant]
        # if multiple rows match, take the first and log a warning
        if isinstance(row, pd.DataFrame):
            logger.warning("Multiple rows found for system_variant '%s'; using first match.", system_variant)
            row = row.iloc[0]

        # compute bom per m2 and expand to floor
        bom_m2 = bom_per_m2_from_system_row(row)
        area = safe_float(floor_area_lookup.get(floor, 0.0))
        floor_bom = expand_bom_to_floor(bom_m2, area, assemblies=1)
        results[int(floor)] = floor_bom

    return results

def materials_per_floor_dataframe_from_combined(combined_floor_boms: dict, materials_df: 'pd.DataFrame' = None) -> 'pd.DataFrame':
    """
    Convert the combined_floor_takeoffs mapping into a tidy DataFrame.
    """
    rows = []

    # Pre-index materials metadata once (big performance + reliability win)
    materials = None
    material_unit_col = None
    if isinstance(materials_df, pd.DataFrame):
        if "material_id" in materials_df.columns:
            materials = materials_df.set_index("material_id", drop=False)
        else:
            # allow index-as-material_id
            materials = materials_df

        for c in ("unit", "uom", "quantity_unit", "measure_unit"):
            if c in materials_df.columns:
                material_unit_col = c
                break

    for floor, bom in (combined_floor_boms or {}).items():
        for mat_key, qty in (bom or {}).items():
            if not mat_key or qty is None:
                continue
            parts = mat_key.split(":", 1)
            if len(parts) == 2:
                mat_type, mat_id = parts[0], parts[1]
            else:
                mat_type, mat_id = parts[0], ""

            # best-effort unit inference
            if mat_type.endswith("_kg") or mat_type == "steel_kg":
                q_unit = "kg"
            elif mat_type.startswith("steel_") and mat_type.endswith("m3"):
                q_unit = "m3"
            elif mat_type in ("concrete", "timber", "screed", "deck") or mat_type.endswith("m3"):
                q_unit = "m3"
            else:
                # fallback: if materials_df has entry for material_id with a unit field, use that
                q_unit = None
                if materials is not None and material_unit_col is not None and mat_id in materials.index:
                    try:
                        q_unit = materials.loc[mat_id].get(material_unit_col, None)
                    except Exception:
                        q_unit = None
                if q_unit is None:
                    q_unit = "unknown"

            rows.append({
                "floor": int(floor),
                "material_key": mat_key,
                "material_type": mat_type,
                "material_id": mat_id,
                "quantity": float(qty),
                "quantity_unit": q_unit,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        # return canonical empty frame with columns
        return pd.DataFrame(columns=["floor","material_key","material_type","material_id","quantity","quantity_unit"])

    # sensible ordering
    df = df.sort_values(["floor","material_type","material_id"]).reset_index(drop=True)
    return df

def materials_per_floor_csv(candidates_df: 'pd.DataFrame',
                            materials_df: 'pd.DataFrame' = None,
                            out_fp: 'Path | str' = None,
                            floor_area_lookup: dict = None,
                            mapping_system_to_floor: dict = None) -> 'pd.DataFrame':
    """
    High-level helper: given the final candidates DataFrame (the one produced by run_edca),
    compute combined floor-level takeoffs and write out a CSV summarising material quantities per floor.

    Heuristics used (best-effort):
      - If mapping_system_to_floor and floor_area_lookup provided, call combined_floor_takeoffs()
      - Else, try to infer mapping from candidates_df: if columns 'floor' and 'system_variant' present,
        build mapping by taking the first system_variant per floor and use any area column found.
    """
    import pandas as pd
    from pathlib import Path
    from math import isnan

    # try to use caller-provided combined mapping if available
    combined = None
    if mapping_system_to_floor is not None and floor_area_lookup is not None:
        combined = combined_floor_takeoffs(candidates_df, floor_area_lookup, mapping_system_to_floor)
    else:
        # heuristics: look for 'floor' and 'system_variant' in candidates
        if isinstance(candidates_df, pd.DataFrame) and "floor" in candidates_df.columns and "system_variant" in candidates_df.columns:
            # try to find an area column
            area_col = None
            for candidate in ("floor_area","area_m2","area","floor_area_m2","plan_area"):
                if candidate in candidates_df.columns:
                    area_col = candidate
                    break
            # build lookups: for each floor, take the first system_variant and average area
            mapping = {}
            area_lookup = {}
            for floor, group in candidates_df.groupby("floor"):
                # choose the most common system_variant for that floor
                sv = group["system_variant"].mode().iloc[0] if not group["system_variant"].mode().empty else group["system_variant"].iloc[0]
                mapping[int(floor)] = str(sv)
                if area_col is not None:
                    # take mean of the area column for rows on that floor (best-effort)
                    try:
                        area_lookup[int(floor)] = float(group[area_col].mean())
                    except Exception:
                        area_lookup[int(floor)] = 0.0
                else:
                    # fallback: try to use a 'gross_floor_area' column
                    area_lookup[int(floor)] = float(group.get("gross_floor_area", group.get("floor_area_m2", 0.0)).mean() if not group.empty else 0.0)
            combined = combined_floor_takeoffs(candidates_df, area_lookup, mapping)
        else:
            logger.warning("materials_per_floor_csv: could not infer floor mapping from candidates_df; returning empty frame")
            combined = {}

    # convert to dataframe
    df = materials_per_floor_dataframe_from_combined(combined, materials_df=materials_df)

    # write to CSV if path provided
    if out_fp:
        try:
            outp = Path(out_fp)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outp, index=False)
            logger.info("[takeoff] Wrote materials per-floor CSV: %s (rows=%d)", outp, len(df))
        except Exception:
            logger.exception("[takeoff] Failed to write materials per-floor CSV to %s", out_fp)

    return df