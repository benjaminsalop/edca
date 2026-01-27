# edca_code/scripts/core/takeoff.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd

# Assumptions:
# - systems_variants rows contain per-m2 volumes for materials:
#    concrete_volume (m3 per m2), steel_volume (m3 per m2 OR kg per m2 if you prefer),
#    timber_volume (m3 per m2)
# - column names match the schema you provided; adapt names if your parquet uses different labels.
# - takeoff here produces quantities per assembly (or per floor area) in canonical units:
#    concrete_m3, steel_kg, timber_m3, etc.
# - If steel_volume is in m3 we convert to kg using material density if available later in carbon.py; here we keep m3 for concrete/timber and m3 or kg for steel depending on the column.

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df

def bom_per_m2_from_system_row(system_row: pd.Series) -> Dict[str, float]:
    """
    Compute a basic BOM (per m2) for a single system variant row.

    Returns a dict mapping material_id -> quantity per m2, with units implied:
      - 'concrete:<material_id>': m3 per m2
      - 'steel:<material_id>': kg per m2  (if variant provides steel_volume in m3 we'll leave as 'steel_m3' key)
      - 'timber:<material_id>': m3 per m2
      - 'swt': swt value (if present) kept as numeric quantity (units depend on your catalog)

    Notes:
      - system_row is a pandas Series with columns: concrete_volume, steel_volume, timber_volume,
        material_concrete_id, material_steel_id, material_timber_id, swt, screed_depth, etc.
      - This function makes minimal assumptions; conversion m3->kg is left to carbon.py where material density is known.
    """
    bom: Dict[str, float] = {}

    # concrete
    conc_m3_per_m2 = float(system_row.get("concrete_volume") or 0.0)
    mat_conc = system_row.get("material_concrete_id") or "concrete"
    if conc_m3_per_m2 and mat_conc:
        bom[f"concrete:{mat_conc}"] = conc_m3_per_m2

    # steel: some catalogs store steel as m3 (steel_volume) or as kg (steel_mass_per_m2) - attempt both
    steel_vol = system_row.get("steel_volume", None)
    steel_mat = system_row.get("material_steel_id") or "steel"
    if steel_vol and steel_mat:
        bom[f"steel_m3:{steel_mat}"] = float(steel_vol)

    # timber
    timber_m3 = float(system_row.get("timber_volume") or 0.0)
    timber_mat = system_row.get("material_timber_id") or "timber"
    if timber_m3 and timber_mat:
        bom[f"timber:{timber_mat}"] = timber_m3

    # screed / topping (if screed depth is given as m thickness and density is known later)
    screed_m = float(system_row.get("screed_depth") or 0.0)
    if screed_m:
        # approximate screed volume per m2 is screed_m (m3/m2)
        screed_mat = system_row.get("material_pt_id") or "screed"
        bom[f"concrete:{screed_mat}"] = bom.get(f"concrete:{screed_mat}", 0.0) + screed_m

    # keep SDL / LL / swt for record (not a material)
    if "swt" in system_row:
        bom["swt"] = float(system_row.get("swt") or 0.0)

    return bom


def expand_bom_to_floor(bom_per_m2: Dict[str, float], floor_area_m2: float, assemblies: int = 1) -> Dict[str, float]:
    """
    Multiply per-m2 BOM to floor-level quantities.
    - floor_area_m2: total gross floor area covered by the assembly or the sum of assembly footprints
    - assemblies: number of repeated assemblies on the floor (if you already account for footprint, keep assemblies=1)
    Returns mapping material_key -> total quantity for the floor.
    """
    factor = float(floor_area_m2) * int(assemblies)
    out = {}
    for k, v in bom_per_m2.items():
        out[k] = float(v) * factor
    return out


def combined_floor_takeoffs(systems_df: pd.DataFrame,
                            floor_area_lookup: Dict[int, float],
                            mapping_system_to_floor: Dict[int, str]) -> Dict[int, Dict[str, float]]:
    """
    For each floor (keyed by floor number) compute the combined floor-level material totals.

    Parameters:
      - systems_df: DataFrame of candidate systems (filter results). Must contain 'system_variant' or a unique id.
      - floor_area_lookup: mapping floor_number -> area in m2
      - mapping_system_to_floor: mapping floor_number -> system_variant string (i.e., which system applies to that floor)

    Returns:
      - dict: floor_number -> {material_key: total_qty}
    """
    results = {}
    # create index by system_variant
    sys_index = systems_df.set_index("system_variant") if "system_variant" in systems_df.columns else systems_df

    for floor, system_variant in mapping_system_to_floor.items():
        if system_variant not in sys_index.index:
            raise KeyError(f"System variant '{system_variant}' for floor {floor} not found in systems catalog")
        row = sys_index.loc[system_variant]
        # if multiple rows match, take the first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        bom_m2 = bom_per_m2_from_system_row(row)
        area = float(floor_area_lookup.get(floor, 0.0))
        # assemblies default to 1: caller can pre-compute effective assemblies if needed
        floor_bom = expand_bom_to_floor(bom_m2, area, assemblies=1)
        results[int(floor)] = floor_bom

    return results
