# edca_code/scripts/core/carbon.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import logging

logger = logging.getLogger("carbon")

# -------------------------
# Utilities
# -------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert x to float, returning default for None/NaN/unconvertible values."""
    if x is None:
        return float(default)
    try:
        if pd.isna(x):
            return float(default)
    except Exception:
        # pd.isna may fail for some types; fall through to try/except
        pass
    try:
        return float(x)
    except Exception:
        return float(default)

# -------------------------
# Materials table loader
# -------------------------
def load_materials_table(path: str) -> pd.DataFrame:
    """
    Load materials properties table from CSV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Materials CSV not found: {p}")

    df = pd.read_csv(p, dtype={"material_id": str})

    # minimal recommended set — we do not require all, but warn / raise if none of the A1-A3 columns exist
    required_candidates = ["ec_a1a3_volumetric", "ec_a1a3_mass"]
    if not any(c in df.columns for c in required_candidates):
        raise ValueError(f"Materials CSV missing A1-A3 columns (expected one of {required_candidates}): {p}")

    # ensure material_id present
    if "material_id" not in df.columns:
        raise ValueError(f"Materials CSV must contain 'material_id' column: {p}")

    # Coerce common numeric columns to numeric where present
    to_numeric_cols = [
        "density",
        "ec_a1a3_volumetric",
        "ec_a1a3_mass",
        "ec_a4_per_ton_km",
        "transport_km",
        "ec_a5_mass",
        "cost_volumetric",
        "cost_mass",]
    for c in to_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("material_id", drop=False)

    # warnings for missing but non-fatal columns
    if "density" not in df.columns:
        logger.warning("load_materials_table: 'density' column missing in materials table; volumetric<->mass conversions may fail.")
    if "ec_a1a3_volumetric" not in df.columns and "ec_a1a3_mass" not in df.columns:
        logger.warning("load_materials_table: no A1-A3 data present (volumetric or mass). Computations will be zero.")
    return df

# -------------------------
# Material-specific helpers
# -------------------------
def material_a1a3_volumetric(material_row: pd.Series) -> float:
    """
    Return A1-A3 intensity as kgCO2e / m3.
    Preference:
      - ec_a1a3_volumetric (kgCO2e / m3)
      - ec_a1a3_mass (kgCO2e / kg) * density (kg / m3)
    Returns 0.0 if data insufficient.
    """
    if pd.notna(material_row.get("ec_a1a3_volumetric")):
        return _safe_float(material_row["ec_a1a3_volumetric"])
    if pd.notna(material_row.get("ec_a1a3_mass")) and pd.notna(material_row.get("density")):
        return _safe_float(material_row["ec_a1a3_mass"]) * _safe_float(material_row["density"])
    logger.warning(
        "material_a1a3_volumetric: missing ec_a1a3_volumetric and/or density for material_id=%s",
        material_row.get("material_id", "(unknown)"),)
    return 0.0

# -------------------------
# Core assembly carbon computation
# -------------------------
def compute_assembly_carbon_from_bom(
    bom: Dict[str, float],
    materials_df: pd.DataFrame,
    include_a4_a5: bool = True,
    default_transport_km: float = 50.0) -> Dict[str, Any]:
    """
    Compute carbon for an assembly BOM.
    Parameters:
      - bom: dict mapping 'category:material_id' -> quantity
          e.g., 'concrete:material_001' -> 2.5  (m3)
                'steel_kg:material_002' -> 150.0 (kg)
      - materials_df: DataFrame loaded from materials CSV
      - include_a4_a5: if True, include transport (A4) and end-of-life (A5) impacts
      - default_transport_km: used if material row transport_km is missing
    Returns:
      { "per_material": [ ... ], "totals": {...} }
    """
    if not isinstance(bom, dict):
        raise TypeError("BOM must be a dict mapping 'category:material_id' -> quantity")

    per_material: List[Dict[str, Any]] = []
    total_a1a3 = 0.0
    total_a4 = 0.0
    total_a5 = 0.0
    total_cost = 0.0

    for key, raw_qty in bom.items():
        if ":" not in key:
            logger.debug("Skipping BOM key without category: %s", key)
            continue
        category, mat_id = key.split(":", 1)
        qty = _safe_float(raw_qty, default=0.0)

        # lookup material row (defensive)
        mat_row: Optional[pd.Series]
        try:
            mat_row = materials_df.loc[mat_id]
        except Exception:
            mat_row = None

        if mat_row is None:
            logger.warning("Material id '%s' not found in materials table; treating quantities as zero-impact for now", mat_id)
            per_material.append({
                "material_id": mat_id,
                "category": category,
                "qty_m3": 0.0,
                "qty_kg": 0.0,
                "a1a3": 0.0,
                "a4": 0.0,
                "a5": 0.0,
                "total": 0.0,
                "cost": 0.0,})
            continue

        density = _safe_float(mat_row.get("density", 0.0))

        # interpret quantities
        qty_m3 = 0.0
        qty_kg = 0.0
        if category in ("concrete", "timber", "steel_m3"):
            qty_m3 = qty
            qty_kg = qty_m3 * density if density > 0.0 else 0.0
        elif category == "steel_kg":
            qty_kg = qty
            qty_m3 = (qty_kg / density) if density > 0.0 else 0.0
        else:
            # default assumption: volumetric
            qty_m3 = qty
            qty_kg = qty_m3 * density if density > 0.0 else 0.0

        # A1-A3
        a1a3_volumetric = material_a1a3_volumetric(mat_row)
        a1a3 = a1a3_volumetric * qty_m3

        # A4 transport
        a4 = 0.0
        if include_a4_a5:
            a4_per_ton_km = _safe_float(mat_row.get("ec_a4_per_ton_km", 0.0))
            transport_km = _safe_float(mat_row.get("transport_km", default_transport_km))
            tonnes = qty_kg / 1000.0 if qty_kg > 0.0 else 0.0
            a4 = a4_per_ton_km * tonnes * transport_km

        # A5 end-of-life (mass-based)
        a5 = 0.0
        if include_a4_a5:
            a5_mass = _safe_float(mat_row.get("ec_a5_mass", 0.0))
            a5 = a5_mass * qty_kg

        # cost (prefer volumetric price)
        cost = 0.0
        if pd.notna(mat_row.get("cost_volumetric")):
            cost = _safe_float(mat_row.get("cost_volumetric")) * qty_m3
        elif pd.notna(mat_row.get("cost_mass")):
            cost = _safe_float(mat_row.get("cost_mass")) * qty_kg

        total = float(a1a3) + float(a4) + float(a5)

        per_material.append({
            "material_id": mat_id,
            "category": category,
            "qty_m3": float(qty_m3),
            "qty_kg": float(qty_kg),
            "a1a3": float(a1a3),
            "a4": float(a4),
            "a5": float(a5),
            "total": float(total),
            "cost": float(cost),})

        total_a1a3 += float(a1a3)
        total_a4 += float(a4)
        total_a5 += float(a5)
        total_cost += float(cost)

    # --- breakdown totals (useful for keeping carbon separate by material / category)
    # Normalise categories so the downstream tables can have stable column names.
    # e.g. steel_kg + steel_m3 -> steel
    totals_by_category: Dict[str, float] = {}
    totals_by_material_id: Dict[str, float] = {}

    for row in per_material:
        cat_raw = str(row.get("category", "") or "")
        cat = "steel" if cat_raw.startswith("steel") else cat_raw  # steel_kg / steel_m3 -> steel
        mat_id = str(row.get("material_id", "") or "")
        tot = _safe_float(row.get("total", 0.0))

        if cat:
            totals_by_category[cat] = totals_by_category.get(cat, 0.0) + tot
        if mat_id:
            totals_by_material_id[mat_id] = totals_by_material_id.get(mat_id, 0.0) + tot

    overall_total = float(total_a1a3 + (total_a4 if include_a4_a5 else 0.0) + (total_a5 if include_a4_a5 else 0.0))

    totals = {
        "total_a1a3": float(total_a1a3),
        "total_a4": float(total_a4),
        "total_a5": float(total_a5),
        "total": overall_total,
        "total_cost": float(total_cost),}

    return {
        "per_material": per_material,
        "totals": totals,
        "totals_by_category": totals_by_category,
        "totals_by_material_id": totals_by_material_id,
        }

# -------------------------
# Convenience wrapper
# -------------------------
# convenience: allow reuse of loaded materials_df to avoid repeated CSV I/O
def assembly_carbon_for_floor(
    bom_floor: Dict[str, float],
    materials_csv: Optional[str] = None,
    materials_df: Optional[pd.DataFrame] = None,
    include_a4_a5: bool = True,
    default_transport_km: float = 50.0,
) -> Dict[str, Any]:
    """
    Compute carbon for a single floor BOM.
    Either provide materials_csv (path) OR materials_df (already loaded DataFrame).
    If both are provided, materials_df is used.
    """
    if materials_df is None:
        if materials_csv is None:
            raise ValueError("Either materials_csv or materials_df must be provided")
        materials_df = load_materials_table(materials_csv)
    return compute_assembly_carbon_from_bom(bom_floor, materials_df, include_a4_a5, default_transport_km)


def assembly_carbon_for_building(
    floor_boms: Dict[int, Dict[str, float]],
    materials_csv: Optional[str] = None,
    materials_df: Optional[pd.DataFrame] = None,
    include_a4_a5: bool = True,
    default_transport_km: float = 50.0,
    return_dataframe: bool = False,
) -> Dict[int, Dict[str, Any]] | pd.DataFrame:
    """
    Compute carbon for multiple floors.
    """
    if materials_df is None:
        if materials_csv is None:
            raise ValueError("Either materials_csv or materials_df must be provided")
        materials_df = load_materials_table(materials_csv)

    results: Dict[int, Dict[str, Any]] = {}
    rows = []
    for floor, bom in floor_boms.items():
        res = compute_assembly_carbon_from_bom(bom, materials_df, include_a4_a5, default_transport_km)
        results[int(floor)] = res
        if return_dataframe:
            # flatten per_material into rows with floor annotated
            for pm in res["per_material"]:
                row = {
                    "floor": int(floor),
                    "material_id": pm["material_id"],
                    "category": pm["category"],
                    "qty_m3": pm["qty_m3"],
                    "qty_kg": pm["qty_kg"],
                    "a1a3": pm["a1a3"],
                    "a4": pm["a4"],
                    "a5": pm["a5"],
                    "total": pm["total"],
                    "cost": pm["cost"],
                }
                rows.append(row)

    if return_dataframe:
        df = pd.DataFrame(rows)
        # you may also add aggregated totals per floor if desired:
        totals_rows = []
        for floor, res in results.items():
            t = res["totals"]
            totals_rows.append({"floor": int(floor), **t})
        totals_df = pd.DataFrame(totals_rows)
        return {"per_material_df": df, "totals_df": totals_df}
    return results

