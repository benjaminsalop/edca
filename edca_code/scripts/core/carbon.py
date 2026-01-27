# edca_code/scripts/core/carbon.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import math
import logging


# Simple utilities
def load_materials_table(path: str) -> pd.DataFrame:
    """
    Load materials CSV and *normalize* column names to the canonical ones the
    carbon functions expect.

    Canonical columns produced by this loader:
      - material_id (index)
      - density_kg_per_m3
      - ec_a1a3_volumetric   (kgCO2e per m3)
      - ec_a1a3_mass         (kgCO2e per kg)
      - ec_a4_per_ton_km     (kgCO2e per tonne-km)
      - transport_km
      - ec_a5_per_kg
      - cost_per_m3
      - cost_per_kg
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Materials CSV not found: {p}")
    df = pd.read_csv(p, dtype={"material_id": str})

    if "material_id" not in df.columns:
        raise ValueError("materials CSV must include material_id column")

    # --- column alias mapping (source names -> canonical names) ---
    alias_map = {
        "density": "density_kg_per_m3",
        "density_kg_m3": "density_kg_per_m3",
        "ec_a1a3_volumetric": "ec_a1a3_volumetric",
        "ec_a1a3_mass": "ec_a1a3_mass",
        "ec_a4_ton_km": "ec_a4_per_ton_km",
        "ec_a4_per_ton_km": "ec_a4_per_ton_km",
        "transport_distance": "transport_km",
        "transport_km": "transport_km",
        "ec_a5_mass": "ec_a5_per_kg",
        "ec_a5_per_kg": "ec_a5_per_kg",
        "cost_volume": "cost_per_m3",
        "cost_per_m3": "cost_per_m3",
        "cost_mass": "cost_per_kg",
        "cost_per_kg": "cost_per_kg",
    }

    # If a canonical column is missing but an alias exists, copy it
    for src, dst in alias_map.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]

    # Ensure canonical numeric columns exist
    canonical_numeric = [
        "density_kg_per_m3",
        "ec_a1a3_volumetric",
        "ec_a1a3_mass",
        "ec_a4_per_ton_km",
        "transport_km",
        "ec_a5_per_kg",
        "cost_per_m3",
        "cost_per_kg",
    ]
    for c in canonical_numeric:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce to numeric (NaN for bad entries)
    for c in canonical_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Density unit normalization heuristics ---
    # Some rows use 24 (kN/m3) for concrete; convert those to kg/m3 (24 -> 2400)
    # If density < 1000 and > 1, multiply by 100 (heuristic: kN/m3 -> kg/m3)
    def _norm_density(x):
        try:
            v = float(x)
        except Exception:
            return float("nan")
        if v >= 1000:
            return v
        if 1 < v < 1000:
            return v * 100.0
        if v <= 1:
            return v * 1000.0
        return float("nan")

    df["density_kg_per_m3"] = df["density_kg_per_m3"].apply(_norm_density)

    # Set index
    df = df.set_index("material_id", drop=False)

    # Final defensive fill: keep numeric NaNs (they'll be handled later), but log if many are missing
    missing_density = df["density_kg_per_m3"].isna().sum()
    missing_a1a3_m3 = df["ec_a1a3_volumetric"].isna().sum()
    if missing_density > 0:
        logging.getLogger(__name__).warning(
            "load_materials_table: %d materials missing normalized density_kg_per_m3", missing_density
        )
    if missing_a1a3_m3 == len(df):
        logging.getLogger(__name__).warning(
            "load_materials_table: no materials have ec_a1a3_volumetric (check column names)"
        )

    return df


def _material_a1a3_for_m3(material_row: pd.Series) -> float:
    """
    Prefer ec_a1a3_volumetric if provided; otherwise compute from ec_a1a3_mass * density.
    Returns kgCO2e per m3 (A1-A3).
    """
    if pd.notna(material_row.get("ec_a1a3_volumetric")):
        return float(material_row["ec_a1a3_volumetric"])
    if pd.notna(material_row.get("ec_a1a3_mass")) and pd.notna(material_row.get("density_kg_per_m3")):
        return float(material_row["ec_a1a3_mass"]) * float(material_row["density_kg_per_m3"])
    # fallback: 0.0 (warning handled by caller if desired)
    return 0.0

def _material_a1a3_for_kg(material_row: pd.Series) -> float:
    """
    Return A1-A3 per kg (kgCO2e/kg). Prefer ec_a1a3_mass else ec_a1a3_volumetric/density.
    """
    if pd.notna(material_row.get("ec_a1a3_mass")):
        return float(material_row["ec_a1a3_mass"])
    if pd.notna(material_row.get("ec_a1a3_volumetric")) and pd.notna(material_row.get("density_kg_per_m3")):
        return float(material_row["ec_a1a3_volumetric"]) / float(material_row["density_kg_per_m3"])
    return 0.0

def compute_assembly_carbon_from_bom(bom: Dict[str, float],
                                    materials_df: pd.DataFrame,
                                    include_a4_a5: bool = True,
                                    default_transport_km: float = 50.0) -> Dict[str, Any]:
    """
    Compute carbon for an assembly or floor BOM mapping keys like:
      - 'concrete:<material_id>': quantity in m3
      - 'timber:<material_id>': quantity in m3
      - 'steel_m3:<material_id>' : quantity in m3 (if given)
      - 'steel_kg:<material_id>' : quantity in kg (if given)

    Defensive: treat missing numeric material fields as 0.0 (do NOT let NaNs propagate).
    Returns dict summarizing per_material and totals (all numeric floats).
    """
    per_material: List[Dict[str, Any]] = []
    total_a1a3 = 0.0
    total_a4 = 0.0
    total_a5 = 0.0
    total_cost = 0.0

    def _coerce_qty(x):
        """Return a safe float for qty: None / NaN -> 0.0, else float(x)."""
        # handle pandas/np.nan and None
        if x is None:
            return 0.0
        # pd.isna covers numpy.nan, pandas NA, None
        if pd.isna(x):
            return 0.0
        try:
            return float(x)
        except Exception:
            return 0.0

    for key, raw_qty in (bom.items() if isinstance(bom, dict) else []):
        if ":" not in key:
            continue
        category, mat_id = key.split(":", 1)
        qty = _coerce_qty(raw_qty)

        mat_row = materials_df.loc[mat_id] if (isinstance(materials_df, pd.DataFrame) and mat_id in materials_df.index) else None
        if mat_row is None:
            per_material.append({
                "material_id": mat_id,
                "category": category,
                "qty_m3": float(qty) if category in ("concrete", "timber", "steel_m3") else 0.0,
                "qty_kg": 0.0,
                "a1a3": 0.0,
                "a4": 0.0,
                "a5": 0.0,
                "total": 0.0,
                "cost": 0.0,
            })
            continue

        # read numeric material properties defensively (treat NaN/missing as 0.0)
        density = float(mat_row["density_kg_per_m3"]) if pd.notna(mat_row.get("density_kg_per_m3")) else 0.0

        # compute qty_m3 / qty_kg depending on category
        qty_m3 = 0.0
        qty_kg = 0.0
        if category in ("concrete", "timber", "steel_m3"):
            qty_m3 = float(qty)
            qty_kg = qty_m3 * density if density > 0.0 else 0.0
        elif category == "steel_kg":
            qty_kg = float(qty)
            qty_m3 = qty_kg / density if density > 0.0 else 0.0
        else:
            qty_m3 = float(qty)
            qty_kg = qty_m3 * density if density > 0.0 else 0.0

        # A1-A3: use functions which already return 0.0 if missing
        a1a3_per_m3 = _material_a1a3_for_m3(mat_row)
        a1a3 = float(a1a3_per_m3) * float(qty_m3)

        # A4: transport - treat missing ec_a4_per_ton_km or transport_km as 0.0
        a4_per_ton_km = float(mat_row["ec_a4_per_ton_km"]) if pd.notna(mat_row.get("ec_a4_per_ton_km")) else 0.0
        transport_km = float(mat_row["transport_km"]) if pd.notna(mat_row.get("transport_km")) else float(default_transport_km or 0.0)
        tonnes = qty_kg / 1000.0 if qty_kg > 0 else 0.0
        a4 = a4_per_ton_km * tonnes * transport_km if include_a4_a5 else 0.0

        # A5: end-of-life - treat missing ec_a5_per_kg as 0.0
        a5_per_kg = float(mat_row["ec_a5_per_kg"]) if pd.notna(mat_row.get("ec_a5_per_kg")) else 0.0
        a5 = a5_per_kg * qty_kg if include_a4_a5 else 0.0

        # cost
        cost = 0.0
        if pd.notna(mat_row.get("cost_per_m3")):
            cost = float(mat_row.get("cost_per_m3")) * qty_m3
        elif pd.notna(mat_row.get("cost_per_kg")):
            cost = float(mat_row.get("cost_per_kg")) * qty_kg

        # ensure totals use numeric 0.0 defaults (avoid NaN propagation)
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
            "cost": float(cost),
        })

        total_a1a3 += float(a1a3)
        total_a4 += float(a4)
        total_a5 += float(a5)
        total_cost += float(cost)

    overall_total = float(total_a1a3 + (total_a4 if include_a4_a5 else 0.0) + (total_a5 if include_a4_a5 else 0.0))

    totals = {
        "total_a1a3": float(total_a1a3),
        "total_a4": float(total_a4),
        "total_a5": float(total_a5),
        "total": overall_total,
        "total_cost": float(total_cost),
    }

    return {"per_material": per_material, "totals": totals}

def assembly_carbon_for_floor(bom_floor: Dict[str, float],
                              materials_csv: str,
                              include_a4_a5: bool = True,
                              default_transport_km: float = 50.0) -> Dict[str, Any]:
    """
    Convenience function: load materials, compute carbon for a floor-level BOM.
    Returns combined result with totals and per-material breakdown.
    """
    materials_df = load_materials_table(materials_csv)
    return compute_assembly_carbon_from_bom(bom_floor, materials_df, include_a4_a5, default_transport_km)
