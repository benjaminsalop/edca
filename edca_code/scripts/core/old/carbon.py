# edca_code/scripts/core/carbon.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import logging

logger = logging.getLogger("carbon")

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
        raise FileNotFoundError(f"Materials file not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        if "material_id" in df.columns:
            df["material_id"] = df["material_id"].astype(str)
    else:
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
    default_transport_km: float = 50.0,
) -> Dict[str, Any]:
    """
    Compute carbon for an assembly BOM.

    Parameters:
      - bom: dict mapping "<category>:<material_id>" -> quantity
          Examples:
            "concrete:mat_conc"   -> 0.20   (m3)
            "steel_kg:mat_steel"  -> 25.0   (kg)
            "rebar_m3:mat_rebar"  -> 0.003  (m3)
            "pt_kg:mat_pt"        -> 4.2    (kg)

      - materials_df: DataFrame loaded from materials CSV (indexed by material_id recommended)
      - include_a4_a5: if True, include A4 transport + A5 (mass-based)
      - default_transport_km: fallback transport distance

    Returns:
      {
        "per_material": [ ... ],
        "totals": {...},
        "totals_by_category": {...},
        "totals_by_material_id": {...},
      }

    Notes:
      - totals_by_category includes a legacy combined "steel" bucket that sums
        structural steel + rebar + PT for backward compatibility.
      - It also includes explicit split buckets:
          "structural_steel", "rebar", "pt"
    """
    if not isinstance(bom, dict):
        raise TypeError("BOM must be a dict mapping '<category>:<material_id>' -> quantity")

    per_material: List[Dict[str, Any]] = []
    total_a1a3 = 0.0
    total_a4 = 0.0
    total_a5 = 0.0
    total_cost = 0.0

    def _parse_category(raw_category: str) -> tuple[str, str | None]:
        """
        Returns (category_base, qty_unit) where qty_unit is 'kg', 'm3', or None.
        Examples:
          steel_kg -> ('steel', 'kg')
          rebar_m3 -> ('rebar', 'm3')
          concrete -> ('concrete', None)  # default treated as volumetric downstream
        """
        c = str(raw_category or "").strip()
        if c.endswith("_kg"):
            return c[:-3], "kg"
        if c.endswith("_m3"):
            return c[:-3], "m3"
        return c, None

    def _material_lookup(material_id: str) -> Optional[pd.Series]:
        try:
            row = materials_df.loc[material_id]
        except Exception:
            return None
        # If duplicate material_id rows exist and .loc returns DataFrame, use first row
        if isinstance(row, pd.DataFrame):
            logger.warning("Multiple material rows found for material_id='%s'; using first match.", material_id)
            return row.iloc[0]
        return row

    for key, raw_qty in bom.items():
        if ":" not in str(key):
            logger.debug("Skipping BOM key without category separator ':': %s", key)
            continue

        raw_category, mat_id = str(key).split(":", 1)
        category_base, qty_unit = _parse_category(raw_category)
        qty = _safe_float(raw_qty, default=0.0)

        # skip non-positive quantities early
        if qty <= 0.0:
            continue

        # lookup material row (defensive)
        mat_row = _material_lookup(mat_id)

        if mat_row is None:
            logger.warning(
                "Material id '%s' not found in materials table; treating quantities as zero-impact for now",
                mat_id
            )
            per_material.append({
                "material_id": mat_id,
                "category": raw_category,          # preserve exact BOM token
                "category_base": category_base,    # normalized category
                "qty_m3": 0.0,
                "qty_kg": 0.0,
                "a1a3": 0.0,
                "a4": 0.0,
                "a5": 0.0,
                "total": 0.0,
                "cost": 0.0,
            })
            continue

        density = _safe_float(mat_row.get("density", 0.0))

        # -------------------------
        # Interpret quantity units
        # -------------------------
        qty_m3 = 0.0
        qty_kg = 0.0

        if qty_unit == "kg":
            qty_kg = qty
            qty_m3 = (qty_kg / density) if density > 0.0 else 0.0
        elif qty_unit == "m3":
            qty_m3 = qty
            qty_kg = qty_m3 * density if density > 0.0 else 0.0
        else:
            # legacy/default convention: unsuffixed categories are volumetric
            qty_m3 = qty
            qty_kg = qty_m3 * density if density > 0.0 else 0.0

        # -------------------------
        # A1-A3 (prefer mass-based if quantity is in kg and factor exists)
        # -------------------------
        a1a3 = 0.0
        ec_a1a3_mass = mat_row.get("ec_a1a3_mass", None)
        ec_a1a3_vol = mat_row.get("ec_a1a3_volumetric", None)

        if qty_kg > 0.0 and pd.notna(ec_a1a3_mass):
            a1a3 = _safe_float(ec_a1a3_mass) * qty_kg
        elif qty_m3 > 0.0:
            # robust fallback: volumetric direct or mass*density via helper
            a1a3 = material_a1a3_volumetric(mat_row) * qty_m3
        elif qty_kg > 0.0 and density > 0.0:
            # final fallback if qty_m3 didn't get set for some reason
            a1a3 = material_a1a3_volumetric(mat_row) * (qty_kg / density)

        # -------------------------
        # A4 transport (mass-based)
        # -------------------------
        a4 = 0.0
        if include_a4_a5:
            a4_per_ton_km = _safe_float(
                mat_row.get("ec_a4_per_ton_km", mat_row.get("ec_a4_ton_km", 0.0))
            )
            transport_km = _safe_float(
                mat_row.get("transport_km", mat_row.get("transport_distance_km", default_transport_km))
            )
            tonnes = qty_kg / 1000.0 if qty_kg > 0.0 else 0.0
            a4 = a4_per_ton_km * tonnes * transport_km

        # -------------------------
        # A5 end-of-life (mass-based)
        # -------------------------
        a5 = 0.0
        if include_a4_a5:
            a5_mass = _safe_float(mat_row.get("ec_a5_mass", 0.0))
            a5 = a5_mass * qty_kg

        # -------------------------
        # Cost (prefer matching quantity basis first)
        # -------------------------
        cost = 0.0
        has_cost_mass = pd.notna(mat_row.get("cost_mass"))
        has_cost_vol = pd.notna(mat_row.get("cost_volumetric", mat_row.get("cost_volume")))

        if qty_unit == "kg" and has_cost_mass:
            cost = _safe_float(mat_row.get("cost_mass")) * qty_kg
        elif (qty_unit in ("m3", None)) and has_cost_vol:
            cost_rate = mat_row.get("cost_volumetric", mat_row.get("cost_volume"))
            cost = _safe_float(cost_rate) * qty_m3
        elif has_cost_vol and qty_m3 > 0.0:
            cost_rate = mat_row.get("cost_volumetric", mat_row.get("cost_volume"))
            cost = _safe_float(cost_rate) * qty_m3
        elif has_cost_mass and qty_kg > 0.0:
            cost = _safe_float(mat_row.get("cost_mass")) * qty_kg

        total = float(a1a3) + float(a4) + float(a5)

        per_material.append({
            "material_id": mat_id,
            "category": raw_category,          # exact BOM key prefix
            "category_base": category_base,    # normalized base
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

    if _dbg_enabled(None):
            logger.debug(
                "[debug] carbon item: material_id=%s category=%s qty_m3=%.6g qty_kg=%.6g a1a3=%.6g a4=%.6g a5=%.6g total=%.6g",
                mat_id, raw_category, qty_m3, qty_kg, a1a3, a4, a5, total
            )

    # -------------------------
    # Breakdown totals (category + material_id)
    # -------------------------
    totals_by_category: Dict[str, float] = {}
    totals_by_material_id: Dict[str, float] = {}

    for row in per_material:
        cat_base = str(row.get("category_base", "") or "")
        mat_id = str(row.get("material_id", "") or "")
        tot = _safe_float(row.get("total", 0.0))

        # Explicit split buckets
        if cat_base == "steel":
            totals_by_category["structural_steel"] = totals_by_category.get("structural_steel", 0.0) + tot
            # Legacy combined bucket (all ferrous goes here; this row is structural steel)
            totals_by_category["steel"] = totals_by_category.get("steel", 0.0) + tot
        elif cat_base == "rebar":
            totals_by_category["rebar"] = totals_by_category.get("rebar", 0.0) + tot
            totals_by_category["steel"] = totals_by_category.get("steel", 0.0) + tot  # legacy combined bucket
        elif cat_base == "pt":
            totals_by_category["pt"] = totals_by_category.get("pt", 0.0) + tot
            totals_by_category["steel"] = totals_by_category.get("steel", 0.0) + tot  # legacy combined bucket
        elif cat_base:
            totals_by_category[cat_base] = totals_by_category.get(cat_base, 0.0) + tot

        if mat_id:
            totals_by_material_id[mat_id] = totals_by_material_id.get(mat_id, 0.0) + tot

    overall_total = float(
        total_a1a3
        + (total_a4 if include_a4_a5 else 0.0)
        + (total_a5 if include_a4_a5 else 0.0)
    )

    totals = {
        "total_a1a3": float(total_a1a3),
        "total_a4": float(total_a4),
        "total_a5": float(total_a5),
        "total": overall_total,
        "total_cost": float(total_cost),
    }

    if _dbg_enabled(None):
        _dbg_kv("carbon.totals", totals, explicit=True, level=logging.INFO)
        _dbg_kv("carbon.totals_by_category", totals_by_category, explicit=True, level=logging.INFO)
    
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

