from pathlib import Path
import pandas as pd

def data_path(base_dir: str | Path, *parts: str) -> Path:
    """
    Join DATA_DIR with subpaths safely.
    Example:
        data_path(cf.data_dir, "materials", "materials.csv")
    """
    return Path(base_dir).joinpath(*parts)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce to numeric, keeping NaN where coercion fails."""
    return pd.to_numeric(series, errors="coerce")

def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder output columns to put the most important ones up front if present.
    """
    prefer = [
        "system_variant",
        "system_family",
        "type",
        "typology",
        "max_span",
        "depth_mm",
        "slab_depth",
        "screed_depth",
        "steel_mass_per_m2",
        "concrete_volume",
        "carbon_total_kgCO2",
        "_for_floor",
    ]
    cols = list(df.columns)
    new = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
    return df.loc[:, new]

def infer_type(row):
                txt = str(row.get("system_variant", "")).lower() + " " + str(row.get("type", "")).lower()
                if "timber" in txt or "wood" in txt:
                    return "Timber"
                if "composite" in txt or "deck" in txt:
                    return "Composite"
                if "steel" in txt:
                    return "Steel"
                if any(s in txt for s in ("concrete", "precast", "pt", "post-tension", "post tension")):
                    return "Concrete"
                return "Other"

def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common alternative column names onto the canonical names that reporting expects.
    This is the #1 reason your plots are missing: columns exist, but under different names.
    """
    if df is None or df.empty:
        return df
    D = df.copy()

    # Map of canonical -> aliases in your pipeline outputs
    aliases = {
        "carbon_per_m2": [
            "ec_per_m2", "embodied_carbon_per_m2", "embodied_carbon_kgco2e_m2",
            "embodied_carbon_intensity", "kgco2e_per_m2", "a1a5_kgco2e_m2",
            "carbon_intensity", "carbon_kgco2e_m2", "co2e_per_m2", "carbon_total_kgco2"
        ],
        "cost_per_m2": [
            "cost_m2", "cost_per_area", "cost_per_sqm", "cost_£_m2", "cost_usd_m2"
        ],
        "max_span": [
            "span", "span_m", "max_span_m", "span_max", "max_span_metres"
        ],
        "system_variant": [
            "variant", "variant_id", "system_id", "system_variant_id"
        ],
        "system_family": [
            "family", "system_brand", "brand", "system_family_name"
        ],
        "type": [
            "system_type", "floor_type", "variant_type"
        ],
        "typology": [
            "structural_typology", "material_typology"
        ],
    }

    # Rename first matching alias for each canonical col
    for canon, alts in aliases.items():
        if canon in D.columns:
            continue
        for a in alts:
            if a in D.columns:
                D = D.rename(columns={a: canon})
                break

    return D