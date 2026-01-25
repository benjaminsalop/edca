# src/edtool/db/loaders.py
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List

from .schemas import (
    Material,
    Occupancy,
    SystemFamily,
    SystemVariant,
)

CANONICAL = Path("inputs/canonical")

def _load_parquet_or_csv(name: str) -> pd.DataFrame:
    pq = CANONICAL / f"{name}.parquet"
    csv = CANONICAL / f"{name}.cleaned.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No parquet or cleaned CSV found for {name}")

def load_materials() -> Tuple[Dict[str, Material], List[str]]:
    df = _load_parquet_or_csv("materials")
    materials = {}
    errors = []

    for i, row in df.iterrows():
        try:
            m = Material(**row.to_dict())
            materials[m.material_id] = m
        except Exception as e:
            errors.append(f"materials row {i}: {e}")

    return materials, errors

def load_system_families() -> Tuple[Dict[str, SystemFamily], List[str]]:
    df = _load_parquet_or_csv("system_families")
    families = {}
    errors = []

    for i, row in df.iterrows():
        try:
            f = SystemFamily(**row.to_dict())
            families[f.system_family_id] = f
        except Exception as e:
            errors.append(f"system_families row {i}: {e}")

    return families, errors

def load_system_variants() -> Tuple[Dict[str, SystemVariant], List[str]]:
    df = _load_parquet_or_csv("system_variants")
    variants = {}
    errors = []

    for i, row in df.iterrows():
        try:
            v = SystemVariant(**row.to_dict())
            variants[v.system_variant] = v
        except Exception as e:
            errors.append(f"system_variants row {i}: {e}")

    return variants, errors

def validate_relationships(
    materials: Dict[str, Material],
    families: Dict[str, SystemFamily],
    variants: Dict[str, SystemVariant],
) -> List[str]:
    errors = []

    for vid, v in variants.items():
        if v.system_family not in families:
            errors.append(f"Variant {vid} references missing family {v.system_family}")

        for mat_id in [
            v.material_concrete_id,
            v.material_steel_id,
            v.material_timber_id,
        ]:
            if mat_id and mat_id not in materials:
                errors.append(f"Variant {vid} references missing material {mat_id}")

    return errors

if __name__ == "__main__":
    # Import loader functions (absolute import to be safe)
    try:
        from edca_tool.code.scripts.setup.loaders import (
            load_materials,
            load_system_families,
            load_system_variants,
            validate_relationships,
        )
    except Exception:
        pass

    try:
        mats, m_errs = load_materials()
        fams, f_errs = load_system_families()
        vars_, v_errs = load_system_variants()
    except Exception as e:
        print("ERROR running loaders:", e)
        raise

    rel_errs = []
    try:
        rel_errs = validate_relationships(mats, fams, vars_)
    except NameError:
        for vid, v in vars_.items():
            if v.system_family not in fams:
                rel_errs.append(f"Variant {vid} references missing family {v.system_family}")
            for mat_id in (v.material_concrete_id, v.material_steel_id, v.material_timber_id):
                if mat_id and mat_id not in mats:
                    rel_errs.append(f"Variant {vid} references missing material {mat_id}")

    # Collect errors
    all_errors = []
    for tag, errs in (("materials", m_errs), ("families", f_errs), ("variants", v_errs)):
        if errs:
            all_errors += [f"{tag}: {e}" for e in errs]

    all_errors += rel_errs

    if all_errors:
        print("DATA VALIDATION ERRORS (count={}):".format(len(all_errors)))
        for e in all_errors:
            print(" -", e)
        # non-zero exit for CI
        import sys
        sys.exit(2)

    print("✅ Data validation passed (materials={}, families={}, variants={})".format(
        len(mats), len(fams), len(vars_)
    ))
