# edca_code/scripts/setup/loaders.py
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Union
import numpy as np
import yaml
from datetime import datetime, timezone

from .schemas import (
    Material,
    Occupancy,
    SystemFamily,
    SystemVariant,
)

CANONICAL = Path("inputs/canonical")


def load_parquet_or_csv(name: str) -> pd.DataFrame:
    """
    Look for inputs/canonical/{name}.parquet or inputs/canonical/{name}.csv
    """
    pq = CANONICAL / f"{name}.parquet"
    csv = CANONICAL / f"{name}.csv"

    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No parquet or cleaned CSV found for {name}")


def regenerate_system_tables(
    systems_source: Union[str, Path, pd.DataFrame, None] = None,
    families_name: str = "system_families",
    variants_name: str = "system_variants",
) -> Tuple[Path, Path]:
    """
    Regenerate system_families and system_variants CSVs and parquet files from systems data.

    Returns: (path_to_families_csv, path_to_variants_csv)
    """
    # Load systems DataFrame
    if isinstance(systems_source, pd.DataFrame):
        df = systems_source.copy()
    else:
        # Try canonical sources first
        try:
            df = load_parquet_or_csv("systems")
        except FileNotFoundError:
            candidate1 = CANONICAL / "systems.csv"
            candidate2 = Path("systems.csv")
            if candidate1.exists():
                df = pd.read_csv(candidate1)
            elif candidate2.exists():
                df = pd.read_csv(candidate2)
            elif systems_source:
                p = Path(systems_source)
                if p.exists():
                    df = pd.read_csv(p)
                else:
                    raise FileNotFoundError(f"Provided systems_source not found: {systems_source}")
            else:
                raise FileNotFoundError("Could not find systems data to regenerate tables.")

    # ensure canonical directory exists
    CANONICAL.mkdir(parents=True, exist_ok=True)

    # Normalize dataframe: keep raw values
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    df = df.replace({np.nan: None})

    # -----------------------------
    # SYSTEM FAMILIES
    # -----------------------------
    desired_family_cols = [
        "system_id",
        "component",
        "category",
        "type",
        "span_behavior",
        "manufacturer",
        "unit",
        "width",
        "material_concrete_id",
        "material_steel_id",
        "material_timber_id",
        "source",
    ]
    family_cols = [c for c in desired_family_cols if c in df.columns]

    if "system_id" not in family_cols:
        raise KeyError("systems data must include 'system_id' to derive families")

    system_families = (
        df[family_cols]
        .drop_duplicates(subset=["system_id"])
        .rename(columns={"system_id": "system_family"})
        .reset_index(drop=True)
    )

    families_csv_out = CANONICAL / f"{families_name}.csv"
    system_families.to_csv(families_csv_out, index=False)

    # also write parquet (best-effort)
    families_parquet_out = CANONICAL / f"{families_name}.parquet"
    try:
        # prefer pyarrow if available
        system_families.to_parquet(families_parquet_out, index=False)
    except Exception as e:
        print(f"Warning: failed to write families parquet ({families_parquet_out}): {e}")


    # -----------------------------
    # SYSTEM VARIANTS
    # -----------------------------
    desired_variant_cols = [
        "system_id",
        "unit",
        "slab_depth",
        "beam_depth",
        "screed_depth",
        "steel_depth",
        "swt",
        "sdl",
        "ll",
        "max_span",
        "concrete_volume",
        "steel_volume",
        "timber_volume",
        "material_concrete_id",
        "material_pt_id",
        "material_steel_id",
        "material_timber_id",
        "ebc_mm",
        "beam_ref",
        "source",
    ]
    variant_cols = [c for c in desired_variant_cols if c in df.columns]

    if "system_id" not in variant_cols:
        raise KeyError("systems data must include 'system_id' to derive variants")

    variants = df[variant_cols].copy()
    variants["system_id"] = variants["system_id"].astype(str)
    variants["variant_index"] = variants.groupby("system_id").cumcount() + 1
    variants["system_variant"] = variants["system_id"] + "_" + variants["variant_index"].astype(str)
    variants = variants.drop(columns="variant_index").rename(columns={"system_id": "system_family"})
    remaining = [c for c in variants.columns if c not in ("system_variant", "system_family")]
    variants = variants[["system_variant", "system_family"] + remaining]

    variants_csv_out = CANONICAL / f"{variants_name}.csv"
    variants.to_csv(variants_csv_out, index=False)

    variants_parquet_out = CANONICAL / f"{variants_name}.parquet"
    try:
        variants.to_parquet(variants_parquet_out, index=False)
    except Exception as e:
        print(f"Warning: failed to write variants parquet ({variants_parquet_out}): {e}")

    # write schema YAMLs so later tooling has schema files to reference
    families_schema_out = CANONICAL / f"{families_name}_schema.yaml"
    variants_schema_out = CANONICAL / f"{variants_name}_schema.yaml"

    try:
        write_simple_schema_yaml(system_families, families_schema_out, families_name)
    except Exception as e:
        print(f"Warning writing families schema yaml: {e}")

    try:
        write_simple_schema_yaml(variants, variants_schema_out, variants_name)
    except Exception as e:
        print(f"Warning writing variants schema yaml: {e}")

    return families_csv_out, variants_csv_out

def df_to_objects(df: pd.DataFrame, model_cls, key_attr: str) -> Tuple[dict, list]:
    """
    Helper: convert DataFrame rows into dataclass/model instances.
    Returns: (dict keyed by key_attr -> instance, list of error messages)
    """
    objs = {}
    errors = []

    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    df = df.replace({np.nan: None})

    for i, row in df.iterrows():
        try:
            inst = model_cls(**row.to_dict())
            key = getattr(inst, key_attr)
            objs[key] = inst
        except Exception as e:
            errors.append(f"{model_cls.__name__} row {i}: {e}")

    return objs, errors


def load_materials() -> Tuple[Dict[str, Material], List[str]]:
    df = load_parquet_or_csv("materials")
    return df_to_objects(df, Material, "material_id")

def load_system_families() -> Tuple[Dict[str, SystemFamily], List[str]]:
    regenerate_system_tables()   # <-- ADD THIS LINE (force)
    df = load_parquet_or_csv("system_families")
    return df_to_objects(df, SystemFamily, "system_family")



def load_system_variants() -> Tuple[Dict[str, SystemVariant], List[str]]:
    regenerate_system_tables()   # <-- ADD THIS LINE (force)
    df = load_parquet_or_csv("system_variants")
    return df_to_objects(df, SystemVariant, "system_variant")

def load_systems() -> pd.DataFrame:
    """
    Load systems. Try canonical parquet/cleaned CSV first, then fallback to canonical/systems.csv
    or ./systems.csv.
    """
    # Preferred: use existing canonical loader for systems (parquet or cleaned csv)
    try:
        return load_parquet_or_csv("systems")
    except FileNotFoundError:
        # fallback to inputs/canonical/systems.csv or ./systems.csv
        cand = CANONICAL / "systems.csv"
        if cand.exists():
            return pd.read_csv(cand)
        if Path("systems.csv").exists():
            return pd.read_csv(Path("systems.csv"))
        raise FileNotFoundError("Could not find systems data (tried canonical and local systems.csv)")

def write_simple_schema_yaml(df: pd.DataFrame, out_path: Path, source_name: str):
    """
    Write a minimal schema YAML describing column names and inferred simple types.
    Matches the pattern {source_name}_schema.yaml that parquets.py uses.
    This is intentionally small: column -> guessed type (string|number|integer|boolean).
    """
    def guess_type(series: pd.Series) -> str:
        # simple heuristics
        if series.dropna().empty:
            return "string"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "number"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        # try parseable numeric
        try:
            pd.to_numeric(series.dropna().iloc[:50])
            # if conversion ok and contains decimals, number, else integer
            s = series.dropna().iloc[:50].astype(float)
            if (s % 1 != 0).any():
                return "number"
            return "integer"
        except Exception:
            return "string"

    cols = []
    for col in df.columns:
        col_type = guess_type(df[col])
        cols.append({"name": col, "type": col_type, "units": None})

    schema = {
        "name": source_name,
        "description": f"Auto-generated schema for {source_name} on {datetime.now(timezone.utc).isoformat()}Z",
        "columns": cols,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(schema, fh, sort_keys=False)


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
            getattr(v, "material_concrete_id", None),
            getattr(v, "material_steel_id", None),
            getattr(v, "material_timber_id", None),
        ]:
            if mat_id and mat_id not in materials:
                errors.append(f"Variant {vid} references missing material {mat_id}")

    return errors


if __name__ == "__main__":
    # Simple CLI-ish behavior for local testing / CI
    try:
        mats, m_errs = load_materials()
        fams, f_errs = load_system_families()
        vars_, v_errs = load_system_variants()
    except Exception as e:
        print("ERROR running loaders:", e)
        raise

    rel_errs = validate_relationships(mats, fams, vars_)

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