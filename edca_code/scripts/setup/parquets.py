"""
scripts/parquets.py
This script converts canonical CSVs to parquet file types, writes basic _schema.yaml files, and normalizes numeric columns heuristically.
Allows faster loading and better type handling in downstream code, while providing a starting point for schema definition.

Run from project root:
    python scripts/parquets.py
"""
import pandas as pd
import re
from pathlib import Path
import yaml

PROJECT_ROOT = Path(".")
SOURCE_DIR = PROJECT_ROOT / "edca_tool" / "inputs" / "source"
CANONICAL_DIR = PROJECT_ROOT / "edca_tool" / "inputs" / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

csv_files = {
    "materials": SOURCE_DIR / "presets" / "materials" / "materials.csv",
    "system_families": SOURCE_DIR / "presets" / "systems" / "system_families.csv",
    "system_variants": SOURCE_DIR / "presets" / "systems" / "system_variants.csv",
    "systems": SOURCE_DIR / "presets" / "systems" / "systems.csv",
}

def infer_col_type(series):
    try:
        pd.to_numeric(series.dropna().head(50))
        return "number"
    except Exception:
        return "string"

def basic_schema_from_df(df):
    schema = {}
    for c in df.columns:
        t = infer_col_type(df[c])
        schema[c] = {"type": t, "units": ""}
    return schema

for name, path in csv_files.items():
    if not path.exists():
        print(f"[WARN] {path} not found — skipping {name}")
        continue
    print(f"[INFO] Loading {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] failed to read with default encoding, trying latin1")
        df = pd.read_csv(path, encoding="latin1")

    # Heuristic numeric coercion for common engineering fields
    numeric_hint = re.compile(r"(depth|volume|weight|kg|mm|m3|m2|span|ll|swt|sdl|max_span|fy|fu|cost|price|ec_|embodied|co2|carbon)", re.I)
    for col in df.columns:
        if numeric_hint.search(col):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_parquet = CANONICAL_DIR / f"{name}.parquet"
    df.to_parquet(out_parquet, index=False, compression="snappy")
    print(f"[INFO] Wrote {out_parquet} ({len(df)} rows)")

    # write a tiny schema YAML you can edit
    schema = {
        "name": name,
        "columns": basic_schema_from_df(df),
        "notes": "Auto-generated basic schema. Please add units & refine types."
    }
    out_schema = CANONICAL_DIR / f"{name}_schema.yaml"
    with open(out_schema, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    print(f"[INFO] Wrote schema {out_schema}")

print("[DONE] Conversion complete. Review schema files and add units where needed.")
