"""
Convert canonical and raw EDCA CSVs to parquet and write lightweight schema YAMLs.

This script also regenerates the derived family / variant tables for:
- floors
- beams
- columns
- lateral systems

Run from project root:
    python scripts/parquets.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from edca_code.scripts.setup.loaders import (
    CANONICAL,
    SYSTEM_CONFIG,
    _load_component_source_df,
    normalize_df,
    regenerate_all_system_tables,
    write_simple_schema_yaml,
)

PROJECT_ROOT = Path(".")
SOURCE_DIR = PROJECT_ROOT / "inputs" / "source"
CANONICAL_DIR = PROJECT_ROOT / "inputs" / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)


STATIC_SOURCES: Dict[str, List[Path]] = {
    "materials": [
        SOURCE_DIR / "presets" / "materials" / "materials.csv",
        PROJECT_ROOT / "materials.csv",
        CANONICAL_DIR / "materials.csv",
    ],
    "occupancies": [
        SOURCE_DIR / "presets" / "loads" / "occupancies.csv",
        SOURCE_DIR / "presets" / "occupancies" / "occupancies.csv",
        PROJECT_ROOT / "occupancies.csv",
        CANONICAL_DIR / "occupancies.csv",
    ],
}


DERIVED_CANONICAL = [
    "floor_families",
    "floor_variants",
    "beam_families",
    "beam_variants",
    "column_families",
    "column_variants",
    "lateral_families",
    "lateral_variants",
]


def first_existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def write_dataset(name: str, df: pd.DataFrame) -> None:
    df = normalize_df(df)
    csv_out = CANONICAL_DIR / f"{name}.csv"
    parquet_out = CANONICAL_DIR / f"{name}.parquet"
    schema_out = CANONICAL_DIR / f"{name}_schema.yaml"

    df.to_csv(csv_out, index=False)
    try:
        df.to_parquet(parquet_out, index=False, compression="snappy")
        parquet_status = f" and {parquet_out}"
    except Exception as exc:
        parquet_status = ""
        print(f"[WARN] Failed to write {parquet_out}: {exc}")
    write_simple_schema_yaml(df, schema_out, name)
    print(f"[INFO] Wrote {csv_out}{parquet_status}")


def convert_static_sources() -> None:
    for name, candidates in STATIC_SOURCES.items():
        source_path = first_existing(candidates)
        if source_path is None:
            print(f"[WARN] No source found for {name} — skipping")
            continue
        df = pd.read_csv(source_path)
        write_dataset(name, df)


def convert_component_sources() -> None:
    for component, cfg in SYSTEM_CONFIG.items():
        try:
            df = _load_component_source_df(component)
        except FileNotFoundError:
            print(f"[WARN] No source found for {cfg.source_name} — skipping")
            continue
        write_dataset(cfg.source_name, df)


def convert_derived_outputs() -> None:
    regenerate_all_system_tables()
    for name in DERIVED_CANONICAL:
        csv_path = CANONICAL_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"[WARN] Expected derived file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        write_dataset(name, df)


def convert_legacy_system_aliases() -> None:
    floor_families_path = CANONICAL_DIR / "floor_families.csv"
    floor_variants_path = CANONICAL_DIR / "floor_variants.csv"
    if not floor_families_path.exists() or not floor_variants_path.exists():
        print("[WARN] Floor family/variant files missing; skipping legacy system aliases")
        return

    families = pd.read_csv(floor_families_path).rename(
        columns={"floor_family_id": "system_family"}
    )
    variants = pd.read_csv(floor_variants_path).rename(
        columns={
            "floor_variant_id": "system_variant",
            "floor_family_id": "system_family",
        }
    )

    if "floor_category" in families.columns and "category" not in families.columns:
        families["category"] = families["floor_category"]
    if "floor_type" in families.columns and "type" not in families.columns:
        families["type"] = families["floor_type"]

    family_cols = [
        col
        for col in families.columns
        if col == "system_family" or col not in variants.columns
    ]
    system_variants = variants.merge(
        families[family_cols],
        on="system_family",
        how="left",
    )

    write_dataset("system_families", families)
    write_dataset("system_variants", system_variants)


if __name__ == "__main__":
    convert_static_sources()
    convert_component_sources()
    convert_derived_outputs()
    convert_legacy_system_aliases()
    print("[DONE] Conversion complete. Review schema YAMLs and add units where needed.")
