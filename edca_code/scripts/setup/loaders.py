from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from .schemas import (
    BeamFamily,
    BeamVariant,
    ColumnFamily,
    ColumnVariant,
    FloorFamily,
    FloorVariant,
    LateralFamily,
    LateralVariant,
    Material,
    Occupancy,
)

CANONICAL = Path("inputs/canonical")
SOURCE_PRESETS = Path("inputs/source/presets")


@dataclass(frozen=True)
class ComponentConfig:
    component: str
    source_name: str
    raw_id: str
    family_id: str
    variant_id: str
    families_name: str
    variants_name: str
    family_model: Type[BaseModel]
    variant_model: Type[BaseModel]
    family_cols: List[str]
    variant_cols: List[str]
    material_fields: List[str]
    rename_map: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class CatalogData:
    materials: Dict[str, Material]
    material_errors: List[str]
    floor_families: Dict[str, FloorFamily]
    floor_family_errors: List[str]
    floor_variants: Dict[str, FloorVariant]
    floor_variant_errors: List[str]
    beam_families: Dict[str, BeamFamily]
    beam_family_errors: List[str]
    beam_variants: Dict[str, BeamVariant]
    beam_variant_errors: List[str]
    column_families: Dict[str, ColumnFamily]
    column_family_errors: List[str]
    column_variants: Dict[str, ColumnVariant]
    column_variant_errors: List[str]
    lateral_families: Dict[str, LateralFamily]
    lateral_family_errors: List[str]
    lateral_variants: Dict[str, LateralVariant]
    lateral_variant_errors: List[str]


SYSTEM_CONFIG: Dict[str, ComponentConfig] = {
    "floor": ComponentConfig(
        component="floor",
        source_name="floor_systems",
        raw_id="floor_system_id",
        family_id="floor_family_id",
        variant_id="floor_variant_id",
        families_name="floor_families",
        variants_name="floor_variants",
        family_model=FloorFamily,
        variant_model=FloorVariant,
        family_cols=[
            "floor_system_id",
            "component",
            "floor_category",
            "floor_type",
            "span_behavior",
            "manufacturer",
            "material_family",
            "construction_method",
            "support_condition",
            "unit",
            "material_concrete_id",
            "material_screed_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "vibration_limit",
            "fire_rating",
            "acoustic_note",
            "requires_rebar",
            "requires_pt",
            "requires_screed",
            "requires_topping",
            "requires_fireproofing",
            "beam_requirements",
            "can_span_to_columns",
            "can_span_to_walls",
            "source",
            "notes",
        ],
        variant_cols=[
            "floor_system_id",
            "slab_length",
            "slab_width",
            "slab_depth",
            "rib_depth",
            "screed_depth",
            "deck_depth",
            "overall_depth",
            "swt",
            "sdl",
            "ll",
            "max_span",
            "secondary_beam_spacing_m",
            "concrete_volume",
            "screed_volume",
            "steel_volume",
            "rebar_volume",
            "pt_volume",
            "timber_volume",
            "material_concrete_id",
            "material_screed_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "ebc_mm",
            "beam_ref",
            "source",
            "notes",
        ],
        material_fields=[
            "material_concrete_id",
            "material_screed_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
        ],
    ),
    "beam": ComponentConfig(
        component="beam",
        source_name="beam_systems",
        raw_id="beam_system_id",
        family_id="beam_family_id",
        variant_id="beam_variant_id",
        families_name="beam_families",
        variants_name="beam_variants",
        family_model=BeamFamily,
        variant_model=BeamVariant,
        family_cols=[
            "beam_system_id",
            "component",
            "beam_category",
            "beam_type",
            "span_behavior",
            "manufacturer",
            "material_family",
            "construction_method",
            "section_type",
            "beam_role",
            "unit",
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "is_composite",
            "requires_rebar",
            "requires_pt",
            "requires_encasement",
            "requires_fireproofing",
            "supports_floor_types",
            "supports_column_types",
            "can_be_primary",
            "can_be_secondary",
            "can_be_edge",
            "directly_supports_floor",
            "source",
            "notes",
        ],
        variant_cols=[
            "beam_system_id",
            "beam_length",
            "beam_width",
            "beam_camber",
            "beam_depth",
            "flange_width",
            "web_thickness",
            "fireproofing_depth",
            "load_capacity",
            "moment_capacity",
            "shear_capacity",
            "deflection_rule",
            "max_span",
            "fire_rating",
            "concrete_volume",
            "steel_volume",
            "rebar_volume",
            "pt_volume",
            "timber_volume",
            "fireproofing_volume",
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "source",
            "notes",
        ],
        material_fields=[
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
        ],
    ),
    "column": ComponentConfig(
        component="column",
        source_name="column_systems",
        raw_id="column_system_id",
        family_id="column_family_id",
        variant_id="column_variant_id",
        families_name="column_families",
        variants_name="column_variants",
        family_model=ColumnFamily,
        variant_model=ColumnVariant,
        family_cols=[
            "column_system_id",
            "component",
            "column_category",
            "column_type",
            "manufacturer",
            "material_family",
            "construction_method",
            "section_type",
            "column_role",
            "unit",
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "is_composite",
            "requires_rebar",
            "requires_pt",
            "requires_encasement",
            "requires_fireproofing",
            "supports_floor_types",
            "supports_beam_types",
            "supports_low_rise",
            "supports_mid_rise",
            "supports_high_rise",
            "source",
            "notes",
        ],
        variant_cols=[
            "column_system_id",
            "column_height",
            "column_width",
            "column_depth",
            "axial_capacity",
            "moment_capacity",
            "maximum_story_count",
            "slenderness_limit",
            "fire_rating",
            "concrete_volume",
            "steel_volume",
            "rebar_volume",
            "pt_volume",
            "timber_volume",
            "fireproofing_volume",
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "source",
            "notes",
        ],
        material_fields=[
            "material_concrete_id",
            "material_pt_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
        ],
    ),
    "lateral": ComponentConfig(
        component="lateral",
        source_name="lateral_systems",
        raw_id="lateral_system_id",
        family_id="lateral_family_id",
        variant_id="lateral_variant_id",
        families_name="lateral_families",
        variants_name="lateral_variants",
        family_model=LateralFamily,
        variant_model=LateralVariant,
        family_cols=[
            "lateral_system_id",
            "component",
            "lateral_category",
            "lateral_type",
            "manufacturer",
            "material_family",
            "construction_method",
            "lateral_mechanism",
            "unit",
            "material_concrete_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "supports_low_rise",
            "supports_mid_rise",
            "supports_high_rise",
            "drift_efficiency",
            "fire_rating",
            "source",
            "notes",
        ],
        variant_cols=[
            "lateral_system_id",
            "wall_thickness",
            "frame_depth",
            "bay_width_default",
            "core_area_ratio_default",
            "core_perimeter_ratio_default",
            "axial_capacity",
            "moment_capacity",
            "maximum_story_count",
            "slenderness_limit",
            "lateral_type_detail",
            "fire_rating_variant",
            "concrete_volume",
            "steel_volume",
            "rebar_volume",
            "timber_volume",
            "material_concrete_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
            "source",
            "notes",
        ],
        material_fields=[
            "material_concrete_id",
            "material_rebar_id",
            "material_steel_id",
            "material_timber_id",
            "material_fireproofing_id",
        ],
        rename_map={
            "lateral_type.1": "lateral_type_detail",
            "fire_rating.1": "fire_rating_variant",
        },
    ),
}


def _default_source_candidates(name: str) -> List[Path]:
    return [
        CANONICAL / f"{name}.parquet",
        CANONICAL / f"{name}.csv",
        SOURCE_PRESETS / name / f"{name}.csv",
        SOURCE_PRESETS / "systems" / f"{name}.csv",
        SOURCE_PRESETS / "materials" / f"{name}.csv",
        SOURCE_PRESETS / "loads" / f"{name}.csv",
        SOURCE_PRESETS / "occupancies" / f"{name}.csv",
        Path(f"{name}.parquet"),
        Path(f"{name}.csv"),
    ]


def load_parquet_or_csv(name: str) -> pd.DataFrame:
    for candidate in _default_source_candidates(name):
        if candidate.exists():
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"No parquet or CSV found for {name}")


def normalize_df(df: pd.DataFrame, rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~out.columns.str.match(r"^Unnamed:")]
    if rename_map:
        out = out.rename(columns=rename_map)
    for col in out.columns:
        if col.startswith("material_") and col.endswith("_id"):
            out[col] = out[col].map(_normalize_material_ref)
    out = out.astype(object)
    out = out.where(pd.notnull(out), None)
    out = out.replace({np.nan: None})
    return out


def _repair_shifted_component_units(df: pd.DataFrame, component: str) -> pd.DataFrame:
    """
    Some section-catalog rows omit the empty role field before unit, so values
    from unit onward shift one column left. Repair rows where role contains a
    unit token and unit is empty, then let normal family/variant splitting run.
    """
    role_col = f"{component}_role"
    if role_col not in df.columns or "unit" not in df.columns:
        return df

    unit_tokens = {"m", "metric", "si", "ft", "imperial"}
    role = df[role_col].astype(str).str.strip().str.lower()
    unit = df["unit"].astype(str).str.strip().str.lower()
    mask = role.isin(unit_tokens) & unit.isin({"", "none", "nan"})
    if not mask.any():
        return df

    out = df.copy()
    cols = list(out.columns)
    role_idx = cols.index(role_col)
    shift_cols = cols[role_idx:-1]

    out.loc[mask, shift_cols[1:]] = out.loc[mask, shift_cols[:-1]].to_numpy()
    out.loc[mask, role_col] = None
    return out


def _default_material_refs_from_quantities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for material_col, quantity_col, default_material in [
        ("material_steel_id", "steel_volume", "steel"),
        ("material_rebar_id", "rebar_volume", "rebar"),
        ("material_pt_id", "pt_volume", "pt"),
        ("material_concrete_id", "concrete_volume", "concrete"),
        ("material_screed_id", "screed_volume", "screed"),
        ("material_timber_id", "timber_volume", "timber"),
    ]:
        if material_col not in out.columns or quantity_col not in out.columns:
            continue
        missing = out[material_col].isna() | out[material_col].astype(str).str.strip().str.lower().isin({"", "none", "nan"})
        positive_quantity = pd.to_numeric(out[quantity_col], errors="coerce").fillna(0.0).gt(0.0)
        out.loc[missing & positive_quantity, material_col] = default_material
    return out


def _normalize_material_ref(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "false", "0"}:
        return None

    aliases = {
        "CONC_30MPA": "concrete",
        "CONC_40MPA": "concrete",
        "CONC_4000PSI": "concrete_4000psi",
        "CONC_5000PSI": "concrete_5000psi",
        "SCREED_GENERIC": "screed",
    }
    return aliases.get(text.upper(), text)


def write_simple_schema_yaml(df: pd.DataFrame, out_path: Path, source_name: str) -> None:
    def guess_type(series: pd.Series) -> str:
        if series.dropna().empty:
            return "string"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "number"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        try:
            sample = pd.to_numeric(series.dropna().iloc[:50])
            return "number" if (sample.astype(float) % 1 != 0).any() else "integer"
        except Exception:
            return "string"

    schema = {
        "name": source_name,
        "description": f"Auto-generated schema for {source_name} on {datetime.now(timezone.utc).isoformat()}",
        "columns": [
            {"name": col, "type": guess_type(df[col]), "units": None}
            for col in df.columns
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(schema, fh, sort_keys=False)


def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        non_null = s.notna()
        if non_null.any():
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric[non_null].notna().all():
                out[col] = numeric
                continue

        if s.dtype == object:
            out[col] = s.map(
                lambda v: (
                    None
                    if v is None or (isinstance(v, float) and np.isnan(v))
                    else str(v)
                )
            )
    return out


def df_to_objects(df: pd.DataFrame, model_cls: Type[BaseModel], key_attr: str) -> Tuple[Dict[str, BaseModel], List[str]]:
    objects: Dict[str, BaseModel] = {}
    errors: List[str] = []

    df = normalize_df(df)
    for idx, row in df.iterrows():
        try:
            instance = model_cls(**row.to_dict())
            key = getattr(instance, key_attr)
            objects[key] = instance
        except Exception as exc:
            errors.append(f"{model_cls.__name__} row {idx}: {exc}")

    return objects, errors


def _load_component_source_df(component: str, systems_source: Union[str, Path, pd.DataFrame, None] = None) -> pd.DataFrame:
    cfg = SYSTEM_CONFIG[component]

    if isinstance(systems_source, pd.DataFrame):
        return _repair_shifted_component_units(normalize_df(systems_source, rename_map=cfg.rename_map), component)

    if systems_source is not None:
        path = Path(systems_source)
        if not path.exists():
            raise FileNotFoundError(f"Provided systems_source not found: {systems_source}")
        if path.suffix == ".parquet":
            return _repair_shifted_component_units(normalize_df(pd.read_parquet(path), rename_map=cfg.rename_map), component)
        return _repair_shifted_component_units(normalize_df(pd.read_csv(path), rename_map=cfg.rename_map), component)

    source_candidates = [
        SOURCE_PRESETS / "systems" / f"{cfg.source_name}.csv",
        SOURCE_PRESETS / cfg.source_name / f"{cfg.source_name}.csv",
        Path(f"{cfg.source_name}.csv"),
        CANONICAL / f"{cfg.source_name}.parquet",
        CANONICAL / f"{cfg.source_name}.csv",
    ]
    for candidate in source_candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            return _repair_shifted_component_units(normalize_df(pd.read_parquet(candidate), rename_map=cfg.rename_map), component)
        return _repair_shifted_component_units(normalize_df(pd.read_csv(candidate, low_memory=False), rename_map=cfg.rename_map), component)

    raise FileNotFoundError(f"Could not find source data for {cfg.source_name}")


def regenerate_component_tables(
    component: str,
    systems_source: Union[str, Path, pd.DataFrame, None] = None,
) -> Tuple[Path, Path]:
    cfg = SYSTEM_CONFIG[component]
    df = _load_component_source_df(component, systems_source=systems_source)
    df = _default_material_refs_from_quantities(df)

    # Fill blank material_family from the component's category field so that
    # entries like KLH CLT (floor_category=timber, material_family="") are
    # correctly grouped in downstream assembly and chart logic.
    _cat_col = {
        "floor":   "floor_category",
        "beam":    "beam_category",
        "column":  "column_category",
        "lateral": "lateral_category",
    }.get(component)
    if _cat_col and _cat_col in df.columns and "material_family" in df.columns:
        blank_mat = df["material_family"].isna() | (df["material_family"].astype(str).str.strip() == "")
        df.loc[blank_mat, "material_family"] = df.loc[blank_mat, _cat_col]

    if cfg.raw_id not in df.columns:
        raise KeyError(f"{cfg.source_name} must include '{cfg.raw_id}'")

    CANONICAL.mkdir(parents=True, exist_ok=True)

    family_cols = [col for col in cfg.family_cols if col in df.columns]
    variant_cols = [col for col in cfg.variant_cols if col in df.columns]

    families = (
        df[family_cols]
        .drop_duplicates(subset=[cfg.raw_id])
        .rename(columns={cfg.raw_id: cfg.family_id})
        .reset_index(drop=True)
    )

    variants = df[variant_cols].copy()
    variants[cfg.raw_id] = variants[cfg.raw_id].astype(str)
    variants["_variant_index"] = variants.groupby(cfg.raw_id).cumcount() + 1
    variants[cfg.variant_id] = variants[cfg.raw_id] + "_" + variants["_variant_index"].astype(str)
    variants = variants.drop(columns=["_variant_index"]).rename(columns={cfg.raw_id: cfg.family_id})
    remaining = [c for c in variants.columns if c not in (cfg.variant_id, cfg.family_id)]
    variants = variants[[cfg.variant_id, cfg.family_id] + remaining]

    families_csv = CANONICAL / f"{cfg.families_name}.csv"
    variants_csv = CANONICAL / f"{cfg.variants_name}.csv"
    families_parquet = CANONICAL / f"{cfg.families_name}.parquet"
    variants_parquet = CANONICAL / f"{cfg.variants_name}.parquet"

    families.to_csv(families_csv, index=False)
    variants.to_csv(variants_csv, index=False)

    try:
        _prepare_for_parquet(families).to_parquet(families_parquet, index=False)
    except Exception as exc:
        print(f"Warning: failed to write {families_parquet}: {exc}")

    try:
        _prepare_for_parquet(variants).to_parquet(variants_parquet, index=False)
    except Exception as exc:
        print(f"Warning: failed to write {variants_parquet}: {exc}")

    write_simple_schema_yaml(families, CANONICAL / f"{cfg.families_name}_schema.yaml", cfg.families_name)
    write_simple_schema_yaml(variants, CANONICAL / f"{cfg.variants_name}_schema.yaml", cfg.variants_name)

    return families_csv, variants_csv


def regenerate_all_system_tables() -> Dict[str, Tuple[Path, Path]]:
    return {
        component: regenerate_component_tables(component)
        for component in SYSTEM_CONFIG
    }


def load_materials() -> Tuple[Dict[str, Material], List[str]]:
    df = load_parquet_or_csv("materials")
    return df_to_objects(df, Material, "material_id")


def load_occupancies() -> Tuple[Dict[str, Occupancy], List[str]]:
    df = load_parquet_or_csv("occupancies")
    return df_to_objects(df, Occupancy, "use")


def load_floor_families() -> Tuple[Dict[str, FloorFamily], List[str]]:
    regenerate_component_tables("floor")
    df = load_parquet_or_csv("floor_families")
    return df_to_objects(df, FloorFamily, "floor_family_id")


def load_floor_variants() -> Tuple[Dict[str, FloorVariant], List[str]]:
    regenerate_component_tables("floor")
    df = load_parquet_or_csv("floor_variants")
    return df_to_objects(df, FloorVariant, "floor_variant_id")


def load_beam_families() -> Tuple[Dict[str, BeamFamily], List[str]]:
    regenerate_component_tables("beam")
    df = load_parquet_or_csv("beam_families")
    return df_to_objects(df, BeamFamily, "beam_family_id")


def load_beam_variants() -> Tuple[Dict[str, BeamVariant], List[str]]:
    regenerate_component_tables("beam")
    df = load_parquet_or_csv("beam_variants")
    return df_to_objects(df, BeamVariant, "beam_variant_id")


def load_column_families() -> Tuple[Dict[str, ColumnFamily], List[str]]:
    regenerate_component_tables("column")
    df = load_parquet_or_csv("column_families")
    return df_to_objects(df, ColumnFamily, "column_family_id")


def load_column_variants() -> Tuple[Dict[str, ColumnVariant], List[str]]:
    regenerate_component_tables("column")
    df = load_parquet_or_csv("column_variants")
    return df_to_objects(df, ColumnVariant, "column_variant_id")


def load_lateral_families() -> Tuple[Dict[str, LateralFamily], List[str]]:
    regenerate_component_tables("lateral")
    df = load_parquet_or_csv("lateral_families")
    return df_to_objects(df, LateralFamily, "lateral_family_id")


def load_lateral_variants() -> Tuple[Dict[str, LateralVariant], List[str]]:
    regenerate_component_tables("lateral")
    df = load_parquet_or_csv("lateral_variants")
    return df_to_objects(df, LateralVariant, "lateral_variant_id")


def load_catalog_data(*, regenerate_systems: bool = True) -> CatalogData:
    """
    Load the setup catalogue in dependency order.

    Materials are loaded first because component families/variants reference
    material IDs. When requested, system-derived family/variant tables are then
    regenerated once before object loading begins.
    """
    materials, material_errors = load_materials()

    if regenerate_systems:
        regenerate_all_system_tables()

    floor_families_df = load_parquet_or_csv("floor_families")
    floor_variants_df = load_parquet_or_csv("floor_variants")
    beam_families_df = load_parquet_or_csv("beam_families")
    beam_variants_df = load_parquet_or_csv("beam_variants")
    column_families_df = load_parquet_or_csv("column_families")
    column_variants_df = load_parquet_or_csv("column_variants")
    lateral_families_df = load_parquet_or_csv("lateral_families")
    lateral_variants_df = load_parquet_or_csv("lateral_variants")

    floor_families, floor_family_errors = df_to_objects(floor_families_df, FloorFamily, "floor_family_id")
    floor_variants, floor_variant_errors = df_to_objects(floor_variants_df, FloorVariant, "floor_variant_id")
    beam_families, beam_family_errors = df_to_objects(beam_families_df, BeamFamily, "beam_family_id")
    beam_variants, beam_variant_errors = df_to_objects(beam_variants_df, BeamVariant, "beam_variant_id")
    column_families, column_family_errors = df_to_objects(column_families_df, ColumnFamily, "column_family_id")
    column_variants, column_variant_errors = df_to_objects(column_variants_df, ColumnVariant, "column_variant_id")
    lateral_families, lateral_family_errors = df_to_objects(lateral_families_df, LateralFamily, "lateral_family_id")
    lateral_variants, lateral_variant_errors = df_to_objects(lateral_variants_df, LateralVariant, "lateral_variant_id")

    return CatalogData(
        materials=materials,
        material_errors=material_errors,
        floor_families=floor_families,
        floor_family_errors=floor_family_errors,
        floor_variants=floor_variants,
        floor_variant_errors=floor_variant_errors,
        beam_families=beam_families,
        beam_family_errors=beam_family_errors,
        beam_variants=beam_variants,
        beam_variant_errors=beam_variant_errors,
        column_families=column_families,
        column_family_errors=column_family_errors,
        column_variants=column_variants,
        column_variant_errors=column_variant_errors,
        lateral_families=lateral_families,
        lateral_family_errors=lateral_family_errors,
        lateral_variants=lateral_variants,
        lateral_variant_errors=lateral_variant_errors,
    )


def _validate_component_relationships(
    materials: Dict[str, Material],
    families: Dict[str, BaseModel],
    variants: Dict[str, BaseModel],
    component: str,
) -> List[str]:
    cfg = SYSTEM_CONFIG[component]
    errors: List[str] = []

    for variant_id, variant in variants.items():
        family_ref = getattr(variant, cfg.family_id, None)
        if family_ref not in families:
            errors.append(f"{component} variant {variant_id} references missing family {family_ref}")

        for material_attr in cfg.material_fields:
            material_id = getattr(variant, material_attr, None)
            if material_id and material_id not in materials:
                errors.append(
                    f"{component} variant {variant_id} references missing material {material_id} via {material_attr}"
                )

    return errors


def validate_all_relationships(
    materials: Dict[str, Material],
    floor_families: Dict[str, FloorFamily],
    floor_variants: Dict[str, FloorVariant],
    beam_families: Dict[str, BeamFamily],
    beam_variants: Dict[str, BeamVariant],
    column_families: Dict[str, ColumnFamily],
    column_variants: Dict[str, ColumnVariant],
    lateral_families: Dict[str, LateralFamily],
    lateral_variants: Dict[str, LateralVariant],
) -> List[str]:
    errors: List[str] = []
    errors.extend(_validate_component_relationships(materials, floor_families, floor_variants, "floor"))
    errors.extend(_validate_component_relationships(materials, beam_families, beam_variants, "beam"))
    errors.extend(_validate_component_relationships(materials, column_families, column_variants, "column"))
    errors.extend(_validate_component_relationships(materials, lateral_families, lateral_variants, "lateral"))
    return errors


if __name__ == "__main__":
    data = load_catalog_data(regenerate_systems=True)
    relationship_errors = validate_all_relationships(
        data.materials,
        data.floor_families,
        data.floor_variants,
        data.beam_families,
        data.beam_variants,
        data.column_families,
        data.column_variants,
        data.lateral_families,
        data.lateral_variants,
    )
    object_errors = (
        data.material_errors
        + data.floor_family_errors
        + data.floor_variant_errors
        + data.beam_family_errors
        + data.beam_variant_errors
        + data.column_family_errors
        + data.column_variant_errors
        + data.lateral_family_errors
        + data.lateral_variant_errors
    )
    if object_errors or relationship_errors:
        print(f"❌ Catalogue validation failed: object_errors={len(object_errors)}, relationship_errors={len(relationship_errors)}")
        raise SystemExit(1)
    print("✅ Loaded materials first, regenerated systems, and validated setup catalogues")
