from __future__ import annotations

from typing import Optional, Dict, Any
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------
# Shared helpers
# -----------------------

def _empty_to_none(value: Any) -> Any:
    if value == "":
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isnan(float(value)):
                return None
        except Exception:
            pass
        if float(value).is_integer() and int(value) in {0, 1}:
            return str(int(value))
    return value


def _coerce_float(value: Any) -> Optional[float]:
    value = _empty_to_none(value)
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) else out


class EDCABaseModel(BaseModel):
    """Base model with forgiving parsing for CSV-backed engineering data."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _strip_empty_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: _empty_to_none(v) for k, v in data.items()}
        return data


# -----------------------
# Occupancy
# -----------------------

class Occupancy(EDCABaseModel):
    use: str
    unit: Optional[str] = None
    sdl: Optional[float] = None
    sdl_partition: Optional[float] = None
    ll: Optional[float] = None
    code: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("sdl", "sdl_partition", "ll", mode="before")
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)


# -----------------------
# Material
# -----------------------

class Material(EDCABaseModel):
    material_id: str
    family: Optional[str] = None
    standard_grade: Optional[str] = None
    concrete_f_ck: Optional[float] = None
    steel_fy: Optional[float] = None
    steel_fu: Optional[float] = None
    timber_fm: Optional[float] = None
    timber_e: Optional[float] = None
    density: Optional[float] = None
    unit: Optional[str] = None
    ec_a1a3_volumetric: Optional[float] = None
    ec_a1a3_mass: Optional[float] = None
    ec_a4_volumetric: Optional[float] = None
    ec_a4_mass: Optional[float] = None
    transport_distance: Optional[float] = None
    ec_a5_volumetric: Optional[float] = None
    ec_a5_mass: Optional[float] = None
    cost_volumetric: Optional[float] = None
    cost_mass: Optional[float] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "concrete_f_ck",
        "steel_fy",
        "steel_fu",
        "timber_fm",
        "timber_e",
        "density",
        "ec_a1a3_volumetric",
        "ec_a1a3_mass",
        "ec_a4_volumetric",
        "ec_a4_mass",
        "transport_distance",
        "ec_a5_volumetric",
        "ec_a5_mass",
        "cost_volumetric",
        "cost_mass",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)

    @model_validator(mode="after")
    def _fill_derived_volumetric_values(self):
        if self.ec_a1a3_volumetric is None and self.ec_a1a3_mass is not None and self.density is not None:
            self.ec_a1a3_volumetric = self.ec_a1a3_mass * self.density
        if self.ec_a4_volumetric is None and self.ec_a4_mass is not None and self.density is not None:
            self.ec_a4_volumetric = self.ec_a4_mass * self.density
        if self.ec_a5_volumetric is None and self.ec_a5_mass is not None and self.density is not None:
            self.ec_a5_volumetric = self.ec_a5_mass * self.density
        if self.cost_volumetric is None and self.cost_mass is not None and self.density is not None:
            self.cost_volumetric = self.cost_mass * self.density
        return self

    @model_validator(mode="after")
    def _check_non_negative(self):
        for field_name in (
            "concrete_f_ck",
            "steel_fy",
            "steel_fu",
            "timber_fm",
            "timber_e",
            "density",
            "transport_distance",
            "ec_a4_volumetric",
            "ec_a4_mass",
            "ec_a5_volumetric",
            "ec_a5_mass",
            "cost_volumetric",
            "cost_mass",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        return self


# -----------------------
# Optional source / EPD metadata
# -----------------------

class Source(EDCABaseModel):
    type: Optional[str] = None
    source: str
    specification_id: Optional[str] = None
    citation: Optional[str] = None
    url: Optional[str] = None
    published: Optional[str] = None
    region: Optional[str] = None


# -----------------------
# Floor family / variant
# -----------------------

class FloorFamily(EDCABaseModel):
    floor_family_id: str
    component: Optional[str] = None
    floor_category: Optional[str] = None
    floor_type: Optional[str] = None
    span_behavior: Optional[str] = None
    material_family: Optional[str] = None
    construction_method: Optional[str] = None
    support_condition: Optional[str] = None
    manufacturer: Optional[str] = None
    unit: Optional[str] = None
    material_concrete_id: Optional[str] = None
    material_screed_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    vibration_limit: Optional[str] = None
    fire_rating: Optional[str] = None
    acoustic_note: Optional[str] = None
    requires_rebar: Optional[str] = None
    requires_pt: Optional[str] = None
    requires_screed: Optional[str] = None
    requires_topping: Optional[str] = None
    requires_fireproofing: Optional[str] = None
    beam_requirements: Optional[str] = None
    can_span_to_columns: Optional[str] = None
    can_span_to_walls: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class FloorVariant(EDCABaseModel):
    floor_variant_id: str
    floor_family_id: str
    slab_length: Optional[float] = None
    slab_width: Optional[float] = None
    slab_depth: Optional[float] = None
    rib_depth: Optional[float] = None
    screed_depth: Optional[float] = None
    deck_depth: Optional[float] = None
    overall_depth: Optional[float] = None
    swt: Optional[float] = None
    sdl: Optional[float] = None
    ll: Optional[float] = None
    max_span: Optional[float] = None
    concrete_volume: Optional[float] = None
    screed_volume: Optional[float] = None
    steel_volume: Optional[float] = None
    rebar_volume: Optional[float] = None
    pt_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_screed_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    ebc_mm: Optional[float] = None
    beam_ref: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
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
        "concrete_volume",
        "screed_volume",
        "steel_volume",
        "rebar_volume",
        "pt_volume",
        "timber_volume",
        "ebc_mm",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)

    @field_validator("max_span")
    @classmethod
    def _validate_max_span(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("max_span must be > 0")
        return value

    def concrete_mass_per_m2(self, material_lookup: Dict[str, Material]) -> Optional[float]:
        if self.concrete_volume is None or not self.material_concrete_id:
            return None
        material = material_lookup.get(self.material_concrete_id)
        if material is None or material.density is None:
            return None
        return float(self.concrete_volume) * float(material.density)

    def embodied_carbon_a1a3_per_m2(self, material_lookup: Dict[str, Material]) -> Optional[float]:
        total = 0.0
        any_known = False

        for volume_attr, material_attr in (
            ("concrete_volume", "material_concrete_id"),
            ("steel_volume", "material_steel_id"),
            ("rebar_volume", "material_rebar_id"),
            ("screed_volume", "material_screed_id"),
            ("pt_volume", "material_pt_id"),
            ("timber_volume", "material_timber_id"),
        ):
            volume = getattr(self, volume_attr, None)
            material_id = getattr(self, material_attr, None)
            if volume is None or material_id is None:
                continue
            material = material_lookup.get(material_id)
            if material is None or material.density is None or material.ec_a1a3_mass is None:
                continue
            total += float(volume) * float(material.density) * float(material.ec_a1a3_mass)
            any_known = True

        return total if any_known else None


# -----------------------
# Beam family / variant
# -----------------------

class BeamFamily(EDCABaseModel):
    beam_family_id: str
    component: Optional[str] = None
    beam_category: Optional[str] = None
    beam_type: Optional[str] = None
    span_behavior: Optional[str] = None
    manufacturer: Optional[str] = None
    material_family: Optional[str] = None
    construction_method: Optional[str] = None
    section_type: Optional[str] = None
    beam_role: Optional[str] = None
    unit: Optional[str] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    is_composite: Optional[str] = None
    requires_rebar: Optional[str] = None
    requires_pt: Optional[str] = None
    requires_encasement: Optional[str] = None
    requires_fireproofing: Optional[str] = None
    supports_floor_types: Optional[str] = None
    supports_column_types: Optional[str] = None
    can_be_primary: Optional[str] = None
    can_be_secondary: Optional[str] = None
    can_be_edge: Optional[str] = None
    directly_supports_floor: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class BeamVariant(EDCABaseModel):
    beam_variant_id: str
    beam_family_id: str
    beam_length: Optional[float] = None
    beam_width: Optional[float] = None
    beam_camber: Optional[float] = None
    beam_depth: Optional[float] = None
    flange_width: Optional[float] = None
    web_thickness: Optional[float] = None
    fireproofing_depth: Optional[float] = None
    load_capacity: Optional[float] = None
    moment_capacity: Optional[float] = None
    shear_capacity: Optional[float] = None
    deflection_rule: Optional[str] = None
    max_span: Optional[float] = None
    fire_rating: Optional[str] = None
    concrete_volume: Optional[float] = None
    steel_volume: Optional[float] = None
    rebar_volume: Optional[float] = None
    pt_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    fireproofing_volume: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
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
        "max_span",
        "concrete_volume",
        "steel_volume",
        "rebar_volume",
        "pt_volume",
        "timber_volume",
        "fireproofing_volume",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)


# -----------------------
# Column family / variant
# -----------------------

class ColumnFamily(EDCABaseModel):
    column_family_id: str
    component: Optional[str] = None
    column_category: Optional[str] = None
    column_type: Optional[str] = None
    manufacturer: Optional[str] = None
    material_family: Optional[str] = None
    construction_method: Optional[str] = None
    section_type: Optional[str] = None
    column_role: Optional[str] = None
    unit: Optional[str] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    is_composite: Optional[str] = None
    requires_rebar: Optional[str] = None
    requires_pt: Optional[str] = None
    requires_encasement: Optional[str] = None
    requires_fireproofing: Optional[str] = None
    supports_floor_types: Optional[str] = None
    supports_beam_types: Optional[str] = None
    supports_low_rise: Optional[str] = None
    supports_mid_rise: Optional[str] = None
    supports_high_rise: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class ColumnVariant(EDCABaseModel):
    column_variant_id: str
    column_family_id: str
    column_height: Optional[float] = None
    column_width: Optional[float] = None
    column_depth: Optional[float] = None
    axial_capacity: Optional[float] = None
    moment_capacity: Optional[float] = None
    maximum_story_count: Optional[float] = None
    slenderness_limit: Optional[float] = None
    fire_rating: Optional[str] = None
    concrete_volume: Optional[float] = None
    steel_volume: Optional[float] = None
    rebar_volume: Optional[float] = None
    pt_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    fireproofing_volume: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "column_height",
        "column_width",
        "column_depth",
        "axial_capacity",
        "moment_capacity",
        "maximum_story_count",
        "slenderness_limit",
        "concrete_volume",
        "steel_volume",
        "rebar_volume",
        "pt_volume",
        "timber_volume",
        "fireproofing_volume",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)


# -----------------------
# Lateral family / variant
# -----------------------

class LateralFamily(EDCABaseModel):
    lateral_family_id: str
    component: Optional[str] = None
    lateral_category: Optional[str] = None
    lateral_type: Optional[str] = None
    manufacturer: Optional[str] = None
    material_family: Optional[str] = None
    construction_method: Optional[str] = None
    lateral_mechanism: Optional[str] = None
    unit: Optional[str] = None
    material_concrete_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    supports_low_rise: Optional[str] = None
    supports_mid_rise: Optional[str] = None
    supports_high_rise: Optional[str] = None
    drift_efficiency: Optional[str] = None
    fire_rating: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class LateralVariant(EDCABaseModel):
    lateral_variant_id: str
    lateral_family_id: str
    wall_thickness: Optional[float] = None
    frame_depth: Optional[float] = None
    bay_width_default: Optional[float] = None
    core_area_ratio_default: Optional[float] = None
    core_perimeter_ratio_default: Optional[float] = None
    axial_capacity: Optional[float] = None
    moment_capacity: Optional[float] = None
    maximum_story_count: Optional[float] = None
    slenderness_limit: Optional[float] = None
    lateral_type_detail: Optional[str] = None
    fire_rating_variant: Optional[str] = None
    concrete_volume: Optional[float] = None
    steel_volume: Optional[float] = None
    rebar_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    material_fireproofing_id: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "wall_thickness",
        "frame_depth",
        "bay_width_default",
        "core_area_ratio_default",
        "core_perimeter_ratio_default",
        "axial_capacity",
        "moment_capacity",
        "maximum_story_count",
        "slenderness_limit",
        "concrete_volume",
        "steel_volume",
        "rebar_volume",
        "timber_volume",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        return _coerce_float(value)
