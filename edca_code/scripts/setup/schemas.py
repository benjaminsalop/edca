from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl, root_validator
import math

# -----------------------
# Occupancy
# -----------------------

class Occupancy(BaseModel):
    use: str
    unit: Optional[str] = None
    sdl: Optional[float] = None
    sdl_partition: Optional[float] = None
    ll: Optional[float] = None
    code: Optional[str] = None
    notes: Optional[str] = None

    @validator("sdl", "sdl_partition", "ll", pre=True)
    def empty_to_none_float(cls, v):
        if v == "" or v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

# -----------------------
# Material
# -----------------------
class Material(BaseModel):
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
    source: Optional[str] = None      # link to sources.yaml / epd_registry

    @validator(
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
        pre=True,
        always=False
        )

    def empty_to_none_float(cls, v):
        if v == "" or v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    @validator("ec_a1a3_volumetric", "ec_a4_volumetric", "ec_a5_volumetric", always=True)
    def fill_a1a3_volumetric_from_mass(cls, v, values):
        # If per_m3 missing but per_kg + density exist, compute per_m3
        if v is None and values.get("ec_a1a3_mass") is not None and values.get("density") is not None:
            return values["ec_a1a3_mass"] * values["density"]
        if v is None and values.get("ec_a4_mass") is not None and values.get("density") is not None:
            return values["ec_a4_mass"] * values["density"]
        if v is None and values.get("ec_a5_mass") is not None and values.get("density") is not None:
            return values["ec_a5_mass"] * values["density"]
        return v
    
    @validator("cost_volumetric", always=True)
    def cost_volumetric_from_mass(cls, v, values):
        # If per_m3 missing but per_kg + density exist, compute per_m3
        if v is None and values.get("cost_mass") is not None and values.get("density") is not None:
            return values["cost_mass"] * values["density"]
        return v

    @validator("concrete_f_ck", "steel_fy", "steel_fu", "timber_fm", "timber_e", "density", "transport_distance", "cost_volumetric", "cost_mass")
    def non_negative(cls, v):
        if v is None:
            return None
        if v < 0:
            raise ValueError("must be non-negative")
        return v

# -----------------------
# Source / EPD metadata
# -----------------------
class Source(BaseModel):
    type: Optional[str] = None          # 'epd' | 'datasheet' | 'book' | 'assumption'
    source: str
    specification_id: Optional[str] = None
    citation: Optional[str] = None
    url: Optional[str] = None
    published: Optional[str] = None     # ISO date string
    region: Optional[str] = None        # e.g. 'UK', 'EU',

# -----------------------
# SystemFamily (catalog-level)
# -----------------------
class SystemFamily(BaseModel):
    system_family: str             # canonical family id (was system_id)
    component: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    span_behavior: Optional[str] = None
    manufacturer: Optional[str] = None
    unit: Optional[str] = None
    width: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

# -----------------------
# SystemVariant (performance row)
# -----------------------
class SystemVariant(BaseModel):
    system_variant: Optional[str] = None    # specific variant id (was system_variant_id)
    system_family: Optional[str] = None
    slab_depth: Optional[float] = None
    beam_depth: Optional[float] = None
    screed_depth: Optional[float] = None
    steel_depth: Optional[float] = None
    swt: Optional[float] = None
    sdl: Optional[float] = None
    ll: Optional[float] = None
    max_span: Optional[float] = None
    concrete_volume: Optional[float] = None   # m3 per m2 (or per unit area) — document units in schema yaml
    steel_volume: Optional[float] = None
    rebar_volume: Optional[float] = None
    pt_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_rebar_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    ebc_mm: Optional[float] = None
    beam_ref: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    # numeric coercion + non-negative checks
    @validator(
        "slab_depth", "beam_depth", "screed_depth", "steel_depth",
        "swt", "sdl", "ll", "max_span", "concrete_volume", "steel_volume", "rebar_volume", "pt_volume", "timber_volume",
        pre=True
    )
    def coerce_to_float_or_none(cls, v):
        if v == "" or v is None:
            return None
        try:
            f = float(v)
            if isinstance(f, float) and math.isnan(f):
                return None
            return f
        except Exception:
            return None

    @validator("max_span")
    def max_span_positive(cls, v):
        if v is None:
            return None
        if v <= 0:
            raise ValueError("max_span must be > 0")
        return v

    # helpers that use material lookup (material_lookup must be dict material_id->Material)
    def concrete_mass_per_m2(self, material_lookup: Dict[str, Material]) -> Optional[float]:
        if self.concrete_volume is None:
            return None
        mat_id = self.material_concrete_id
        if not mat_id:
            return None
        mat = material_lookup.get(mat_id)
        if mat is None or mat.density is None:
            return None
        return float(self.concrete_volume) * float(mat.density)

    def embodied_carbon_a1a3_per_m2(self, material_lookup: Dict[str, Material]) -> Optional[float]:
        total = 0.0
        any_known = False
        cm = self.concrete_mass_per_m2(material_lookup)
        if cm is not None:
            mat = material_lookup.get(self.material_concrete_id)
            if mat and mat.ec_a1a3_mass is not None:
                total += cm * float(mat.ec_a1a3_mass)
                any_known = True
        # steel
        if self.steel_volume is not None:
            mat = material_lookup.get(self.material_steel_id)
            if mat and mat.density is not None and mat.ec_a1a3_mass is not None:
                steel_mass = float(self.steel_volume) * float(mat.density)
                total += steel_mass * float(mat.ec_a1a3_mass)
                any_known = True
        # rebar
        if self.rebar_volume is not None:
            mat = material_lookup.get(self.material_rebar_id)
            if mat and mat.density is not None and mat.ec_a1a3_mass is not None:
                rebar_mass = float(self.rebar_volume) * float(mat.density)
                total += rebar_mass * float(mat.ec_a1a3_mass)
                any_known = True
        # pt steel
        if self.pt_volume is not None and self.material_pt_id is not None:
            mat = material_lookup.get(self.material_pt_id)
            if mat and mat.density is not None and mat.ec_a1a3_mass is not None:
                pt_mass = float(self.pt_volume) * float(mat.density)
                total += pt_mass * float(mat.ec_a1a3_mass)
                any_known = True
        # timber
        if self.timber_volume is not None:
            mat = material_lookup.get(self.material_timber_id)
            if mat and mat.density is not None and mat.ec_a1a3_mass is not None:
                timber_mass = float(self.timber_volume) * float(mat.density)
                total += timber_mass * float(mat.ec_a1a3_mass)
                any_known = True

        return total if any_known else None

# IMPORTANT: forward refs for methods that reference other models
SystemVariant.update_forward_refs()