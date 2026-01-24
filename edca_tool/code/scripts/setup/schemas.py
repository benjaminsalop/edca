# src/edtool/db/schemas.py
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator, HttpUrl

class Occupancy(BaseModel):
    use: str
    unit: Optional[str] = None
    sdl: Optional[float] = None         # superimposed dead load (kN/m2 or as defined)
    sdl_partition: Optional[float] = None
    ll: Optional[float] = None          # live load (kN/m2)
    code: Optional[str] = None
    notes: Optional[str] = None

    @validator("sdl", "sdl_partition", "ll", pre=True)
    def empty_to_none_float(cls, v):
        if v == "" or v is None:
            return None
        return float(v)

class Material(BaseModel):
    material_id: str
    family: Optional[str] = None
    concrete_psi: Optional[float] = None
    steel_ksi: Optional[float] = None
    density_kg_per_m3: Optional[float] = None
    ec_a1a3_per_m3: Optional[float] = None
    ec_a1a3_per_kg: Optional[float] = None
    ec_a4_per_ton_km: Optional[float] = None
    transport_km: Optional[float] = None
    ec_a5_per_kg: Optional[float] = None
    cost_per_m3: Optional[float] = None
    cost_per_kg: Optional[float] = None

    @validator(
        "concrete_psi",
        "steel_ksi",
        "density_kg_per_m3",
        "ec_a1a3_per_m3",
        "ec_a1a3_per_kg",
        "ec_a4_per_ton_km",
        "transport_km",
        "ec_a5_per_kg",
        "cost_per_m3",
        "cost_per_kg",
        pre=True,
    )
    def empty_to_none_float(cls, v):
        if v == "" or v is None:
            return None
        return float(v)

    @validator("ec_a1a3_per_m3", always=True)
    def fill_a1a3_per_m3_from_per_kg(cls, v, values):
        # If per_m3 missing but per_kg + density exist, compute per_m3
        if v is None and values.get("ec_a1a3_per_kg") is not None and values.get("density_kg_per_m3") is not None:
            return values["ec_a1a3_per_kg"] * values["density_kg_per_m3"]
        return v

class SystemRow(BaseModel):
    component: Optional[str] = None
    system_id: str
    category: Optional[str] = None
    type: Optional[str] = None
    span_behavior: Optional[str] = None
    manufacturer: Optional[str] = None
    unit: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    slab_depth: Optional[float] = None
    beam_depth: Optional[float] = None
    screed_depth: Optional[float] = None
    steel_depth: Optional[float] = None
    swt: Optional[float] = None
    sdl: Optional[float] = None
    ll: Optional[float] = None
    max_span: Optional[float] = None
    material_concrete_id: Optional[str] = None
    material_pt_id: Optional[str] = None
    material_steel_id: Optional[str] = None
    material_timber_id: Optional[str] = None
    ebc_mm: Optional[float] = None
    beam_ref: Optional[str] = None
    concrete_volume: Optional[float] = None
    steel_volume: Optional[float] = None
    timber_volume: Optional[float] = None
    total_depth: Optional[float] = None

    @validator(
        "length", "width", "slab_depth", "beam_depth", "screed_depth", "steel_depth",
        "swt", "sdl", "ll", "max_span", "ebc_mm", "concrete_volume", "steel_volume",
        "timber_volume", "total_depth", pre=True
    )
    def empty_to_none_float(cls, v):
        if v == "" or v is None:
            return None
        return float(v)
