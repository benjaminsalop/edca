from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DesignCodeContext:
    code_family: str
    code_standard: Optional[str] = None
    design_basis: Optional[str] = None
    region: Optional[str] = None
    unit_system: str = "metric"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaterialContext:
    concrete_material_id: Optional[str] = None
    rebar_material_id: Optional[str] = None
    steel_material_id: Optional[str] = None
    timber_material_id: Optional[str] = None
    screed_material_id: Optional[str] = None
    resolved: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadEffect:
    case_id: str
    limit_state: str
    action_type: str
    value: float
    units: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentDesignInput:
    component_type: str
    family_id: str
    variant_id: Optional[str] = None
    role: Optional[str] = None
    geometry: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    loads: List[LoadEffect] = field(default_factory=list)
    materials: MaterialContext = field(default_factory=MaterialContext)
    code_context: Optional[DesignCodeContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SlabDesignInput(ComponentDesignInput):
    clear_span_m: Optional[float] = None
    strip_width_m: float = 1.0
    slab_depth_m: Optional[float] = None
    screed_depth_m: float = 0.0
    support_condition: Optional[str] = None
    distribution_method: Optional[str] = None


@dataclass(slots=True)
class BeamDesignInput(ComponentDesignInput):
    span_m: Optional[float] = None
    unbraced_length_m: Optional[float] = None


@dataclass(slots=True)
class ColumnDesignInput(ComponentDesignInput):
    storey_height_m: Optional[float] = None
    effective_length_factor: Optional[float] = None


@dataclass(slots=True)
class LateralSystemDesignInput(ComponentDesignInput):
    storey_shear_kN: Optional[float] = None
    overturning_moment_kNm: Optional[float] = None
    torsion_kNm: Optional[float] = None


@dataclass(slots=True)
class BuildingSystemDesignInput:
    project_id: Optional[str] = None
    building_name: Optional[str] = None
    code_context: Optional[DesignCodeContext] = None
    storey_count: Optional[int] = None
    torsion_irregularity_ratio: Optional[float] = None
    drift_by_storey: Dict[str, float] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
