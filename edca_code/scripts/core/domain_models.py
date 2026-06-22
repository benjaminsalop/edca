from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ComponentType(str, Enum):
    FLOOR = "floor"
    BEAM = "beam"
    COLUMN = "column"
    LATERAL = "lateral"
    WALL = "wall"


class BeamRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EDGE = "edge"


class RuleSeverity(str, Enum):
    HARD_DISALLOW = "hard_disallow"
    SOFT_DISCOURAGE = "soft_discourage"


class LoadPathMethod(str, Enum):
    BEAMLESS = "beamless"
    ONE_WAY_WITH_PRIMARY_ONLY = "one_way_with_primary_only"
    TWO_WAY_WITH_SECONDARY_AND_PRIMARY = "two_way_with_secondary_and_primary"
    CUSTOM = "custom"


CodeFamily = Literal["EC", "ASCE"]
Region = Literal["UK", "US", "EU"]
MaterialFamily = Literal["precast", "concrete", "steel", "timber", "hybrid"]


@dataclass(slots=True)
class GeometryContext:
    span_x_m: float | None = None
    span_y_m: float | None = None
    floor_to_floor_m: float | None = None
    storey_count: int | None = None
    bay_area_m2: float | None = None


@dataclass(slots=True)
class ProjectContext:
    project_id: str | None = None
    region: Region = "UK"
    code_family: CodeFamily = "EC"
    typology_id: str | None = None
    occupancy_id: str | None = None
    geometry: GeometryContext = field(default_factory=GeometryContext)
    design_options: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaterialRef:
    material_id: str
    material_class: str | None = None
    standard_or_grade: str | None = None


@dataclass(slots=True)
class MaterialSet:
    material_set_id: str
    concrete: MaterialRef | None = None
    rebar: MaterialRef | None = None
    steel: MaterialRef | None = None
    timber: MaterialRef | None = None
    alternatives: dict[str, list[MaterialRef]] = field(default_factory=dict)


@dataclass(slots=True)
class MaterialMappingRule:
    when: dict[str, Any]
    use_material_set: str


@dataclass(slots=True)
class CapabilityFlags:
    requires_primary_beams: bool | None = None
    requires_secondary_beams: bool | None = None
    can_beamless: bool | None = None
    can_span_to_columns_directly: bool | None = None
    can_span_to_walls: bool | None = None
    can_be_primary: bool | None = None
    can_be_secondary: bool | None = None
    can_be_edge_beam: bool | None = None


@dataclass(slots=True)
class SystemFamily:
    component: ComponentType
    family_id: str
    material_family: MaterialFamily | None = None
    type_name: str | None = None
    capabilities: CapabilityFlags = field(default_factory=CapabilityFlags)
    defaults: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VariantMetrics:
    cost: float | None = None
    embodied_carbon: float | None = None
    weight_per_m: float | None = None
    weight_per_m2: float | None = None
    steel_volume_per_m: float | None = None
    concrete_volume_per_m3eq: float | None = None


@dataclass(slots=True)
class GeometricProperties:
    depth_overall: float | None = None
    width: float | None = None
    thickness: float | None = None
    area: float | None = None
    inertia_major: float | None = None
    inertia_minor: float | None = None
    section_modulus_major: float | None = None
    section_modulus_minor: float | None = None


@dataclass(slots=True)
class SpanLimits:
    min_span_m: float | None = None
    max_span_m: float | None = None
    max_cantilever_m: float | None = None


@dataclass(slots=True)
class SystemVariant:
    component: ComponentType
    variant_id: str
    family_id: str
    material_family: MaterialFamily | None = None
    code_basis: list[str] = field(default_factory=list)
    geometry: GeometricProperties = field(default_factory=GeometricProperties)
    span_limits: SpanLimits = field(default_factory=SpanLimits)
    metrics: VariantMetrics = field(default_factory=VariantMetrics)
    material_refs: dict[str, str] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TypologyConstraints:
    occupancy: str | None = None
    allowed_material_families: list[MaterialFamily] = field(default_factory=list)
    max_storey_count: int | None = None
    span_x_m: tuple[float, float] | None = None
    span_y_m: tuple[float, float] | None = None
    floor_to_floor_m: tuple[float, float] | None = None


@dataclass(slots=True)
class Typology:
    typology_id: str
    description: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    constraints: TypologyConstraints = field(default_factory=TypologyConstraints)
    preferences: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankingField:
    field: str
    direction: Literal["asc", "desc"]


@dataclass(slots=True)
class SystemPreferenceRule:
    name: str
    when: dict[str, Any]
    filters: dict[str, list[Any] | Any] = field(default_factory=dict)
    rank: list[RankingField] = field(default_factory=list)


@dataclass(slots=True)
class CompatibilityTarget:
    floor_family: str | None = None
    primary_beam_family: str | None = None
    secondary_beam_family: str | None = None
    beam_family: str | None = None
    column_family: str | None = None
    lateral_family: str | None = None


@dataclass(slots=True)
class CompatibilityRule:
    severity: RuleSeverity
    target: CompatibilityTarget
    reason: str | None = None
    penalty: float | None = None


@dataclass(slots=True)
class AssemblyPenalty:
    source: str
    points: float
    reason: str | None = None


@dataclass(slots=True)
class AssemblyCandidate:
    candidate_id: str
    typology_id: str | None
    floor_family_id: str
    floor_variant_id: str | None = None
    primary_beam_family_id: str | None = None
    primary_beam_variant_id: str | None = None
    secondary_beam_family_id: str | None = None
    secondary_beam_variant_id: str | None = None
    column_family_id: str | None = None
    column_variant_id: str | None = None
    lateral_family_id: str | None = None
    lateral_variant_id: str | None = None
    load_path_method: LoadPathMethod = LoadPathMethod.CUSTOM
    material_mix_label: str | None = None
    penalties: list[AssemblyPenalty] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_penalty(self) -> float:
        return sum(p.points for p in self.penalties)


@dataclass(slots=True)
class Occupancy:
    occupancy_id: str
    code_family: CodeFamily | None = None
    category: str | None = None
    description: str | None = None
    load_values: dict[str, float] = field(default_factory=dict)
    reduction_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadAction:
    action: str
    factor: float | str


@dataclass(slots=True)
class LoadCombinationExpression:
    actions: list[LoadAction] = field(default_factory=list)


@dataclass(slots=True)
class LoadCombinationTemplate:
    code_family: str
    design_basis: str
    combination_id: str
    description: str | None = None
    expressions: list[LoadCombinationExpression] = field(default_factory=list)


@dataclass(slots=True)
class EvaluatedLoadCombination:
    code_family: str
    design_basis: str
    combination_id: str
    resolved_factors: dict[str, float]
    source_expression_index: int = 0


@dataclass(slots=True)
class SourceBundle:
    floor_families: list[dict[str, Any]] = field(default_factory=list)
    floor_variants: list[dict[str, Any]] = field(default_factory=list)
    beam_families: list[dict[str, Any]] = field(default_factory=list)
    beam_variants: list[dict[str, Any]] = field(default_factory=list)
    column_families: list[dict[str, Any]] = field(default_factory=list)
    column_variants: list[dict[str, Any]] = field(default_factory=list)
    lateral_families: list[dict[str, Any]] = field(default_factory=list)
    lateral_variants: list[dict[str, Any]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    occupancies: list[dict[str, Any]] = field(default_factory=list)
    typologies_yaml: dict[str, Any] = field(default_factory=dict)
    system_types_yaml: dict[str, Any] = field(default_factory=dict)
    system_preferences_yaml: dict[str, Any] = field(default_factory=dict)
    system_compatibility_yaml: dict[str, Any] = field(default_factory=dict)
    assembly_generation_yaml: dict[str, Any] = field(default_factory=dict)
    material_mapping_yaml: dict[str, Any] = field(default_factory=dict)
    material_sets_yaml: dict[str, Any] = field(default_factory=dict)
    materials_config_yaml: dict[str, Any] = field(default_factory=dict)
    materials_methods_yaml: dict[str, Any] = field(default_factory=dict)
    load_combinations_yaml: dict[str, Any] = field(default_factory=dict)
    load_values_yaml: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DomainRepository:
    typologies: dict[str, Typology] = field(default_factory=dict)
    families: dict[tuple[ComponentType, str], SystemFamily] = field(default_factory=dict)
    variants: dict[tuple[ComponentType, str], SystemVariant] = field(default_factory=dict)
    occupancies: dict[str, Occupancy] = field(default_factory=dict)
    material_sets: dict[str, MaterialSet] = field(default_factory=dict)
    material_mapping_rules: list[MaterialMappingRule] = field(default_factory=list)
    preference_rules: list[SystemPreferenceRule] = field(default_factory=list)
    compatibility_rules: list[CompatibilityRule] = field(default_factory=list)
    load_combinations: list[LoadCombinationTemplate] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
