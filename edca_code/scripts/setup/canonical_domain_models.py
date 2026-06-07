from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# -----------------------------
# Core enums / identifiers
# -----------------------------


class ComponentType(str, Enum):
    FLOOR = "floor"
    BEAM = "beam"
    COLUMN = "column"
    LATERAL = "lateral"
    WALL = "wall"  # reserved for upcoming expansion


class BeamRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EDGE = "edge"


class LimitState(str, Enum):
    ULS = "ULS"
    SLS = "SLS"
    LRFD = "LRFD"
    ASD = "ASD"
    STRENGTH = "strength"
    SERVICEABILITY = "serviceability"
    STABILITY = "stability"


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


# -----------------------------
# Project / run context
# -----------------------------


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


# -----------------------------
# Materials
# -----------------------------


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


# -----------------------------
# System catalogs
# -----------------------------


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


# -----------------------------
# Typologies / preferences
# -----------------------------


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


# -----------------------------
# Compatibility / assembly
# -----------------------------


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


# -----------------------------
# Loads / occupancies / combinations
# -----------------------------


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


# -----------------------------
# Repository container
# -----------------------------


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


# -----------------------------
# Loader source bundle
# -----------------------------


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


# -----------------------------
# Service boundaries
# -----------------------------


class DomainError(Exception):
    """Raised when source data cannot be normalized into the canonical model."""


class DomainRepositoryBuilder:
    """
    Adapter boundary.

    This class should be the only place that knows how to read raw parquet/csv/yaml
    files and normalize them into the canonical in-memory model.
    """

    def build(self, raw_sources: SourceBundle) -> DomainRepository:
        repo = DomainRepository()
        repo.typologies = self._load_typologies(raw_sources.typologies_yaml)
        repo.families.update(self._load_system_families(ComponentType.FLOOR, raw_sources.floor_families))
        repo.families.update(self._load_system_families(ComponentType.BEAM, raw_sources.beam_families))
        repo.families.update(self._load_system_families(ComponentType.COLUMN, raw_sources.column_families))
        repo.families.update(self._load_system_families(ComponentType.LATERAL, raw_sources.lateral_families))

        repo.variants.update(self._load_system_variants(ComponentType.FLOOR, raw_sources.floor_variants))
        repo.variants.update(self._load_system_variants(ComponentType.BEAM, raw_sources.beam_variants))
        repo.variants.update(self._load_system_variants(ComponentType.COLUMN, raw_sources.column_variants))
        repo.variants.update(self._load_system_variants(ComponentType.LATERAL, raw_sources.lateral_variants))

        repo.occupancies = self._load_occupancies(raw_sources.occupancies)
        repo.material_sets = self._load_material_sets(raw_sources.material_sets_yaml)
        repo.material_mapping_rules = self._load_material_mapping_rules(raw_sources.material_mapping_yaml)
        repo.preference_rules = self._load_preference_rules(raw_sources.system_preferences_yaml)
        repo.compatibility_rules = self._load_compatibility_rules(raw_sources.system_compatibility_yaml)
        repo.load_combinations = self._load_load_combinations(raw_sources.load_combinations_yaml)

        repo.config["assembly_generation"] = raw_sources.assembly_generation_yaml
        repo.config["system_types"] = raw_sources.system_types_yaml
        repo.config["materials_config"] = raw_sources.materials_config_yaml
        repo.config["materials_methods"] = raw_sources.materials_methods_yaml
        repo.config["load_values"] = raw_sources.load_values_yaml

        self._validate_cross_references(repo)
        return repo

    def _load_typologies(self, data: dict[str, Any]) -> dict[str, Typology]:
        result: dict[str, Typology] = {}
        for typology_id, payload in data.get("typologies", {}).items():
            constraints = payload.get("constraints", {})
            result[typology_id] = Typology(
                typology_id=typology_id,
                description=payload.get("description"),
                defaults=payload.get("defaults", {}),
                constraints=TypologyConstraints(
                    occupancy=constraints.get("occupancy"),
                    allowed_material_families=constraints.get("allowed_material_families", []),
                    max_storey_count=constraints.get("max_storey_count"),
                    span_x_m=self._as_range_tuple(constraints.get("span_x_m")),
                    span_y_m=self._as_range_tuple(constraints.get("span_y_m")),
                    floor_to_floor_m=self._as_range_tuple(constraints.get("floor_to_floor_m")),
                ),
                preferences=payload.get("preferences", {}),
            )
        return result

    def _load_system_families(
        self,
        component: ComponentType,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[ComponentType, str], SystemFamily]:
        result: dict[tuple[ComponentType, str], SystemFamily] = {}
        for row in rows:
            family_id = self._pick_first(row, [f"{component.value}_family_id", "family_id", f"{component.value}_type", "system_family_id"])
            if not family_id:
                raise DomainError(f"Missing family id for component={component.value}: {row}")
            result[(component, family_id)] = SystemFamily(
                component=component,
                family_id=family_id,
                material_family=row.get("material_family"),
                type_name=self._pick_first(row, [f"{component.value}_type", "type_name", "system_type"]),
                defaults={},
                metadata=dict(row),
            )
        return result

    def _load_system_variants(
        self,
        component: ComponentType,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[ComponentType, str], SystemVariant]:
        result: dict[tuple[ComponentType, str], SystemVariant] = {}
        for row in rows:
            variant_id = self._pick_first(row, [f"{component.value}_variant_id", "variant_id", "system_variant_id"])
            family_id = self._pick_first(row, [f"{component.value}_family_id", "family_id", f"{component.value}_type", "system_family_id"])
            if not variant_id or not family_id:
                raise DomainError(f"Missing variant/family id for component={component.value}: {row}")
            result[(component, variant_id)] = SystemVariant(
                component=component,
                variant_id=variant_id,
                family_id=family_id,
                material_family=row.get("material_family"),
                code_basis=self._as_list(row.get("code_basis")),
                geometry=GeometricProperties(
                    depth_overall=self._as_float(self._pick_first(row, ["depth_overall", "overall_depth", "depth"])),
                    width=self._as_float(row.get("width")),
                    thickness=self._as_float(row.get("thickness")),
                    area=self._as_float(row.get("area")),
                    inertia_major=self._as_float(self._pick_first(row, ["inertia_major", "ix", "I_major"])),
                    inertia_minor=self._as_float(self._pick_first(row, ["inertia_minor", "iy", "I_minor"])),
                    section_modulus_major=self._as_float(self._pick_first(row, ["section_modulus_major", "zx", "S_major"])),
                    section_modulus_minor=self._as_float(self._pick_first(row, ["section_modulus_minor", "zy", "S_minor"])),
                ),
                span_limits=SpanLimits(
                    min_span_m=self._as_float(row.get("min_span_m")),
                    max_span_m=self._as_float(row.get("max_span_m")),
                    max_cantilever_m=self._as_float(row.get("max_cantilever_m")),
                ),
                metrics=VariantMetrics(
                    cost=self._as_float(row.get("cost")),
                    embodied_carbon=self._as_float(row.get("embodied_carbon")),
                    weight_per_m=self._as_float(row.get("weight_per_m")),
                    weight_per_m2=self._as_float(row.get("weight_per_m2")),
                    steel_volume_per_m=self._as_float(row.get("steel_volume_per_m")),
                    concrete_volume_per_m3eq=self._as_float(row.get("concrete_volume_per_m3eq")),
                ),
                material_refs={},
                properties=dict(row),
                raw=dict(row),
            )
        return result

    def _load_occupancies(self, rows: list[dict[str, Any]]) -> dict[str, Occupancy]:
        result: dict[str, Occupancy] = {}
        for row in rows:
            occupancy_id = self._pick_first(row, ["occupancy_id", "id", "occupancy"])
            if not occupancy_id:
                raise DomainError(f"Missing occupancy id: {row}")
            result[occupancy_id] = Occupancy(
                occupancy_id=occupancy_id,
                code_family=row.get("code_family"),
                category=row.get("category"),
                description=row.get("description"),
                load_values={
                    k: v for k, v in row.items()
                    if isinstance(v, (int, float)) and k not in {"storey_count"}
                },
                reduction_group=row.get("reduction_group"),
                metadata=dict(row),
            )
        return result

    def _load_material_sets(self, data: dict[str, Any]) -> dict[str, MaterialSet]:
        result: dict[str, MaterialSet] = {}
        for material_set_id, payload in data.get("material_sets", {}).items():
            result[material_set_id] = MaterialSet(
                material_set_id=material_set_id,
                concrete=self._material_ref_from_payload(payload.get("concrete")),
                rebar=self._material_ref_from_payload(payload.get("rebar")),
                steel=self._material_ref_from_payload(payload.get("steel")),
                timber=self._material_ref_from_payload(payload.get("timber")),
                alternatives=self._material_alternatives_from_payload(payload),
            )
        return result

    def _load_material_mapping_rules(self, data: dict[str, Any]) -> list[MaterialMappingRule]:
        return [
            MaterialMappingRule(
                when=rule.get("when", {}),
                use_material_set=rule["use_material_set"],
            )
            for rule in data.get("rules", [])
        ]

    def _load_preference_rules(self, data: dict[str, Any]) -> list[SystemPreferenceRule]:
        rules: list[SystemPreferenceRule] = []
        for rule in data.get("rules", []):
            rules.append(
                SystemPreferenceRule(
                    name=rule["name"],
                    when=rule.get("when", {}),
                    filters=rule.get("filters", {}),
                    rank=[
                        RankingField(field=item["field"], direction=item["direction"])
                        for item in rule.get("rank", [])
                    ],
                )
            )
        return rules

    def _load_compatibility_rules(self, data: dict[str, Any]) -> list[CompatibilityRule]:
        rules: list[CompatibilityRule] = []
        for entry in data.get("global_rules", {}).get("hard_disallow", []):
            rules.append(
                CompatibilityRule(
                    severity=RuleSeverity.HARD_DISALLOW,
                    target=CompatibilityTarget(
                        floor_family=entry.get("floor_family"),
                        primary_beam_family=entry.get("primary_beam_family"),
                        secondary_beam_family=entry.get("secondary_beam_family"),
                        beam_family=entry.get("beam_family"),
                        column_family=entry.get("column_family"),
                        lateral_family=entry.get("lateral_family"),
                    ),
                    reason=entry.get("reason"),
                )
            )
        for entry in data.get("global_rules", {}).get("soft_discourage", []):
            rules.append(
                CompatibilityRule(
                    severity=RuleSeverity.SOFT_DISCOURAGE,
                    target=CompatibilityTarget(
                        floor_family=entry.get("floor_family"),
                        primary_beam_family=entry.get("primary_beam_family"),
                        secondary_beam_family=entry.get("secondary_beam_family"),
                        beam_family=entry.get("beam_family"),
                        column_family=entry.get("column_family"),
                        lateral_family=entry.get("lateral_family"),
                    ),
                    reason=entry.get("reason"),
                )
            )
        return rules

    def _load_load_combinations(self, data: dict[str, Any]) -> list[LoadCombinationTemplate]:
        combinations: list[LoadCombinationTemplate] = []
        for code_family, payload in data.items():
            if code_family == "meta" or not isinstance(payload, dict):
                continue
            for design_basis, combo_map in payload.items():
                if not isinstance(combo_map, dict):
                    continue
                for combination_id, combo_payload in combo_map.items():
                    expressions: list[LoadCombinationExpression] = []
                    if "expression" in combo_payload:
                        expressions.append(self._parse_expression(combo_payload["expression"]))
                    for alt in combo_payload.get("alternatives", []):
                        expressions.append(self._parse_expression(alt.get("expression", [])))
                    combinations.append(
                        LoadCombinationTemplate(
                            code_family=code_family,
                            design_basis=design_basis,
                            combination_id=combination_id,
                            description=combo_payload.get("description"),
                            expressions=expressions,
                        )
                    )
        return combinations

    def _parse_expression(self, rows: list[dict[str, Any]]) -> LoadCombinationExpression:
        return LoadCombinationExpression(
            actions=[LoadAction(action=row["action"], factor=row["factor"]) for row in rows]
        )

    def _validate_cross_references(self, repo: DomainRepository) -> None:
        for typology in repo.typologies.values():
            defaults = typology.defaults
            self._validate_family_ref(repo, ComponentType.FLOOR, defaults.get("floor_system_family"), typology.typology_id)
            self._validate_family_ref(repo, ComponentType.BEAM, defaults.get("primary_beam_system_family"), typology.typology_id)
            self._validate_family_ref(repo, ComponentType.BEAM, defaults.get("secondary_beam_system_family"), typology.typology_id)
            self._validate_family_ref(repo, ComponentType.COLUMN, defaults.get("column_system_family"), typology.typology_id)
            self._validate_family_ref(repo, ComponentType.LATERAL, defaults.get("lateral_system_family"), typology.typology_id)

    def _validate_family_ref(
        self,
        repo: DomainRepository,
        component: ComponentType,
        family_id: str | None,
        source_name: str,
    ) -> None:
        if family_id is None:
            return
        if (component, family_id) not in repo.families:
            raise DomainError(
                f"Unknown {component.value} family '{family_id}' referenced by {source_name}"
            )

    def _material_ref_from_payload(self, payload: dict[str, Any] | None) -> MaterialRef | None:
        if not payload or not payload.get("default"):
            return None
        return MaterialRef(material_id=payload["default"])

    def _material_alternatives_from_payload(self, payload: dict[str, Any]) -> dict[str, list[MaterialRef]]:
        result: dict[str, list[MaterialRef]] = {}
        for key, value in payload.items():
            alts = value.get("alternatives", []) if isinstance(value, dict) else []
            result[key] = [MaterialRef(material_id=item) for item in alts]
        return result

    def _pick_first(self, row: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = row.get(key)
            if value is not None:
                return value
        return None

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _as_range_tuple(self, value: Any) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise DomainError(f"Expected 2-item range, got: {value}")

    def _as_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)


class CandidateGenerator:
    def generate(
        self,
        project: ProjectContext,
        repo: DomainRepository,
    ) -> list[AssemblyCandidate]:
        raise NotImplementedError


class CompatibilityEngine:
    def apply(
        self,
        candidates: list[AssemblyCandidate],
        repo: DomainRepository,
    ) -> list[AssemblyCandidate]:
        raise NotImplementedError


class PreferenceEngine:
    def rank(
        self,
        candidates: list[AssemblyCandidate],
        project: ProjectContext,
        repo: DomainRepository,
    ) -> list[AssemblyCandidate]:
        raise NotImplementedError


class LoadCombinationEngine:
    def resolve(
        self,
        project: ProjectContext,
        occupancy: Occupancy,
        repo: DomainRepository,
    ) -> list[EvaluatedLoadCombination]:
        raise NotImplementedError


# -----------------------------
# Normalization notes
# -----------------------------

# 1. Raw files stay raw. Do not let design engines read parquet/yaml directly.
# 2. Normalize family-vs-variant identity once at load time.
# 3. Keep family-level compatibility separate from variant-level sizing properties.
# 4. Keep symbolic load combinations separate from resolved numeric combinations.
# 5. AssemblyCandidate is the handoff object from selection logic into design/sizing.
