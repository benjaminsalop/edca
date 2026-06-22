from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .domain_models import (
    CapabilityFlags,
    CompatibilityRule,
    CompatibilityTarget,
    ComponentType,
    DomainRepository,
    GeometricProperties,
    LoadAction,
    LoadCombinationExpression,
    LoadCombinationTemplate,
    MaterialMappingRule,
    MaterialRef,
    MaterialSet,
    Occupancy,
    RankingField,
    RuleSeverity,
    SourceBundle,
    SpanLimits,
    SystemFamily,
    SystemPreferenceRule,
    SystemVariant,
    Typology,
    TypologyConstraints,
    VariantMetrics,
)
from .exceptions import DomainError, RepositoryError
from .validation import validate_repository


class RepositoryBuilder:
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

        self._apply_system_type_metadata(repo, raw_sources.system_types_yaml)
        self._apply_compatibility_defaults(repo, raw_sources.system_compatibility_yaml)

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
        repo.config["materials"] = raw_sources.materials

        validate_repository(repo)
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
            # Prefer explicit material_family; fall back to the category column.
            # Construction-method categories ("precast", "cast_in_place", "composite")
            # are normalised to their structural material so typology filters
            # (which use [concrete, steel, timber, hybrid]) work correctly.
            _CAT_TO_MAT = {
                "precast": "concrete",
                "cast_in_place": "concrete",
                "composite": "steel",
                "timber": "timber",
                "clt": "timber",
                "lvl": "timber",
            }
            raw_cat = (
                row.get("material_family")
                or row.get(f"{component.value}_category")
                or row.get("category")
            )
            mat_family = _CAT_TO_MAT.get(str(raw_cat).lower(), raw_cat) if raw_cat else None
            family = SystemFamily(
                component=component,
                family_id=str(family_id),
                material_family=mat_family or None,
                type_name=self._pick_first(row, [f"{component.value}_type", "type_name", "system_type"]),
                defaults={},
                metadata=dict(row),
            )
            # For floor families, initialise capabilities from the CSV beam_requirements
            # field.  The YAML _apply_compatibility_defaults() step will override this for
            # families that appear in system_compatibility.yaml.  For families that are
            # only in the CSV (e.g. ec_cip_twoway_*, aci_cip_twoway_*), the CSV value
            # is the sole source of beam framing requirements.
            if component == ComponentType.FLOOR:
                beam_req = str(row.get("beam_requirements") or "").strip().lower()
                if beam_req in ("none", "no beams required", "no beams"):
                    family.capabilities = CapabilityFlags(
                        can_beamless=True,
                        requires_primary_beams=False,
                        requires_secondary_beams=False,
                    )
                elif beam_req == "secondary":
                    family.capabilities = CapabilityFlags(
                        can_beamless=False,
                        requires_primary_beams=True,
                        requires_secondary_beams=True,
                    )
                elif beam_req in ("primary", "supporting beams or walls", "integral ribs/joists"):
                    family.capabilities = CapabilityFlags(
                        can_beamless=False,
                        requires_primary_beams=True,
                        requires_secondary_beams=False,
                    )
            result[(component, str(family_id))] = family
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
            result[(component, str(variant_id))] = SystemVariant(
                component=component,
                variant_id=str(variant_id),
                family_id=str(family_id),
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

    def _apply_system_type_metadata(self, repo: DomainRepository, data: dict[str, Any]) -> None:
        aliases = data.get("aliases", {})
        types = data.get("types", {})
        family_map: dict[str, tuple[ComponentType, str, str]] = {}
        for component_name, material_map in types.items():
            component = ComponentType(component_name)
            for material_family, family_ids in material_map.items():
                for family_id in family_ids:
                    family_map[str(family_id)] = (component, str(family_id), str(material_family))
        for alias, canonical in aliases.items():
            entry = family_map.get(str(canonical))
            if not entry:
                continue
            component, family_id, _material_family = entry
            family = repo.families.get((component, family_id))
            if family and alias not in family.aliases:
                family.aliases.append(str(alias))
        for (component, family_id), family in repo.families.items():
            # Try matching by family_id first, then by type_name (e.g. "beam_block").
            mapped = family_map.get(family_id) or (
                family_map.get(family.type_name) if family.type_name else None
            )
            if mapped and not family.material_family:
                family.material_family = mapped[2]

    def _apply_compatibility_defaults(self, repo: DomainRepository, data: dict[str, Any]) -> None:
        families_section = data.get("families", {})

        floor_section = families_section.get("floor", {})
        for family_id, payload in floor_section.items():
            family = repo.families.get((ComponentType.FLOOR, family_id))
            if not family:
                continue
            defaults = payload.get("defaults", {})
            family.defaults.update(defaults)
            family.capabilities = CapabilityFlags(
                requires_primary_beams=defaults.get("requires_primary_beams"),
                requires_secondary_beams=defaults.get("requires_secondary_beams"),
                can_beamless=defaults.get("can_beamless"),
                can_span_to_columns_directly=defaults.get("can_span_to_columns_directly"),
                can_span_to_walls=defaults.get("can_span_to_walls"),
            )
            family.metadata.setdefault("compatibility", {}).update(payload)

        beam_section = families_section.get("beam", {})
        for family_id, payload in beam_section.items():
            family = repo.families.get((ComponentType.BEAM, family_id))
            if not family:
                continue
            props = payload.get("properties", {})
            family.capabilities.can_be_primary = props.get("can_be_primary")
            family.capabilities.can_be_secondary = props.get("can_be_secondary")
            family.capabilities.can_be_edge_beam = props.get("can_be_edge_beam")
            family.metadata.setdefault("compatibility", {}).update(payload)

        for component_name in ("column", "lateral"):
            component = ComponentType(component_name)
            section = families_section.get(component_name, {})
            for family_id, payload in section.items():
                family = repo.families.get((component, family_id))
                if family:
                    family.metadata.setdefault("compatibility", {}).update(payload)

    def _load_occupancies(self, rows: list[dict[str, Any]]) -> dict[str, Occupancy]:
        result: dict[str, Occupancy] = {}
        for row in rows:
            occupancy_id = self._pick_first(row, ["occupancy_id", "id", "occupancy", "use"])
            if not occupancy_id:
                raise DomainError(f"Missing occupancy id: {row}")
            result[str(occupancy_id)] = Occupancy(
                occupancy_id=str(occupancy_id),
                code_family=row.get("code_family"),
                category=row.get("category"),
                description=row.get("description"),
                load_values={
                    str(k): float(v) for k, v in row.items()
                    if isinstance(v, (int, float)) and k not in {"storey_count"}
                },
                reduction_group=row.get("reduction_group"),
                metadata=dict(row),
            )
        return result

    def _load_material_sets(self, data: dict[str, Any]) -> dict[str, MaterialSet]:
        result: dict[str, MaterialSet] = {}
        for material_set_id, payload in data.get("material_sets", {}).items():
            result[str(material_set_id)] = MaterialSet(
                material_set_id=str(material_set_id),
                concrete=self._material_ref_from_payload(payload.get("concrete")),
                rebar=self._material_ref_from_payload(payload.get("rebar")),
                steel=self._material_ref_from_payload(payload.get("steel")),
                timber=self._material_ref_from_payload(payload.get("timber")),
                alternatives=self._material_alternatives_from_payload(payload),
            )
        return result

    def _load_material_mapping_rules(self, data: dict[str, Any]) -> list[MaterialMappingRule]:
        return [
            MaterialMappingRule(when=rule.get("when", {}), use_material_set=str(rule["use_material_set"]))
            for rule in data.get("rules", [])
        ]

    def _load_preference_rules(self, data: dict[str, Any]) -> list[SystemPreferenceRule]:
        rules: list[SystemPreferenceRule] = []
        for rule in data.get("rules", []):
            rules.append(
                SystemPreferenceRule(
                    name=str(rule["name"]),
                    when=rule.get("when", {}),
                    filters=rule.get("filters", {}),
                    rank=[RankingField(field=str(item["field"]), direction=str(item["direction"])) for item in rule.get("rank", [])],
                )
            )
        return rules

    def _load_compatibility_rules(self, data: dict[str, Any]) -> list[CompatibilityRule]:
        rules: list[CompatibilityRule] = []
        for entry in data.get("global_rules", {}).get("hard_disallow", []):
            rules.append(
                CompatibilityRule(
                    severity=RuleSeverity.HARD_DISALLOW,
                    target=self._compatibility_target_from_entry(entry),
                    reason=entry.get("reason"),
                )
            )
        for entry in data.get("global_rules", {}).get("soft_discourage", []):
            rules.append(
                CompatibilityRule(
                    severity=RuleSeverity.SOFT_DISCOURAGE,
                    target=self._compatibility_target_from_entry(entry),
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
                    if not isinstance(combo_payload, dict):
                        continue
                    expressions: list[LoadCombinationExpression] = []
                    if "expression" in combo_payload:
                        expressions.append(self._parse_expression(combo_payload["expression"]))
                    for alt in combo_payload.get("alternatives", []):
                        expressions.append(self._parse_expression(alt.get("expression", [])))
                    combinations.append(
                        LoadCombinationTemplate(
                            code_family=str(code_family),
                            design_basis=str(design_basis),
                            combination_id=str(combination_id),
                            description=combo_payload.get("description"),
                            expressions=expressions,
                        )
                    )
        return combinations

    def _parse_expression(self, rows: list[dict[str, Any]]) -> LoadCombinationExpression:
        return LoadCombinationExpression(actions=[LoadAction(action=str(row["action"]), factor=row["factor"]) for row in rows])

    def _compatibility_target_from_entry(self, entry: dict[str, Any]) -> CompatibilityTarget:
        return CompatibilityTarget(
            floor_family=entry.get("floor_family"),
            primary_beam_family=entry.get("primary_beam_family"),
            secondary_beam_family=entry.get("secondary_beam_family"),
            beam_family=entry.get("beam_family"),
            column_family=entry.get("column_family"),
            lateral_family=entry.get("lateral_family"),
        )

    def _material_ref_from_payload(self, payload: dict[str, Any] | None) -> MaterialRef | None:
        if not payload or not payload.get("default"):
            return None
        return MaterialRef(material_id=str(payload["default"]))

    def _material_alternatives_from_payload(self, payload: dict[str, Any]) -> dict[str, list[MaterialRef]]:
        result: dict[str, list[MaterialRef]] = {}
        for key, value in payload.items():
            alts = value.get("alternatives", []) if isinstance(value, dict) else []
            result[str(key)] = [MaterialRef(material_id=str(item)) for item in alts]
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


class RepositoryLoader:
    def load_bundle_from_paths(self, **paths: str | Path | None) -> SourceBundle:
        bundle = SourceBundle()
        for name, value in paths.items():
            if value is None:
                continue
            path = Path(value)
            if not path.exists():
                raise RepositoryError(f"Path does not exist for '{name}': {path}")
            loaded = self._load_path(path)
            setattr(bundle, name, loaded)
        return bundle

    def _load_path(self, path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if suffix == ".csv":
            return pd.read_csv(path).to_dict(orient="records")
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path).to_dict(orient="records")
        raise RepositoryError(f"Unsupported file type: {path}")


class RepositoryQueryService:
    def __init__(self, repo: DomainRepository):
        self.repo = repo

    def get_family(self, component: ComponentType, family_id: str) -> SystemFamily:
        try:
            return self.repo.families[(component, family_id)]
        except KeyError as exc:
            raise RepositoryError(f"Unknown {component.value} family '{family_id}'") from exc

    def get_variant(self, component: ComponentType, variant_id: str) -> SystemVariant:
        try:
            return self.repo.variants[(component, variant_id)]
        except KeyError as exc:
            raise RepositoryError(f"Unknown {component.value} variant '{variant_id}'") from exc

    def get_variants_for_family(self, component: ComponentType, family_id: str) -> list[SystemVariant]:
        return [variant for (comp, _), variant in self.repo.variants.items() if comp == component and variant.family_id == family_id]

    def get_typology(self, typology_id: str) -> Typology:
        try:
            return self.repo.typologies[typology_id]
        except KeyError as exc:
            raise RepositoryError(f"Unknown typology '{typology_id}'") from exc

    def resolve_material_set(self, region: str | None = None, code_family: str | None = None, typology_id: str | None = None) -> MaterialSet | None:
        for rule in self.repo.material_mapping_rules:
            when = rule.when
            if region is not None and when.get("region") not in (None, region):
                continue
            if code_family is not None and when.get("code_family") not in (None, code_family):
                continue
            if typology_id is not None and when.get("typology") not in (None, typology_id):
                continue
            return self.repo.material_sets.get(rule.use_material_set)
        return None

    def get_preference_rules(self, *, typology_id: str | None = None, component: str | None = None, role: str | None = None) -> list[SystemPreferenceRule]:
        rules = self.repo.preference_rules
        if typology_id is not None:
            rules = [r for r in rules if r.when.get("typology") == typology_id]
        if component is not None:
            rules = [r for r in rules if r.when.get("component") == component]
        if role is not None:
            rules = [r for r in rules if r.when.get("role") == role]
        return rules

    def get_compatibility_rules(self, severity: RuleSeverity | None = None) -> list[CompatibilityRule]:
        if severity is None:
            return list(self.repo.compatibility_rules)
        return [rule for rule in self.repo.compatibility_rules if rule.severity == severity]

    def iter_families(self, component: ComponentType | None = None) -> Iterable[SystemFamily]:
        for (comp, _), family in self.repo.families.items():
            if component is None or comp == component:
                yield family
