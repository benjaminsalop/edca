from __future__ import annotations

import logging
from typing import Iterable

from .domain_models import ComponentType, DomainRepository, RuleSeverity
from .exceptions import ValidationError

logger = logging.getLogger(__name__)

VALID_COMPONENT_STRINGS = {component.value for component in ComponentType}


def validate_repository(repo: DomainRepository) -> None:
    errors: list[str] = []
    errors.extend(_validate_variant_family_refs(repo))
    errors.extend(_validate_load_combinations(repo))
    errors.extend(_validate_preference_rules(repo))
    if errors:
        raise ValidationError("Repository validation failed:\n- " + "\n- ".join(errors))

    # These generate warnings only — typologies and compat rules use generic
    # type names (e.g. "flat_slab") that may not match specific catalog IDs.
    for msg in _validate_typology_defaults(repo):
        logger.debug("[validation] %s", msg)
    for msg in _validate_compatibility_rules(repo):
        logger.debug("[validation] %s", msg)
    for msg in _validate_material_sets(repo):
        logger.debug("[validation] %s", msg)


def _family_id_or_type_exists(repo: DomainRepository, component: ComponentType, ref_id: str) -> bool:
    """True if ref_id matches any family_id or type_name for the component."""
    if (component, ref_id) in repo.families:
        return True
    return any(
        f.type_name == ref_id
        for (comp, _), f in repo.families.items()
        if comp == component
    )


def _validate_typology_defaults(repo: DomainRepository) -> list[str]:
    warnings: list[str] = []
    default_map = {
        "floor_system_family": ComponentType.FLOOR,
        "primary_beam_system_family": ComponentType.BEAM,
        "secondary_beam_system_family": ComponentType.BEAM,
        "column_system_family": ComponentType.COLUMN,
        "lateral_system_family": ComponentType.LATERAL,
    }
    for typology in repo.typologies.values():
        for key, component in default_map.items():
            family_id = typology.defaults.get(key)
            if family_id is None:
                continue
            if not _family_id_or_type_exists(repo, component, family_id):
                warnings.append(
                    f"Typology '{typology.typology_id}' references unknown {component.value} "
                    f"family/type '{family_id}' (no catalog entry found — typology default will be ignored)"
                )
    return warnings


def _validate_variant_family_refs(repo: DomainRepository) -> list[str]:
    errors: list[str] = []
    for (component, variant_id), variant in repo.variants.items():
        if (component, variant.family_id) not in repo.families:
            errors.append(
                f"Variant '{variant_id}' ({component.value}) references unknown family '{variant.family_id}'"
            )
    return errors


def _validate_material_sets(repo: DomainRepository) -> list[str]:
    errors: list[str] = []
    material_ids = {str(row.get("material_id")) for row in repo.config.get("materials", []) if row.get("material_id") is not None}
    if not material_ids:
        return errors
    for material_set in repo.material_sets.values():
        refs = [material_set.concrete, material_set.rebar, material_set.steel, material_set.timber]
        for ref in refs:
            if ref and ref.material_id not in material_ids:
                errors.append(
                    f"Material set '{material_set.material_set_id}' references unknown material_id '{ref.material_id}'"
                )
        for category, alternatives in material_set.alternatives.items():
            for ref in alternatives:
                if ref.material_id not in material_ids:
                    errors.append(
                        f"Material set '{material_set.material_set_id}' alternative '{category}' references unknown material_id '{ref.material_id}'"
                    )
    return errors


def _validate_compatibility_rules(repo: DomainRepository) -> list[str]:
    warnings: list[str] = []
    family_fields = {
        "floor_family": ComponentType.FLOOR,
        "primary_beam_family": ComponentType.BEAM,
        "secondary_beam_family": ComponentType.BEAM,
        "beam_family": ComponentType.BEAM,
        "column_family": ComponentType.COLUMN,
        "lateral_family": ComponentType.LATERAL,
    }
    for rule in repo.compatibility_rules:
        for field_name, component in family_fields.items():
            family_id = getattr(rule.target, field_name)
            if family_id is None:
                continue
            if not _family_id_or_type_exists(repo, component, family_id):
                warnings.append(
                    f"Compatibility rule references unknown {component.value} family '{family_id}' via '{field_name}'"
                )
    return warnings


def _validate_load_combinations(repo: DomainRepository) -> list[str]:
    errors: list[str] = []
    for combo in repo.load_combinations:
        if not combo.combination_id:
            errors.append("Load combination missing combination_id")
        if not combo.code_family:
            errors.append(f"Load combination '{combo.combination_id}' missing code_family")
        if not combo.design_basis:
            errors.append(f"Load combination '{combo.combination_id}' missing design_basis")
        if not combo.expressions:
            errors.append(f"Load combination '{combo.combination_id}' has no expressions")
        for expression in combo.expressions:
            for action in expression.actions:
                if not action.action:
                    errors.append(f"Load combination '{combo.combination_id}' contains blank action name")
    return errors


def _validate_preference_rules(repo: DomainRepository) -> list[str]:
    errors: list[str] = []
    for rule in repo.preference_rules:
        typology_id = rule.when.get("typology")
        if typology_id and typology_id not in repo.typologies:
            errors.append(f"Preference rule '{rule.name}' references unknown typology '{typology_id}'")
        component = rule.when.get("component")
        if component and component not in VALID_COMPONENT_STRINGS:
            errors.append(f"Preference rule '{rule.name}' references invalid component '{component}'")
        for rank_item in rule.rank:
            if rank_item.direction not in {"asc", "desc"}:
                errors.append(
                    f"Preference rule '{rule.name}' has invalid rank direction '{rank_item.direction}'"
                )
    return errors


def iter_repository_errors(repo: DomainRepository) -> Iterable[str]:
    try:
        validate_repository(repo)
    except ValidationError as exc:
        text = str(exc)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                yield line[2:]
