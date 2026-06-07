from __future__ import annotations

from dataclasses import replace
from itertools import islice
from typing import Iterable

from .domain_models import (
    AssemblyCandidate,
    AssemblyPenalty,
    ComponentType,
    LoadPathMethod,
    ProjectContext,
    SystemFamily,
)
from .exceptions import RepositoryError
from .repositories import RepositoryQueryService


class CandidateGenerator:
    """Generate assembly-level candidates from normalized repository data.

    This is intentionally family-first. Variant selection happens later in the
    evaluation flow unless the caller explicitly asks for representative variants.
    """

    def __init__(self, query: RepositoryQueryService):
        self.query = query
        self.repo = query.repo

    def generate(self, project: ProjectContext) -> list[AssemblyCandidate]:
        typology = self.query.get_typology(project.typology_id) if project.typology_id else None
        floor_families = self._select_floor_families(project)

        generated: list[AssemblyCandidate] = []
        for floor_family in floor_families:
            load_path_method, primary_required, secondary_required = self._resolve_floor_framing(floor_family)

            primary_beam_families = self._select_beam_families(
                project,
                role="primary",
                floor_family=floor_family,
                required=primary_required,
            )
            secondary_beam_families = self._select_beam_families(
                project,
                role="secondary",
                floor_family=floor_family,
                required=secondary_required,
            )
            column_families = self._select_column_families(project, floor_family=floor_family)
            lateral_families = self._select_lateral_families(project, floor_family=floor_family)

            combos = self._candidate_iterables(
                floor_family=floor_family,
                primary_beam_families=primary_beam_families,
                secondary_beam_families=secondary_beam_families,
                column_families=column_families,
                lateral_families=lateral_families,
                load_path_method=load_path_method,
            )
            generated.extend(combos)

        deduped = self._deduplicate_candidates(generated)
        limit = self.repo.config.get("assembly_generation", {}).get("pruning", {}).get("max_candidates_per_typology")
        if isinstance(limit, int) and limit > 0:
            deduped = list(islice(deduped, limit))

        if not deduped and typology is not None:
            raise RepositoryError(
                f"No assembly candidates generated for typology '{typology.typology_id}'"
            )
        return deduped

    def _select_floor_families(self, project: ProjectContext) -> list[SystemFamily]:
        families = list(self.query.iter_families(ComponentType.FLOOR))
        typology = self.query.get_typology(project.typology_id) if project.typology_id else None

        families = [f for f in families if self._matches_allowed_materials(f, typology)]
        families = [f for f in families if self._matches_typology_constraints(project, typology, f)]

        default_floor_family = typology.defaults.get("floor_system_family") if typology else None
        if default_floor_family:
            families = self._prioritize_default_family(families, default_floor_family)
        return families

    def _select_beam_families(
        self,
        project: ProjectContext,
        *,
        role: str,
        floor_family: SystemFamily,
        required: bool,
    ) -> list[SystemFamily | None]:
        if not required:
            return [None]

        families = list(self.query.iter_families(ComponentType.BEAM))
        typology = self.query.get_typology(project.typology_id) if project.typology_id else None
        compatible_ids = set(
            floor_family.metadata.get("compatibility", {})
            .get("compatible_with", {})
            .get(f"{role}_beam_families", [])
        )
        disallowed_ids = set(
            floor_family.metadata.get("compatibility", {})
            .get("disallow", {})
            .get(f"{role}_beam_families", [])
        )

        result = []
        for family in families:
            if compatible_ids and family.family_id not in compatible_ids:
                continue
            if family.family_id in disallowed_ids:
                continue
            if role == "primary" and family.capabilities.can_be_primary is False:
                continue
            if role == "secondary" and family.capabilities.can_be_secondary is False:
                continue
            if not self._matches_allowed_materials(family, typology):
                continue
            result.append(family)
        return result

    def _select_column_families(self, project: ProjectContext, *, floor_family: SystemFamily) -> list[SystemFamily]:
        families = list(self.query.iter_families(ComponentType.COLUMN))
        typology = self.query.get_typology(project.typology_id) if project.typology_id else None
        compatible_ids = set(
            floor_family.metadata.get("compatibility", {})
            .get("compatible_with", {})
            .get("column_families", [])
        )
        disallowed_ids = set(
            floor_family.metadata.get("compatibility", {})
            .get("disallow", {})
            .get("column_families", [])
        )

        result = []
        for family in families:
            if compatible_ids and family.family_id not in compatible_ids:
                continue
            if family.family_id in disallowed_ids:
                continue
            if not self._matches_allowed_materials(family, typology):
                continue
            result.append(family)
        return result

    def _select_lateral_families(self, project: ProjectContext, *, floor_family: SystemFamily) -> list[SystemFamily]:
        families = list(self.query.iter_families(ComponentType.LATERAL))
        typology = self.query.get_typology(project.typology_id) if project.typology_id else None
        compatible_ids = set(
            floor_family.metadata.get("compatibility", {})
            .get("compatible_with", {})
            .get("lateral_families", [])
        )
        result = []
        for family in families:
            if compatible_ids and family.family_id not in compatible_ids:
                continue
            if not self._matches_allowed_materials(family, typology):
                continue
            result.append(family)
        return result

    def _candidate_iterables(
        self,
        *,
        floor_family: SystemFamily,
        primary_beam_families: list[SystemFamily | None],
        secondary_beam_families: list[SystemFamily | None],
        column_families: list[SystemFamily],
        lateral_families: list[SystemFamily],
        load_path_method: LoadPathMethod,
    ) -> list[AssemblyCandidate]:
        candidates: list[AssemblyCandidate] = []
        for primary in primary_beam_families:
            for secondary in secondary_beam_families:
                for column in column_families:
                    for lateral in lateral_families:
                        candidate = AssemblyCandidate(
                            candidate_id=self._build_candidate_id(
                                floor_family_id=floor_family.family_id,
                                primary_beam_family_id=primary.family_id if primary else None,
                                secondary_beam_family_id=secondary.family_id if secondary else None,
                                column_family_id=column.family_id,
                                lateral_family_id=lateral.family_id,
                            ),
                            typology_id=None,
                            floor_family_id=floor_family.family_id,
                            primary_beam_family_id=primary.family_id if primary else None,
                            secondary_beam_family_id=secondary.family_id if secondary else None,
                            column_family_id=column.family_id,
                            lateral_family_id=lateral.family_id,
                            load_path_method=load_path_method,
                            material_mix_label=self._infer_material_mix(
                                floor_material=floor_family.material_family,
                                primary_material=primary.material_family if primary else None,
                                column_material=column.material_family,
                            ),
                            metadata={
                                "generated_from": "CandidateGenerator",
                                "floor_defaults": dict(floor_family.defaults),
                            },
                        )
                        candidates.append(candidate)
        return candidates

    def _resolve_floor_framing(self, floor_family: SystemFamily) -> tuple[LoadPathMethod, bool, bool]:
        caps = floor_family.capabilities
        if caps.can_beamless and not caps.requires_primary_beams and not caps.requires_secondary_beams:
            return LoadPathMethod.BEAMLESS, False, False
        if caps.requires_primary_beams and not caps.requires_secondary_beams:
            return LoadPathMethod.ONE_WAY_WITH_PRIMARY_ONLY, True, False
        if caps.requires_primary_beams and caps.requires_secondary_beams:
            return LoadPathMethod.TWO_WAY_WITH_SECONDARY_AND_PRIMARY, True, True
        return LoadPathMethod.CUSTOM, bool(caps.requires_primary_beams), bool(caps.requires_secondary_beams)

    def _matches_allowed_materials(self, family: SystemFamily, typology) -> bool:
        if typology is None:
            return True
        allowed = typology.constraints.allowed_material_families
        if not allowed or family.material_family is None:
            return True
        return family.material_family in allowed

    def _matches_typology_constraints(self, project: ProjectContext, typology, family: SystemFamily) -> bool:
        if typology is None:
            return True
        constraints = typology.constraints
        storey_count = project.geometry.storey_count
        if constraints.max_storey_count is not None and storey_count is not None and storey_count > constraints.max_storey_count:
            return False
        return True

    def _prioritize_default_family(self, families: list[SystemFamily], default_family_id: str) -> list[SystemFamily]:
        def _is_default(f: SystemFamily) -> int:
            return 0 if (f.family_id == default_family_id or f.type_name == default_family_id) else 1
        return sorted(families, key=_is_default)

    def _build_candidate_id(
        self,
        *,
        floor_family_id: str,
        primary_beam_family_id: str | None,
        secondary_beam_family_id: str | None,
        column_family_id: str | None,
        lateral_family_id: str | None,
    ) -> str:
        bits = [
            floor_family_id or "null",
            primary_beam_family_id or "null",
            secondary_beam_family_id or "null",
            column_family_id or "null",
            lateral_family_id or "null",
        ]
        return "__".join(bits)

    def _infer_material_mix(
        self,
        *,
        floor_material: str | None,
        primary_material: str | None,
        column_material: str | None,
    ) -> str | None:
        labels = self.repo.config.get("assembly_generation", {}).get("material_mix_labels", {})
        for label, definition in labels.items():
            if not isinstance(definition, dict):
                continue
            if definition.get("floor_material_family") not in (None, floor_material):
                continue
            if definition.get("primary_beam_material_family") not in (None, primary_material):
                continue
            if definition.get("column_material_family") not in (None, column_material):
                continue
            return label
        families = {floor_material, primary_material, column_material} - {None}
        if len(families) == 1:
            return next(iter(families)) + "_only"
        if len(families) > 1:
            return "hybrid"
        return None

    def _deduplicate_candidates(self, candidates: Iterable[AssemblyCandidate]) -> list[AssemblyCandidate]:
        seen: set[str] = set()
        deduped: list[AssemblyCandidate] = []
        for candidate in candidates:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            deduped.append(candidate)
        return deduped
