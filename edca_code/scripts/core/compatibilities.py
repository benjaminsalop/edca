from __future__ import annotations

from dataclasses import replace
from typing import Any

from .domain_models import (
    AssemblyCandidate,
    AssemblyPenalty,
    ComponentType,
    RankingField,
    RuleSeverity,
    SystemPreferenceRule,
)
from .exceptions import CompatibilityError
from .repositories import RepositoryQueryService


class CompatibilityPreferenceEngine:
    """Apply hard compatibility checks and soft preference scoring."""

    def __init__(self, query: RepositoryQueryService):
        self.query = query
        self.repo = query.repo

    def apply(self, candidates: list[AssemblyCandidate], *, typology_id: str | None = None) -> list[AssemblyCandidate]:
        filtered = self.apply_hard_rules(candidates)
        filtered = self.apply_soft_rules(filtered)
        filtered = self.apply_preference_rules(filtered, typology_id=typology_id)
        filtered.sort(key=self._sort_key)
        return filtered

    def apply_hard_rules(self, candidates: list[AssemblyCandidate]) -> list[AssemblyCandidate]:
        hard_rules = self.query.get_compatibility_rules(RuleSeverity.HARD_DISALLOW)
        accepted: list[AssemblyCandidate] = []
        for candidate in candidates:
            if any(self._matches_rule(candidate, rule.target) for rule in hard_rules):
                continue
            accepted.append(candidate)
        return accepted

    def apply_soft_rules(self, candidates: list[AssemblyCandidate]) -> list[AssemblyCandidate]:
        soft_rules = self.query.get_compatibility_rules(RuleSeverity.SOFT_DISCOURAGE)
        penalty_value = float(
            self.repo.config.get("assembly_generation", {})
            .get("ranking_penalties", {})
            .get("soft_discouraged_pairing", 25.0)
        )
        updated: list[AssemblyCandidate] = []
        for candidate in candidates:
            penalties = list(candidate.penalties)
            for rule in soft_rules:
                if self._matches_rule(candidate, rule.target):
                    penalties.append(
                        AssemblyPenalty(
                            source="compatibility_rule",
                            points=float(rule.penalty or penalty_value),
                            reason=rule.reason,
                        )
                    )
            updated.append(replace(candidate, penalties=penalties))
        return updated

    def apply_preference_rules(
        self,
        candidates: list[AssemblyCandidate],
        *,
        typology_id: str | None,
    ) -> list[AssemblyCandidate]:
        if typology_id is None:
            return candidates

        updated: list[AssemblyCandidate] = []
        for candidate in candidates:
            scored = candidate
            scored = self._apply_component_preferences(scored, typology_id, component="floor")
            scored = self._apply_component_preferences(scored, typology_id, component="beam", role="primary")
            scored = self._apply_component_preferences(scored, typology_id, component="beam", role="secondary")
            scored = self._apply_component_preferences(scored, typology_id, component="column")
            scored = self._apply_component_preferences(scored, typology_id, component="lateral")
            updated.append(scored)
        return updated

    def require_nonempty(self, candidates: list[AssemblyCandidate], *, message: str = "No compatible candidates remain") -> list[AssemblyCandidate]:
        if not candidates:
            raise CompatibilityError(message)
        return candidates

    def _apply_component_preferences(
        self,
        candidate: AssemblyCandidate,
        typology_id: str,
        *,
        component: str,
        role: str | None = None,
    ) -> AssemblyCandidate:
        rules = self.query.get_preference_rules(typology_id=typology_id, component=component, role=role)
        if not rules:
            return candidate

        family = self._family_for_candidate_component(candidate, component, role)
        if family is None:
            return candidate

        penalties = list(candidate.penalties)
        rank_fields: list[dict[str, str]] = list(candidate.metadata.get("rank_fields", []))
        for rule in rules:
            if not self._matches_preference_filters(family.metadata, rule):
                penalties.append(
                    AssemblyPenalty(
                        source=f"preference_filter:{rule.name}",
                        points=50.0,
                        reason=f"Did not satisfy typology preference filter for {component}",
                    )
                )
            rank_fields.extend({"field": field.field, "direction": field.direction} for field in rule.rank)
        metadata = dict(candidate.metadata)
        if rank_fields:
            metadata["rank_fields"] = self._dedupe_rank_fields(rank_fields)
        return replace(candidate, penalties=penalties, metadata=metadata)

    def _family_for_candidate_component(self, candidate: AssemblyCandidate, component: str, role: str | None):
        if component == "floor":
            return self.query.get_family(ComponentType.FLOOR, candidate.floor_family_id)
        if component == "beam" and role == "primary" and candidate.primary_beam_family_id:
            return self.query.get_family(ComponentType.BEAM, candidate.primary_beam_family_id)
        if component == "beam" and role == "secondary" and candidate.secondary_beam_family_id:
            return self.query.get_family(ComponentType.BEAM, candidate.secondary_beam_family_id)
        if component == "column" and candidate.column_family_id:
            return self.query.get_family(ComponentType.COLUMN, candidate.column_family_id)
        if component == "lateral" and candidate.lateral_family_id:
            return self.query.get_family(ComponentType.LATERAL, candidate.lateral_family_id)
        return None

    def _matches_rule(self, candidate: AssemblyCandidate, target) -> bool:
        checks = {
            "floor_family": candidate.floor_family_id,
            "primary_beam_family": candidate.primary_beam_family_id,
            "secondary_beam_family": candidate.secondary_beam_family_id,
            "beam_family": candidate.primary_beam_family_id,
            "column_family": candidate.column_family_id,
            "lateral_family": candidate.lateral_family_id,
        }
        for field_name, actual in checks.items():
            expected = getattr(target, field_name)
            if expected is not None and actual != expected:
                return False
        return True

    def _matches_preference_filters(self, metadata: dict[str, Any], rule: SystemPreferenceRule) -> bool:
        for field_name, expected in rule.filters.items():
            actual = metadata.get(field_name)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _dedupe_rank_fields(self, rank_fields: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        output: list[dict[str, str]] = []
        for item in rank_fields:
            key = (item["field"], item["direction"])
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    def _sort_key(self, candidate: AssemblyCandidate) -> tuple[float, str]:
        return (candidate.total_penalty, candidate.candidate_id)
