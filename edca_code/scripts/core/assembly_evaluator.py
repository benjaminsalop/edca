from __future__ import annotations

from dataclasses import replace

from .analysis_models import AssemblyAnalysisInput
from .design_results import AssemblyDesignResult, ComponentDesignResult, ResultWarning
from .domain_models import AssemblyCandidate, BeamRole, ComponentType, ProjectContext
from .exceptions import AssemblyEvaluationError
from .repositories import RepositoryQueryService


class AssemblyEvaluator:
    """Coordinator for converting an assembly candidate into design results.

    This is a skeleton orchestrator. It does not perform code checks itself;
    instead it wires together demand generation and downstream component design
    hooks that the existing code-check package can plug into later.
    """

    def __init__(self, query: RepositoryQueryService, load_engine=None, component_designers: dict[str, object] | None = None):
        self.query = query
        self.load_engine = load_engine
        self.component_designers = component_designers or {}

    def evaluate(self, project: ProjectContext, candidate: AssemblyCandidate) -> AssemblyDesignResult:
        analysis_input = self.build_analysis_input(project, candidate)
        result = AssemblyDesignResult(candidate_id=candidate.candidate_id, metadata={"candidate": candidate.metadata})

        result.floor = self._evaluate_component("floor", candidate.floor_family_id, candidate.floor_variant_id, analysis_input.floor)
        result.primary_beam = self._evaluate_component("primary_beam", candidate.primary_beam_family_id, candidate.primary_beam_variant_id, analysis_input.primary_beam)
        result.secondary_beam = self._evaluate_component("secondary_beam", candidate.secondary_beam_family_id, candidate.secondary_beam_variant_id, analysis_input.secondary_beam)
        result.column = self._evaluate_component("column", candidate.column_family_id, candidate.column_variant_id, analysis_input.column)
        result.lateral = self._evaluate_component("lateral", candidate.lateral_family_id, candidate.lateral_variant_id, analysis_input.lateral)

        if analysis_input.torsion is not None:
            result.warnings.append(
                ResultWarning(
                    code="TORSION_PRESENT",
                    message="Assembly includes torsional demand placeholder that should be checked by the lateral design module.",
                    metadata={"source": analysis_input.torsion.source},
                )
            )

        result.compute_totals()
        result.compute_passed()
        return result

    def build_analysis_input(self, project: ProjectContext, candidate: AssemblyCandidate) -> AssemblyAnalysisInput:
        if self.load_engine is None:
            return AssemblyAnalysisInput(
                candidate_id=candidate.candidate_id,
                load_path_method=candidate.load_path_method,
                metadata={"status": "placeholder", "reason": "No load engine provided"},
            )
        analysis_input = self.load_engine.build_analysis_input(project=project, candidate=candidate)
        if analysis_input is None:
            raise AssemblyEvaluationError(
                f"Load engine returned no analysis input for candidate '{candidate.candidate_id}'"
            )
        return analysis_input

    def _evaluate_component(self, component_key: str, family_id: str | None, variant_id: str | None, demand) -> ComponentDesignResult | None:
        if family_id is None:
            return None
        designer = self.component_designers.get(component_key)
        if designer is None:
            return ComponentDesignResult(
                component=component_key,
                family_id=family_id,
                variant_id=variant_id,
                passed=False,
                warnings=[
                    ResultWarning(
                        code="DESIGNER_NOT_ATTACHED",
                        message=f"No designer attached for component '{component_key}'.",
                    )
                ],
                metadata={"demand_present": demand is not None},
            )
        return designer.design(family_id=family_id, variant_id=variant_id, demand=demand)
