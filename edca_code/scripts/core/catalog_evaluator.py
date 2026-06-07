from __future__ import annotations

from typing import Any

from .assembly_takeoff import AssemblyTakeoffEngine
from .carbon import CarbonCostEngine
from .design_results import AssemblyDesignResult, ComponentDesignResult
from .domain_models import AssemblyCandidate, ComponentType, ProjectContext, SystemFamily
from .repositories import RepositoryQueryService
from .structural_class import infer_structural_class
from .variant_selector import VariantExpander


class CatalogEvaluator:
    """Screening-mode evaluator: variant expansion + catalog takeoff + carbon.

    This does NOT perform structural code checks.  Instead it uses the
    pre-computed material volume fields from the variant catalogs (m³/m²,
    m³/linear-m) to estimate embodied carbon and cost for every viable
    variant combination at a given bay span.

    Candidates are always marked passed=True because span/storey viability
    is already enforced by VariantExpander before evaluation.

    Typical use
    -----------
    evaluator = CatalogEvaluator(query, carbon_engine)
    for span_x, span_y in span_pairs:
        results = evaluator.evaluate_all(ranked_candidates, project, span_x, span_y)
        # collect into a flat DataFrame with build_results_dataframe()
    """

    def __init__(
        self,
        query: RepositoryQueryService,
        carbon_engine: CarbonCostEngine,
        *,
        max_variants_per_component: int = 20,
    ) -> None:
        self.query = query
        self.expander = VariantExpander(query, max_variants_per_component)
        self.takeoff = AssemblyTakeoffEngine(query)
        self.carbon = carbon_engine

    def evaluate_all(
        self,
        candidates: list[AssemblyCandidate],
        project: ProjectContext,
        span_x_m: float,
        span_y_m: float,
    ) -> list[AssemblyDesignResult]:
        # Pre-compute both factored and unfactored loads; the variant expander
        # uses the factored load (conservative screening for beam/column capacity).
        # Per-category basis only affects which candidates survive span-screening
        # when their catalog capacity field represents a service load.
        factored_load   = self._screening_area_load(project, factored=True)
        unfactored_load = self._screening_area_load(project, factored=False)
        storey_count = project.geometry.storey_count or 1

        primary_line_load   = factored_load * span_y_m if factored_load is not None else None
        secondary_line_load = factored_load * span_x_m if factored_load is not None else None
        column_axial        = factored_load * span_x_m * span_y_m * storey_count if factored_load is not None else None

        # Store both loads in project design_options for downstream use
        _loads_context = {
            "screening_factored_kpa":   factored_load,
            "screening_unfactored_kpa": unfactored_load,
        }

        results: list[AssemblyDesignResult] = []
        for family_candidate in candidates:
            for variant_candidate in self.expander.expand(
                family_candidate, project, span_x_m, span_y_m,
                primary_beam_line_load_kn_per_m=primary_line_load,
                secondary_beam_line_load_kn_per_m=secondary_line_load,
                column_axial_kn=column_axial,
            ):
                results.append(self._evaluate_one(variant_candidate, project, span_x_m, span_y_m))
        return results

    def _screening_area_load(self, project: ProjectContext,
                             factored: bool = True) -> float | None:
        """Return screening area load (kN/m²) from the governing occupancy.

        When ``factored=True`` (ULS):
          EC:   1.35 × (sdl + sdl_partition) + 1.5 × ll
          ASCE: 1.2  × (sdl + sdl_partition) + 1.6 × ll
        When ``factored=False`` (characteristic/service):
          Returns (sdl + sdl_partition + ll) unfactored.

        Structural self-weight is not included (not known at screening time).
        """
        occ_id = project.occupancy_id
        if not occ_id:
            return None
        occ = self.query.repo.occupancies.get(occ_id)
        if occ is None:
            return None
        lv   = occ.load_values or {}
        meta = dict(occ.metadata or {})

        def _get(primary: str, *aliases: str) -> float:
            for key in (primary,) + aliases:
                val = lv.get(key) if lv.get(key) is not None else meta.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            return 0.0

        dead = _get("sdl",           "dead_kpa", "dead")
        part = _get("sdl_partition", "partition_kpa", "partition")
        live = _get("ll",            "live_kpa", "live")

        if not factored:
            return dead + part + live

        code = str(meta.get("code") or "EC").strip().upper()
        if "ASCE" in code:
            return 1.2 * (dead + part) + 1.6 * live
        else:  # Eurocode default
            return 1.35 * (dead + part) + 1.5 * live

    def floor_category_load(self, project: ProjectContext,
                            floor_category: str) -> float | None:
        """Return the appropriate screening load for a given floor_category.

        Looks up LOAD_BASIS_BY_CATEGORY from project.design_options to decide
        whether to return factored or unfactored load.  Defaults:
          precast / timber → unfactored
          cast_in_place / composite / pt / other → factored
        """
        basis_map: dict[str, str] = dict(
            project.design_options.get("load_basis_by_category") or {}
        )
        # Hard defaults if the map is empty
        if not basis_map:
            basis_map = {
                "cast_in_place": "factored",
                "composite":     "factored",
                "pt":            "factored",
                "precast":       "unfactored",
                "timber":        "unfactored",
                "other":         "factored",
            }
        basis = basis_map.get(str(floor_category).lower(), "factored")
        return self._screening_area_load(project, factored=(basis == "factored"))

    # ------------------------------------------------------------------
    def _evaluate_one(
        self,
        candidate: AssemblyCandidate,
        project: ProjectContext,
        span_x_m: float,
        span_y_m: float,
    ) -> AssemblyDesignResult:
        bom = self.takeoff.compute_bom(candidate, project, span_x_m, span_y_m)
        lines = self.carbon.evaluate_bom(bom)

        # Accumulate carbon + cost per component category
        by_cat: dict[str, tuple[float, float]] = {}
        for line in lines:
            ec, cost = by_cat.get(line.category, (0.0, 0.0))
            by_cat[line.category] = (ec + line.embodied_carbon, cost + line.cost)

        structural_class = self._get_structural_class(candidate)
        result = AssemblyDesignResult(
            candidate_id=candidate.candidate_id,
            passed=True,
            floor=_comp("floor", candidate.floor_family_id, candidate.floor_variant_id, by_cat),
            primary_beam=_comp("primary_beam", candidate.primary_beam_family_id, candidate.primary_beam_variant_id, by_cat),
            secondary_beam=_comp("secondary_beam", candidate.secondary_beam_family_id, candidate.secondary_beam_variant_id, by_cat),
            column=_comp("column", candidate.column_family_id, candidate.column_variant_id, by_cat),
            lateral=_comp("lateral", candidate.lateral_family_id, candidate.lateral_variant_id, by_cat),
            metadata={
                **candidate.metadata,
                "span_x_m": span_x_m,
                "span_y_m": span_y_m,
                "bom_lines": len(bom),
                "total_penalty": candidate.total_penalty,
                "structural_class": structural_class,
            },
        )
        result.compute_totals()
        return result


    def _get_structural_class(self, candidate: AssemblyCandidate) -> str:
        def _fam(component: ComponentType, family_id: str | None) -> SystemFamily | None:
            if not family_id:
                return None
            try:
                return self.query.get_family(component, family_id)
            except Exception:
                return None

        floor_fam = _fam(ComponentType.FLOOR, candidate.floor_family_id)
        beam_fam = _fam(ComponentType.BEAM, candidate.primary_beam_family_id)
        col_fam = _fam(ComponentType.COLUMN, candidate.column_family_id)
        return infer_structural_class(floor_fam, beam_fam, col_fam)


def _comp(
    key: str,
    family_id: str | None,
    variant_id: str | None,
    by_cat: dict[str, tuple[float, float]],
) -> ComponentDesignResult | None:
    if family_id is None:
        return None
    ec, cost = by_cat.get(key, (0.0, 0.0))
    return ComponentDesignResult(
        component=key,
        family_id=family_id,
        variant_id=variant_id,
        passed=True,
        embodied_carbon=ec,
        cost=cost,
    )
