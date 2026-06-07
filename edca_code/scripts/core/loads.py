from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis_models import (
    AssemblyAnalysisInput,
    BeamDemand,
    ColumnDemand,
    DemandEnvelope,
    FloorPanelDemand,
    ForceEffects,
    LateralDemand,
    StoreyForces,
    TorsionalDemand,
    WallDemand,
)
from .domain_models import (
    AssemblyCandidate,
    EvaluatedLoadCombination,
    LoadCombinationExpression,
    LoadPathMethod,
    Occupancy,
    ProjectContext,
)
from .exceptions import LoadResolutionError, UnsupportedConfigurationError
from .repositories import RepositoryQueryService


@dataclass(slots=True)
class ResolvedLoadCase:
    occupancy_id: str
    dead_kpa: float
    live_kpa: float
    partition_kpa: float = 0.0
    roof_live_kpa: float = 0.0
    snow_kpa: float = 0.0
    rain_kpa: float = 0.0
    wind_kpa: float = 0.0
    seismic_kpa: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def superimposed_dead_kpa(self) -> float:
        return self.dead_kpa + self.partition_kpa

    @property
    def total_gravity_unfactored_kpa(self) -> float:
        return self.dead_kpa + self.partition_kpa + self.live_kpa


@dataclass(slots=True)
class DistributionParameters:
    span_x_m: float | None = None
    span_y_m: float | None = None
    tributary_width_m: float | None = None
    tributary_area_m2: float | None = None
    storey_count: int | None = None
    floor_to_floor_m: float | None = None
    accidental_torsion_ratio: float = 0.05


class LoadCombinationEngine:
    """Resolves symbolic repository load-combination templates into numeric factors."""

    def __init__(self, query: RepositoryQueryService):
        self.query = query

    def resolve_for_project(self, project: ProjectContext, occupancy: Occupancy) -> list[EvaluatedLoadCombination]:
        code_family = self._canonical_code_family(project.code_family or occupancy.code_family)
        load_values = self.query.repo.config.get("load_values", {}) or {}
        out: list[EvaluatedLoadCombination] = []

        for template in self.query.repo.load_combinations:
            if self._canonical_code_family(template.code_family) != code_family:
                continue
            for i, expression in enumerate(template.expressions):
                out.append(
                    EvaluatedLoadCombination(
                        code_family=code_family,
                        design_basis=template.design_basis,
                        combination_id=template.combination_id,
                        resolved_factors=self._resolve_expression_factors(
                            expression=expression,
                            project=project,
                            occupancy=occupancy,
                            load_values=load_values,
                            code_family=code_family,
                        ),
                        source_expression_index=i,
                    )
                )
        if not out:
            raise LoadResolutionError(
                f"No load combinations available for code family '{code_family}'"
            )
        return out

    def _resolve_expression_factors(
        self,
        *,
        expression: LoadCombinationExpression,
        project: ProjectContext,
        occupancy: Occupancy,
        load_values: dict[str, Any],
        code_family: str,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for action in expression.actions:
            result[action.action] = self._resolve_factor_value(
                action.factor,
                project=project,
                occupancy=occupancy,
                load_values=load_values,
                code_family=code_family,
            )
        return result

    def _resolve_factor_value(
        self,
        factor: float | str,
        *,
        project: ProjectContext,
        occupancy: Occupancy,
        load_values: dict[str, Any],
        code_family: str,
    ) -> float:
        if isinstance(factor, (int, float)):
            return float(factor)

        expr = str(factor).strip()
        if not expr:
            raise LoadResolutionError("Encountered empty load factor expression.")

        occupancy_token = self._occupancy_factor_key(project=project, occupancy=occupancy, code_family=code_family)
        expr = expr.replace("{occupancy}", occupancy_token)
        terms = [t.strip() for t in expr.split("*")]
        value = 1.0
        for term in terms:
            if not term:
                continue
            value *= self._resolve_single_term(term, load_values)
        return float(value)

    def _resolve_single_term(self, term: str, load_values: dict[str, Any]) -> float:
        try:
            return float(term)
        except ValueError:
            pass

        current: Any = load_values
        for part in term.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise LoadResolutionError(f"Could not resolve load factor path '{term}'")
        try:
            return float(current)
        except Exception as exc:
            raise LoadResolutionError(f"Resolved non-numeric load factor '{term}' -> {current!r}") from exc

    def _canonical_code_family(self, value: str | None) -> str:
        token = str(value or "EC").strip().upper()
        if token in {"EC", "EN", "EUROCODE", "EN1990"}:
            return "EN1990"
        if token in {"ASCE", "ASCE7", "ASD", "LRFD"}:
            return "ASCE7"
        return token

    def _occupancy_factor_key(self, *, project: ProjectContext, occupancy: Occupancy, code_family: str) -> str:
        explicit = occupancy.reduction_group or occupancy.metadata.get("reduction_group")
        if explicit:
            return str(explicit)

        base = (
            occupancy.metadata.get("load_category")
            or occupancy.category
            or occupancy.occupancy_id
            or project.occupancy_id
            or "office"
        )
        base_token = str(base).strip().lower().replace(" ", "_")
        if code_family == "EN1990":
            if not base_token.endswith("_ec"):
                base_token = f"{base_token}_EC"
            return base_token
        return base_token.upper()


class LoadDistributionEngine:
    """Converts area loads and lateral placeholders into member-level demand objects."""

    def build_analysis_input(
        self,
        *,
        project: ProjectContext,
        candidate: AssemblyCandidate,
        resolved_case: ResolvedLoadCase,
        load_combinations: list[EvaluatedLoadCombination],
    ) -> AssemblyAnalysisInput:
        params = self._distribution_parameters(project)
        governing_gravity = self._governing_gravity_combo(load_combinations, resolved_case)
        governing_lateral = self._governing_lateral_combo(load_combinations, resolved_case)

        floor = self._build_floor_demand(candidate, resolved_case, params, governing_gravity)
        primary_beam = self._build_primary_beam_demand(candidate, resolved_case, params, governing_gravity)
        secondary_beam = self._build_secondary_beam_demand(candidate, resolved_case, params, governing_gravity)
        column = self._build_column_demand(candidate, resolved_case, params, governing_gravity)
        wall = self._build_wall_demand(candidate, resolved_case, params, governing_gravity)
        lateral = self._build_lateral_demand(candidate, resolved_case, params, governing_lateral)
        torsion = self._build_torsional_demand(candidate, resolved_case, params, lateral)

        return AssemblyAnalysisInput(
            candidate_id=candidate.candidate_id,
            load_path_method=candidate.load_path_method,
            load_combinations=load_combinations,
            floor=floor,
            primary_beam=primary_beam,
            secondary_beam=secondary_beam,
            column=column,
            wall=wall,
            lateral=lateral,
            torsion=torsion,
            metadata={
                "occupancy_id": resolved_case.occupancy_id,
                "distribution_parameters": params.__dict__,
            },
        )

    def _distribution_parameters(self, project: ProjectContext) -> DistributionParameters:
        span_x = project.geometry.span_x_m
        span_y = project.geometry.span_y_m
        tributary_width = min(v for v in [span_x, span_y] if v is not None) if any(v is not None for v in [span_x, span_y]) else None
        tributary_area = None
        if span_x is not None and span_y is not None:
            tributary_area = span_x * span_y
        elif project.geometry.bay_area_m2 is not None:
            tributary_area = project.geometry.bay_area_m2

        ratio = float(project.design_options.get("accidental_torsion_ratio", 0.05) or 0.05)
        return DistributionParameters(
            span_x_m=span_x,
            span_y_m=span_y,
            tributary_width_m=tributary_width,
            tributary_area_m2=tributary_area,
            storey_count=project.geometry.storey_count,
            floor_to_floor_m=project.geometry.floor_to_floor_m,
            accidental_torsion_ratio=ratio,
        )

    def _governing_gravity_combo(
        self,
        load_combinations: list[EvaluatedLoadCombination],
        resolved_case: ResolvedLoadCase,
    ) -> tuple[EvaluatedLoadCombination | None, float]:
        best_combo = None
        best_load = 0.0
        for combo in load_combinations:
            total = self._gravity_area_load_for_combo(combo, resolved_case)
            if total >= best_load:
                best_combo = combo
                best_load = total
        return best_combo, best_load

    def _governing_lateral_combo(
        self,
        load_combinations: list[EvaluatedLoadCombination],
        resolved_case: ResolvedLoadCase,
    ) -> tuple[EvaluatedLoadCombination | None, float]:
        best_combo = None
        best_pressure = 0.0
        for combo in load_combinations:
            total = self._lateral_pressure_for_combo(combo, resolved_case)
            if total >= best_pressure:
                best_combo = combo
                best_pressure = total
        return best_combo, best_pressure

    def _gravity_area_load_for_combo(self, combo: EvaluatedLoadCombination, case: ResolvedLoadCase) -> float:
        f = combo.resolved_factors
        dead = f.get("G", f.get("G_unf", f.get("D", 0.0))) * case.dead_kpa
        dead += f.get("G", f.get("G_unf", f.get("D", 0.0))) * case.partition_kpa
        live = f.get("Q", f.get("Q_lead", f.get("L", 0.0))) * case.live_kpa
        live += f.get("Lr", 0.0) * case.roof_live_kpa
        live += f.get("S", 0.0) * case.snow_kpa
        live += f.get("R", 0.0) * case.rain_kpa
        return float(dead + live)

    def _lateral_pressure_for_combo(self, combo: EvaluatedLoadCombination, case: ResolvedLoadCase) -> float:
        f = combo.resolved_factors
        wind = f.get("W", 0.0) * case.wind_kpa + f.get("Wt", 0.0) * case.wind_kpa
        seismic = f.get("E", 0.0) * case.seismic_kpa
        return float(wind + seismic)

    def _build_floor_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> FloorPanelDemand:
        combo, factored = governing
        return FloorPanelDemand(
            tributary_area_m2=params.tributary_area_m2,
            unfactored_area_load_kpa=case.total_gravity_unfactored_kpa,
            factored_area_load_kpa=factored,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(),
                metadata={"component": "floor", "occupancy_id": case.occupancy_id},
            ),
        )

    def _framing_mode(self, candidate: AssemblyCandidate) -> LoadPathMethod:
        mode = candidate.load_path_method
        if mode != LoadPathMethod.CUSTOM:
            return mode
        if candidate.primary_beam_family_id is None:
            return LoadPathMethod.BEAMLESS
        if candidate.secondary_beam_family_id is None:
            return LoadPathMethod.ONE_WAY_WITH_PRIMARY_ONLY
        return LoadPathMethod.TWO_WAY_WITH_SECONDARY_AND_PRIMARY

    def _primary_beam_tributary_width(self, candidate: AssemblyCandidate, params: DistributionParameters) -> float | None:
        mode = self._framing_mode(candidate)
        if mode == LoadPathMethod.BEAMLESS:
            return None
        if mode == LoadPathMethod.ONE_WAY_WITH_PRIMARY_ONLY:
            return params.span_y_m or params.tributary_width_m
        if mode == LoadPathMethod.TWO_WAY_WITH_SECONDARY_AND_PRIMARY:
            return (params.span_y_m / 2.0) if params.span_y_m is not None else params.tributary_width_m
        return params.tributary_width_m

    def _secondary_beam_tributary_width(self, candidate: AssemblyCandidate, params: DistributionParameters) -> float | None:
        mode = self._framing_mode(candidate)
        if mode != LoadPathMethod.TWO_WAY_WITH_SECONDARY_AND_PRIMARY:
            return None
        return params.span_x_m or params.tributary_width_m

    def _column_tributary_area(self, params: DistributionParameters) -> float | None:
        if params.span_x_m is not None and params.span_y_m is not None:
            return params.span_x_m * params.span_y_m
        return params.tributary_area_m2

    def _build_primary_beam_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> BeamDemand | None:
        if candidate.primary_beam_family_id is None:
            return None
        combo, factored_area = governing
        tributary_width = self._primary_beam_tributary_width(candidate, params)
        line_load = factored_area * tributary_width if tributary_width is not None else None
        span = params.span_x_m
        moment = line_load * span**2 / 8.0 if line_load is not None and span is not None else None
        shear = line_load * span / 2.0 if line_load is not None and span is not None else None
        return BeamDemand(
            role="primary",
            span_m=span,
            tributary_width_m=tributary_width,
            unfactored_line_load_kn_per_m=(case.total_gravity_unfactored_kpa * tributary_width if tributary_width is not None else None),
            factored_line_load_kn_per_m=line_load,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(moment_major=moment, shear_major=shear),
                metadata={"component": "primary_beam", "framing_mode": self._framing_mode(candidate).value},
            ),
        )

    def _build_secondary_beam_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> BeamDemand | None:
        if candidate.secondary_beam_family_id is None:
            return None
        combo, factored_area = governing
        tributary_width = self._secondary_beam_tributary_width(candidate, params)
        line_load = factored_area * tributary_width if tributary_width is not None else None
        span = params.span_y_m
        moment = line_load * span**2 / 8.0 if line_load is not None and span is not None else None
        shear = line_load * span / 2.0 if line_load is not None and span is not None else None
        return BeamDemand(
            role="secondary",
            span_m=span,
            tributary_width_m=tributary_width,
            unfactored_line_load_kn_per_m=(case.total_gravity_unfactored_kpa * tributary_width if tributary_width is not None else None),
            factored_line_load_kn_per_m=line_load,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(moment_major=moment, shear_major=shear),
                metadata={"component": "secondary_beam", "framing_mode": self._framing_mode(candidate).value},
            ),
        )

    def _build_column_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> ColumnDemand | None:
        if candidate.column_family_id is None:
            return None
        combo, factored_area = governing
        tributary_area = self._column_tributary_area(params)
        storeys = params.storey_count or 1
        axial_total = factored_area * tributary_area * storeys if tributary_area is not None else None
        dead = case.superimposed_dead_kpa * tributary_area * storeys if tributary_area is not None else None
        live = case.live_kpa * tributary_area * storeys if tributary_area is not None else None
        return ColumnDemand(
            storey=1,
            tributary_area_m2=tributary_area,
            axial_dead_kn=dead,
            axial_live_kn=live,
            effective_length_m=params.floor_to_floor_m,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(axial=axial_total),
                metadata={
                    "component": "column",
                    "storeys_accumulated": storeys,
                    "framing_mode": self._framing_mode(candidate).value,
                    "tributary_area_basis_m2": tributary_area,
                },
            ),
        )

    def _build_wall_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> WallDemand | None:
        if candidate.floor_family_id is None:
            return None
        floor_family = candidate.floor_family_id
        spans_to_wall = floor_family is not None and candidate.primary_beam_family_id is None
        if not spans_to_wall:
            return None
        combo, factored_area = governing
        tributary_width = params.span_x_m or params.span_y_m
        axial = factored_area * params.tributary_area_m2 if params.tributary_area_m2 is not None else None
        return WallDemand(
            storey=1,
            tributary_width_m=tributary_width,
            axial_kn=axial,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(axial=axial),
                metadata={"component": "wall", "path": "gravity_support_wall"},
            ),
        )

    def _build_lateral_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, governing: tuple[EvaluatedLoadCombination | None, float]) -> LateralDemand | None:
        if candidate.lateral_family_id is None:
            return None
        combo, governing_pressure = governing
        span_x = params.span_x_m or 0.0
        span_y = params.span_y_m or 0.0
        floor_to_floor = params.floor_to_floor_m or 0.0
        storeys = params.storey_count or 1
        plan_dim = max(span_x, span_y, 1.0)
        storey_area = params.tributary_area_m2 or max(span_x * span_y, 1.0)
        total_height = floor_to_floor * storeys if floor_to_floor else float(storeys)
        base_shear = governing_pressure * storey_area * storeys
        overturning = base_shear * total_height / 2.0 if total_height else None

        storey_forces: list[StoreyForces] = []
        if storeys > 0:
            triangular_den = sum(range(1, storeys + 1))
            for i in range(1, storeys + 1):
                share = i / triangular_den
                shear = base_shear * share
                storey_forces.append(
                    StoreyForces(
                        storey=i,
                        elevation_m=(floor_to_floor * i if floor_to_floor else None),
                        seismic_base_share_kn=(shear if case.seismic_kpa > 0 else None),
                        wind_shear_kn=(shear if case.wind_kpa > 0 else None),
                        torsional_moment_knm=shear * params.accidental_torsion_ratio * plan_dim,
                    )
                )

        return LateralDemand(
            total_base_shear_kn=base_shear,
            overturning_moment_knm=overturning,
            accidental_torsion_eccentricity_m=params.accidental_torsion_ratio * plan_dim,
            storey_forces=storey_forces,
            envelope=DemandEnvelope(
                governing_combination_id=(combo.combination_id if combo else None),
                effects=ForceEffects(shear_major=base_shear, moment_major=overturning),
                metadata={"component": "lateral", "governing_pressure_kpa": governing_pressure},
            ),
        )

    def _build_torsional_demand(self, candidate: AssemblyCandidate, case: ResolvedLoadCase, params: DistributionParameters, lateral: LateralDemand | None) -> TorsionalDemand | None:
        if lateral is None or lateral.total_base_shear_kn is None:
            return None
        plan_dim = max(params.span_x_m or 0.0, params.span_y_m or 0.0, 1.0)
        ecc = params.accidental_torsion_ratio * plan_dim
        return TorsionalDemand(
            source="accidental_torsion",
            design_eccentricity_m=ecc,
            torsional_moment_knm=lateral.total_base_shear_kn * ecc,
            metadata={"lateral_family_id": candidate.lateral_family_id},
        )


class StructuralLoadEngine:
    """Facade used by AssemblyEvaluator for combination resolution and demand generation."""

    def __init__(self, query: RepositoryQueryService):
        self.query = query
        self.combo_engine = LoadCombinationEngine(query)
        self.distribution_engine = LoadDistributionEngine()

    def build_analysis_input(self, *, project: ProjectContext, candidate: AssemblyCandidate) -> AssemblyAnalysisInput:
        occupancy = self._resolve_occupancy(project)
        resolved_case = self._resolve_load_case(project=project, occupancy=occupancy)
        load_combinations = self.combo_engine.resolve_for_project(project=project, occupancy=occupancy)
        return self.distribution_engine.build_analysis_input(
            project=project,
            candidate=candidate,
            resolved_case=resolved_case,
            load_combinations=load_combinations,
        )

    def _resolve_occupancy(self, project: ProjectContext) -> Occupancy:
        occupancy_id = project.occupancy_id
        if not occupancy_id and project.typology_id:
            typology = self.query.get_typology(project.typology_id)
            occupancy_id = typology.constraints.occupancy
        if not occupancy_id:
            raise LoadResolutionError("ProjectContext is missing occupancy_id and typology occupancy fallback.")
        try:
            return self.query.repo.occupancies[str(occupancy_id)]
        except KeyError as exc:
            raise LoadResolutionError(f"Unknown occupancy '{occupancy_id}'") from exc

    def _resolve_load_case(self, *, project: ProjectContext, occupancy: Occupancy) -> ResolvedLoadCase:
        row = dict(occupancy.metadata)
        lv = occupancy.load_values
        dead = self._pick_numeric(row, lv, ["dead_kpa", "dead_load_kpa", "sdl", "super_dead_kpa", "gk"]) or 0.0
        live = self._pick_numeric(row, lv, ["live_kpa", "live_load_kpa", "ll", "qk"]) or 0.0
        partition = self._pick_numeric(row, lv, ["partition_kpa", "partition_load_kpa", "partition", "partitions"]) or 0.0
        roof_live = self._pick_numeric(row, lv, ["roof_live_kpa", "roof_live_load_kpa", "lr", "lr_kpa"]) or 0.0
        snow = self._pick_numeric(row, lv, ["snow_kpa", "snow_load_kpa", "s", "s_kpa"]) or 0.0
        rain = self._pick_numeric(row, lv, ["rain_kpa", "rain_load_kpa", "r", "r_kpa"]) or 0.0
        wind = self._pick_numeric(row, lv, ["wind_kpa", "wind_pressure_kpa", "w", "w_kpa"]) or 0.0
        seismic = self._pick_numeric(row, lv, ["seismic_kpa", "eq_kpa", "e", "e_kpa"]) or 0.0

        dead += float(project.overrides.get("dead_kpa", 0.0) or 0.0)
        live += float(project.overrides.get("live_kpa", 0.0) or 0.0)
        partition += float(project.overrides.get("partition_kpa", 0.0) or 0.0)
        wind = float(project.overrides.get("wind_kpa", wind) or 0.0)
        seismic = float(project.overrides.get("seismic_kpa", seismic) or 0.0)

        return ResolvedLoadCase(
            occupancy_id=occupancy.occupancy_id,
            dead_kpa=dead,
            live_kpa=live,
            partition_kpa=partition,
            roof_live_kpa=roof_live,
            snow_kpa=snow,
            rain_kpa=rain,
            wind_kpa=wind,
            seismic_kpa=seismic,
            metadata={"occupancy_description": occupancy.description},
        )

    def _pick_numeric(self, primary: dict[str, Any], secondary: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            for source in (primary, secondary):
                if key not in source:
                    continue
                value = source.get(key)
                try:
                    if value is None or value == "":
                        continue
                    return float(value)
                except Exception:
                    continue
        return None


def build_load_engine(query: RepositoryQueryService) -> StructuralLoadEngine:
    return StructuralLoadEngine(query)
