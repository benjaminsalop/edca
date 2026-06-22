from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GoverningCase:
    combination_id: str | None = None
    limit_state: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None


@dataclass(slots=True)
class ResultWarning:
    code: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckResult:
    check_name: str
    passed: bool
    demand: float | None = None
    capacity: float | None = None
    unity_ratio: float | None = None
    governing_case: GoverningCase | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentDesignResult:
    component: str
    family_id: str | None = None
    variant_id: str | None = None
    passed: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    governing_case: GoverningCase | None = None
    selected_section: str | None = None
    utilization_max: float | None = None
    warnings: list[ResultWarning] = field(default_factory=list)
    quantities: dict[str, float] = field(default_factory=dict)
    cost: float | None = None
    embodied_carbon: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssemblyDesignResult:
    candidate_id: str
    passed: bool = False
    floor: ComponentDesignResult | None = None
    primary_beam: ComponentDesignResult | None = None
    secondary_beam: ComponentDesignResult | None = None
    column: ComponentDesignResult | None = None
    wall: ComponentDesignResult | None = None
    lateral: ComponentDesignResult | None = None
    total_cost: float | None = None
    total_embodied_carbon: float | None = None
    warnings: list[ResultWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def component_results(self) -> list[ComponentDesignResult]:
        results = [
            self.floor,
            self.primary_beam,
            self.secondary_beam,
            self.column,
            self.wall,
            self.lateral,
        ]
        return [result for result in results if result is not None]

    def compute_passed(self) -> bool:
        components = self.component_results
        if not components:
            self.passed = False
        else:
            self.passed = all(component.passed for component in components)
        return self.passed

    def compute_totals(self) -> tuple[float, float]:
        cost = 0.0
        carbon = 0.0
        for component in self.component_results:
            cost += float(component.cost or 0.0)
            carbon += float(component.embodied_carbon or 0.0)
        self.total_cost = cost
        self.total_embodied_carbon = carbon
        return cost, carbon
