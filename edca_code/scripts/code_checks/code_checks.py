from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from edca_code.scripts.code_checks.design_inputs import (
    BeamDesignInput,
    BuildingSystemDesignInput,
    ColumnDesignInput,
    ComponentDesignInput,
    LateralSystemDesignInput,
    SlabDesignInput,
)
from edca_code.scripts.core.design_results import CheckResult, ComponentDesignResult

logger = logging.getLogger("code_checks")

CheckerFn = Callable[..., ComponentDesignResult]


class CodeCheckError(Exception):
    pass


class CodeCheckRegistry:
    def __init__(self) -> None:
        self._registry: Dict[Tuple[str, Optional[str]], CheckerFn] = {}

    def register(self, component_type: str, family_id: Optional[str], fn: CheckerFn) -> None:
        self._registry[(component_type, family_id)] = fn

    def get(self, component_type: str, family_id: Optional[str]) -> Optional[CheckerFn]:
        return self._registry.get((component_type, family_id)) or self._registry.get((component_type, None))


registry = CodeCheckRegistry()


def _fallback_result(component: ComponentDesignInput, reason: str) -> ComponentDesignResult:
    return ComponentDesignResult(
        component=component.component_type,
        family_id=component.family_id,
        variant_id=component.variant_id,
        checks=[
            CheckResult(
                check_name="no_registered_checker",
                passed=False,
                demand=None,
                capacity=None,
                unity_ratio=None,
                metadata={"reason": reason},
            )
        ],
        passed=False,
        utilization_max=None,
        warnings=[],
        metadata={"input": asdict(component)},
    )


def run_component_code_check(component: ComponentDesignInput, **kwargs: Any) -> ComponentDesignResult:
    checker = registry.get(component.component_type, component.family_id)
    if checker is None:
        return _fallback_result(
            component,
            f"No checker registered for component_type={component.component_type!r}, family_id={component.family_id!r}",
        )
    return checker(component, **kwargs)


# Backward-compatible batch wrapper for the new pipeline.
def run_code_checks_on_components(
    components: Iterable[ComponentDesignInput],
    **kwargs: Any,
) -> List[ComponentDesignResult]:
    return [run_component_code_check(component, **kwargs) for component in components]


# Temporary legacy wrapper. It preserves the old entry point name but does not depend on DataFrames.
def run_code_checks_on_candidates(candidates: Iterable[Any], **kwargs: Any) -> List[ComponentDesignResult]:
    normalized: List[ComponentDesignInput] = []
    for item in candidates:
        if isinstance(item, ComponentDesignInput):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            normalized.append(
                ComponentDesignInput(
                    component_type=str(item.get("component_type") or item.get("component") or "floor"),
                    family_id=str(item.get("family_id") or item.get("system_family") or "unknown"),
                    variant_id=item.get("variant_id") or item.get("system_variant"),
                    role=item.get("role"),
                    geometry=item.get("geometry", {}),
                    properties=item,
                    metadata={"legacy_candidate": True},
                )
            )
            continue
        raise CodeCheckError(f"Unsupported candidate item type: {type(item).__name__}")
    return run_code_checks_on_components(normalized, **kwargs)


# Placeholders for future whole-building/lateral checks.
def run_building_code_check(building: BuildingSystemDesignInput, **kwargs: Any) -> Optional[CheckResult]:
    _ = kwargs
    if building.torsion_irregularity_ratio is None and not building.drift_by_storey:
        return None
    passes = True
    details: Dict[str, Any] = {}
    if building.torsion_irregularity_ratio is not None:
        details["torsion_irregularity_ratio"] = building.torsion_irregularity_ratio
        passes = passes and building.torsion_irregularity_ratio <= 1.2
    return CheckResult(
        check_name="building_global_stability_placeholder",
        passed=passes,
        demand=building.torsion_irregularity_ratio,
        capacity=1.2 if building.torsion_irregularity_ratio is not None else None,
        unity_ratio=(building.torsion_irregularity_ratio / 1.2) if building.torsion_irregularity_ratio is not None else None,
        metadata=details,
    )
