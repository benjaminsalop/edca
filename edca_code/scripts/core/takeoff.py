from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edca_code.scripts.core.design_results import AssemblyDesignResult, ComponentDesignResult


@dataclass(slots=True)
class TakeoffLine:
    category: str
    material_id: str
    quantity: float
    unit: str
    source_component_id: str | None = None


@dataclass(slots=True)
class TakeoffResult:
    candidate_id: str
    lines: list[TakeoffLine]

    @property
    def totals_by_material(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for line in self.lines:
            key = f"{line.material_id}|{line.unit}"
            totals[key] = totals.get(key, 0.0) + line.quantity
        return totals


class TakeoffEngine:
    def build_component_takeoff(self, component: ComponentDesignResult) -> list[TakeoffLine]:
        lines: list[TakeoffLine] = []
        payload = component.takeoff or {}
        for item in payload.get("lines", []):
            material_id = str(item.get("material_id") or "").strip()
            if not material_id:
                continue
            lines.append(
                TakeoffLine(
                    category=str(item.get("category") or component.component_type),
                    material_id=material_id,
                    quantity=float(item.get("quantity") or 0.0),
                    unit=str(item.get("unit") or "kg"),
                    source_component_id=component.component_id,
                )
            )
        return lines

    def build_assembly_takeoff(self, assembly: AssemblyDesignResult) -> TakeoffResult:
        lines: list[TakeoffLine] = []
        for component in assembly.component_results:
            lines.extend(self.build_component_takeoff(component))
        return TakeoffResult(candidate_id=assembly.candidate_id, lines=lines)


def takeoff_result_to_rows(result: TakeoffResult) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": result.candidate_id,
            "category": line.category,
            "material_id": line.material_id,
            "quantity": line.quantity,
            "unit": line.unit,
            "source_component_id": line.source_component_id,
        }
        for line in result.lines
    ]
