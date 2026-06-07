from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import logging

from edca_code.scripts.core.design_results import AssemblyDesignResult, ComponentDesignResult
from edca_code.scripts.core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MaterialRecord:
    material_id: str
    material_class: str | None = None
    standard_or_grade: str | None = None
    density: float | None = None
    cost_per_kg: float | None = None
    cost_per_m3: float | None = None
    cost_per_m2: float | None = None
    ec_per_kg: float | None = None
    ec_per_m3: float | None = None
    ec_per_m2: float | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class BomLine:
    category: str
    material_id: str
    quantity: float
    unit: str
    source: str | None = None


@dataclass(slots=True)
class CarbonCostLine:
    category: str
    material_id: str
    quantity: float
    unit: str
    embodied_carbon: float
    cost: float
    source: str | None = None


class MaterialsTable:
    def __init__(self, records: dict[str, MaterialRecord]) -> None:
        self.records = records

    def get(self, material_id: str) -> MaterialRecord:
        try:
            return self.records[material_id]
        except KeyError as exc:
            raise RepositoryError(f"Unknown material_id '{material_id}'") from exc

    @classmethod
    def from_csv(cls, path: str | Path) -> "MaterialsTable":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Materials file not found: {path}")
        rows: list[dict] = []
        if path.suffix.lower() == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            rows = df.where(df.notna(), other=None).to_dict(orient="records")
        else:
            for enc in ("utf-8-sig", "latin-1", "cp1252"):
                try:
                    with path.open("r", encoding=enc, newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    break
                except UnicodeDecodeError:
                    continue
        _LB_FT3_TO_KG_M3 = 16.01846

        records: dict[str, MaterialRecord] = {}
        for row in rows:
            material_id = str(row.get("material_id") or "").strip()
            if not material_id:
                continue
            density = _as_float(row.get("density"))
            if density is not None and str(row.get("unit") or "").strip().lower() == "imperial":
                density = density * _LB_FT3_TO_KG_M3
            records[material_id] = MaterialRecord(
                material_id=material_id,
                material_class=_clean_str(row.get("material_class")),
                standard_or_grade=_clean_str(row.get("standard_or_grade")),
                density=density,
                cost_per_kg=_first_float(row, ["cost_per_kg", "cost_mass"]),
                cost_per_m3=_first_float(row, ["cost_per_m3", "cost_volumetric"]),
                cost_per_m2=_first_float(row, ["cost_per_m2", "cost_areal"]),
                ec_per_kg=_first_float(row, ["ec_per_kg", "ec_a1a3_mass"]),
                ec_per_m3=_first_float(row, ["ec_per_m3", "ec_a1a3_volumetric"]),
                ec_per_m2=_first_float(row, ["ec_per_m2", "ec_a1a3_areal"]),
                raw=dict(row),
            )
        return cls(records)


class CarbonCostEngine:
    def __init__(self, materials: MaterialsTable) -> None:
        self.materials = materials

    def evaluate_bom(self, bom: Iterable[BomLine]) -> list[CarbonCostLine]:
        lines: list[CarbonCostLine] = []
        for line in bom:
            try:
                record = self.materials.get(line.material_id)
            except RepositoryError:
                logger.debug("Unknown material_id '%s' — skipping BOM line (category=%s)",
                             line.material_id, line.category)
                continue
            try:
                embodied_carbon = self._evaluate_intensity(record, line.quantity, line.unit, kind="carbon")
                cost = self._evaluate_intensity(record, line.quantity, line.unit, kind="cost")
            except RepositoryError as exc:
                logger.debug("Cannot evaluate intensity for material '%s': %s", line.material_id, exc)
                continue
            lines.append(
                CarbonCostLine(
                    category=line.category,
                    material_id=line.material_id,
                    quantity=line.quantity,
                    unit=line.unit,
                    embodied_carbon=embodied_carbon,
                    cost=cost,
                    source=line.source,
                )
            )
        return lines

    def evaluate_component(self, result: ComponentDesignResult) -> list[CarbonCostLine]:
        return self.evaluate_bom(_extract_bom_lines(result))

    def evaluate_assembly(self, result: AssemblyDesignResult) -> tuple[list[CarbonCostLine], dict[str, float]]:
        bom: list[BomLine] = []
        for component in result.component_results:
            bom.extend(_extract_bom_lines(component))
        lines = self.evaluate_bom(bom)
        totals = {
            "carbon_total": sum(line.embodied_carbon for line in lines),
            "cost_total": sum(line.cost for line in lines),
        }
        return lines, totals

    def _evaluate_intensity(self, record: MaterialRecord, quantity: float, unit: str, *, kind: str) -> float:
        if quantity == 0:
            return 0.0
        if kind == "carbon":
            per_kg = record.ec_per_kg
            per_m3 = record.ec_per_m3
            per_m2 = record.ec_per_m2
        else:
            per_kg = record.cost_per_kg
            per_m3 = record.cost_per_m3
            per_m2 = record.cost_per_m2

        unit_norm = unit.strip().lower()
        if unit_norm in {"kg", "kilogram", "kilograms"}:
            if per_kg is None:
                raise RepositoryError(f"Material '{record.material_id}' missing per-kg {kind} intensity")
            return quantity * per_kg
        if unit_norm in {"m3", "cubic_meter", "cubic_metre"}:
            if per_m3 is not None:
                return quantity * per_m3
            # Fallback: convert m³ → kg via density, then apply per-kg intensity.
            if per_kg is not None and record.density is not None:
                return quantity * record.density * per_kg
            raise RepositoryError(
                f"Material '{record.material_id}' has no per-m3 {kind} intensity "
                f"and cannot auto-convert (missing density or per-kg intensity)"
            )
        if unit_norm in {"m2", "square_meter", "square_metre"}:
            if per_m2 is None:
                raise RepositoryError(f"Material '{record.material_id}' missing per-m2 {kind} intensity")
            return quantity * per_m2
        raise RepositoryError(f"Unsupported BOM unit '{unit}' for material '{record.material_id}'")


def assembly_result_to_summary_row(result: AssemblyDesignResult, totals: dict[str, float]) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "pass_overall": result.pass_overall,
        "governing_utilization": result.governing_utilization,
        "governing_limit_state": result.governing_limit_state,
        "carbon_total": totals.get("carbon_total", 0.0),
        "cost_total": totals.get("cost_total", 0.0),
        "warnings": len(result.warnings),
    }


def write_carbon_cost_report(
    result: AssemblyDesignResult,
    lines: list[CarbonCostLine],
    totals: dict[str, float],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines_path = out_dir / f"{result.candidate_id}_carbon_cost_lines.json"
    summary_path = out_dir / f"{result.candidate_id}_carbon_cost_summary.json"
    with lines_path.open("w", encoding="utf-8") as handle:
        json.dump([line.__dict__ for line in lines], handle, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(assembly_result_to_summary_row(result, totals), handle, indent=2)
    return lines_path, summary_path


def _extract_bom_lines(result: ComponentDesignResult) -> list[BomLine]:
    bom = result.takeoff or {}
    lines: list[BomLine] = []
    for item in bom.get("lines", []):
        material_id = str(item.get("material_id") or "").strip()
        if not material_id:
            continue
        lines.append(
            BomLine(
                category=str(item.get("category") or result.component_type),
                material_id=material_id,
                quantity=float(item.get("quantity") or 0.0),
                unit=str(item.get("unit") or "kg"),
                source=result.component_id,
            )
        )
    return lines


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None
