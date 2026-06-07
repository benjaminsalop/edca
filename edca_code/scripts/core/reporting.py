from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import logging

import pandas as pd

from edca_code.scripts.core.assembly_summary import build_assembly_rankings
from edca_code.scripts.core.design_results import AssemblyDesignResult, ComponentDesignResult
from edca_code.scripts.core.output_schema import ASSEMBLY_RESULT_COLUMNS, COMPONENT_RESULT_COLUMNS
from edca_code.scripts.core.visualize import plot_material_breakdown_comparison, plot_structural_class_comparison

logger = logging.getLogger(__name__)


class ReportingEngine:
    def write_results(
        self,
        assembly_results: Iterable[AssemblyDesignResult],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        assembly_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        for assembly in assembly_results:
            assembly_rows.append(self._assembly_row(assembly))
            summaries.append(self._assembly_summary(assembly))
            for component in assembly.component_results:
                component_rows.append(self._component_row(assembly, component))

        assembly_csv = out_dir / "assembly_results.csv"
        component_csv = out_dir / "component_results.csv"
        summary_json = out_dir / "run_summary.json"

        _write_csv(assembly_csv, assembly_rows, ASSEMBLY_RESULT_COLUMNS)
        _write_csv(component_csv, component_rows, COMPONENT_RESULT_COLUMNS)
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump({"assemblies": summaries}, handle, indent=2)

        return {
            "assembly_results": assembly_csv,
            "component_results": component_csv,
            "run_summary": summary_json,
        }

    def _assembly_row(self, assembly: AssemblyDesignResult) -> dict[str, Any]:
        return {
            "candidate_id": assembly.candidate_id,
            "pass_overall": assembly.pass_overall,
            "governing_utilization": assembly.governing_utilization,
            "governing_limit_state": assembly.governing_limit_state,
            "governing_combination": assembly.governing_combination,
            "warnings_count": len(assembly.warnings),
            "components_count": len(assembly.component_results),
        }

    def _component_row(self, assembly: AssemblyDesignResult, component: ComponentDesignResult) -> dict[str, Any]:
        return {
            "candidate_id": assembly.candidate_id,
            "component_id": component.component_id,
            "component_type": component.component_type,
            "selected_family_id": component.selected_family_id,
            "selected_variant_id": component.selected_variant_id,
            "pass_component": component.pass_component,
            "governing_utilization": component.governing_utilization,
            "governing_limit_state": component.governing_limit_state,
            "governing_combination": component.governing_combination,
            "checks_count": len(component.checks),
        }

    def _assembly_summary(self, assembly: AssemblyDesignResult) -> dict[str, Any]:
        return {
            "candidate_id": assembly.candidate_id,
            "pass_overall": assembly.pass_overall,
            "warnings": [asdict(item) for item in assembly.warnings],
            "components": [component.component_id for component in assembly.component_results],
        }


def write_comparison_chart(
    out_dir: str | Path,
    *,
    floors_df: pd.DataFrame | None = None,
    beams_df: pd.DataFrame | None = None,
    columns_df: pd.DataFrame | None = None,
    laterals_df: pd.DataFrame | None = None,
    assemblies_df: pd.DataFrame | None = None,
    title: str = "Superstructure Options — Embodied Carbon (kgCO₂e/m² GFA)",
    subtitle: str | None = None,
) -> Path | None:
    """Generate the structural-class comparison bar chart and save as PNG.

    Accepts either:
    - A pre-built ``assemblies_df`` (output of :func:`build_assembly_rankings`), OR
    - The four per-component ``summary_ranked_all`` DataFrames, which are
      assembled on-the-fly.

    Returns the path to the saved chart, or None if nothing was generated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if assemblies_df is None or assemblies_df.empty:
        if any(df is not None for df in (floors_df, beams_df, columns_df, laterals_df)):
            assemblies_df = build_assembly_rankings(
                floors_df=floors_df or pd.DataFrame(),
                beams_df=beams_df,
                columns_df=columns_df,
                laterals_df=laterals_df,
            )

    if assemblies_df is None or assemblies_df.empty:
        logger.warning("write_comparison_chart: no assembly data available — chart skipped")
        return None

    chart_path = out_dir / "comparison_chart.png"
    try:
        plot_structural_class_comparison(
            assemblies_df,
            title=title,
            subtitle=subtitle,
            out_path=chart_path,
        )
        logger.info("Wrote comparison_chart.png (%d structural classes)", len(assemblies_df))
        return chart_path
    except Exception:
        logger.exception("write_comparison_chart: failed to generate chart")
        return None


def write_material_breakdown_chart(
    out_dir: str | Path,
    *,
    assemblies_df: pd.DataFrame,
    title: str = "Superstructure Options — Embodied Carbon by Material (kgCO₂e/m² GFA)",
    subtitle: str | None = None,
) -> Path | None:
    """Generate the material-breakdown stacked bar chart and save as PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if assemblies_df is None or assemblies_df.empty:
        logger.warning("write_material_breakdown_chart: no assembly data — chart skipped")
        return None

    chart_path = out_dir / "material_breakdown_chart.png"
    try:
        plot_material_breakdown_comparison(
            assemblies_df,
            title=title,
            subtitle=subtitle,
            out_path=chart_path,
        )
        logger.info("Wrote material_breakdown_chart.png (%d structural classes)", len(assemblies_df))
        return chart_path
    except Exception:
        logger.exception("write_material_breakdown_chart: failed to generate chart")
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
