#!/usr/bin/env python3
"""core/run_edca.py — Refactored EDCA assembly engine.

Generates and compares complete structural assemblies (floor + beams +
columns + lateral) across all viable variant combinations and a range of
bay spans, ranking by embodied carbon per m² of floor plate.

Invocation (from repo root)
---------------------------
python -m edca_code.scripts.core.run_edca \
    --control  setup/control_files/control_file.yaml \
    --canonical-dir  inputs/canonical \
    --presets-dir    inputs/source/presets/systems \
    --materials      inputs/canonical/materials.csv \
    --occupancies    inputs/canonical/occupancies.csv \
    --out            outputs/edca_assembly_run

Span sweep (optional — overrides the SPANS list in the control file)
---------------------------
    --span-min 5.0 --span-max 12.0 --span-step 0.5

The engine produces one 'assembly_results_all.csv' per run plus several
pre-grouped summary tables (by material mix, by floor family, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from edca_code.scripts.core.carbon import CarbonCostEngine, MaterialsTable
from edca_code.scripts.core.candidates import CandidateGenerator
from edca_code.scripts.core.catalog_evaluator import CatalogEvaluator
from edca_code.scripts.core.compatibilities import CompatibilityPreferenceEngine
from edca_code.scripts.core.design_results import AssemblyDesignResult
from edca_code.scripts.core.parse import ControlFile
from edca_code.scripts.core.rank import (
    build_summary_tables,
    candidates_to_dataframe,
    catalog_results_to_dataframe,
)
from edca_code.scripts.core.visualize import plot_structural_class_comparison
from edca_code.scripts.core.repositories import RepositoryBuilder, RepositoryLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the EDCA assembly-comparison engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--control", "-c", required=True, help="Path to control_file.yaml")
    p.add_argument("--out", "-o", default="outputs/edca_assembly_run", help="Output directory")

    # Shorthand directory arguments (auto-discover CSVs / YAMLs)
    p.add_argument("--canonical-dir", default="inputs/canonical",
                   help="Directory containing per-component *_families.* and *_variants.* files")
    p.add_argument("--presets-dir", default="inputs/source/presets/systems",
                   help="Directory containing typologies.yaml, system_types.yaml, etc.")
    p.add_argument("--loads-dir", default="inputs/source/presets/loads",
                   help="Directory containing load_combinations.yaml, load_values.yaml, occupancies.csv")

    # Individual file overrides (override the shorthand defaults)
    p.add_argument("--materials",   default=None, help="Override path to materials CSV")
    p.add_argument("--occupancies", default=None, help="Override path to occupancies CSV")

    # Span sweep
    p.add_argument("--span-min",  type=float, default=None, help="Override sweep start (m)")
    p.add_argument("--span-max",  type=float, default=None, help="Override sweep end (m)")
    p.add_argument("--span-step", type=float, default=0.5,  help="Sweep step size (m)")

    # Evaluation options
    p.add_argument("--max-variants", type=int, default=20,
                   help="Max variants per component before Cartesian product")
    p.add_argument("--top-n-families", type=int, default=None,
                   help="Evaluate only the top-N ranked family candidates (None = all)")
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find(directory: Path, stem: str) -> Path | None:
    """Return the first existing file matching <stem>.{parquet,csv} in directory."""
    for ext in (".parquet", ".csv"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _require(directory: Path, stem: str, label: str) -> str:
    p = _find(directory, stem)
    if p is None:
        raise FileNotFoundError(
            f"Required {label} file not found in {directory}: "
            f"tried {stem}.parquet, {stem}.csv"
        )
    return str(p)


def _find_yaml(directory: Path, stem: str) -> Path | None:
    for ext in (".yaml", ".yml"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _require_yaml(directory: Path, stem: str, label: str) -> str:
    p = _find_yaml(directory, stem)
    if p is None:
        raise FileNotFoundError(
            f"Required {label} YAML not found in {directory}: tried {stem}.yaml"
        )
    return str(p)


# ---------------------------------------------------------------------------
# Span sweep resolution
# ---------------------------------------------------------------------------

def _resolve_spans(
    project_spans: list[float],
    span_min: float | None,
    span_max: float | None,
    step: float,
) -> list[float]:
    """Build the list of spans to evaluate.

    If --span-min / --span-max given: generate a range.
    Otherwise fall back to the SPANS list from the control file.
    If that is empty: use a single default span of 7.5 m.
    """
    if span_min is not None and span_max is not None:
        spans: list[float] = []
        s = span_min
        while s <= span_max + 1e-9:
            spans.append(round(s, 4))
            s += step
        return spans or [span_min]
    if project_spans:
        return list(project_spans)
    return [7.5]


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = Path(args.canonical_dir)
    presets = Path(args.presets_dir)
    loads = Path(args.loads_dir)

    # ---- 1. Load control file & project context ----
    control = ControlFile.from_path(args.control)
    project = control.to_project_context()
    logger.info("Project: %s | typology: %s | storeys: %s",
                project.project_id, project.typology_id, project.geometry.storey_count)

    # ---- 2. Resolve spans ----
    project_spans: list[float] = [float(s) for s in (project.design_options.get("spans") or [])]
    span_list = _resolve_spans(project_spans, args.span_min, args.span_max, args.span_step)
    logger.info("Span sweep: %s values — %s … %s m", len(span_list), span_list[0], span_list[-1])

    # ---- 3. Build data bundle ----
    materials_path = args.materials or str(_require(canonical, "materials", "materials"))
    occupancies_path = args.occupancies or str(_require(canonical, "occupancies", "occupancies"))

    bundle_paths: dict[str, str] = {
        "floor_families":  _require(canonical, "floor_families",  "floor families"),
        "floor_variants":  _require(canonical, "floor_variants",  "floor variants"),
        "beam_families":   _require(canonical, "beam_families",   "beam families"),
        "beam_variants":   _require(canonical, "beam_variants",   "beam variants"),
        "column_families": _require(canonical, "column_families", "column families"),
        "column_variants": _require(canonical, "column_variants", "column variants"),
        "lateral_families": _require(canonical, "lateral_families", "lateral families"),
        "lateral_variants": _require(canonical, "lateral_variants", "lateral variants"),
        "materials":   materials_path,
        "occupancies": occupancies_path,
        "typologies_yaml":            _require_yaml(presets, "typologies",            "typologies"),
        "system_types_yaml":          _require_yaml(presets, "system_types",          "system_types"),
        "system_preferences_yaml":    _require_yaml(presets, "system_preferences",    "system_preferences"),
        "system_compatibility_yaml":  _require_yaml(presets, "system_compatibility",  "system_compatibility"),
        "assembly_generation_yaml":   _require_yaml(presets, "assembly_generation",   "assembly_generation"),
    }

    # Optional YAML files — pass None if absent so the loader skips them
    for stem, key in (
        ("material_mapping",   "material_mapping_yaml"),
        ("material_sets",      "material_sets_yaml"),
        ("materials_config",   "materials_config_yaml"),
        ("materials_methods",  "materials_methods_yaml"),
    ):
        p = _find_yaml(presets, stem)
        if p:
            bundle_paths[key] = str(p)

    for stem, key in (
        ("load_combinations", "load_combinations_yaml"),
        ("load_values",       "load_values_yaml"),
    ):
        p = _find_yaml(loads, stem)
        if p:
            bundle_paths[key] = str(p)

    loader = RepositoryLoader()
    bundle = loader.load_bundle_from_paths(**bundle_paths)
    repo = RepositoryBuilder().build(bundle)

    from edca_code.scripts.core.repositories import RepositoryQueryService
    query = RepositoryQueryService(repo)

    # ---- 4. Generate & rank family-level candidates ----
    generator = CandidateGenerator(query)
    compatibility = CompatibilityPreferenceEngine(query)

    raw_candidates = generator.generate(project)
    ranked_candidates = compatibility.apply(raw_candidates, typology_id=project.typology_id)
    logger.info("Generated %d family candidates → %d after compatibility filter",
                len(raw_candidates), len(ranked_candidates))

    if args.top_n_families:
        ranked_candidates = ranked_candidates[: args.top_n_families]
        logger.info("Trimmed to top-%d family candidates", args.top_n_families)

    # Write the family-level candidate table immediately
    candidate_df = candidates_to_dataframe(ranked_candidates)
    candidate_df.to_csv(out_dir / "family_candidates_ranked.csv", index=False)

    # ---- 5. Build carbon engine ----
    materials_table = MaterialsTable.from_csv(materials_path)
    carbon_engine = CarbonCostEngine(materials_table)
    evaluator = CatalogEvaluator(query, carbon_engine, max_variants_per_component=args.max_variants)

    # ---- 6. Span sweep → collect all results ----
    all_results: list[AssemblyDesignResult] = []

    _one_way_irregular = project.design_options.get("one_way_irregular", False)
    _beam_span = project.design_options.get("one_way_beam_min_span")
    _slab_span = project.design_options.get("one_way_slab_min_span")

    for span in span_list:
        if _one_way_irregular and _beam_span is not None and _slab_span is not None:
            span_x = float(_beam_span)
            span_y = float(_slab_span)
        else:
            span_x = span
            span_y = span
        results = evaluator.evaluate_all(ranked_candidates, project, span_x, span_y)
        all_results.extend(results)
        logger.debug("span=%.2f m  → %d assemblies evaluated", span, len(results))

    logger.info("Total assembly variants evaluated: %d", len(all_results))

    # ---- 7. Build results DataFrame ----
    df_all = catalog_results_to_dataframe(all_results, ranked_candidates)

    # Attach material_mix_label from family candidates where missing
    if "family_combo_id" in df_all.columns and "material_mix_label" in candidate_df.columns:
        mix_map = candidate_df.set_index("candidate_id")["material_mix_label"].to_dict()
        df_all["material_mix_label"] = df_all["material_mix_label"].fillna(
            df_all["family_combo_id"].map(mix_map)
        )

    df_all.to_csv(out_dir / "assembly_results_all.csv", index=False)
    logger.info("Wrote assembly_results_all.csv  (%d rows)", len(df_all))

    # ---- 8. Build summary tables ----
    tables = build_summary_tables(df_all)
    for name, df_table in tables.items():
        if df_table is not None and not df_table.empty:
            fp = out_dir / f"summary_{name}.csv"
            df_table.to_csv(fp, index=False)
            logger.info("Wrote summary_%s.csv  (%d rows)", name, len(df_table))

    # ---- 9. Generate comparison chart ----
    structural_class_table = tables.get("best_by_structural_class")
    chart_path: Path | None = None
    if structural_class_table is not None and not structural_class_table.empty:
        chart_path = out_dir / "comparison_chart.png"
        span_label = f"{span_list[0]:.1f}–{span_list[-1]:.1f} m" if len(span_list) > 1 else f"{span_list[0]:.1f} m"
        try:
            plot_structural_class_comparison(
                structural_class_table,
                title=f"Superstructure Options — Embodied Carbon (kgCO₂e/m² GFA)",
                subtitle=f"Span sweep: {span_label}  ·  Best assembly per structural class",
                out_path=chart_path,
            )
            logger.info("Wrote comparison_chart.png")
        except Exception as exc:
            logger.warning("Could not generate chart: %s", exc)
            chart_path = None

    # ---- 10. Write run summary JSON ----
    summary: dict[str, Any] = {
        "project_id": project.project_id,
        "typology_id": project.typology_id,
        "storey_count": project.geometry.storey_count,
        "spans_evaluated": span_list,
        "family_candidates_generated": len(raw_candidates),
        "family_candidates_after_filter": len(ranked_candidates),
        "assembly_variants_evaluated": len(all_results),
        "output_directory": str(out_dir),
        "tables_written": [
            "family_candidates_ranked.csv",
            "assembly_results_all.csv",
        ] + [f"summary_{name}.csv" for name, t in tables.items() if t is not None and not t.empty],
        "chart": str(chart_path) if chart_path else None,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Run complete → %s", out_dir)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
