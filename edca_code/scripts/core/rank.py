from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import pandas as pd

from .design_results import AssemblyDesignResult, ComponentDesignResult
from .domain_models import AssemblyCandidate
from .output_schema import ASSEMBLY_RESULT_COLUMNS, COMPONENT_RESULT_COLUMNS

# AssemblyCandidate imported above is also used in catalog_results_to_dataframe


def candidates_to_dataframe(candidates: Iterable[AssemblyCandidate]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rank_fields = candidate.metadata.get("rank_fields", [])
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "typology_id": candidate.typology_id,
                "floor_family_id": candidate.floor_family_id,
                "floor_variant_id": candidate.floor_variant_id,
                "primary_beam_family_id": candidate.primary_beam_family_id,
                "primary_beam_variant_id": candidate.primary_beam_variant_id,
                "secondary_beam_family_id": candidate.secondary_beam_family_id,
                "secondary_beam_variant_id": candidate.secondary_beam_variant_id,
                "column_family_id": candidate.column_family_id,
                "column_variant_id": candidate.column_variant_id,
                "lateral_family_id": candidate.lateral_family_id,
                "lateral_variant_id": candidate.lateral_variant_id,
                "load_path_method": candidate.load_path_method.value,
                "material_mix_label": candidate.material_mix_label,
                "total_penalty": candidate.total_penalty,
                "penalty_count": len(candidate.penalties),
                "penalty_sources": ";".join(p.source for p in candidate.penalties),
                "rank_fields": ";".join(f"{item['field']}:{item['direction']}" for item in rank_fields),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["total_penalty", "candidate_id"], ascending=[True, True]).reset_index(drop=True)
    return df


def component_result_to_row(candidate_id: str, result: ComponentDesignResult) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "component": result.component,
        "family_id": result.family_id,
        "variant_id": result.variant_id,
        "passed": result.passed,
        "utilization_max": result.utilization_max,
        "selected_section": result.selected_section,
        "cost": result.cost,
        "embodied_carbon": result.embodied_carbon,
        "warning_count": len(result.warnings),
    }


def assembly_results_to_dataframe(results: Iterable[AssemblyDesignResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "candidate_id": result.candidate_id,
            "passed": result.passed,
            "total_cost": result.total_cost,
            "total_embodied_carbon": result.total_embodied_carbon,
            "warning_count": len(result.warnings),
        }
        for name in ("floor", "primary_beam", "secondary_beam", "column", "lateral"):
            component = getattr(result, name)
            prefix = f"{name}_"
            row[prefix + "family_id"] = component.family_id if component else None
            row[prefix + "variant_id"] = component.variant_id if component else None
            row[prefix + "passed"] = component.passed if component else None
            row[prefix + "utilization_max"] = component.utilization_max if component else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        preferred = [c for c in ASSEMBLY_RESULT_COLUMNS if c in df.columns]
        remainder = [c for c in df.columns if c not in preferred]
        df = df.loc[:, preferred + remainder]
    return df


def component_results_to_dataframe(results: Iterable[AssemblyDesignResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for component in result.component_results:
            rows.append(component_result_to_row(result.candidate_id, component))
    df = pd.DataFrame(rows)
    if not df.empty:
        preferred = [c for c in COMPONENT_RESULT_COLUMNS if c in df.columns]
        remainder = [c for c in df.columns if c not in preferred]
        df = df.loc[:, preferred + remainder]
    return df


def rank_assembly_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ASSEMBLY_RESULT_COLUMNS)

    ranked = df.copy()
    if "passed" in ranked.columns:
        ranked["_passed_rank"] = (~ranked["passed"].fillna(False)).astype(int)
    else:
        ranked["_passed_rank"] = 1

    numeric_sort_cols: list[str] = []
    for col in ["total_embodied_carbon", "total_cost"]:
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
            numeric_sort_cols.append(col)

    sort_cols = ["_passed_rank"] + numeric_sort_cols + (["candidate_id"] if "candidate_id" in ranked.columns else [])
    ranked = ranked.sort_values(sort_cols, ascending=[True] * len(sort_cols)).drop(columns=["_passed_rank"])
    return ranked.reset_index(drop=True)


# Transitional wrappers for old reporting-oriented names.
def rank_candidates_by_carbon(candidates_df: pd.DataFrame, carbon_total_col: str = "total_embodied_carbon", group_by: list[str] | None = None, take_lowest_per_group: bool = True) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=list(candidates_df.columns) if candidates_df is not None else [])
    df = candidates_df.copy()
    if carbon_total_col not in df.columns:
        return df
    df[carbon_total_col] = pd.to_numeric(df[carbon_total_col], errors="coerce")
    if group_by and take_lowest_per_group:
        idx = df.groupby(group_by, dropna=False)[carbon_total_col].idxmin()
        df = df.loc[idx]
    return df.sort_values(carbon_total_col, ascending=True).reset_index(drop=True)


def rank_candidates_by_cost(candidates_df: pd.DataFrame, cost_col: str = "total_cost") -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty or cost_col not in candidates_df.columns:
        return pd.DataFrame(columns=list(candidates_df.columns) if candidates_df is not None else [])
    df = candidates_df.copy()
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")
    return df.sort_values(cost_col, ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Catalog-evaluation result helpers
# ---------------------------------------------------------------------------

def catalog_results_to_dataframe(
    results: Iterable[AssemblyDesignResult],
    candidates: list[AssemblyCandidate] | None = None,
) -> pd.DataFrame:
    """Flatten CatalogEvaluator results into a wide DataFrame.

    One row per variant-resolved assembly candidate.  Includes span, all
    family/variant IDs, per-component carbon, and total carbon/cost — all
    normalised to per m² of gross floor plate per storey.
    """
    candidate_index: dict[str, AssemblyCandidate] = {}
    if candidates:
        for c in candidates:
            candidate_index[c.candidate_id] = c

    rows: list[dict[str, Any]] = []
    for result in results:
        meta = result.metadata or {}
        # Recover the original family-level candidate ID so we can group later
        expanded_from = meta.get("expanded_from", result.candidate_id)

        row: dict[str, Any] = {
            "candidate_id": result.candidate_id,
            "family_combo_id": expanded_from,
            "span_x_m": meta.get("span_x_m"),
            "span_y_m": meta.get("span_y_m"),
            "total_penalty": meta.get("total_penalty", 0.0),
            "total_embodied_carbon_per_m2": result.total_embodied_carbon,
            "total_cost_per_m2": result.total_cost,
        }

        for name in ("floor", "primary_beam", "secondary_beam", "column", "lateral"):
            comp = getattr(result, name, None)
            row[f"{name}_family_id"] = comp.family_id if comp else None
            row[f"{name}_variant_id"] = comp.variant_id if comp else None
            row[f"{name}_carbon_per_m2"] = comp.embodied_carbon if comp else None
            row[f"{name}_cost_per_m2"] = comp.cost if comp else None

        # Pull material_mix_label from metadata if available
        row["material_mix_label"] = meta.get("material_mix_label")
        row["load_path_method"] = meta.get("load_path_method")
        row["structural_class"] = meta.get("structural_class")

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Coerce numeric columns
    num_cols = [c for c in df.columns if "carbon" in c or "cost" in c or "span" in c or "penalty" in c]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add rank by carbon
    df = df.sort_values("total_embodied_carbon_per_m2", ascending=True).reset_index(drop=True)
    df["rank_carbon"] = df.index + 1
    cols = ["rank_carbon"] + [c for c in df.columns if c != "rank_carbon"]
    return df[cols]


def build_summary_tables(df_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all comparison summary tables from the full results DataFrame.

    Returns a dict keyed by table name; each value is a DataFrame.

    Tables produced
    ---------------
    best_overall
        Single best assembly (lowest carbon_per_m²).
    best_by_structural_class
        One row per structural class label (e.g. "Precast Hollowcore · Steel Frame").
        This is the primary comparison table for the visualisation.
    best_by_family_combo
        One row per unique (floor_family × beam_family × column_family × lateral_family),
        selecting the variant/span combination with the lowest carbon.
    best_by_material_mix
        One row per material_mix_label.
    best_by_floor_family
        One row per floor_family_id.
    best_by_column_family
        One row per column_family_id.
    best_by_lateral_family
        One row per lateral_family_id.
    span_sweep
        All rows, sorted by (family_combo_id, span_x_m) — useful for plotting
        how carbon varies with span for each system combination.
    """
    if df_all is None or df_all.empty:
        empty: dict[str, pd.DataFrame] = {
            name: pd.DataFrame()
            for name in (
                "best_overall", "best_by_family_combo", "best_by_material_mix",
                "best_by_floor_family", "best_by_column_family", "best_by_lateral_family",
                "span_sweep",
            )
        }
        return empty

    carbon_col = "total_embodied_carbon_per_m2"
    df = df_all.copy()
    df[carbon_col] = pd.to_numeric(df[carbon_col], errors="coerce")

    def _best_per_group(group_cols: list[str]) -> pd.DataFrame:
        valid = [c for c in group_cols if c in df.columns]
        if not valid:
            return pd.DataFrame()
        idx = df.dropna(subset=[carbon_col]).groupby(valid, dropna=False)[carbon_col].idxmin()
        return df.loc[idx].sort_values(carbon_col).reset_index(drop=True)

    # family_combo_id encodes the full family combination
    family_combo_cols = ["family_combo_id"]
    if "material_mix_label" not in df.columns:
        df["material_mix_label"] = None

    tables: dict[str, pd.DataFrame] = {
        "best_overall": df.dropna(subset=[carbon_col]).sort_values(carbon_col).head(1).reset_index(drop=True),
        "best_by_structural_class": _best_per_group(["structural_class"]),
        "best_by_family_combo": _best_per_group(family_combo_cols),
        "best_by_material_mix": _best_per_group(["material_mix_label"]),
        "best_by_floor_family": _best_per_group(["floor_family_id"]),
        "best_by_column_family": _best_per_group(["column_family_id"]),
        "best_by_lateral_family": _best_per_group(["lateral_family_id"]),
        "span_sweep": df.sort_values(["family_combo_id", "span_x_m"]).reset_index(drop=True) if "span_x_m" in df.columns else df,
    }

    # Add rank columns to the by-* summaries
    for name in ("best_by_structural_class", "best_by_family_combo", "best_by_material_mix", "best_by_floor_family"):
        t = tables[name]
        if not t.empty and carbon_col in t.columns:
            t = t.sort_values(carbon_col).reset_index(drop=True)
            t["rank_carbon"] = t.index + 1
            cols = ["rank_carbon"] + [c for c in t.columns if c != "rank_carbon"]
            tables[name] = t[cols]

    return tables
