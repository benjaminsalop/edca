from __future__ import annotations

"""Compatibility shims for the old systems module.

The old engine used a monolithic systems.py that mixed raw file loading,
filtering, and gravity-only candidate selection. The refactored engine splits
those responsibilities across repositories.py, candidates.py, and
compatibilities.py.

This module intentionally provides a small set of bridge helpers so older call
sites can be migrated incrementally without preserving the old architecture.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import CandidateGenerator
from .compatibilities import CompatibilityPreferenceEngine
from .domain_models import AssemblyCandidate, ProjectContext, SourceBundle
from .repositories import RepositoryBuilder, RepositoryLoader, RepositoryQueryService


class SystemsService:
    """Thin facade around the refactored repository and candidate layers."""

    def __init__(self, query: RepositoryQueryService):
        self.query = query
        self.generator = CandidateGenerator(query)
        self.compatibility = CompatibilityPreferenceEngine(query)

    def generate_candidates(self, project: ProjectContext) -> list[AssemblyCandidate]:
        candidates = self.generator.generate(project)
        return self.compatibility.apply(candidates, typology_id=project.typology_id)

    def generate_candidates_df(self, project: ProjectContext) -> pd.DataFrame:
        return candidates_to_dataframe(self.generate_candidates(project))


def build_repository_from_paths(**paths: str | Path | None):
    loader = RepositoryLoader()
    bundle = loader.load_bundle_from_paths(**paths)
    repo = RepositoryBuilder().build(bundle)
    return repo


def build_query_service_from_paths(**paths: str | Path | None) -> RepositoryQueryService:
    repo = build_repository_from_paths(**paths)
    return RepositoryQueryService(repo)


def generate_candidates(project: ProjectContext, query: RepositoryQueryService) -> list[AssemblyCandidate]:
    service = SystemsService(query)
    return service.generate_candidates(project)


def candidates_to_dataframe(candidates: list[AssemblyCandidate]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {
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
            "tags": ";".join(candidate.tags),
        }
        rank_fields = candidate.metadata.get("rank_fields", [])
        row["rank_fields"] = ";".join(f"{item['field']}:{item['direction']}" for item in rank_fields)
        row["metadata"] = candidate.metadata
        rows.append(row)
    return pd.DataFrame(rows)


def load_systems_catalog(path_or_fp: str | Path, unit_filter: str | None = None):
    """Backwards-compatible loader for a single catalog file.

    This keeps the old function name alive for transitional code, but it now
    simply returns the loaded DataFrame and does not try to infer family or
    variant siblings. New code should use RepositoryLoader instead.
    """
    path = Path(path_or_fp)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if unit_filter and "unit" in df.columns:
        df = df[df["unit"].astype(str).str.lower() == str(unit_filter).lower()].copy()
    return df, None, None


def filter_systems(*args, **kwargs):
    raise NotImplementedError(
        "filter_systems() belonged to the gravity-only pipeline and has been removed. "
        "Use RepositoryQueryService + CandidateGenerator + CompatibilityPreferenceEngine instead."
    )
