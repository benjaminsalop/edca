from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from .domain_models import AssemblyCandidate, ComponentType, LoadPathMethod, ProjectContext, SystemVariant
from .repositories import RepositoryQueryService

# Hard cap per component before the Cartesian product to prevent runaway expansion.
# E.g. 20 × 20 × 20 × 10 = 80 000 candidates per family combo — manageable.
_DEFAULT_MAX_VARIANTS_PER_COMPONENT = 20


class VariantExpander:
    """Expand a family-level AssemblyCandidate into all viable variant combinations.

    For each component the viable set is filtered by geometric constraints:
      - floor / beams  : max_span   >= required span
      - columns / lateral : maximum_story_count >= project storey_count (when set)

    The result is the Cartesian product of those filtered sets, one
    AssemblyCandidate per combination with all *_variant_id fields populated.
    Candidates where a family has NO viable variants still appear with
    variant_id=None so downstream carbon totals for that component are zero
    (they act as "family present, variant unresolved" markers).
    """

    def __init__(
        self,
        query: RepositoryQueryService,
        max_variants_per_component: int = _DEFAULT_MAX_VARIANTS_PER_COMPONENT,
    ) -> None:
        self.query = query
        self.max = max_variants_per_component

    def expand(
        self,
        candidate: AssemblyCandidate,
        project: ProjectContext,
        span_x_m: float,
        span_y_m: float,
        primary_beam_line_load_kn_per_m: float | None = None,
        secondary_beam_line_load_kn_per_m: float | None = None,
        column_axial_kn: float | None = None,
    ) -> list[AssemblyCandidate]:
        storey_count = project.geometry.storey_count or 1
        # For two-way RC slabs with secondary beams the effective slab span is the
        # beam-to-beam distance (≈ half the short bay dimension when one intermediate
        # secondary beam is used).  Using the full bay span would unfairly eliminate
        # thin slab variants that are perfectly adequate at the shorter panel span.
        floor_type = ""
        if candidate.floor_family_id:
            try:
                _ff = self.query.get_family(ComponentType.FLOOR, candidate.floor_family_id)
                floor_type = str(_ff.metadata.get("floor_type") or "").lower()
            except Exception:
                pass
        if candidate.secondary_beam_family_id is not None and floor_type == "two_way_slab":
            # effective panel span = half the short bay dimension (1 intermediate secondary beam)
            span_floor = min(span_x_m, span_y_m) / 2.0
        else:
            # Floor must span the longer direction; beams span their own direction.
            span_floor = max(span_x_m, span_y_m)

        floor_vids = self._span_filtered(ComponentType.FLOOR, candidate.floor_family_id, span_floor)
        pb_vids = self._span_filtered(ComponentType.BEAM, candidate.primary_beam_family_id, span_x_m,
                                      line_load_kn_per_m=primary_beam_line_load_kn_per_m)
        sb_vids = self._span_filtered(ComponentType.BEAM, candidate.secondary_beam_family_id, span_y_m,
                                      line_load_kn_per_m=secondary_beam_line_load_kn_per_m)
        col_vids = self._storey_filtered(ComponentType.COLUMN, candidate.column_family_id, storey_count,
                                         axial_demand_kn=column_axial_kn)
        lat_vids = self._storey_filtered(ComponentType.LATERAL, candidate.lateral_family_id, storey_count)

        span_tag = f"sx{span_x_m:.2f}_sy{span_y_m:.2f}"
        expanded: list[AssemblyCandidate] = []

        for fv, pbv, sbv, cv, lv in product(floor_vids, pb_vids, sb_vids, col_vids, lat_vids):
            load_path_method = _resolve_load_path_method(self.query, candidate, fv)
            parts = [
                candidate.candidate_id,
                fv or "N", pbv or "N", sbv or "N", cv or "N", lv or "N",
                span_tag,
            ]
            vid = "__".join(parts)
            expanded.append(replace(
                candidate,
                candidate_id=vid,
                floor_variant_id=fv,
                primary_beam_variant_id=pbv,
                secondary_beam_variant_id=sbv,
                column_variant_id=cv,
                lateral_variant_id=lv,
                load_path_method=load_path_method,
                metadata={
                    **candidate.metadata,
                    "span_x_m": span_x_m,
                    "span_y_m": span_y_m,
                    "expanded_from": candidate.candidate_id,
                },
            ))
        return expanded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _span_filtered(
        self,
        component: ComponentType,
        family_id: str | None,
        required_span: float,
        line_load_kn_per_m: float | None = None,
    ) -> list[str | None]:
        if family_id is None:
            return [None]
        variants = self.query.get_variants_for_family(component, family_id)
        viable = [v for v in variants if _ok_span(v, required_span)]
        if component == ComponentType.BEAM and line_load_kn_per_m is not None:
            viable = [v for v in viable if _ok_beam_capacity(v, required_span, line_load_kn_per_m)]
        # Sort lightest-adequate first so the 20-cap preserves the minimum-carbon candidates.
        viable.sort(key=lambda v: _as_float(v.properties.get("moment_capacity")) or float("inf"))
        viable = viable[: self.max]
        return [v.variant_id for v in viable] if viable else [None]

    def _storey_filtered(
        self,
        component: ComponentType,
        family_id: str | None,
        storeys: int,
        axial_demand_kn: float | None = None,
    ) -> list[str | None]:
        if family_id is None:
            return [None]
        variants = self.query.get_variants_for_family(component, family_id)

        if component == ComponentType.COLUMN and axial_demand_kn is not None:
            # When we have an actual demand, use it as the primary filter.
            # Fall back to the reference-load storey check only for variants
            # that have no axial_capacity declared (e.g. concrete columns
            # whose capacity is not yet populated in the catalog).
            viable = [
                v for v in variants
                if _ok_column_capacity_or_storeys(v, axial_demand_kn, storeys)
            ]
        else:
            viable = [v for v in variants if _ok_storeys(v, storeys)]

        # Sort lightest-adequate first so the 20-cap preserves the minimum-carbon candidates.
        viable.sort(key=lambda v: _as_float(v.properties.get("axial_capacity")) or float("inf"))
        viable = viable[: self.max]
        return [v.variant_id for v in viable] if viable else [None]


# ------------------------------------------------------------------
# Span / storey checks — read from properties dict because the
# repository builder maps "max_span_m" from the CSV but the actual
# column name is "max_span"; similarly "maximum_story_count".
# ------------------------------------------------------------------

def _ok_beam_capacity(v: SystemVariant, span_m: float, line_load_kn_per_m: float) -> bool:
    moment_cap = _as_float(v.properties.get("moment_capacity"))
    shear_cap  = _as_float(v.properties.get("shear_capacity"))
    if moment_cap is None and shear_cap is None:
        return True  # no capacity declared → backward compatible
    M_demand = line_load_kn_per_m * span_m ** 2 / 8.0
    V_demand = line_load_kn_per_m * span_m / 2.0
    if moment_cap is not None and M_demand > moment_cap:
        return False
    if shear_cap is not None and V_demand > shear_cap:
        return False
    return True


def _ok_column_capacity(v: SystemVariant, axial_demand_kn: float) -> bool:
    cap = _as_float(v.properties.get("axial_capacity"))
    if cap is None:
        return True
    return cap >= axial_demand_kn


def _ok_column_capacity_or_storeys(v: SystemVariant, axial_demand_kn: float, storeys: int) -> bool:
    """Use actual demand check when capacity is known; fall back to storey proxy otherwise.

    The maximum_story_count value in the catalog is derived from a reference
    load (typically 15 kPa × 100 m² = 1500 kN/storey).  It is NOT suitable
    as a primary filter when the actual project tributary area and loads are
    known — it will over-size columns for smaller bays and under-size for
    larger ones.  Only use it as a fallback for variants (e.g. un-enriched
    concrete columns) that have no axial_capacity declared.
    """
    cap = _as_float(v.properties.get("axial_capacity"))
    if cap is not None:
        return cap >= axial_demand_kn
    # No capacity data → fall back to reference-load storey count
    return _ok_storeys(v, storeys)


def _ok_span(v: SystemVariant, required: float) -> bool:
    """Return True if this floor variant can span the required distance.

    For composite deck variants (identified by ``secondary_beam_spacing_m``
    being set — the deck trough pitch), the deck panel span is determined by
    secondary beam placement, not the full bay span.  These variants are always
    retained here; ``_secondary_beam_count`` determines how many secondary beams
    are needed so the deck panels stay within max_span.

    For all other floor variants, max_span is checked against the required span.
    """
    # Composite deck profiles: secondary_beam_spacing_m is the deck trough pitch,
    # not a structural spacing.  Span adequacy is handled by _secondary_beam_count.
    secondary_spacing = _as_float(v.properties.get("secondary_beam_spacing_m"))
    if secondary_spacing is not None and secondary_spacing > 0:
        return True

    # Standard path: max_span vs. required bay (or effective panel) span.
    raw = _as_float(v.properties.get("max_span"))
    if raw is None:
        raw = v.span_limits.max_span_m
    if raw is None:
        return True  # no limit declared → always viable
    return raw >= required


def _ok_storeys(v: SystemVariant, required: int) -> bool:
    raw = _as_float(v.properties.get("maximum_story_count"))
    if raw is None:
        return True  # no limit declared → always viable
    return raw >= required  # compare as float; int() would truncate 3.999 → 3


def _resolve_load_path_method(
    query: RepositoryQueryService,
    candidate: AssemblyCandidate,
    floor_variant_id: str | None,
) -> LoadPathMethod:
    if not floor_variant_id:
        return candidate.load_path_method
    try:
        floor_variant = query.get_variant(ComponentType.FLOOR, floor_variant_id)
    except Exception:
        return candidate.load_path_method

    raw = str(floor_variant.properties.get("beam_requirements", "") or "").strip().lower()
    if raw in {"none", "no beams required", "no beams"}:
        return LoadPathMethod.BEAMLESS
    if raw == "secondary":
        return LoadPathMethod.TWO_WAY_WITH_SECONDARY_AND_PRIMARY
    if raw in {"primary", "supporting beams or walls", "integral ribs/joists", ""}:
        return LoadPathMethod.ONE_WAY_WITH_PRIMARY_ONLY
    return candidate.load_path_method


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f  # drop NaN
    except (TypeError, ValueError):
        return None
