from __future__ import annotations

import math
from typing import Any

from .carbon import BomLine
from .domain_models import AssemblyCandidate, ComponentType, ProjectContext, SystemVariant
from .repositories import RepositoryQueryService

# Volume columns present in the per-component CSVs and their companion
# material-ID columns.  Order matters: first non-null material_id wins when
# a fallback is used.
_MATERIAL_COLUMNS: list[tuple[str, str]] = [
    ("concrete_volume",     "material_concrete_id"),
    ("screed_volume",       "material_screed_id"),
    ("steel_volume",        "material_steel_id"),
    ("rebar_volume",        "material_rebar_id"),
    ("pt_volume",           "material_pt_id"),
    ("timber_volume",       "material_timber_id"),
    ("fireproofing_volume", "material_fireproofing_id"),
]

# Generic fallback material IDs used when the CSV cell is blank.
_FALLBACK_IDS: dict[str, str] = {
    "concrete_volume":      "concrete",
    "screed_volume":        "screed",
    "steel_volume":         "steel",
    "rebar_volume":         "rebar",
    "pt_volume":            "pt",
    "timber_volume":        "timber",
    "fireproofing_volume":  "fireproofing",
}


class AssemblyTakeoffEngine:
    """Compute a per-m² BOM for a variant-resolved assembly from catalog data.

    All quantities are normalised to **per m² of gross floor plate per storey**
    so results are directly comparable across assemblies regardless of building
    size.  Scale up by (total_floor_area × storey_count) for absolute totals.

    Scaling conventions (matching quantity_basis in the catalogs):
      floor    → per_m2  : volume × 1.0
                           (catalog values are already per m² of floor)
      beam     → per_lm  : one beam of length `span` per bay, divided by bay area
                           scale = span / (span_x × span_y) = 1 / perp_span
      column   → per_lm  : one column of height `ftf` per bay, divided by bay area
                           scale = floor_to_floor / bay_area
      lateral  → per_m2  : volume × 1.0
                           (catalog values are per m² of floor plate)
    """

    def __init__(self, query: RepositoryQueryService) -> None:
        self.query = query

    def compute_bom(
        self,
        candidate: AssemblyCandidate,
        project: ProjectContext,
        span_x_m: float,
        span_y_m: float,
        rebar_overrides: dict[str, float] | None = None,
        concrete_overrides: dict[str, float] | None = None,
        steel_overrides: dict[str, float] | None = None,
    ) -> list[BomLine]:
        """Return a BOM in m³ per m² of gross floor plate per storey.

        Override dicts map variant_id → volume (in the catalog's units: m³/linear-m
        for beams/columns, m³/m² for floors).  When provided, the quantity used is
        max(catalog_value, override), so code-check results can only INCREASE the
        volume relative to the catalog heuristic, never reduce it.

        - rebar_overrides:    override for rebar_volume column
        - concrete_overrides: override for concrete_volume column (used when the
                              RC section has been upsized by the design engine)
        - steel_overrides:    override for steel_volume column (used when a steel
                              section has been upsized by the beam design engine)
        """
        bay_area = max(span_x_m * span_y_m, 1e-6)
        ftf = project.geometry.floor_to_floor_m or 3.5
        rebar_ov    = rebar_overrides or {}
        concrete_ov = concrete_overrides or {}
        steel_ov    = steel_overrides or {}
        bom: list[BomLine] = []

        if candidate.floor_variant_id:
            v = self.query.get_variant(ComponentType.FLOOR, candidate.floor_variant_id)
            bom += self._volumes(v, scale=1.0, category="floor",
                                 rebar_overrides=rebar_ov, concrete_overrides=concrete_ov, steel_overrides=steel_ov)

        if candidate.primary_beam_variant_id:
            v = self.query.get_variant(ComponentType.BEAM, candidate.primary_beam_variant_id)
            bom += self._volumes(v, scale=span_x_m / bay_area, category="primary_beam",
                                 rebar_overrides=rebar_ov, concrete_overrides=concrete_ov, steel_overrides=steel_ov)

        if candidate.secondary_beam_variant_id:
            v = self.query.get_variant(ComponentType.BEAM, candidate.secondary_beam_variant_id)
            n_sec = self._secondary_beam_count(candidate, span_x_m, span_y_m)
            bom += self._volumes(v, scale=n_sec * span_y_m / bay_area, category="secondary_beam",
                                 rebar_overrides=rebar_ov, concrete_overrides=concrete_ov, steel_overrides=steel_ov)

        if candidate.column_variant_id:
            v = self.query.get_variant(ComponentType.COLUMN, candidate.column_variant_id)
            bom += self._volumes(v, scale=ftf / bay_area, category="column",
                                 rebar_overrides=rebar_ov, concrete_overrides=concrete_ov, steel_overrides=steel_ov)

        if candidate.lateral_variant_id:
            v = self.query.get_variant(ComponentType.LATERAL, candidate.lateral_variant_id)
            bom += self._volumes(v, scale=1.0, category="lateral",
                                 rebar_overrides=rebar_ov, concrete_overrides=concrete_ov, steel_overrides=steel_ov)

        return bom

    # ------------------------------------------------------------------
    def _secondary_beam_count(self, candidate: AssemblyCandidate,
                              span_x_m: float, span_y_m: float) -> int:
        """Number of secondary beams between primary beams in one bay.

        The floor slab spans between secondary beams (perpendicular to them).
        Secondary beams run parallel to the short bay dimension (span_y) and
        are spaced at intervals along span_x.

        Spacing is derived from the floor variant's ``max_span`` (capped at
        4.5 m per SCI P354 vibration guidance for offices).  The catalog field
        ``secondary_beam_spacing_m`` is the deck trough pitch, not a structural
        secondary beam spacing, and is intentionally ignored here.

        Total secondary beams = ceil(span_x / spacing).
        This equals (n_intermediate + 2 column-line beams), but since the
        column-line beams at span_x = 0 and span_x = span_x are already
        represented by the primary-beam framing, only intermediate beams
        are counted: n_intermediate = ceil(span_x / spacing) - 1.

        Returns at least 1.  If neither field is populated, defaults to 1.
        """
        if not candidate.floor_variant_id:
            return 1
        try:
            fv = self.query.get_variant(ComponentType.FLOOR, candidate.floor_variant_id)
        except Exception:
            return 1
        props = fv.properties
        # Use max_span to determine secondary beam spacing.
        # NOTE: secondary_beam_spacing_m in the catalog is the deck trough pitch (e.g. 1.65m),
        # NOT a structural secondary beam spacing — ignore it here.
        fallback = _as_positive_float(props.get("max_span"))
        # SCI P354 vibration limit for offices: secondary beam spacing ≤ 4.5 m
        spacing = min(fallback, 4.5) if fallback else None
        if spacing is None or spacing <= 0:
            return 1
        # Intermediate secondary beams between the two primary-beam column lines.
        n_intermediate = math.ceil(span_x_m / spacing) - 1
        return max(1, n_intermediate)

    # ------------------------------------------------------------------
    def _volumes(
        self,
        variant: SystemVariant,
        *,
        scale: float,
        category: str,
        rebar_overrides: dict[str, float],
        concrete_overrides: dict[str, float],
        steel_overrides: dict[str, float],
    ) -> list[BomLine]:
        lines: list[BomLine] = []
        props = variant.properties
        for vol_col, mat_col in _MATERIAL_COLUMNS:
            qty = _as_positive_float(props.get(vol_col))
            # Code-check overrides: take max(catalog, override) for rebar/concrete/steel.
            if vol_col == "rebar_volume" and variant.variant_id in rebar_overrides:
                qty = max(qty or 0.0, rebar_overrides[variant.variant_id]) or None
            elif vol_col == "concrete_volume" and variant.variant_id in concrete_overrides:
                qty = max(qty or 0.0, concrete_overrides[variant.variant_id]) or None
            elif vol_col == "steel_volume" and variant.variant_id in steel_overrides:
                qty = max(qty or 0.0, steel_overrides[variant.variant_id]) or None
            # Timber beams: derive cross-section volume from dimensions if the
            # catalog row was not populated (e.g. West Fraser LVL entries that
            # have beam_width / beam_depth but no timber_volume field).
            elif vol_col == "timber_volume" and qty is None:
                bw = _as_positive_float(props.get("beam_width"))
                bd = _as_positive_float(props.get("beam_depth"))
                if bw and bd:
                    qty = bw * bd   # m² cross-section → m³ per linear m of beam
            if qty is None:
                continue
            mat_id = _clean_str(props.get(mat_col)) or _FALLBACK_IDS.get(vol_col)
            if not mat_id:
                continue
            scaled = qty * scale
            if scaled > 0:
                lines.append(BomLine(
                    category=category,
                    material_id=mat_id,
                    quantity=scaled,
                    unit="m3",
                    source=variant.variant_id,
                ))
        return lines


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _as_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if f > 0 and f == f else None  # reject zero, negative, NaN
    except (TypeError, ValueError):
        return None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in {"none", "nan", "na", ""} else None
