"""
Post-ranking design refinement: re-evaluate the winning candidates with the
column and beam design engines and produce a refined summary CSV.

Inputs:
  - summary_assemblies_ranked.csv  (output of run_edca pipeline)
  - beam_variants.csv, column_variants.csv  (catalog raw data)

For each row in the input summary:
  1. Reconstruct the column demand from per-storey factored load × tributary area
     (the tributary area is approximated as span_x × span_y from the row).
  2. Reconstruct the beam demand (factored line load, span).
  3. Look up the column variant; run design_column_full_height() to get sized
     concrete & rebar volumes per linear m of column.
  4. Look up each beam variant; run design_beam() to get sized concrete/steel/
     rebar volumes per linear m of beam.
  5. Compute the per-m² delta from the row's existing per-m² volumes (using the
     same scaling formulas as AssemblyTakeoffEngine: span/bay_area for beams,
     ftf/bay_area for columns).
  6. Derive the new carbon for those components by multiplying by an effective
     carbon factor derived from the row's existing volume → carbon mapping.
  7. Write a refined summary CSV with updated volumes and carbons.

The refined CSV has the same schema as the input.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("design_refinement")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s — %(message)s')


# ============================================================================
# Lightweight mocks: SystemVariant + QueryService + Project + ColumnDemand
# (avoid bootstrapping the full edca_code repository machinery)
# ============================================================================

@dataclass
class _Variant:
    variant_id: str
    family_id: str
    properties: dict


@dataclass
class _Geom:
    storey_count: int
    floor_to_floor_m: float


@dataclass
class _Project:
    geometry: _Geom
    design_options: dict = field(default_factory=dict)


@dataclass
class _Effects:
    axial: float | None = None
    moment_major: float | None = None
    moment_minor: float | None = None


@dataclass
class _Envelope:
    effects: _Effects = field(default_factory=_Effects)


@dataclass
class _ColumnDemand:
    axial_dead_kn: float | None = None
    axial_live_kn: float | None = None
    envelope: _Envelope = field(default_factory=_Envelope)


@dataclass
class _BeamDemand:
    role: str = "primary"
    span_m: float | None = None
    tributary_width_m: float | None = None
    factored_line_load_kn_per_m: float | None = None
    unfactored_line_load_kn_per_m: float | None = None
    envelope: _Envelope = field(default_factory=_Envelope)


class _QueryService:
    """Looks up variants from in-memory catalog DataFrames."""

    def __init__(self, beam_df: pd.DataFrame, column_df: pd.DataFrame):
        # Index by variant_id for lookups
        self.beams   = beam_df.set_index('beam_variant_id', drop=False)
        self.columns = column_df.set_index('column_variant_id', drop=False)

    def _as_variant(self, row: pd.Series, id_col: str, fam_col: str) -> _Variant:
        return _Variant(
            variant_id = str(row.get(id_col, '')),
            family_id  = str(row.get(fam_col, '')),
            properties = {k: row.get(k) for k in row.index},
        )

    def get_variant(self, ct, vid: str) -> _Variant:
        # ct is ComponentType-like; we route by the id naming
        if vid in self.beams.index:
            return self._as_variant(self.beams.loc[vid], 'beam_variant_id', 'beam_family_id')
        if vid in self.columns.index:
            return self._as_variant(self.columns.loc[vid], 'column_variant_id', 'column_family_id')
        raise KeyError(f"variant {vid!r} not found")

    def get_variants_for_family(self, ct, family_id: str) -> list[_Variant]:
        out: list[_Variant] = []
        if family_id in self.beams['beam_family_id'].values:
            sub = self.beams[self.beams['beam_family_id'] == family_id]
            for _, row in sub.iterrows():
                out.append(self._as_variant(row, 'beam_variant_id', 'beam_family_id'))
        if family_id in self.columns['column_family_id'].values:
            sub = self.columns[self.columns['column_family_id'] == family_id]
            for _, row in sub.iterrows():
                out.append(self._as_variant(row, 'column_variant_id', 'column_family_id'))
        return out


def _get_all_beam_variants(query: _QueryService) -> list[_Variant]:
    out = []
    for _, row in query.beams.iterrows():
        out.append(query._as_variant(row, 'beam_variant_id', 'beam_family_id'))
    return out


# Monkey-patch beam_design_engine's _get_all_beam_variants to use our query
# (the real one tries to access query.repo which we don't have)
import edca_code.scripts.core.beam_design_engine as _bde
_bde._get_all_beam_variants = _get_all_beam_variants


# Stub ComponentType for the design engines' isinstance/equality checks
class _CT:
    BEAM = type("CT_BEAM", (), {"value": "beam"})()
    COLUMN = type("CT_COLUMN", (), {"value": "column"})()
    FLOOR = type("CT_FLOOR", (), {"value": "floor"})()
    LATERAL = type("CT_LATERAL", (), {"value": "lateral"})()


# Monkey-patch column_design_engine's ComponentType import (used in
# _find_section_for_load to call query.get_variants_for_family).  The engine
# imports `from .domain_models import ComponentType as CT` lazily; we substitute
# the lookup with our stub via wrapping.
import edca_code.scripts.core.column_design_engine as _cde
_orig_find = _cde._find_section_for_load
def _find_section_patched(query, base_variant, n_ed_kN, f_ck_MPa, n_max=0.7):
    # The original imports ComponentType from domain_models; we provide an
    # alternative by directly calling our query with our CT stub.
    from .column_design_engine import _column_section_area_m2, _DEFAULT_N_MAX
    a_c_min_m2 = (n_ed_kN * 1000) / (n_max * f_ck_MPa * 1e6)
    base_area = _column_section_area_m2(base_variant) or 0.0
    if base_area >= a_c_min_m2:
        return base_variant
    candidates = query.get_variants_for_family(_CT.COLUMN, base_variant.family_id)
    # If family-only doesn't find a larger size, fall back to scanning the catalog
    # for variants whose ID shares a meaningful prefix (e.g. ec_gravity_column_)
    if not any(_column_section_area_m2(v) and _column_section_area_m2(v) >= a_c_min_m2
               for v in candidates):
        # Broaden search: variants whose ID starts with the family prefix's "category"
        # e.g. for ec_gravity_column_350x350 → prefix = ec_gravity_column_
        fam_str = str(base_variant.family_id or "")
        # find last underscore before the size token
        import re
        m = re.match(r'^(.*_)(\d+x\d+)$', fam_str)
        prefix = m.group(1) if m else (fam_str + "_")
        all_cols = []
        for _, row in query.columns.iterrows():
            cv = query._as_variant(row, 'column_variant_id', 'column_family_id')
            if str(cv.variant_id).startswith(prefix):
                all_cols.append(cv)
        candidates = all_cols if all_cols else candidates

    viable: list[tuple[float, _Variant]] = []
    for v in candidates:
        a = _column_section_area_m2(v)
        if a is None: continue
        if a >= a_c_min_m2:
            viable.append((a, v))
    if viable:
        viable.sort(key=lambda x: x[0])
        sized = viable[0][1]
        if sized.variant_id != base_variant.variant_id:
            logger.info("[col_design] Upsized '%s' (A_c=%.4fm²) → '%s' (A_c=%.4fm²) for N_Ed=%.0fkN",
                        base_variant.variant_id, base_area, sized.variant_id, viable[0][0], n_ed_kN)
        return sized
    # No larger variant — return largest available
    all_with_area = []
    for v in candidates:
        a = _column_section_area_m2(v)
        if a is not None:
            all_with_area.append((a, v))
    if all_with_area:
        all_with_area.sort(key=lambda x: x[0], reverse=True)
        largest = all_with_area[0][1]
        logger.warning("[col_design] No variant ≥ A_c=%.4fm² found; using largest '%s' (A_c=%.4fm²)",
                       a_c_min_m2, largest.variant_id, all_with_area[0][0])
        return largest
    return base_variant
_cde._find_section_for_load = _find_section_patched


# ============================================================================
# Refinement constants
# ============================================================================

# Column rebar detailing uplift applied to the EC2 structural+links result from
# design_column_full_height.  Accounts for elements not in the EC2 section check:
#   • Beam-column joint hooks: anchor tails extending 40φ beyond column face
#   • Storey-height lap splice: bars overlap ≈ 30φ, effectively doubling steel
#     over a ~0.5 m zone at mid-height
#   • Practical rounding / bar-count adjustments
# The design-engine already includes EC2 9.5.3 links, so 1.60 ≈ structural+links
# × ~1.33 detailing factor (consistent with run_edca._REBAR_DETAILING_MULT_COL).
_COL_DETAILING_MULT = 1.60

# ============================================================================
# Refinement logic
# ============================================================================

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _refine_row(row: pd.Series, query: _QueryService, project: _Project,
                ftf: float, span_x_m: float, span_y_m: float,
                bay_area: float) -> dict:
    """Refine a single summary row. Returns dict of changes to apply."""
    changes: dict[str, Any] = {}

    # ---- COLUMN ----
    column_vid = str(row.get('column_variant', '') or '')
    if column_vid and column_vid != 'nan':
        try:
            col_variant = query.get_variant(_CT.COLUMN, column_vid)
        except KeyError:
            col_variant = None

        # Skip non-RC columns
        is_rc_col = (col_variant is not None
                     and str(col_variant.properties.get('material_rebar_id') or '').lower()
                         not in {'', 'nan', 'none'})
        if is_rc_col:
            # Per-storey demand from row: g+q area loads × tributary area
            # demand_factored_load_kpa is ULS combined; for storey-by-storey we
            # need g_k and q_k separately. Approximate by splitting from total_unfactored.
            unfactored = _safe(row.get('demand_total_unfactored_kpa'))
            # Typical g:q split for offices ≈ 2:1
            g_kpa = unfactored * 2 / 3
            q_kpa = unfactored * 1 / 3
            trib_area = bay_area   # m²
            demand = _ColumnDemand(
                axial_dead_kn=g_kpa * trib_area,
                axial_live_kn=q_kpa * trib_area,
            )

            try:
                result = _cde.design_column_full_height(
                    base_variant=col_variant,
                    query=query,
                    column_demand_per_storey=demand,
                    project=project,
                )
                if result.get('success'):
                    sized_id = result['sized_variant_id']
                    new_conc_per_m = result['concrete_volume_m3_per_m']
                    # Apply detailing uplift: EC2 structural+links × 1.60 to account
                    # for beam-column joint anchor tails, storey-height lap zones, and
                    # practical bar-count rounding (see _COL_DETAILING_MULT above).
                    new_reb_per_m  = result['rebar_volume_m3_per_m'] * _COL_DETAILING_MULT

                    # Convert per-m of column to per-m² of floor: scale = ftf / bay_area
                    scale = ftf / bay_area
                    new_conc_per_m2 = new_conc_per_m * scale
                    new_reb_per_m2  = new_reb_per_m  * scale

                    # Compute carbon factors from existing row (kgCO2 / m³)
                    cf_conc = (_safe(row.get('column_carbon_concrete_per_m2'))
                               / max(_safe(row.get('column_concrete_volume_per_m2')), 1e-12))
                    cf_reb  = (_safe(row.get('column_carbon_rebar_per_m2'))
                               / max(_safe(row.get('column_rebar_volume_per_m2')), 1e-12))

                    new_conc_carbon = new_conc_per_m2 * cf_conc
                    new_reb_carbon  = new_reb_per_m2  * cf_reb

                    # Take max(old, new) for both
                    changes['column_concrete_volume_per_m2'] = max(_safe(row.get('column_concrete_volume_per_m2')), new_conc_per_m2)
                    changes['column_rebar_volume_per_m2']    = max(_safe(row.get('column_rebar_volume_per_m2')),    new_reb_per_m2)
                    changes['column_carbon_concrete_per_m2'] = max(_safe(row.get('column_carbon_concrete_per_m2')), new_conc_carbon)
                    changes['column_carbon_rebar_per_m2']    = max(_safe(row.get('column_carbon_rebar_per_m2')),    new_reb_carbon)
                    # Overwrite the variant_id only if the engine actually upsized
                    if sized_id and sized_id != column_vid:
                        changes['column_variant'] = sized_id
                    changes['_col_sized_to']                  = sized_id
                    changes['_col_rho_avg']                   = result.get('rho_pct_avg', 0)
            except Exception as exc:
                logger.warning("Column design failed for variant %s: %s", column_vid, exc)

    # ---- BEAMS ----
    for prefix, beam_id_col, role in [
        ('beam',     'beam_variant',           'primary'),
        ('sec_beam', 'secondary_beam_variant', 'secondary'),
    ]:
        beam_vid = str(row.get(beam_id_col, '') or '')
        if not beam_vid or beam_vid == 'nan':
            continue
        try:
            beam_variant = query.get_variant(_CT.BEAM, beam_vid)
        except KeyError:
            continue

        span_m = _safe(row.get('demand_span_m'))
        trib_w = _safe(row.get('demand_trib_width_m'))
        factored = _safe(row.get('demand_factored_load_kpa'))
        line_load = factored * trib_w   # kN/m

        demand = _BeamDemand(
            role=role, span_m=span_m, tributary_width_m=trib_w,
            factored_line_load_kn_per_m=line_load,
        )

        try:
            result = _bde.design_beam(
                base_variant=beam_variant,
                query=query,
                beam_demand=demand,
                project=project,
            )
            if not result.get('success'):
                continue

            # Per-m of beam → per-m² of floor: scale = span / bay_area
            beam_span = span_m if role == 'primary' else span_y_m
            scale = beam_span / bay_area   # rough — same as compute_bom logic

            cat_conc_per_m  = _safe(beam_variant.properties.get('concrete_volume'))
            cat_reb_per_m   = _safe(beam_variant.properties.get('rebar_volume'))
            cat_steel_per_m = _safe(beam_variant.properties.get('steel_volume'))

            new_conc_per_m  = max(cat_conc_per_m,  result['concrete_volume_m3_per_m'])
            # Include shear links: result['rebar_volume_m3_per_m'] is longitudinal only;
            # link_volume_m3_per_m must be added so total beam rebar is not undercounted.
            _beam_link_vol  = result.get('link_volume_m3_per_m') or 0.0
            new_reb_per_m   = max(cat_reb_per_m,   result['rebar_volume_m3_per_m'] + _beam_link_vol)
            new_steel_per_m = max(cat_steel_per_m, result['steel_volume_m3_per_m'])

            new_conc_per_m2  = new_conc_per_m  * scale
            new_reb_per_m2   = new_reb_per_m   * scale
            new_steel_per_m2 = new_steel_per_m * scale

            # Carbon factors from existing row
            for mat, vol_key, carb_key, new_vol_per_m2 in [
                ('concrete', f'{prefix}_concrete_volume_per_m2',         f'{prefix}_carbon_concrete_per_m2',         new_conc_per_m2),
                ('rebar',    f'{prefix}_rebar_volume_per_m2',            f'{prefix}_carbon_rebar_per_m2',            new_reb_per_m2),
                ('steel',    f'{prefix}_structural_steel_volume_per_m2', f'{prefix}_carbon_structural_steel_per_m2', new_steel_per_m2),
            ]:
                old_vol = _safe(row.get(vol_key))
                old_carb = _safe(row.get(carb_key))
                if old_vol > 1e-12:
                    cf = old_carb / old_vol   # kgCO2 / m³
                else:
                    cf = 0.0   # no existing carbon → can't extrapolate
                new_vol = max(old_vol, new_vol_per_m2)
                # Only refine carbon if we have a valid carbon factor
                new_carb = new_vol * cf if cf > 0 else old_carb
                changes[vol_key]  = new_vol
                changes[carb_key] = max(old_carb, new_carb)

            changes[f'_{prefix}_sized_to'] = result.get('sized_variant_id')
        except Exception as exc:
            logger.warning("Beam design failed for variant %s: %s", beam_vid, exc)

    return changes


def refine_summary(input_csv: Path, output_csv: Path,
                   beam_variants_csv: Path, column_variants_csv: Path,
                   ftf: float = 4.0, num_floors: int = 4,
                   winners_only: bool = True,
                   bay_span_x_m: float | None = None,
                   bay_span_y_m: float | None = None) -> pd.DataFrame:
    """Read a summary CSV, refine via design engines, write a refined CSV."""
    df = pd.read_csv(input_csv)
    beams = pd.read_csv(beam_variants_csv)
    cols  = pd.read_csv(column_variants_csv)
    query = _QueryService(beams, cols)
    project = _Project(geometry=_Geom(storey_count=num_floors, floor_to_floor_m=ftf))

    rows_to_refine = df
    if winners_only:
        # Take top 1 per floor_family + structural_class combination
        rows_to_refine = df.sort_values('total_embodied_carbon_per_m2').drop_duplicates(
            subset=['floor_family', 'structural_class'], keep='first')

    logger.info("Refining %d / %d rows (winners_only=%s)", len(rows_to_refine), len(df), winners_only)

    refined = df.copy()
    # Material densities (kg/m³) used to back-fill mass fields when only volume is overridden
    DENSITY = {'concrete': 2400.0, 'rebar': 7850.0, 'structural_steel': 7850.0,
               'pt': 7850.0, 'timber': 500.0, 'screed': 2200.0}

    for idx in rows_to_refine.index:
        row = df.loc[idx]
        # Prefer caller-supplied bay dimensions (since the summary CSV has the SLAB demand,
        # not the BEAM grid).  Fall back to demand fields if not supplied.
        span_x = bay_span_x_m if bay_span_x_m is not None else _safe(row.get('demand_span_m'))
        span_y = bay_span_y_m if bay_span_y_m is not None else (_safe(row.get('demand_trib_width_m')) or span_x)
        bay_area = max(span_x * span_y, 1e-6)
        changes = _refine_row(row, query, project, ftf, span_x, span_y, bay_area)
        for k, v in changes.items():
            if k.startswith('_'):
                continue
            refined.at[idx, k] = v

        # Sync mass fields with refined volumes (mass = volume × density)
        for comp in ['floor', 'beam', 'sec_beam', 'column', 'lateral']:
            for mat, density in DENSITY.items():
                vol_key  = f'{comp}_{mat}_volume_per_m2'
                mass_key = f'{comp}_{mat}_mass_per_m2'
                if vol_key in refined.columns and mass_key in refined.columns:
                    new_mass = _safe(refined.at[idx, vol_key]) * density
                    if new_mass > _safe(refined.at[idx, mass_key]):
                        refined.at[idx, mass_key] = new_mass

        # Recompute total_embodied_carbon_per_m2 and total_carbon_per_component
        for comp in ['floor', 'beam', 'sec_beam', 'column', 'lateral']:
            comp_carbon = 0.0
            for mat in DENSITY:
                ck = f'{comp}_carbon_{mat}_per_m2'
                if ck in refined.columns:
                    comp_carbon += _safe(refined.at[idx, ck])
            comp_total_key = f'{comp}_carbon_per_m2'
            if comp_total_key in refined.columns:
                refined.at[idx, comp_total_key] = comp_carbon

        # Recompute totals across all components/materials
        new_carbon = 0.0
        for comp in ['floor', 'beam', 'sec_beam', 'column', 'lateral']:
            for mat in DENSITY:
                ck = f'{comp}_carbon_{mat}_per_m2'
                if ck in refined.columns:
                    new_carbon += _safe(refined.at[idx, ck])
        refined.at[idx, 'total_embodied_carbon_per_m2'] = new_carbon

        # Recompute total mass/volume by material
        for mat, density in DENSITY.items():
            for unit, agg_key in [('volume_per_m2', f'total_{mat}_volume_per_m2'),
                                  ('mass_per_m2',   f'total_{mat}_mass_per_m2')]:
                if agg_key in refined.columns:
                    total = 0.0
                    for comp in ['floor', 'beam', 'sec_beam', 'column', 'lateral']:
                        ck = f'{comp}_{mat}_{unit}'
                        if ck in refined.columns:
                            total += _safe(refined.at[idx, ck])
                    refined.at[idx, agg_key] = total

    refined.to_csv(output_csv, index=False)
    logger.info("Wrote refined summary: %s (%d rows)", output_csv, len(refined))
    return refined


def main():
    p = argparse.ArgumentParser(description="Post-rank design refinement")
    p.add_argument('--input', '-i', required=True, help='summary_assemblies_ranked.csv')
    p.add_argument('--output', '-o', required=True, help='refined output path')
    p.add_argument('--beams', default='/Users/benjaminsalop/Desktop/Oxford/Research/edca/inputs/canonical/beam_variants.csv')
    p.add_argument('--columns', default='/Users/benjaminsalop/Desktop/Oxford/Research/edca/inputs/canonical/column_variants.csv')
    p.add_argument('--ftf', type=float, default=4.0)
    p.add_argument('--floors', type=int, default=4)
    p.add_argument('--all-rows', action='store_true', help='refine all rows, not just winners')
    p.add_argument('--bay-x', type=float, default=None, help='bay span_x (m); if omitted uses demand_span_m from row')
    p.add_argument('--bay-y', type=float, default=None, help='bay span_y (m)')
    args = p.parse_args()

    refine_summary(
        input_csv=Path(args.input), output_csv=Path(args.output),
        beam_variants_csv=Path(args.beams), column_variants_csv=Path(args.columns),
        ftf=args.ftf, num_floors=args.floors, winners_only=not args.all_rows,
        bay_span_x_m=args.bay_x, bay_span_y_m=args.bay_y,
    )


if __name__ == '__main__':
    main()
