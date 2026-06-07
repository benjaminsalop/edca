"""assembly_summary.py — Build assembly-level carbon rankings from per-component summaries.

Takes the per-component ``summary_ranked_all.csv`` DataFrames that the EDCA
pipeline produces and combines them into full building assemblies:

    Floor  +  [Primary Beam]  +  [Secondary Beam]  +  Column  +  Lateral
    ───────────────────────────────────────────────────────────────────────
    → one assembly row per (floor_type × beam_material × column_material)

Beam need is read from the floor variant's ``beam_requirements`` field:
  - "none" / "no beams required"       → no beams added
  - "primary" / "supporting beams or walls" / "integral ribs/joists" / NaN
                                        → primary beam only
  - "secondary"                         → primary + secondary beams

Additionally, floors that cannot span the full bay (needs_secondary_beam flag)
always receive a secondary span-extension beam regardless of beam_requirements.

All carbon values are expected to be already normalised to kgCO₂e / m² GFA.
All beam × column material combinations are tried for each floor type.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from .structural_class import build_assembly_label, infer_structural_class

logger = logging.getLogger(__name__)

_BEAM_SIMPLE_MOMENT_COEFF = 1.0 / 8.0
_BEAM_INTERIOR_SUPPORT_MOMENT_COEFF = 0.106
_BEAM_INTERIOR_SHEAR_COEFF = 0.63
_STEEL_DENSITY_KG_M3 = 7850.0
_G_KN_PER_KG = 9.81 / 1000.0
_CONCRETE_UNIT_WEIGHT_KN_M3 = 25.0


# ---------------------------------------------------------------------------
# beam_requirements vocabulary — handles both old and new values
# ---------------------------------------------------------------------------

def _parse_beam_requirements(floor_row: pd.Series) -> str:
    """Return 'none', 'primary', or 'secondary' from a floor variant row.

    Accepts the new vocabulary ("none", "primary", "secondary") as well as the
    legacy values used in the existing data ("no beams required",
    "supporting beams or walls", "integral ribs/joists").
    """
    raw = str(floor_row.get("beam_requirements", "") or "").strip().lower()
    if raw in ("none", "no beams required", "no beams"):
        return "none"
    if raw in ("secondary",):
        return "secondary"
    # "primary", "supporting beams or walls", "integral ribs/joists", NaN/empty
    # all default to needing at least a primary beam
    return "primary"


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col)
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _secondary_spacing_from_floor(floor_row: pd.Series, span_x_m: float) -> tuple[float | None, str | None]:
    """Return the structural secondary-beam spacing implied by a floor row.

    Floor-system presets can state an explicit secondary support spacing.  Where
    they do not, use the floor's span capacity as the maximum supported module,
    with a 4.5 m upper bound to avoid treating vibration-sensitive floors as if
    arbitrarily wide secondary spacing is acceptable.
    """
    explicit = _row_float(floor_row, "secondary_beam_spacing_m")
    if explicit > 0:
        return explicit, "secondary_beam_spacing_m"

    max_span = _row_float(floor_row, "max_span")
    if max_span > 0:
        return min(max_span, 4.5), "max_span_capped_4.5m"

    stored = _row_float(floor_row, "secondary_beam_spacing_used_m")
    if stored > 0:
        return stored, "stored_spacing"

    if span_x_m > 0:
        return span_x_m, "bay_span_fallback"
    return None, None


def _beam_layout_factors(
    floor_row: pd.Series,
    total_gfa_m2: float | None,
    *,
    needs_primary: bool,
    needs_secondary: bool,
) -> dict[str, float | int | str | None]:
    """Estimate grid-line quantity multipliers for primary and secondary beams.

    Component beam rows are normalised as one line of beam per ideal bay
    (1/span_y).  This converts that ideal-bay basis to an approximate finite
    floor plate: primary beams include perimeter grid lines; secondary beams are
    counted as intermediate support lines inside each bay based on floor-system
    support spacing.
    """
    span_x = _row_float(floor_row, "eval_span_x_m", _row_float(floor_row, "demand_span_m"))
    span_y = _row_float(floor_row, "eval_span_y_m", _row_float(floor_row, "demand_trib_width_m", span_x))
    floorplates = max(1.0, _row_float(floor_row, "demand_storeys", 1.0))
    floor_area = (float(total_gfa_m2 or 0.0) / floorplates) if total_gfa_m2 else 0.0

    n_x = n_y = 1
    if span_x > 0 and span_y > 0 and floor_area > 0:
        plate_x = math.sqrt(floor_area * span_x / span_y)
        plate_y = floor_area / plate_x if plate_x > 0 else 0.0
        n_x = max(1, math.ceil(plate_x / span_x))
        n_y = max(1, math.ceil(plate_y / span_y))

    primary_factor = ((n_y + 1) / n_y) if needs_primary else 0.0

    spacing, spacing_source = _secondary_spacing_from_floor(floor_row, span_x)
    n_secondary = 0
    if needs_secondary and span_x > 0 and spacing and spacing > 0:
        n_secondary = max(0, math.ceil(span_x / spacing) - 1)

    # A secondary beam row has the same one-line-per-bay basis as a primary beam
    # row.  Intermediate support lines have intensity n/span_x, so convert to
    # the beam row basis of 1/span_y.
    secondary_factor = (
        float(n_secondary) * span_y / span_x
        if needs_secondary and span_x > 0 and span_y > 0
        else 0.0
    )
    # Edge/spandrel beams in the perpendicular direction are commonly present
    # even when the primary grid is modelled in one direction. Use the secondary
    # member when available; otherwise the caller can fold this into the primary.
    perimeter_factor = (2.0 / n_x) if needs_primary and n_x > 0 else 0.0

    return {
        "beam_bays_x": n_x,
        "beam_bays_y": n_y,
        "primary_beam_layout_factor": primary_factor,
        "secondary_beam_layout_factor": secondary_factor,
        "perimeter_beam_layout_factor": perimeter_factor,
        "n_secondary_beams": n_secondary,
        "secondary_beam_spacing_m": spacing,
        "secondary_beam_spacing_source": spacing_source,
    }


def _floor_grid_from_row(floor_row: pd.Series, total_gfa_m2: float | None) -> dict[str, float | int]:
    span_x = _row_float(floor_row, "eval_span_x_m", _row_float(floor_row, "demand_span_m"))
    span_y = _row_float(floor_row, "eval_span_y_m", _row_float(floor_row, "demand_trib_width_m", span_x))
    floorplates = max(1.0, _row_float(floor_row, "demand_storeys", 1.0))
    floor_area = (float(total_gfa_m2 or 0.0) / floorplates) if total_gfa_m2 else 0.0
    n_x = n_y = 1
    if span_x > 0 and span_y > 0 and floor_area > 0:
        plate_x = math.sqrt(floor_area * span_x / span_y)
        plate_y = floor_area / plate_x if plate_x > 0 else 0.0
        n_x = max(1, math.ceil(plate_x / span_x))
        n_y = max(1, math.ceil(plate_y / span_y))
    nodes = (n_x + 1) * (n_y + 1)
    return {
        "floor_area_m2": floor_area,
        "beam_bays_x": n_x,
        "beam_bays_y": n_y,
        "column_nodes_per_floor": nodes,
        "column_density_per_m2": (nodes / floor_area) if floor_area > 0 else 0.0,
    }


def _safe_float_col(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Return the first non-null numeric value from a column, or default."""
    if col not in df.columns:
        return default
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[0]) if not s.empty else default


def _interior_beam_demands(uls_area_load_kpa: float, span_m: float, trib_width_m: float) -> tuple[float, float, float]:
    """Return line load, moment, and shear for an interior continuous beam line."""
    line_load = uls_area_load_kpa * trib_width_m
    moment = max(
        _BEAM_SIMPLE_MOMENT_COEFF * line_load * span_m ** 2,
        _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * line_load * span_m ** 2,
    )
    shear = _BEAM_INTERIOR_SHEAR_COEFF * line_load * span_m
    return line_load, moment, shear


def _interior_column_moment_demand(uls_area_load_kpa: float, span_x_m: float, span_y_m: float) -> float:
    """Approximate gravity frame end moment demand on an interior column.

    Each principal direction has beams framing from both sides.  Equal spans will
    partly balance in a real frame analysis, but without frame stiffness data the
    safer catalog screen is to require the column moment capacity to cover the
    two-sided beam end-moment envelope about a principal axis.
    """
    if uls_area_load_kpa <= 0 or span_x_m <= 0 or span_y_m <= 0:
        return 0.0
    m_x = _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * (uls_area_load_kpa * span_y_m) * span_x_m ** 2
    m_y = _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF * (uls_area_load_kpa * span_x_m) * span_y_m ** 2
    return max(m_x, m_y)


def _rc_beam_code_passes(df: pd.DataFrame, span_m: float, line_load_kn_m: float) -> pd.Series:
    """Return EC2 RC-beam pass/fail values for rows without catalog capacities."""
    passes = pd.Series(True, index=df.index)
    if span_m <= 0 or line_load_kn_m <= 0 or not {"beam_width", "beam_depth"}.issubset(df.columns):
        return passes
    material = (
        df["material_family"].fillna("").astype(str).str.strip().str.lower()
        if "material_family" in df.columns
        else pd.Series("", index=df.index)
    )
    conc_id = (
        df["material_concrete_id"].fillna("").astype(str).str.strip().str.lower()
        if "material_concrete_id" in df.columns
        else pd.Series("", index=df.index)
    )
    is_rc = material.eq("concrete") | (conc_id.ne("") & ~conc_id.isin({"nan", "none"}))
    if not is_rc.any():
        return passes

    from edca_code.scripts.code_checks.rc_beam import check_rc_beam

    unit_col = df["unit"].fillna("metric").str.strip().str.lower() if "unit" in df.columns else pd.Series("metric", index=df.index)

    for idx, row in df[is_rc].iterrows():
        try:
            bw = pd.to_numeric(pd.Series([row.get("beam_width")]), errors="coerce").iloc[0]
            bd = pd.to_numeric(pd.Series([row.get("beam_depth")]), errors="coerce").iloc[0]
            if pd.isna(bw) or pd.isna(bd) or float(bw) <= 0 or float(bd) <= 0:
                passes.loc[idx] = False
                continue
            _unit = unit_col.loc[idx] if idx in unit_col.index else "metric"
            if _unit == "imperial":
                b_m = float(bw) * 0.0254  # inches → metres
                h_m = float(bd) * 0.0254
            else:
                b_m = float(bw) / 1000.0 if float(bw) > 10.0 else float(bw)
                h_m = float(bd) / 1000.0 if float(bd) > 10.0 else float(bd)
            result = check_rc_beam(h_m=h_m, b_m=b_m, L_m=span_m, n_ULS=line_load_kn_m)
            passes.loc[idx] = bool(result.get("success")) and bool(result.get("deflection_pass", True))
        except Exception:
            logger.debug("[assembly] RC beam code check failed for row %s", idx, exc_info=True)
            passes.loc[idx] = False
    return passes


def _filter_beams_sw(
    df: pd.DataFrame,
    uls_area_load_kpa: float,
    *,
    secondary: bool = False,
    floor_row: pd.Series | None = None,
    steel_sls_checks: bool = True,
    steel_max_span_depth_ratio: float = 16.0,
    steel_moment_capacity_factor: float = 1.0,
    timber_moment_capacity_factor: float = 0.64,
    steel_include_self_weight: bool = True,
    rc_span_depth_checks: bool = True,
    rc_max_span_depth_ratio: float = 12.0,
) -> pd.DataFrame:
    """Return the subset of beam rows whose moment and shear capacities meet the
    SW-adjusted demand derived from *uls_area_load_kpa* (kN/m²).

    Span and tributary width are read from the beam rows themselves
    (``eval_span_x_m`` / ``eval_span_y_m``), so no external geometry is needed.
    For a secondary beam, the tributary width comes from the floor-system support
    spacing where available rather than a generic half-bay assumption.
    Rows with null capacities are kept (conservative — they can't be checked).
    """
    if not {"moment_capacity", "shear_capacity"}.issubset(df.columns):
        return df  # can't check — pass through

    span = _safe_float_col(df, "eval_span_x_m",
           _safe_float_col(df, "demand_span_m", 0.0))
    trib = _safe_float_col(df, "eval_span_y_m",
           _safe_float_col(df, "demand_trib_width_m", 0.0))
    if secondary:
        spacing, _ = _secondary_spacing_from_floor(floor_row, span) if floor_row is not None else (None, None)
        trib = float(spacing) if spacing and spacing > 0 else trib
    if span <= 0 or trib <= 0:
        return df  # geometry unknown — pass through

    material = (
        df["material_family"].fillna("").astype(str).str.strip().str.lower()
        if "material_family" in df.columns
        else pd.Series("", index=df.index)
    )
    is_steel = material.eq("steel")

    base_line_load = uls_area_load_kpa * trib
    line_load = pd.Series(base_line_load, index=df.index, dtype="float64")
    if steel_include_self_weight and "steel_volume" in df.columns:
        steel_sw_uls = (
            1.35
            * pd.to_numeric(df["steel_volume"], errors="coerce").fillna(0.0)
            * _STEEL_DENSITY_KG_M3
            * _G_KN_PER_KG
        )
        line_load = line_load + steel_sw_uls.where(is_steel, 0.0)
    if "concrete_volume" in df.columns:
        concrete_sw_uls = (
            1.35
            * pd.to_numeric(df["concrete_volume"], errors="coerce").fillna(0.0)
            * _CONCRETE_UNIT_WEIGHT_KN_M3
        )
        line_load = line_load + concrete_sw_uls.where(material.eq("concrete"), 0.0)

    moment_coeff = max(_BEAM_SIMPLE_MOMENT_COEFF, _BEAM_INTERIOR_SUPPORT_MOMENT_COEFF)
    M_dem = line_load * span ** 2 * moment_coeff
    V_dem = line_load * span * _BEAM_INTERIOR_SHEAR_COEFF

    m_cap = pd.to_numeric(df["moment_capacity"], errors="coerce")
    v_cap = pd.to_numeric(df["shear_capacity"],  errors="coerce")
    if steel_moment_capacity_factor > 0:
        m_cap = m_cap.where(~is_steel, m_cap * steel_moment_capacity_factor)
    # Timber catalog stores characteristic moment capacity (f_m,k × W_y).
    # Apply kmod/γM to convert to design capacity before comparing against
    # the ULS demand.  Default: kmod=0.8 (medium-duration), γM=1.25 → 0.64.
    is_timber = material.eq("timber")
    if timber_moment_capacity_factor > 0 and is_timber.any():
        m_cap = m_cap.where(~is_timber, m_cap * timber_moment_capacity_factor)

    passes = (m_cap.isna() | m_cap.ge(M_dem)) & (v_cap.isna() | v_cap.ge(V_dem))
    passes = passes & _rc_beam_code_passes(df, span, base_line_load)
    if rc_span_depth_checks and "beam_depth" in df.columns:
        depth = pd.to_numeric(df["beam_depth"], errors="coerce")
        depth_m = depth.where(depth.le(10.0), depth / 1000.0)
        min_depth = span / max(float(rc_max_span_depth_ratio or 12.0), 1.0)
        passes = passes & (~material.eq("concrete") | depth_m.isna() | depth_m.ge(min_depth))
    if steel_sls_checks and "beam_depth" in df.columns:
        depth = pd.to_numeric(df["beam_depth"], errors="coerce")
        depth_m = depth.where(depth.le(10.0), depth / 1000.0)
        min_depth = span / max(float(steel_max_span_depth_ratio or 16.0), 1.0)
        passes = passes & (~material.eq("steel") | depth_m.isna() | depth_m.ge(min_depth))
    filtered = df[passes]
    if filtered.empty:
        logger.debug(
            "[assembly] SW beam filter: no beams pass M_dem=%.1f-%.1f kNm / V_dem=%.1f-%.1f kN "
            "(uls=%.2f kPa, span=%.1f m, trib=%.1f m%s); relaxing to pre-SW pool",
            float(M_dem.min()), float(M_dem.max()), float(V_dem.min()), float(V_dem.max()),
            uls_area_load_kpa, span, trib,
            ", secondary" if secondary else "",
        )
        return df  # fall back to full set rather than leaving empty
    return filtered


def _filter_columns_sw(df: pd.DataFrame, uls_area_load_kpa: float) -> pd.DataFrame:
    """Return the subset of column rows whose axial capacity meets the
    SW-adjusted demand derived from *uls_area_load_kpa* (kN/m²).

    Tributary bay area and storey count are read from the column rows.
    Rows with null capacities are kept (conservative).
    """
    if "axial_capacity" not in df.columns:
        return df

    span_x  = _safe_float_col(df, "eval_span_x_m",
               _safe_float_col(df, "demand_span_m", 0.0))
    span_y  = _safe_float_col(df, "eval_span_y_m",
               _safe_float_col(df, "demand_trib_width_m", span_x))
    storeys = _safe_float_col(df, "demand_storeys", 1.0)
    bay_area = span_x * span_y
    if bay_area <= 0 or storeys <= 0:
        return df

    N_dem = uls_area_load_kpa * bay_area * storeys   # kN

    a_cap = pd.to_numeric(df["axial_capacity"], errors="coerce")
    passes = a_cap.isna() | a_cap.ge(N_dem)

    filtered = df[passes]
    if filtered.empty:
        logger.debug(
            "[assembly] SW column filter: no columns pass N_dem=%.0f kN "
            "(uls=%.2f kPa, bay=%.1f×%.1f m, %d storeys); relaxing to pre-SW pool",
            N_dem, uls_area_load_kpa, span_x, span_y, int(storeys),
        )
        return df
    return filtered


def build_assembly_rankings(
    floors_df: pd.DataFrame,
    beams_df: pd.DataFrame | None,
    columns_df: pd.DataFrame | None,
    laterals_df: pd.DataFrame | None,
    *,
    passing_only: bool = True,
    top_n_per_type: int | None = None,
    total_gfa_m2: float = 0.0,
    storey_height_m: float = 0.0,
    columns_by_floor: bool = True,
    moment_frame_columns: bool = False,
    steel_beam_sls_checks: bool = True,
    steel_beam_max_span_depth_ratio: float = 16.0,
    steel_secondary_beam_max_span_depth_ratio: float = 16.0,
    steel_primary_beam_moment_capacity_factor: float = 1.0,
    steel_secondary_beam_moment_capacity_factor: float = 1.0,
    steel_beam_moment_capacity_factor: float | None = None,
    steel_beam_include_self_weight: bool = True,
    timber_beam_moment_capacity_factor: float = 0.75,
    rc_beam_span_depth_checks: bool = True,
    rc_beam_max_span_depth_ratio: float = 12.0,
    rc_secondary_beam_max_span_depth_ratio: float = 12.0,
) -> pd.DataFrame:
    """Build assembly-level carbon rankings from per-component summary tables.

    Parameters
    ----------
    floors_df, beams_df, columns_df, laterals_df:
        Per-component ``summary_ranked_all`` DataFrames.
        Any of the non-floor DataFrames may be None / empty.
    passing_only:
        If True, only use variants where ``pass_overall == True``.
    top_n_per_type:
        If set, limit each structural type to the N lowest-carbon floor variants
        (useful to cap very large catalogs).

    Returns
    -------
    pd.DataFrame
        One row per best (floor_type × beam_material × column_material) assembly,
        sorted by ``total_embodied_carbon_per_m2`` ascending.
    """
    floors   = _prep(floors_df,   "floor",   passing_only)
    beams    = _prep(beams_df,    "beam",    passing_only)
    # Secondary beams are sized for half the tributary width (each beam carries
    # slab_span/2 when a mid-span secondary splits the bay), so use the
    # secondary-specific pass flags when available.
    sec_beams = _prep_secondary_beams(beams_df, passing_only)
    # Columns are selected storey-by-storey below.  Do not pre-filter on the
    # whole-building axial pass flag here, because upper storeys may correctly
    # use smaller variants that would fail the ground-floor cumulative load.
    columns  = _prep(columns_df,  "column",  passing_only=False)
    laterals = _prep(laterals_df, "lateral", passing_only)

    # Best lateral overall (lowest carbon — used across all assemblies)
    best_lateral = _best_row(laterals)

    # All available beam and column material families from the passing catalogs
    beam_materials: list[str] = (
        sorted(beams["material_family"].dropna().unique().tolist()) if not beams.empty else []
    )
    col_materials: list[str] = (
        sorted(columns["material_family"].dropna().unique().tolist()) if not columns.empty else []
    )

    rows: list[dict[str, Any]] = []

    if steel_beam_moment_capacity_factor is not None:
        steel_primary_beam_moment_capacity_factor = float(steel_beam_moment_capacity_factor)
        steel_secondary_beam_moment_capacity_factor = float(steel_beam_moment_capacity_factor)

    # Pre-index subsets by material_family so the inner loop doesn't re-filter on every iteration.
    # _prep() already lowercases material_family so equality comparisons work directly.
    col_by_mat: dict[str, pd.DataFrame] = (
        {mat: columns[columns["material_family"] == mat] for mat in col_materials}
        if not columns.empty else {}
    )
    beam_by_mat: dict[str, pd.DataFrame] = (
        {mat: beams[beams["material_family"] == mat] for mat in beam_materials}
        if not beams.empty else {}
    )
    sec_beam_by_mat: dict[str, pd.DataFrame] = (
        {mat: sec_beams[sec_beams["material_family"] == mat] for mat in beam_materials}
        if not sec_beams.empty else {}
    )

    if floors.empty:
        logger.warning("No passing floor variants found; skipping assembly ranking.")
        return pd.DataFrame()

    for (floor_cat, floor_type), grp in floors.groupby(
        [
            _col(floors, "floor_category", "category"),
            _col(floors, "floor_type", "type"),
        ],
        dropna=True,
    ):
        floor_variants_all = grp.sort_values("carbon_per_m2")
        first_req = _parse_beam_requirements(floor_variants_all.iloc[0])
        # Prefer full-span variants only for systems whose support rules do not
        # inherently require secondary beams.  For systems such as two-way slabs
        # with beam_requirements="secondary", the secondary-supported panel is
        # the intended design mode; forcing a full-bay slab here oversizes it.
        if first_req != "secondary" and "needs_secondary_beam" in floor_variants_all.columns:
            full_span = floor_variants_all[
                ~floor_variants_all["needs_secondary_beam"].fillna(False).astype(bool)
            ]
        else:
            full_span = floor_variants_all
        floor_variants = full_span if not full_span.empty else floor_variants_all
        if top_n_per_type:
            floor_variants = floor_variants.head(top_n_per_type)

        best_floor = floor_variants.iloc[0] if not floor_variants.empty else None
        if best_floor is None:
            continue

        # ── SW-adjusted ULS area load for beam / column sizing ──────────────
        # Each floor variant carries its own structural self-weight (swt, kN/m²)
        # authored by the EC2/ACI generators.  Beams and columns must carry
        # γ_G×(SDL+SW) + γ_Q×LL, not just γ_G×SDL + γ_Q×LL.
        # We reconstruct the correct factored area load from the floor row and
        # use it to re-filter beam/column candidates below.
        # Applied for all metric floors where any load data is present (including
        # timber/precast floors whose structural SW is embedded in span-table
        # capacity — swt=0 for those, so only SDL+LL enter the EN1990 combo).
        _floor_swt  = float(best_floor.get("swt",            0.0) or 0.0)
        _floor_sdl  = float(best_floor.get("demand_sdl_kpa",
                            best_floor.get("sdl",            0.0)) or 0.0)
        _floor_ll   = float(best_floor.get("demand_ll_kpa",
                            best_floor.get("ll",             0.0)) or 0.0)
        _floor_unit = str(best_floor.get("unit", "metric") or "metric").strip()
        # EN1990 ULS combination: γ_G×(SDL+SW) + γ_Q×LL
        # Only applied when the floor has an explicit structural self-weight (swt>0),
        # i.e. concrete/steel decks whose SW must be factored.  Timber/precast floors
        # embed their SW in the span-table capacity so swt=0; for those floors the
        # catalog loads (SDL+LL) are already the effective demand and we leave the
        # beam demand as the unfactored sum (SW-ULS is not applied).
        _sw_uls_val = 1.35 * (_floor_sdl + _floor_swt) + 1.5 * _floor_ll
        _sw_uls: float | None = (
            _sw_uls_val
            if (_floor_unit == "metric" and _floor_swt > 0)
            else None
        )
        _beam_uls = _sw_uls if _sw_uls is not None else float(best_floor.get("demand_factored_load_kpa", 0.0) or 0.0)
        if _sw_uls is not None:
            logger.debug(
                "[assembly] %s/%s: floor swt=%.2f kN/m² → SW-adjusted ULS=%.2f kPa "
                "(SDL=%.2f, LL=%.2f)",
                floor_cat, floor_type, _floor_swt, _sw_uls, _floor_sdl, _floor_ll,
            )

        # Determine beam requirements from the best floor variant
        beam_req = _parse_beam_requirements(best_floor)
        needs_primary   = beam_req in ("primary", "secondary")
        needs_sec_inherent = beam_req == "secondary"

        # Span-extension secondary beam: floor can't reach full bay
        floor_needs_sec_span = bool(best_floor.get("needs_secondary_beam", False))

        # If the floor physically cannot span the bay without support,
        # it needs beams regardless of its beam_requirements declaration —
        # unless the system is explicitly beamless (flat slab / PT flat plate).
        if floor_needs_sec_span and beam_req != "none":
            needs_primary = True

        # Determine which beam materials to iterate over
        iter_beam_mats: list[str | None] = beam_materials if needs_primary else [None]

        for beam_mat in iter_beam_mats:
            for col_mat in col_materials:
                # --- Column ---
                col_subset = col_by_mat.get(col_mat, pd.DataFrame())
                col_design = _select_column_schedule(
                    col_subset,
                    _sw_uls if _sw_uls is not None else float(best_floor.get("demand_factored_load_kpa", 0.0) or 0.0),
                    best_floor=best_floor,
                    total_gfa_m2=total_gfa_m2,
                    columns_by_floor=columns_by_floor,
                    moment_frame_columns=moment_frame_columns,
                    storey_height_m_override=storey_height_m,
                )
                best_col = col_design.get("representative")
                if best_col is None:
                    continue

                # --- Primary beam ---
                best_beam: pd.Series | None = None
                if needs_primary and beam_mat and not beams.empty:
                    beam_subset = beam_by_mat.get(beam_mat, pd.DataFrame())
                    if "can_be_primary" in beam_subset.columns:
                        eligible = beam_subset[beam_subset["can_be_primary"].fillna(True).astype(bool)]
                        if not eligible.empty:
                            beam_subset = eligible
                    # Re-filter beams against SW-adjusted moment/shear demand.
                    if _beam_uls > 0 and not beam_subset.empty:
                        beam_subset = _filter_beams_sw(
                            beam_subset,
                            _beam_uls,
                            secondary=False,
                            floor_row=best_floor,
                            steel_sls_checks=steel_beam_sls_checks,
                            steel_max_span_depth_ratio=steel_beam_max_span_depth_ratio,
                            steel_moment_capacity_factor=steel_primary_beam_moment_capacity_factor,
                            timber_moment_capacity_factor=timber_beam_moment_capacity_factor,
                            steel_include_self_weight=steel_beam_include_self_weight,
                            rc_span_depth_checks=rc_beam_span_depth_checks,
                            rc_max_span_depth_ratio=rc_beam_max_span_depth_ratio,
                        )
                    best_beam = _best_row(beam_subset)

                # --- Secondary beam (inherent requirement OR span extension) ---
                # Uses sec_beam_by_mat which is filtered on pass_overall_secondary
                # (half-trib demand), so a lighter variant can be selected here
                # compared to the primary beam catalog.
                best_sec_beam: pd.Series | None = None
                if (needs_sec_inherent or floor_needs_sec_span) and beam_mat:
                    sec_pool = sec_beam_by_mat if sec_beam_by_mat else beam_by_mat
                    sec_subset = sec_pool.get(beam_mat, pd.DataFrame())
                    if sec_subset.empty:
                        # Fall back to primary beam pool if secondary-specific pool is empty
                        sec_subset = beam_by_mat.get(beam_mat, pd.DataFrame())
                    if "can_be_secondary" in sec_subset.columns:
                        eligible = sec_subset[sec_subset["can_be_secondary"].fillna(True).astype(bool)]
                        if not eligible.empty:
                            sec_subset = eligible
                    # Secondary beam demand uses the selected floor system's
                    # support spacing as tributary width.
                    if _beam_uls > 0 and not sec_subset.empty:
                        sec_subset = _filter_beams_sw(
                            sec_subset,
                            _beam_uls,
                            secondary=True,
                            floor_row=best_floor,
                            steel_sls_checks=steel_beam_sls_checks,
                            steel_max_span_depth_ratio=steel_secondary_beam_max_span_depth_ratio,
                            steel_moment_capacity_factor=steel_secondary_beam_moment_capacity_factor,
                            timber_moment_capacity_factor=timber_beam_moment_capacity_factor,
                            steel_include_self_weight=steel_beam_include_self_weight,
                            rc_span_depth_checks=rc_beam_span_depth_checks,
                            rc_max_span_depth_ratio=rc_secondary_beam_max_span_depth_ratio,
                        )
                    best_sec_beam = _best_row(sec_subset)

                _layout = _beam_layout_factors(
                    best_floor,
                    total_gfa_m2,
                    needs_primary=best_beam is not None,
                    needs_secondary=best_sec_beam is not None,
                )
                _primary_factor = float(_layout["primary_beam_layout_factor"] or 0.0)
                _secondary_factor = float(_layout["secondary_beam_layout_factor"] or 0.0)
                _perimeter_factor = float(_layout["perimeter_beam_layout_factor"] or 0.0)
                _beam_bays_x = int(_layout["beam_bays_x"] or 0)
                _orthogonal_grid_factor = (
                    ((_beam_bays_x + 1.0) / _beam_bays_x)
                    if _beam_bays_x > 0
                    else _perimeter_factor
                )
                # Beam-and-column frames need primary-sized beams on the
                # orthogonal column grid as well. Secondary beams remain infill
                # members between those primary lines.
                _is_two_way_frame = str(beam_mat or "").strip().lower() in {"steel", "concrete"}
                if best_sec_beam is not None:
                    if _is_two_way_frame:
                        _primary_factor += _orthogonal_grid_factor
                    else:
                        _secondary_factor += _perimeter_factor
                    if str(beam_mat or "").strip().lower() == "steel":
                        # Steel secondary beams are reported as the full support
                        # module count in the validation schedule. The base
                        # layout counts only intermediate lines, so add one
                        # shared support module per bay to avoid under-reporting
                        # the secondary beam bucket while leaving primary beams
                        # unchanged.
                        _sec_span_x = _row_float(best_floor, "eval_span_x_m", _row_float(best_floor, "demand_span_m", 0.0))
                        _sec_span_y = _row_float(best_floor, "eval_span_y_m", _row_float(best_floor, "demand_trib_width_m", 0.0))
                        if _sec_span_x > 0 and _sec_span_y > 0:
                            _secondary_factor += _sec_span_y / _sec_span_x
                else:
                    _primary_factor += (
                        _orthogonal_grid_factor
                        if _is_two_way_frame
                        else _perimeter_factor
                    )
                _n_sec = int(_layout["n_secondary_beams"] or 0)

                # --- Carbon totals ---
                floor_c    = float(best_floor["carbon_per_m2"])
                beam_c     = float(best_beam["carbon_per_m2"]) * _primary_factor if best_beam is not None else 0.0
                sec_beam_c = float(best_sec_beam["carbon_per_m2"]) * _secondary_factor if best_sec_beam is not None else 0.0
                col_c      = float(col_design.get("carbon_per_m2", 0.0) or 0.0)
                lat_c      = float(best_lateral["carbon_per_m2"]) if best_lateral  is not None else 0.0
                total_c    = floor_c + beam_c + sec_beam_c + col_c + lat_c

                # --- Per-component material-category carbon, volume, and mass ---
                _mat_cats  = ["concrete", "structural_steel", "rebar", "pt", "timber", "screed"]
                _vol_mats  = ["concrete", "structural_steel", "rebar", "pt", "timber", "screed"]
                _mass_mats = ["structural_steel", "rebar", "pt"]
                _comp_sources: list[tuple[str, pd.Series | None]] = [
                    ("floor",    best_floor),
                    ("beam",     best_beam),
                    ("sec_beam", best_sec_beam),
                    ("column",   best_col),
                    ("lateral",  best_lateral),
                ]
                comp_mat: dict[str, float] = {}
                mat_totals:  dict[str, float] = {m: 0.0 for m in _mat_cats}
                vol_totals:  dict[str, float] = {m: 0.0 for m in _vol_mats}
                mass_totals: dict[str, float] = {m: 0.0 for m in _mass_mats}
                for comp_prefix, src in _comp_sources:
                    if comp_prefix == "beam":
                        _qty_mult = _primary_factor
                    elif comp_prefix == "sec_beam":
                        _qty_mult = _secondary_factor
                    else:
                        _qty_mult = 1.0
                    for mat in _mat_cats:
                        src_col = f"carbon_{mat}_per_m2"
                        if comp_prefix == "column":
                            val = float(col_design.get(src_col, 0.0) or 0.0)
                        else:
                            val = float(src.get(src_col, 0.0) or 0.0) * _qty_mult if src is not None else 0.0
                        comp_mat[f"{comp_prefix}_carbon_{mat}_per_m2"] = val
                        mat_totals[mat] += val
                    for mat in _vol_mats:
                        src_col = f"{mat}_volume_per_m2"
                        if comp_prefix == "column":
                            val = float(col_design.get(src_col, 0.0) or 0.0)
                        else:
                            val = float(src.get(src_col, 0.0) or 0.0) * _qty_mult if src is not None else 0.0
                        comp_mat[f"{comp_prefix}_{mat}_volume_per_m2"] = val
                        vol_totals[mat] += val
                    for mat in _mass_mats:
                        src_col = f"{mat}_mass_per_m2"
                        if comp_prefix == "column":
                            val = float(col_design.get(src_col, 0.0) or 0.0)
                        else:
                            val = float(src.get(src_col, 0.0) or 0.0) * _qty_mult if src is not None else 0.0
                        comp_mat[f"{comp_prefix}_{mat}_mass_per_m2"] = val
                        mass_totals[mat] += val
                # Totals per m² across all components
                for mat in _vol_mats:
                    comp_mat[f"total_{mat}_volume_per_m2"] = vol_totals[mat]
                for mat in _mass_mats:
                    comp_mat[f"total_{mat}_mass_per_m2"] = mass_totals[mat]
                # Absolute building totals (per_m² × GFA)
                if total_gfa_m2 > 0:
                    for mat in _vol_mats:
                        comp_mat[f"total_{mat}_volume_m3"] = vol_totals[mat] * total_gfa_m2
                    for mat in _mass_mats:
                        comp_mat[f"total_{mat}_mass_kg"] = mass_totals[mat] * total_gfa_m2

                # Density-derived mass for volume-based materials (no pre-computed mass field)
                _density_kg_m3 = {"concrete": 2400.0, "timber": 500.0, "screed": 1800.0}
                for comp_prefix, _ in _comp_sources:
                    for mat, density in _density_kg_m3.items():
                        vol = comp_mat.get(f"{comp_prefix}_{mat}_volume_per_m2", 0.0)
                        comp_mat[f"{comp_prefix}_{mat}_mass_per_m2"] = vol * density
                for mat, density in _density_kg_m3.items():
                    total_vol = vol_totals.get(mat, 0.0)
                    comp_mat[f"total_{mat}_mass_per_m2"] = total_vol * density
                    if total_gfa_m2 > 0:
                        comp_mat[f"total_{mat}_mass_kg"] = total_vol * density * total_gfa_m2

                # --- Structural class label (verbose) ---
                floor_meta = _mock_family(
                    material_family=best_floor.get("material_family"),
                    floor_category=floor_cat,
                    floor_type=floor_type,
                )
                beam_meta = _mock_family(
                    material_family=beam_mat,
                    beam_category=best_beam.get("beam_category") if best_beam is not None else None,
                ) if best_beam is not None else None
                sec_beam_meta = _mock_family(
                    material_family=beam_mat,
                    beam_category=best_sec_beam.get("beam_category") if best_sec_beam is not None else None,
                ) if best_sec_beam is not None else None
                col_meta = _mock_family(
                    material_family=col_mat,
                    column_category=best_col.get("column_category") if best_col is not None else None,
                )

                structural_class = build_assembly_label(  # type: ignore[arg-type]
                    floor_meta, beam_meta, col_meta, sec_beam_meta
                )

                row: dict[str, Any] = {
                    "structural_class":           structural_class,
                    "floor_category":             floor_cat,
                    "floor_type":                 floor_type,
                    "beam_material":              beam_mat,
                    "column_material":            col_mat,
                    "assembly_family":            _assembly_family_label(floor_cat, floor_type, beam_mat, col_mat),
                    "governing_material":         _dominant_material_label(mat_totals),
                    "demand_span_m":              best_floor.get("demand_span_m"),
                    "demand_trib_width_m":        best_floor.get("demand_trib_width_m"),
                    "demand_total_unfactored_kpa": best_floor.get("demand_total_unfactored_kpa"),
                    "demand_factored_load_kpa":   best_floor.get("demand_factored_load_kpa"),
                    "demand_storeys":             best_floor.get("demand_storeys"),
                    "floor_overall_depth":        best_floor.get("overall_depth", best_floor.get("slab_depth")),
                    "beam_depth":                 best_beam.get("beam_depth") if best_beam is not None else None,
                    "column_depth":               col_design.get("avg_column_depth", best_col.get("column_depth") if best_col is not None else None),
                    "column_moment_demand_kNm":   col_design.get("column_moment_demand_kNm"),
                    "beam_required":              best_beam is not None,
                    "secondary_beam_required":    best_sec_beam is not None,
                    "n_secondary_beams":          _n_sec if best_sec_beam is not None else 0,
                    "secondary_beam_spacing_m":   _layout["secondary_beam_spacing_m"] if best_sec_beam is not None else None,
                    "secondary_beam_spacing_source": _layout["secondary_beam_spacing_source"] if best_sec_beam is not None else None,
                    "primary_beam_layout_factor": _primary_factor,
                    "secondary_beam_layout_factor": _secondary_factor,
                    "perimeter_beam_layout_factor": _perimeter_factor,
                    "beam_bays_x":                _layout["beam_bays_x"],
                    "beam_bays_y":                _layout["beam_bays_y"],
                    # component identifiers
                    "floor_variant":  best_floor.get("system_variant",   best_floor.get("floor_variant_id")),
                    "floor_family":   best_floor.get("system_family",    best_floor.get("floor_family_id")),
                    "beam_variant":   best_beam.get("system_variant",    best_beam.get("beam_variant_id"))       if best_beam     is not None else None,
                    "beam_family":    best_beam.get("system_family",     best_beam.get("beam_family_id"))        if best_beam     is not None else None,
                    "secondary_beam_variant": best_sec_beam.get("system_variant", best_sec_beam.get("beam_variant_id")) if best_sec_beam is not None else None,
                    "secondary_beam_family":  best_sec_beam.get("system_family",  best_sec_beam.get("beam_family_id"))  if best_sec_beam is not None else None,
                    "column_variant": col_design.get("variant_summary"),
                    "column_family":  col_design.get("family_summary"),
                    "column_schedule_by_floor": col_design.get("schedule_by_floor"),
                    "column_count_per_floor": col_design.get("column_count_per_floor"),
                    "column_grid_bays_x": col_design.get("column_grid_bays_x"),
                    "column_grid_bays_y": col_design.get("column_grid_bays_y"),
                    "column_density_per_m2": col_design.get("column_density_per_m2"),
                    "column_schedule_unique_variants": col_design.get("unique_variants"),
                    "lateral_variant": best_lateral.get("system_variant", best_lateral.get("lateral_variant_id")) if best_lateral is not None else None,
                    "lateral_family":  best_lateral.get("system_family",  best_lateral.get("lateral_family_id"))  if best_lateral is not None else None,
                    # per-component carbon
                    "floor_carbon_per_m2":          floor_c,
                    "beam_carbon_per_m2":           beam_c,
                    "secondary_beam_carbon_per_m2": sec_beam_c,
                    "column_carbon_per_m2":         col_c,
                    "lateral_carbon_per_m2":        lat_c,
                    # per-material-category carbon (sum across all components)
                    "mat_concrete_per_m2":         mat_totals["concrete"],
                    "mat_structural_steel_per_m2": mat_totals["structural_steel"],
                    "mat_rebar_per_m2":            mat_totals["rebar"],
                    "mat_pt_per_m2":               mat_totals["pt"],
                    "mat_timber_per_m2":           mat_totals["timber"],
                    "mat_screed_per_m2":           mat_totals["screed"],
                    # total
                    "total_embodied_carbon_per_m2": total_c,
                    # per-component × per-material breakdown
                    **comp_mat,
                }
                rows.append(row)

    if not rows:
        logger.warning(
            "build_assembly_rankings: no assembly rows generated — "
            "check that per-component DataFrames have passing variants"
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Keep the best-carbon assembly per structural class label
    idx_best = df.groupby("structural_class", dropna=False)["total_embodied_carbon_per_m2"].idxmin()
    df_best = df.loc[idx_best].sort_values("total_embodied_carbon_per_m2").reset_index(drop=True)
    df_best["rank_carbon"] = df_best.index + 1

    front = [
        "rank_carbon", "structural_class", "total_embodied_carbon_per_m2",
        "floor_carbon_per_m2", "beam_carbon_per_m2", "secondary_beam_carbon_per_m2",
        "column_carbon_per_m2", "lateral_carbon_per_m2",
        "mat_concrete_per_m2", "mat_structural_steel_per_m2", "mat_rebar_per_m2",
        "mat_pt_per_m2", "mat_timber_per_m2", "mat_screed_per_m2",
        "floor_category", "floor_type", "beam_material", "column_material",
        "assembly_family", "governing_material",
        "demand_span_m", "demand_trib_width_m", "demand_total_unfactored_kpa",
        "demand_factored_load_kpa", "demand_storeys",
        "floor_overall_depth", "beam_depth", "column_depth",
    ]
    rest = [c for c in df_best.columns if c not in front]
    return df_best[front + rest]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prep(df: pd.DataFrame | None, component: str, passing_only: bool) -> pd.DataFrame:
    return _prep_with_flag(df, component, passing_only, pass_col="pass_overall")


def _prep_secondary_beams(df: pd.DataFrame | None, passing_only: bool) -> pd.DataFrame:
    """Like _prep("beam") but filters on pass_overall_secondary when available.

    pass_overall_secondary uses trib = slab_span/2, matching the actual demand on
    each beam (primary and secondary) when a mid-span secondary beam splits the bay.
    Falls back to pass_overall when the secondary-specific flag is absent.
    """
    flag = "pass_overall_secondary" if (
        df is not None and hasattr(df, "columns") and "pass_overall_secondary" in df.columns
    ) else "pass_overall"
    return _prep_with_flag(df, "beam", passing_only, pass_col=flag)


def _prep_with_flag(
    df: pd.DataFrame | None, component: str, passing_only: bool, *, pass_col: str
) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()
    df = df.copy()
    if "carbon_per_m2" not in df.columns and "carbon_total_per_m2" in df.columns:
        df["carbon_per_m2"] = pd.to_numeric(df["carbon_total_per_m2"], errors="coerce")
    else:
        df["carbon_per_m2"] = pd.to_numeric(df.get("carbon_per_m2"), errors="coerce")
    if "material_family" in df.columns:
        df["material_family"] = df["material_family"].astype(str).str.strip().str.lower()
    if passing_only and pass_col in df.columns:
        mask = df[pass_col].fillna(False).astype(bool)
        passed = df[mask]
        if not passed.empty:
            return passed
        logger.debug("%s: no passing variants found with %s; relaxing filter", component, pass_col)
    return df


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    valid = df.dropna(subset=["carbon_per_m2"])
    if valid.empty:
        return None
    return valid.sort_values("carbon_per_m2").iloc[0]


_COLUMN_AVG_COLS = [
    "carbon_per_m2",
    "carbon_concrete_per_m2",
    "carbon_structural_steel_per_m2",
    "carbon_rebar_per_m2",
    "carbon_pt_per_m2",
    "carbon_timber_per_m2",
    "carbon_screed_per_m2",
    "concrete_volume_per_m2",
    "structural_steel_volume_per_m2",
    "rebar_volume_per_m2",
    "pt_volume_per_m2",
    "timber_volume_per_m2",
    "screed_volume_per_m2",
    "structural_steel_mass_per_m2",
    "rebar_mass_per_m2",
    "pt_mass_per_m2",
]


def _material_density_kg_m3(material: str) -> float:
    return {
        "concrete": 2400.0,
        "structural_steel": 7850.0,
        "rebar": 7850.0,
        "pt": 7850.0,
        "timber": 500.0,
        "screed": 1800.0,
    }.get(material, 0.0)


def _column_raw_quantity_per_m2(row: pd.Series, material: str, storey_height_m: float, column_density_per_m2: float) -> float | None:
    """Scale raw per-linear-m column quantity to per-floorplate m2.

    Component summaries may be stale if the upstream design override changed.
    Prefer raw catalog/code-check quantities where available because they carry
    the actual column area per linear metre.
    """
    source_col = {
        "concrete": "concrete_volume",
        "structural_steel": "steel_volume",
        "rebar": "rebar_volume",
        "pt": "pt_volume",
        "timber": "timber_volume",
        "screed": "screed_volume",
    }.get(material)
    if not source_col or source_col not in row.index or storey_height_m <= 0 or column_density_per_m2 <= 0:
        return None
    value = pd.to_numeric(pd.Series([row.get(source_col)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    # RC concrete columns are shared between adjacent storeys (column centreline sits
    # at mid-height), so each floor "owns" half a storey height worth of column.
    # Timber and steel catalog volumes already encode the per-column-per-floor
    # quantity correctly, so they use the full recovered storey height.
    h = storey_height_m * 0.5 if material == "concrete" else storey_height_m

    # Imperial column catalog volumes are in ft³/ft (per linear foot of column height).
    # storey_height_m is in metres, so we must convert to feet before multiplying.
    # "ft" catalogs (AISC sections) are also imperial.
    _row_unit = str(row.get("unit", "metric") or "metric").strip().lower()
    _is_imperial_col = _row_unit in ("imperial", "ft")
    if _is_imperial_col:
        h = h / 0.3048  # metres → feet, so ft³/ft × ft × col/m² = ft³/m²

    raw_qty = float(value) * h * column_density_per_m2  # ft³/m² (imperial) or m³/m² (metric)

    # Convert ft³/m² → m³/m² so that downstream mass × density (kg/m³) gives kg/m².
    _FT3_TO_M3 = 0.028317
    return raw_qty * _FT3_TO_M3 if _is_imperial_col else raw_qty


def _steel_column_buckling_capacity_kn(row: pd.Series, clear_height_m: float) -> float | None:
    """Approximate EC3 buckling capacity for steel columns from area and depth.

    Steel catalog axial_capacity values are often squash loads.  For slender
    building columns, the governing resistance is buckling about the weak axis.
    This estimate intentionally uses only robust fields available in the catalog:
    cross-sectional area from steel_volume and an approximate minor radius of
    gyration from the smaller outside dimension.
    """
    material = str(row.get("material_family", "") or "").strip().lower()
    steel_vol = pd.to_numeric(pd.Series([row.get("steel_volume")]), errors="coerce").iloc[0]
    if material != "steel" or pd.isna(steel_vol) or float(steel_vol) <= 0 or clear_height_m <= 0:
        return None

    width = pd.to_numeric(pd.Series([row.get("column_width")]), errors="coerce").iloc[0]
    depth = pd.to_numeric(pd.Series([row.get("column_depth")]), errors="coerce").iloc[0]
    if pd.isna(width) or pd.isna(depth) or float(width) <= 0 or float(depth) <= 0:
        return None

    a_m2 = float(steel_vol)
    minor_dim_m = min(float(width), float(depth))
    # Conservative minor-axis radius.  Thin-walled CHS/SHS/RHS sections are
    # around D/3 to D/sqrt(8); using D/4 avoids overstating catalog capacity when
    # wall thickness/section tables are unavailable.
    i_m = minor_dim_m / 4.0
    if i_m <= 0:
        return None

    fy_mpa = 355.0
    e_mpa = 210_000.0
    alpha = 0.49  # EC3 curve c, appropriate/conservative for many hollow/UC columns
    lambda_bar = (clear_height_m / i_m) / (math.pi * math.sqrt(e_mpa / fy_mpa))
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    disc = max(phi ** 2 - lambda_bar ** 2, 0.0)
    chi = min(1.0, 1.0 / (phi + math.sqrt(disc)))
    return chi * a_m2 * fy_mpa * 1e6 / 1e3


def _timber_column_design_capacity_kn(row: pd.Series, clear_height_m: float) -> float | None:
    """EC5 design axial capacity for timber columns (kN).

    Timber catalog rows store the gross characteristic compression capacity
    (fc,0,g,k × A) with no material partial factor, load-duration reduction, or
    column-buckling reduction applied.  This function converts that value to the
    EC5 design capacity:

        N_d = kc × kmod/γM × N_char

    where kc is the instability factor from EC5 §6.3.2 and kmod/γM = 0.64
    (medium-duration loading, glulam service class 1, γM = 1.25).

    E₀,₀₅ ≈ 400 × fc,0,g,k holds for GL24h, GL28h and GL32h to within ±5%.
    """
    material = str(row.get("material_family", "") or "").strip().lower()
    axial_cap = pd.to_numeric(pd.Series([row.get("axial_capacity")]), errors="coerce").iloc[0]
    if material != "timber" or pd.isna(axial_cap) or float(axial_cap) <= 0 or clear_height_m <= 0:
        return None

    width = pd.to_numeric(pd.Series([row.get("column_width")]), errors="coerce").iloc[0]
    depth = pd.to_numeric(pd.Series([row.get("column_depth")]), errors="coerce").iloc[0]
    if pd.isna(width) or pd.isna(depth) or float(width) <= 0 or float(depth) <= 0:
        return None

    b_m = float(width)
    h_m = float(depth)
    a_m2 = b_m * h_m
    min_dim_m = min(b_m, h_m)

    # Radius of gyration for the minor axis (rectangular section)
    i_m = min_dim_m / math.sqrt(12.0)
    if i_m <= 0:
        return None

    # Derive fc,0,g,k from catalog squash load (N/mm²)
    fc0k_mpa = float(axial_cap) * 1000.0 / (a_m2 * 1e6)
    if fc0k_mpa <= 0:
        return None

    # E0,05 ≈ 400 × fc,0,g,k for standard structural glulam grades
    e005_mpa = 400.0 * fc0k_mpa

    # EC5 §6.3.2 relative slenderness
    lambda_rel = (clear_height_m / i_m) / (math.pi * math.sqrt(e005_mpa / fc0k_mpa))

    # EC5 imperfection factor βc = 0.1 for glulam (0.2 for solid sawn timber)
    beta_c = 0.1
    k = 0.5 * (1.0 + beta_c * (lambda_rel - 0.3) + lambda_rel ** 2)
    disc = max(k ** 2 - lambda_rel ** 2, 0.0)
    kc = min(1.0, 1.0 / (k + math.sqrt(disc)))

    # kmod/γM = 0.8/1.25 = 0.64 (medium-duration load, service class 1)
    kmod_over_gm = 0.64
    return kmod_over_gm * kc * float(axial_cap)


def _column_capacity_kn(row: pd.Series, clear_height_m: float = 0.0) -> float | None:
    """Return axial capacity in kN, including RC estimate, steel buckling and
    timber EC5 design capacity."""
    cap = pd.to_numeric(pd.Series([row.get("axial_capacity")]), errors="coerce").iloc[0]
    candidates: list[float] = []
    steel_buckling = _steel_column_buckling_capacity_kn(row, clear_height_m)
    timber_design  = _timber_column_design_capacity_kn(row, clear_height_m)
    if steel_buckling is not None and steel_buckling > 0:
        candidates.append(float(steel_buckling))
    elif timber_design is not None and timber_design > 0:
        candidates.append(float(timber_design))
    elif pd.notna(cap) and float(cap) > 0:
        candidates.append(float(cap))

    # Catalog gap/underestimate fallback used elsewhere in the pipeline:
    # Nrd = 0.8 * (fcd * Ac + fyd * As)
    ac = pd.to_numeric(pd.Series([row.get("concrete_volume")]), errors="coerce").iloc[0]
    if pd.notna(ac) and float(ac) > 0:
        as_ = pd.to_numeric(pd.Series([row.get("rebar_volume")]), errors="coerce").iloc[0]
        as_m2 = float(as_) if pd.notna(as_) else 0.0
        candidates.append(0.8 * (20.0 * float(ac) * 1e6 + 435.0 * as_m2 * 1e6) / 1e3)

    return max(candidates) if candidates else None


def _select_column_schedule(
    columns: pd.DataFrame,
    uls_area_load_kpa: float,
    *,
    best_floor: pd.Series,
    total_gfa_m2: float,
    columns_by_floor: bool = True,
    moment_frame_columns: bool = False,
    storey_height_m_override: float = 0.0,
) -> dict[str, Any]:
    """Pick column variants for the building.

    If columns_by_floor is true, pick the lightest adequate variant for each
    vertical storey segment.  Otherwise, pick one representative variant using
    the average axial demand over all column storeys.
    """
    if columns is None or columns.empty:
        return {}

    pool = columns.dropna(subset=["carbon_per_m2"]).copy()
    if pool.empty:
        return {}

    floorplates = int(float(best_floor.get("demand_storeys", 0.0) or 0.0))
    if floorplates <= 0:
        floorplates = int(float(_safe_float_col(pool, "demand_storeys", 1.0) or 1.0))
    # demand_storeys/NUM_FLOORS is the number of stacked floor loads carried by
    # the ground-storey column.  Do not subtract one here: the component-level
    # demand path already uses the same convention.
    storeys = max(floorplates, 1)

    # Prefer the column pool's eval spans: for irregular bays (e.g. ONE_WAY_IRREGULAR)
    # column rows carry the correct beam_span × slab_span, while the floor row stores
    # span_x = span_y = slab_span (giving a spuriously squared bay area).
    span_x = _safe_float_col(pool, "eval_span_x_m", _safe_float_col(pool, "demand_span_m", 0.0))
    span_y = _safe_float_col(pool, "eval_span_y_m", _safe_float_col(pool, "demand_trib_width_m", span_x))
    if span_x <= 0:
        span_x = float(best_floor.get("eval_span_x_m", best_floor.get("demand_span_m", 0.0)) or 0.0)
    if span_y <= 0:
        span_y = float(best_floor.get("eval_span_y_m", best_floor.get("demand_trib_width_m", span_x)) or 0.0)
    bay_area = span_x * span_y if span_x > 0 and span_y > 0 else 0.0
    if bay_area <= 0:
        representative = _best_row(pool)
        return {"representative": representative}
    # Prefer the explicitly-passed storey height (always in metres, already converted
    # from imperial by parse.py).  Fall back to back-calculating from the BOM-tracked
    # volume ratio only when no override is provided (metric-only pipelines).
    if storey_height_m_override > 0:
        storey_height_m = float(storey_height_m_override)
    else:
        storey_height_m = 0.0
        for raw_col, scaled_col in [
            ("concrete_volume", "concrete_volume_per_m2"),
            ("steel_volume", "structural_steel_volume_per_m2"),
            ("timber_volume", "timber_volume_per_m2"),
        ]:
            if raw_col in pool.columns and scaled_col in pool.columns:
                raw = pd.to_numeric(pool[raw_col], errors="coerce")
                scaled = pd.to_numeric(pool[scaled_col], errors="coerce")
                valid = raw.gt(0) & scaled.gt(0)
                if valid.any():
                    storey_height_m = float((scaled[valid] * bay_area / raw[valid]).median())
                    break
        if storey_height_m <= 0:
            storey_height_m = _safe_float_col(pool, "column_height", 0.0)

    grid = _floor_grid_from_row(best_floor, total_gfa_m2)
    floor_area_m2 = float(grid.get("floor_area_m2", 0.0) or 0.0)
    column_count = float(grid.get("column_nodes_per_floor", 0.0) or 0.0) if floor_area_m2 > 0 else None
    column_density = float(grid.get("column_density_per_m2", 0.0) or 0.0)

    caps = pool.apply(lambda r: _column_capacity_kn(r, storey_height_m), axis=1)
    # Preserve legacy permissiveness for rows without capacity data by keeping
    # them available after all testable options.  This avoids dropping catalogs
    # that do not yet carry axial capacities, while preferring checked rows.
    pool["_capacity_kn"] = caps
    pool["_capacity_sort"] = pool["_capacity_kn"].fillna(float("inf"))
    if moment_frame_columns and "moment_capacity" in pool.columns:
        pool["_moment_capacity_knm"] = pd.to_numeric(pool["moment_capacity"], errors="coerce")
        pool["_moment_capacity_sort"] = pool["_moment_capacity_knm"].fillna(float("inf"))
    else:
        pool["_moment_capacity_sort"] = float("inf")
    moment_demand_knm = (
        _interior_column_moment_demand(float(uls_area_load_kpa or 0.0), span_x, span_y)
        if moment_frame_columns
        else 0.0
    )

    selected: list[pd.Series] = []
    schedule: list[dict[str, Any]] = []

    demand_levels = (
        [(floor_index, storeys - floor_index) for floor_index in range(storeys)]
        if columns_by_floor
        else [(None, (storeys + 1) / 2.0)]
    )

    for floor_index, floors_supported in demand_levels:
        demand_kn = float(uls_area_load_kpa or 0.0) * bay_area * floors_supported
        adequate = pool[
            pool["_capacity_sort"].ge(demand_kn)
            & pool["_moment_capacity_sort"].ge(moment_demand_knm)
        ]
        if adequate.empty:
            adequate = pool.sort_values(
                ["_capacity_sort", "_moment_capacity_sort"],
                ascending=[False, False],
            ).head(1)
        choice = adequate.sort_values("carbon_per_m2", ascending=True).iloc[0]
        selected.append(choice)
        if columns_by_floor:
            schedule.append({
                "floor": floor_index,
                "column": choice.get("system_family", choice.get("column_family_id")),
            })
        else:
            schedule.append({
                "floor": "average",
                "column": choice.get("system_family", choice.get("column_family_id")),
            })

    _internal_cols = ["_capacity_kn", "_capacity_sort", "_moment_capacity_knm", "_moment_capacity_sort"]
    selected_df = pd.DataFrame([s.drop(labels=[c for c in _internal_cols if c in s.index]).to_dict() for s in selected])
    representative = selected[0].drop(labels=[c for c in _internal_cols if c in selected[0].index])

    out: dict[str, Any] = {"representative": representative}
    segment_weight = storeys / floorplates if floorplates > 0 else 1.0
    for col in _COLUMN_AVG_COLS:
        vals = None
        if storey_height_m > 0 and bay_area > 0:
            for material in ["concrete", "structural_steel", "rebar", "pt", "timber", "screed"]:
                if col == f"{material}_volume_per_m2":
                    raw_vals = [
                        _column_raw_quantity_per_m2(row, material, storey_height_m, column_density)
                        for _, row in selected_df.iterrows()
                    ]
                    if any(v is not None for v in raw_vals):
                        vals = pd.Series([float(v or 0.0) for v in raw_vals])
                    break
                if col == f"{material}_mass_per_m2":
                    raw_vals = [
                        _column_raw_quantity_per_m2(row, material, storey_height_m, column_density)
                        for _, row in selected_df.iterrows()
                    ]
                    density = _material_density_kg_m3(material)
                    if density > 0 and any(v is not None for v in raw_vals):
                        vals = pd.Series([float(v or 0.0) * density for v in raw_vals])
                    break
        if vals is None:
            vals = (
                pd.to_numeric(selected_df[col], errors="coerce").fillna(0.0)
                if col in selected_df.columns
                else pd.Series([0.0] * len(selected_df))
            )
        out[col] = float(vals.mean() * segment_weight)

    for col, key in [("column_depth", "avg_column_depth"), ("column_width", "avg_column_width")]:
        if col in selected_df.columns:
            vals = pd.to_numeric(selected_df[col], errors="coerce").dropna()
            if not vals.empty:
                out[key] = float(vals.mean())

    variants = [str(item["column"]) for item in schedule]
    families = [str(item["column"]) for item in schedule]
    unique_variants = list(dict.fromkeys(variants))
    unique_families = list(dict.fromkeys(families))
    out["variant_summary"] = unique_variants[0] if len(unique_variants) == 1 else " | ".join(unique_variants)
    out["family_summary"] = unique_families[0] if len(unique_families) == 1 else " | ".join(unique_families)
    out["unique_variants"] = len(unique_variants)
    out["column_count_per_floor"] = None if column_count is None else float(column_count)
    out["column_grid_bays_x"] = int(grid.get("beam_bays_x", 0) or 0)
    out["column_grid_bays_y"] = int(grid.get("beam_bays_y", 0) or 0)
    out["column_density_per_m2"] = float(column_density)
    out["column_moment_demand_kNm"] = float(moment_demand_knm)
    if columns_by_floor:
        out["schedule_by_floor"] = "; ".join(
            f"floor {item['floor']}: {item['column']}"
            for item in schedule
        )
    else:
        out["schedule_by_floor"] = f"average axial load: {schedule[0]['column']}" if schedule else None
    return out


def _dominant_material_label(mat_totals: dict[str, float]) -> str:
    if not mat_totals:
        return "Unknown"
    key = max(mat_totals, key=lambda mat: abs(float(mat_totals.get(mat, 0.0) or 0.0)))
    return {
        "concrete": "Concrete",
        "structural_steel": "Structural Steel",
        "rebar": "Rebar",
        "pt": "Post-Tension",
        "timber": "Timber",
        "screed": "Screed",
    }.get(key, str(key).replace("_", " ").title())


def _assembly_family_label(floor_cat: Any, floor_type: Any, beam_mat: Any, col_mat: Any) -> str:
    floor = str(floor_type or floor_cat or "floor").replace("_", " ").title()
    col = str(col_mat or "Column").replace("_", " ").title()
    if beam_mat is None or str(beam_mat).lower() in {"", "nan", "none"}:
        return f"{floor} + {col} Columns"
    beam = str(beam_mat).replace("_", " ").title()
    if beam.lower() == col.lower():
        return f"{floor} + {col} Frame"
    return f"{floor} + {beam} Beam/{col} Col"


def _col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    return names[0]


class _MockFamily:
    """Lightweight stand-in for SystemFamily accepted by infer_structural_class."""
    def __init__(self, material_family: str | None, **meta: Any) -> None:
        self.material_family = material_family
        self.metadata = {k: v for k, v in meta.items() if v is not None}
        self.family_id = meta.get("floor_category") or meta.get("beam_category") or material_family or "unknown"


def _mock_family(material_family: str | None = None, **meta: Any) -> _MockFamily:
    return _MockFamily(material_family=material_family, **meta)
