"""
Rebar code-check integration layer.

Bridges the structural code-check modules (rc_beam, rc_column) with the
AssemblyTakeoffEngine.  Provides two public functions:

  dispatch_rebar_check(variant, component_type, beam_demand, column_demand, project)
      → rebar_volume_m3_per_lm | None

  compute_rebar_overrides(candidate, analysis_input, query, project, span_x_m, span_y_m)
      → dict[variant_id, rebar_volume_m3_per_lm]

The override dict is passed directly to AssemblyTakeoffEngine.compute_bom()
which applies max(catalog, override) for every rebar_volume line.

Unit conventions
----------------
All catalog beam/column dimensions use mm for EC variants (beam_depth=500 → 0.5 m).
ACI variants (identified by material_rebar_id containing 'ksi') use inches.
rebar_volume in the catalog is m³ per linear metre for beams and columns.
The code-check functions return rebar_volume_m3_per_m in the same units.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analysis_models import BeamDemand, ColumnDemand
    from .domain_models import AssemblyCandidate, ComponentType, ProjectContext, SystemVariant
    from .repositories import RepositoryQueryService
    from .analysis_models import AssemblyAnalysisInput

logger = logging.getLogger(__name__)

# Default material strengths for EC2 checks (overridable via project.design_options)
_DEFAULT_FCK   = 30.0   # MPa
_DEFAULT_FYK   = 500.0  # MPa

# Nominal cover and bar diameters assumed when not in variant properties
_DEFAULT_C_NOM_MM       = 35.0
_DEFAULT_PHI_MAIN_MM    = 20.0
_DEFAULT_PHI_LINK_MM    = 8.0


# ---------------------------------------------------------------------------
# Unit detection helpers
# ---------------------------------------------------------------------------

def _is_aci_variant(variant: "SystemVariant") -> bool:
    """True if the variant uses US-customary units (ACI)."""
    mat_rebar = str(variant.properties.get("material_rebar_id") or "").lower()
    vid = str(variant.variant_id or "").lower()
    return "ksi" in mat_rebar or vid.startswith("aci_")


def _beam_dims_m(variant: "SystemVariant") -> tuple[float, float] | None:
    """Return (h_m, b_m) for a beam variant, converting from mm or inches as needed."""
    props = variant.properties
    h_raw = _as_float(props.get("beam_depth"))
    b_raw = _as_float(props.get("beam_width"))
    if h_raw is None or b_raw is None or h_raw <= 0 or b_raw <= 0:
        return None
    if _is_aci_variant(variant):
        return h_raw * 0.0254, b_raw * 0.0254   # inches → m
    # EC variants: dimensions in mm
    return h_raw / 1000.0, b_raw / 1000.0


def _column_dims_m(variant: "SystemVariant") -> tuple[float, float] | None:
    """Return (h_m, b_m) for a column variant."""
    props = variant.properties
    h_raw = _as_float(props.get("column_depth"))
    b_raw = _as_float(props.get("column_width"))
    if h_raw is None or b_raw is None or h_raw <= 0 or b_raw <= 0:
        return None
    if _is_aci_variant(variant):
        return h_raw * 0.0254, b_raw * 0.0254
    return h_raw / 1000.0, b_raw / 1000.0


def _column_height_m(variant: "SystemVariant", ftf_default: float = 3.5) -> float:
    """Return clear storey height for a column variant (m)."""
    raw = _as_float(variant.properties.get("column_height"))
    if raw and raw > 0:
        # column_height is stored in m for both EC and ACI variants
        return raw
    return ftf_default


# ---------------------------------------------------------------------------
# Dispatch: single variant → rebar_volume_m3_per_lm
# ---------------------------------------------------------------------------

def dispatch_rebar_check(
    variant: "SystemVariant",
    component_type: "ComponentType",
    *,
    beam_demand: "BeamDemand | None" = None,
    column_demand: "ColumnDemand | None" = None,
    project: "ProjectContext | None" = None,
) -> float | None:
    """Run the appropriate EC2 code check for an RC beam or column variant.

    Returns the design rebar_volume in m³ per linear metre, or None if:
      - the variant has no rebar (steel section, timber, etc.)
      - required demand data is missing
      - the code check fails unexpectedly

    Only EC2 variants are checked (ACI variants are skipped — code-check
    modules implement EC2 only).
    """
    from .domain_models import ComponentType as CT

    # Non-RC variants: no rebar material → nothing to override
    rebar_mat = str(variant.properties.get("material_rebar_id") or "").strip()
    if not rebar_mat or rebar_mat.lower() in {"none", "nan", ""}:
        return None

    # ACI variants: EC2 code checks don't apply
    if _is_aci_variant(variant):
        logger.debug("[rebar_check] Skipping ACI variant '%s' (EC2 checks only).", variant.variant_id)
        return None

    design_opts = project.design_options if project else {}
    f_ck = float(design_opts.get("f_ck_MPa", _DEFAULT_FCK))
    f_yk = float(design_opts.get("f_yk_MPa", _DEFAULT_FYK))
    c_nom = float(design_opts.get("c_nom_mm", _DEFAULT_C_NOM_MM))
    phi_main = float(design_opts.get("phi_main_mm", _DEFAULT_PHI_MAIN_MM))
    phi_link = float(design_opts.get("phi_link_mm", _DEFAULT_PHI_LINK_MM))

    if component_type == CT.BEAM:
        return _check_beam(variant, beam_demand, f_ck=f_ck, f_yk=f_yk,
                           c_nom=c_nom, phi_main=phi_main, phi_link=phi_link)

    if component_type == CT.COLUMN:
        ftf = project.geometry.floor_to_floor_m if project and project.geometry else 3.5
        return _check_column(variant, column_demand, f_ck=f_ck, f_yk=f_yk,
                             c_nom=c_nom, phi_main=phi_main, phi_link=phi_link, ftf=ftf or 3.5)

    return None


def _check_beam(
    variant: "SystemVariant",
    demand: "BeamDemand | None",
    *,
    f_ck: float,
    f_yk: float,
    c_nom: float,
    phi_main: float,
    phi_link: float,
) -> float | None:
    dims = _beam_dims_m(variant)
    if dims is None:
        logger.warning("[rebar_check] Beam variant '%s': missing depth/width.", variant.variant_id)
        return None

    h_m, b_m = dims

    # Extract ULS line load and span from demand
    n_ULS: float | None = None
    L_m: float | None = None
    if demand is not None:
        n_ULS = _as_float(demand.factored_line_load_kn_per_m)
        L_m   = _as_float(demand.span_m)

    if L_m is None or L_m <= 0:
        # Fall back to catalog max_span as the design span
        L_m = _as_float(variant.properties.get("max_span"))

    if L_m is None or L_m <= 0:
        logger.warning("[rebar_check] Beam variant '%s': no span available.", variant.variant_id)
        return None

    try:
        from edca_code.scripts.code_checks.rc_beam import check_rc_beam
        result = check_rc_beam(
            h_m=h_m, b_m=b_m, L_m=L_m,
            f_ck_MPa=f_ck, f_yk_MPa=f_yk,
            c_nom_mm=c_nom, phi_main_mm=phi_main, phi_link_mm=phi_link,
            n_ULS=n_ULS,   # None → function computes from g_k/q_k defaults
        )
    except Exception:
        logger.exception("[rebar_check] rc_beam check failed for variant '%s'.", variant.variant_id)
        return None

    if not result.get("success"):
        logger.warning("[rebar_check] Beam '%s' check FAILED: %s", variant.variant_id, result.get("error"))
        return None

    long_vol = result["rebar_volume_m3_per_m"]
    link_vol = result.get("link_volume_m3_per_m") or 0.0
    return long_vol + link_vol


def _check_column(
    variant: "SystemVariant",
    demand: "ColumnDemand | None",
    *,
    f_ck: float,
    f_yk: float,
    c_nom: float,
    phi_main: float,
    phi_link: float,
    ftf: float,
) -> float | None:
    dims = _column_dims_m(variant)
    if dims is None:
        logger.warning("[rebar_check] Column variant '%s': missing depth/width.", variant.variant_id)
        return None

    h_m, b_m = dims
    clear_height = _column_height_m(variant, ftf_default=ftf)

    # Extract N_Ed and M_Ed from demand
    N_Ed_kN: float | None = None
    M_Ed_kNm: float | None = None

    if demand is not None:
        g = _as_float(demand.axial_dead_kn)
        q = _as_float(demand.axial_live_kn)
        if g is not None and q is not None:
            N_Ed_kN = 1.35 * g + 1.5 * q
        elif demand.envelope.effects.axial is not None:
            N_Ed_kN = abs(demand.envelope.effects.axial)

        if demand.envelope.effects.moment_major is not None:
            M_Ed_kNm = abs(demand.envelope.effects.moment_major)

    # Fall back to catalog moment_capacity as M_Ed if demand moment unknown
    if M_Ed_kNm is None:
        M_Ed_kNm = _as_float(variant.properties.get("moment_capacity"))

    if N_Ed_kN is None or N_Ed_kN <= 0:
        logger.warning("[rebar_check] Column '%s': no axial demand — skipping check.", variant.variant_id)
        return None

    if M_Ed_kNm is None or M_Ed_kNm < 0:
        M_Ed_kNm = 0.0

    try:
        from edca_code.scripts.code_checks.rc_column import check_rc_column
        result = check_rc_column(
            h_m=h_m, b_m=b_m,
            clear_height_m=clear_height,
            N_Ed_kN=N_Ed_kN,
            M_02_kNm=M_Ed_kNm,
            f_ck_MPa=f_ck, f_yk_MPa=f_yk,
            c_nom_mm=c_nom, phi_main_mm=phi_main, phi_link_mm=phi_link,
        )
    except Exception:
        logger.exception("[rebar_check] rc_column check failed for variant '%s'.", variant.variant_id)
        return None

    if not result.get("success"):
        logger.warning("[rebar_check] Column '%s' check FAILED: %s", variant.variant_id, result.get("error"))
        return None

    # Return combined longitudinal + EC2 9.5.3 link rebar volume
    long_vol = result["rebar_volume_m3_per_m"]
    link_vol = result.get("link_rebar_volume_m3_per_m") or 0.0
    return long_vol + link_vol


# ---------------------------------------------------------------------------
# Integration: whole candidate → override dict
# ---------------------------------------------------------------------------

def compute_rebar_overrides(
    candidate: "AssemblyCandidate",
    analysis_input: "AssemblyAnalysisInput | None",
    query: "RepositoryQueryService",
    project: "ProjectContext",
    span_x_m: float,
    span_y_m: float,
) -> dict[str, float]:
    """Compute rebar_volume overrides for all RC beam variants in a candidate.

    For columns, use compute_column_design_overrides() instead — it returns
    BOTH rebar and concrete overrides (with storey-by-storey design and
    auto-upsizing of the section).

    Returns a dict mapping variant_id → rebar_volume_m3_per_lm suitable for
    passing to AssemblyTakeoffEngine.compute_bom(rebar_overrides=...).
    """
    from .domain_models import ComponentType as CT

    beam_demand   = analysis_input.primary_beam   if analysis_input else None
    sec_demand    = analysis_input.secondary_beam  if analysis_input else None

    overrides: dict[str, float] = {}

    beam_checks: list[tuple[str | None, "ComponentType", Any]] = [
        (candidate.primary_beam_variant_id,   CT.BEAM, beam_demand),
        (candidate.secondary_beam_variant_id, CT.BEAM, sec_demand),
    ]

    for vid, ct, demand in beam_checks:
        if not vid:
            continue
        try:
            variant = query.get_variant(ct, vid)
        except Exception:
            logger.debug("[rebar_check] Variant '%s' not found in repository; skipping.", vid)
            continue

        check_result = dispatch_rebar_check(variant, ct, beam_demand=demand, project=project)
        if check_result is None:
            continue

        catalog_rebar = _as_float(variant.properties.get("rebar_volume")) or 0.0
        if check_result > catalog_rebar:
            overrides[vid] = check_result
            logger.info(
                "[rebar_check] %s '%s': code-check rebar %.6f > catalog %.6f m³/m — overriding.",
                ct.value, vid, check_result, catalog_rebar,
            )
        else:
            logger.debug(
                "[rebar_check] %s '%s': catalog %.6f >= code-check %.6f — no override.",
                ct.value, vid, catalog_rebar, check_result,
            )

    return overrides


def compute_beam_design_overrides(
    candidate: "AssemblyCandidate",
    analysis_input: "AssemblyAnalysisInput | None",
    query: "RepositoryQueryService",
    project: "ProjectContext",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run the beam design engine on primary and secondary beams.

    Returns three override dicts (all keyed by ORIGINAL catalog variant_id):
      rebar_overrides    : for RC beams
      concrete_overrides : for RC beams (if section upsized)
      steel_overrides    : for steel beams (if section upsized — keyed for steel_volume)
    """
    from .domain_models import ComponentType as CT
    from .beam_design_engine import design_beam

    rebar_ov: dict[str, float]    = {}
    concrete_ov: dict[str, float] = {}
    steel_ov: dict[str, float]    = {}

    beam_pairs = [
        (candidate.primary_beam_variant_id,   analysis_input.primary_beam   if analysis_input else None),
        (candidate.secondary_beam_variant_id, analysis_input.secondary_beam if analysis_input else None),
    ]

    for vid, demand in beam_pairs:
        if not vid or demand is None:
            continue
        try:
            variant = query.get_variant(CT.BEAM, vid)
        except Exception:
            continue

        result = design_beam(base_variant=variant, query=query,
                             beam_demand=demand, project=project)
        if not result.get("success"):
            continue

        cat_concrete = _as_float(variant.properties.get("concrete_volume")) or 0.0
        cat_rebar    = _as_float(variant.properties.get("rebar_volume"))    or 0.0
        cat_steel    = _as_float(variant.properties.get("steel_volume"))    or 0.0

        new_concrete = result["concrete_volume_m3_per_m"]
        new_rebar    = result["rebar_volume_m3_per_m"]
        new_steel    = result["steel_volume_m3_per_m"]

        if new_concrete > cat_concrete:
            concrete_ov[vid] = new_concrete
        if new_rebar > cat_rebar:
            rebar_ov[vid] = new_rebar
        if new_steel > cat_steel:
            steel_ov[vid] = new_steel

    return rebar_ov, concrete_ov, steel_ov


def compute_column_design_overrides(
    candidate: "AssemblyCandidate",
    analysis_input: "AssemblyAnalysisInput | None",
    query: "RepositoryQueryService",
    project: "ProjectContext",
) -> tuple[dict[str, float], dict[str, float], str | None]:
    """Run storey-by-storey column design and return (rebar_overrides, concrete_overrides, sized_variant_id).

    Both dicts are keyed by the ORIGINAL catalog variant_id so that
    AssemblyTakeoffEngine.compute_bom() can pick up the overrides when looking
    up that variant.  If the design engine upsized the section, the override
    values reflect the larger geometry — but the BOM still uses the original
    variant_id (we only override its volume fields, not the variant itself).
    """
    from .domain_models import ComponentType as CT
    from .column_design_engine import design_column_full_height

    rebar_ov: dict[str, float]    = {}
    concrete_ov: dict[str, float] = {}
    sized_id: str | None          = None

    vid = candidate.column_variant_id
    if not vid:
        return rebar_ov, concrete_ov, sized_id

    try:
        variant = query.get_variant(CT.COLUMN, vid)
    except Exception:
        logger.debug("[col_design] Column variant '%s' not found; skipping.", vid)
        return rebar_ov, concrete_ov, sized_id

    # Skip non-RC variants (steel/timber sections handled by their own checks)
    rebar_mat = str(variant.properties.get("material_rebar_id") or "").strip()
    if not rebar_mat or rebar_mat.lower() in {"none", "nan", ""}:
        return rebar_ov, concrete_ov, sized_id

    column_demand = analysis_input.column if analysis_input else None
    result = design_column_full_height(
        base_variant=variant,
        query=query,
        column_demand_per_storey=column_demand,
        project=project,
    )

    if not result.get("success"):
        logger.warning("[col_design] '%s' design failed: %s", vid, result.get("error"))
        return rebar_ov, concrete_ov, sized_id

    sized_id = result["sized_variant_id"]
    new_rebar    = result["rebar_volume_m3_per_m"]
    new_concrete = result["concrete_volume_m3_per_m"]
    cat_rebar    = _as_float(variant.properties.get("rebar_volume"))    or 0.0
    cat_concrete = _as_float(variant.properties.get("concrete_volume")) or 0.0

    if new_rebar > cat_rebar:
        rebar_ov[vid] = new_rebar
        logger.info("[col_design] Column '%s': rebar %.6f → %.6f m³/m (ρ_avg=%.2f%%)",
                    vid, cat_rebar, new_rebar, result["rho_pct_avg"])
    if new_concrete > cat_concrete:
        concrete_ov[vid] = new_concrete
        logger.info("[col_design] Column '%s': concrete %.6f → %.6f m³/m (upsized to '%s')",
                    vid, cat_concrete, new_concrete, sized_id)

    return rebar_ov, concrete_ov, sized_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else f   # drop NaN
    except (TypeError, ValueError):
        return None
