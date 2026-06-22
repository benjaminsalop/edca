"""
Beam design engine — checks the catalog beam against actual design demand
(moment + shear + deflection) and upsizes the section if the check fails.

Supports both RC beams (EC2) and steel beams (EC3).  For each material:

  1. Identify the beam material (RC vs steel) from material_* IDs.
  2. Run the matching code check at the actual demand.
  3. If the utilisation > 1.0 (moment, shear, or deflection), find the
     smallest variant in the same naming-category whose code check passes,
     and return that variant's volumes as the override.

Family/category logic
---------------------
Beam catalogs use a per-section family structure (e.g. `uk_ub_457x152x52` is
ONE family containing one section).  Upsizing therefore needs a wider search
than the family alone — we scan all variants whose `variant_id` shares a
common prefix:

  uk_ub_*               → UK Universal Beams
  uk_uc_*               → UK Universal Columns (when used as beams)
  uk_hfchs_*            → UK hot-finished CHS
  uk_chs_*, uk_rhs_*    → other UK hollow sections
  aisc_w*               → AISC W-shapes
  ec_primary_beam_*     → EC RC primary beams
  ec_secondary_beam_*   → EC RC secondary beams
  ec_spandrel_beam_*    → EC RC spandrel beams
  hasslacher_*, buckland_*, west_fraser_*  → glulam / LVL (timber — skipped)

For each prefix, candidates are ranked by mass (steel) or section area (RC)
and the smallest variant whose check passes is returned.
"""
from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .analysis_models import BeamDemand
    from .domain_models import ComponentType, ProjectContext, SystemVariant
    from .repositories import RepositoryQueryService

logger = logging.getLogger(__name__)

# Defaults (overridable via project.design_options)
_DEFAULT_FCK = 30.0
_DEFAULT_FYK = 500.0
_DEFAULT_FY  = 355.0
_DEFAULT_CNOM    = 35.0
_DEFAULT_PHI_MAIN = 25.0
_DEFAULT_PHI_LINK = 10.0

# Utilisation thresholds: bump section if any of these exceeds 1.0
_UTIL_TARGET_M     = 1.0
_UTIL_TARGET_V     = 1.0
_UTIL_TARGET_DEFL  = 1.0

# Naming-category prefixes (longest first to avoid uk_ub matching uk_ub_xxx
# before uk_uc_ etc.)
_BEAM_CATEGORY_PREFIXES = [
    "ec_primary_beam_",
    "ec_secondary_beam_",
    "ec_spandrel_beam_",
    "uk_ub_",
    "uk_uc_",
    "uk_hfchs_",
    "uk_chs_",
    "uk_rhs_",
    "uk_shs_",
    "aisc_w",
]


# ----------------------------------------------------------------------------
# Identification & unit helpers
# ----------------------------------------------------------------------------

def _as_float(v: Any) -> float | None:
    if v is None: return None
    try:
        f = float(v)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _is_rc_beam(variant: "SystemVariant") -> bool:
    rebar_mat = str(variant.properties.get("material_rebar_id") or "").strip().lower()
    conc_mat  = str(variant.properties.get("material_concrete_id") or "").strip().lower()
    return bool(rebar_mat and rebar_mat not in {"nan", "none", ""}) or \
           bool(conc_mat and conc_mat not in {"nan", "none", ""})


def _is_steel_beam(variant: "SystemVariant") -> bool:
    steel_mat = str(variant.properties.get("material_steel_id") or "").strip().lower()
    return bool(steel_mat and steel_mat not in {"nan", "none", ""}) and not _is_rc_beam(variant)


def _is_aci_variant(variant: "SystemVariant") -> bool:
    mat_rebar = str(variant.properties.get("material_rebar_id") or "").lower()
    vid = str(variant.variant_id or "").lower()
    return "ksi" in mat_rebar or vid.startswith("aci_")


def _rc_beam_dims_m(variant: "SystemVariant") -> tuple[float, float] | None:
    h = _as_float(variant.properties.get("beam_depth"))
    b = _as_float(variant.properties.get("beam_width"))
    if h is None or b is None or h <= 0 or b <= 0:
        return None
    if _is_aci_variant(variant):
        return h * 0.0254, b * 0.0254
    return h / 1000.0, b / 1000.0


def _steel_beam_section_props(variant: "SystemVariant") -> dict[str, float] | None:
    """Extract steel section properties from a variant.

    Returns a dict with keys h_mm, b_mm, t_w_mm, t_f_mm, A_cm2, I_y_cm4, W_pl_y_cm3
    (best-effort — some catalogs may not have all fields; fall back to
    steel_volume × density for mass).
    """
    p = variant.properties
    # Beam depth/width are in metres for AISC and UK catalogs; convert to mm
    h_m = _as_float(p.get("beam_depth"))
    b_m = _as_float(p.get("beam_width"))
    t_w = _as_float(p.get("web_thickness"))
    t_f = _as_float(p.get("flange_width"))   # NOT thickness! Catalog convention.
    if h_m is None or h_m <= 0:
        return None
    h_mm = h_m * 1000
    b_mm = (b_m or 0.0) * 1000
    t_w_mm = (t_w or 0.0) * 1000
    # Approximate flange thickness from h_mm / 20 if not available
    t_f_mm = (t_f or h_mm * 0.05) if isinstance(t_f, (int, float)) else h_mm * 0.05
    # If steel_volume present, mass per metre = vol × 7850
    steel_vol_per_m = _as_float(p.get("steel_volume")) or 0.0  # m³/m
    A_m2 = steel_vol_per_m   # m³/m = m² of cross-section (since per linear m)
    A_cm2 = A_m2 * 1e4
    # I and W_pl rarely populated; approximate W_pl_y ≈ A × h × 0.4 (rule of thumb for I-sections)
    W_pl_y_cm3 = A_cm2 * (h_mm / 10) * 0.4   # cm³
    # I_y ≈ A × h² × 0.4 / 4 (very crude)
    I_y_cm4 = A_cm2 * (h_mm / 10) ** 2 * 0.1
    moment_cap = _as_float(p.get("moment_capacity"))   # kNm — preferred if populated
    shear_cap  = _as_float(p.get("shear_capacity"))    # kN
    return {
        "h_mm": h_mm, "b_mm": b_mm, "t_w_mm": t_w_mm, "t_f_mm": t_f_mm,
        "A_cm2": A_cm2, "I_y_cm4": I_y_cm4, "W_pl_y_cm3": W_pl_y_cm3,
        "moment_capacity_kNm": moment_cap, "shear_capacity_kN": shear_cap,
        "steel_volume_m3_per_m": steel_vol_per_m,
    }


def _beam_category_prefix(variant_id: str) -> str | None:
    vid = str(variant_id or "").lower()
    for prefix in _BEAM_CATEGORY_PREFIXES:
        if vid.startswith(prefix):
            return prefix
    return None


# ----------------------------------------------------------------------------
# Code-check wrappers
# ----------------------------------------------------------------------------

def _check_rc(variant: "SystemVariant", L_m: float, n_ULS: float | None,
              f_ck: float, f_yk: float, c_nom: float, phi_main: float, phi_link: float) -> dict | None:
    dims = _rc_beam_dims_m(variant)
    if dims is None:
        return None
    h_m, b_m = dims
    from edca_code.scripts.code_checks.rc_beam import check_rc_beam
    try:
        return check_rc_beam(
            h_m=h_m, b_m=b_m, L_m=L_m,
            f_ck_MPa=f_ck, f_yk_MPa=f_yk,
            c_nom_mm=c_nom, phi_main_mm=phi_main, phi_link_mm=phi_link,
            n_ULS=n_ULS,
        )
    except Exception:
        logger.exception("[beam_design] rc_beam check failed for '%s'", variant.variant_id)
        return None


def _check_steel(variant: "SystemVariant", L_m: float, g_k: float, q_k: float,
                 f_y: float, fully_restrained: bool) -> dict | None:
    sp = _steel_beam_section_props(variant)
    if sp is None:
        return None
    from edca_code.scripts.code_checks.steel_beam import check_steel_beam
    try:
        return check_steel_beam(
            section_name=variant.variant_id,
            A_cm2=sp["A_cm2"], h_mm=sp["h_mm"], b_mm=sp["b_mm"],
            t_w_mm=sp["t_w_mm"], t_f_mm=sp["t_f_mm"],
            I_y_cm4=sp["I_y_cm4"], W_pl_y_cm3=sp["W_pl_y_cm3"],
            L_m=L_m, g_k_kNm=g_k, q_k_kNm=q_k, f_y_MPa=f_y,
            fully_restrained=fully_restrained,
        )
    except Exception:
        logger.exception("[beam_design] steel_beam check failed for '%s'", variant.variant_id)
        return None


def _utilisations_steel(result: dict) -> tuple[float, float, float]:
    return (
        _as_float(result.get("util_M"))    or 0.0,
        _as_float(result.get("util_V"))    or 0.0,
        _as_float(result.get("util_defl")) or 0.0,
    )


def _passes_steel(result: dict) -> bool:
    uM, uV, uD = _utilisations_steel(result)
    return uM <= _UTIL_TARGET_M and uV <= _UTIL_TARGET_V and uD <= _UTIL_TARGET_DEFL


def _passes_rc(result: dict) -> bool:
    return bool(result.get("success")) and result.get("deflection_pass", True)


# ----------------------------------------------------------------------------
# Cross-category section search
# ----------------------------------------------------------------------------

def _get_all_beam_variants(query: "RepositoryQueryService") -> list["SystemVariant"]:
    """Return every BEAM variant in the repository.  Falls back to scanning
    families if the repo doesn't expose a get_all_variants helper."""
    from .domain_models import ComponentType as CT
    # Prefer a direct accessor if available
    if hasattr(query, "get_all_variants"):
        try:
            return list(query.get_all_variants(CT.BEAM))
        except Exception:
            pass
    # Fallback: iterate the repo's variants map
    repo = getattr(query, "repo", None)
    if repo is not None and hasattr(repo, "variants"):
        return [v for (ct, _vid), v in repo.variants.items() if ct == CT.BEAM]
    return []


def _find_smallest_passing(
    query: "RepositoryQueryService",
    base_variant: "SystemVariant",
    passes: Callable[["SystemVariant"], bool],
    rank_key: Callable[["SystemVariant"], float],
) -> "SystemVariant":
    """Find the smallest variant in the same naming-category that passes the check.

    Returns base_variant unchanged if no suitable larger variant is found.
    """
    prefix = _beam_category_prefix(base_variant.variant_id)
    if not prefix:
        return base_variant
    all_beams = _get_all_beam_variants(query)
    candidates = [v for v in all_beams
                  if str(v.variant_id or "").lower().startswith(prefix)]
    if not candidates:
        return base_variant

    # Sort lightest first (by mass / section area), then take the smallest that passes
    candidates.sort(key=rank_key)
    for v in candidates:
        if passes(v):
            if v.variant_id != base_variant.variant_id:
                logger.info("[beam_design] Upsized '%s' → '%s' (smallest passing in category '%s')",
                            base_variant.variant_id, v.variant_id, prefix)
            return v
    logger.warning("[beam_design] No variant in category '%s' passes the check; keeping '%s' (may be overstressed).",
                   prefix, base_variant.variant_id)
    return base_variant


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def design_beam(
    *,
    base_variant: "SystemVariant",
    query: "RepositoryQueryService",
    beam_demand: "BeamDemand | None",
    project: "ProjectContext | None" = None,
) -> dict[str, Any]:
    """Design a beam against actual demand.  Upsizes within the same naming
    category if the catalog section fails the EC2/EC3 check.

    Returns:
      success                  : bool
      sized_variant_id         : str
      concrete_volume_m3_per_m : float  (m³ per linear metre — RC only)
      rebar_volume_m3_per_m    : float  (m³ per linear metre — RC only)
      steel_volume_m3_per_m    : float  (m³ per linear metre — steel only)
      util_M / util_V / util_defl : final utilisations (steel) or None
      check_status             : str
    """
    out: dict[str, Any] = {
        "success": False,
        "sized_variant_id": base_variant.variant_id,
        "concrete_volume_m3_per_m": _as_float(base_variant.properties.get("concrete_volume")) or 0.0,
        "rebar_volume_m3_per_m":    _as_float(base_variant.properties.get("rebar_volume"))    or 0.0,
        "steel_volume_m3_per_m":    _as_float(base_variant.properties.get("steel_volume"))    or 0.0,
        "error": "",
    }

    # ACI variants: code checks are EC2/EC3 only
    if _is_aci_variant(base_variant):
        out["error"] = "ACI variant — EC code checks not applicable"
        return out

    if beam_demand is None:
        out["error"] = "No beam demand provided"
        return out

    L_m = _as_float(beam_demand.span_m)
    if L_m is None or L_m <= 0:
        out["error"] = "Beam demand missing span"
        return out

    n_uls = _as_float(beam_demand.factored_line_load_kn_per_m)
    # Approximate g_k / q_k split if only factored is available
    if n_uls is not None and n_uls > 0:
        # Assume typical g/q ratio of 2:1 by characteristic load
        # n_uls = 1.35g + 1.5q with g ≈ 2q → n_uls = (1.35*2 + 1.5)*q = 4.2q → q = n/4.2
        q_k = n_uls / 4.2
        g_k = 2 * q_k
    else:
        g_k = _as_float(beam_demand.unfactored_line_load_kn_per_m) or 0.0
        q_k = 0.0
        n_uls = 1.35 * g_k + 1.5 * q_k

    design_opts = (project.design_options if project else {}) or {}
    f_ck = float(design_opts.get("f_ck_MPa", _DEFAULT_FCK))
    f_yk = float(design_opts.get("f_yk_MPa", _DEFAULT_FYK))
    f_y  = float(design_opts.get("f_y_MPa",  _DEFAULT_FY))
    c_nom    = float(design_opts.get("c_nom_mm", _DEFAULT_CNOM))
    phi_main = float(design_opts.get("phi_main_mm", _DEFAULT_PHI_MAIN))
    phi_link = float(design_opts.get("phi_link_mm", _DEFAULT_PHI_LINK))

    # Dispatch on material
    if _is_rc_beam(base_variant):
        # RC beam: run check, upsize within ec_*_beam category if needed
        check = _check_rc(base_variant, L_m, n_uls, f_ck, f_yk, c_nom, phi_main, phi_link)
        if check is None:
            out["error"] = "RC beam check failed to run"
            return out

        # If the base section passes flexure and deflection, use its catalog values
        # (with code-check rebar as a possible override).
        def rc_passes(v: "SystemVariant") -> bool:
            r = _check_rc(v, L_m, n_uls, f_ck, f_yk, c_nom, phi_main, phi_link)
            return r is not None and _passes_rc(r)

        def rc_rank(v: "SystemVariant") -> float:
            dims = _rc_beam_dims_m(v)
            return (dims[0] * dims[1]) if dims else 1e9   # by section area

        sized = _find_smallest_passing(query, base_variant, rc_passes, rc_rank) \
                if not _passes_rc(check) else base_variant

        # Recompute the check on sized variant
        final = _check_rc(sized, L_m, n_uls, f_ck, f_yk, c_nom, phi_main, phi_link) or check

        sized_dims = _rc_beam_dims_m(sized)
        sized_conc = sized_dims[0] * sized_dims[1] if sized_dims else 0.0

        # Rebar: max(code-check, sized variant catalog)
        sized_catalog_rebar = _as_float(sized.properties.get("rebar_volume")) or 0.0
        code_check_rebar    = _as_float(final.get("rebar_volume_m3_per_m"))   or 0.0
        rebar_final = max(sized_catalog_rebar, code_check_rebar)

        out["sized_variant_id"]         = sized.variant_id
        out["concrete_volume_m3_per_m"] = sized_conc
        out["rebar_volume_m3_per_m"]    = rebar_final
        out["check_status"]             = final.get("status", "?")
        out["util_M"] = (final.get("M_support_kNm", 0.0) / final.get("M_b_Rd_kNm", 1.0)) if final.get("M_b_Rd_kNm") else None
        out["util_V"] = None
        out["util_defl"] = (0.0 if final.get("deflection_pass") else 1.0)
        out["success"] = bool(final.get("success"))
        logger.info("[beam_design] RC %s → sized=%s | L=%.2fm | rebar %.6f m³/m | concrete %.4f m³/m",
                    base_variant.variant_id, sized.variant_id, L_m, rebar_final, sized_conc)
        return out

    if _is_steel_beam(base_variant):
        # Steel beam: run EC3 check, upsize within uk_ub_/aisc_w/... category if needed
        check = _check_steel(base_variant, L_m, g_k, q_k, f_y, fully_restrained=True)
        if check is None:
            out["error"] = "Steel beam check failed to run"
            return out

        def steel_passes(v: "SystemVariant") -> bool:
            r = _check_steel(v, L_m, g_k, q_k, f_y, fully_restrained=True)
            return r is not None and _passes_steel(r)

        def steel_rank(v: "SystemVariant") -> float:
            return _as_float(v.properties.get("steel_volume")) or 1e9   # by section area

        sized = _find_smallest_passing(query, base_variant, steel_passes, steel_rank) \
                if not _passes_steel(check) else base_variant

        final = _check_steel(sized, L_m, g_k, q_k, f_y, fully_restrained=True) or check
        sized_steel_vol = _as_float(sized.properties.get("steel_volume")) or 0.0

        uM, uV, uD = _utilisations_steel(final)
        out["sized_variant_id"]      = sized.variant_id
        out["steel_volume_m3_per_m"] = sized_steel_vol
        out["util_M"]    = uM
        out["util_V"]    = uV
        out["util_defl"] = uD
        out["check_status"] = final.get("status", "?")
        out["success"] = bool(final.get("success"))
        logger.info("[beam_design] STEEL %s → sized=%s | L=%.2fm | steel_vol %.6f m³/m | "
                    "util M=%.2f V=%.2f δ=%.2f",
                    base_variant.variant_id, sized.variant_id, L_m, sized_steel_vol, uM, uV, uD)
        return out

    # Timber / unknown — skip
    out["error"] = f"Unsupported beam material for variant '{base_variant.variant_id}'"
    return out
