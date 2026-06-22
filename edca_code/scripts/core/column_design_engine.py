"""
Multi-storey RC column design engine.

For a column running the full height of the building, this engine:
  1. Computes the cumulative axial load at each storey (top → ground).
  2. Checks the catalog section against the worst-case (ground-floor) load
     and **upsizes** to a larger section in the same family if the
     normalised axial load n = N_Ed/(A_c·f_ck) > n_max (default 0.7).
  3. Runs the EC2 code check at every storey with the (possibly upsized)
     section to get the As_required there.
  4. Returns volume-weighted averages over the column height for both:
       - concrete_volume_m3_per_m   (constant — section is uniform)
       - rebar_volume_m3_per_m      (averaged over height; upper storeys
                                     need less rebar than ground floor)

This is what an industry tool would do for embodied carbon assessment:
constant section through height (typical construction), but rebar quantities
that reflect the real per-storey demand.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analysis_models import ColumnDemand
    from .domain_models import ComponentType, ProjectContext, SystemVariant
    from .repositories import RepositoryQueryService

logger = logging.getLogger(__name__)

# EC2 design defaults
_DEFAULT_FCK   = 30.0   # MPa
_DEFAULT_FYK   = 500.0  # MPa
_DEFAULT_N_MAX = 0.7    # Target n = N_Ed/(A_c·f_ck); above this, upsize section


# ----------------------------------------------------------------------------
# Geometry helpers (mirror rebar_code_check)
# ----------------------------------------------------------------------------

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _is_aci_variant(variant: "SystemVariant") -> bool:
    mat_rebar = str(variant.properties.get("material_rebar_id") or "").lower()
    vid = str(variant.variant_id or "").lower()
    return "ksi" in mat_rebar or vid.startswith("aci_")


def _column_dims_m(variant: "SystemVariant") -> tuple[float, float] | None:
    """Return (h_m, b_m). Returns None if either dimension is missing."""
    h = _as_float(variant.properties.get("column_depth"))
    b = _as_float(variant.properties.get("column_width"))
    if h is None or b is None or h <= 0 or b <= 0:
        return None
    if _is_aci_variant(variant):
        return h * 0.0254, b * 0.0254
    return h / 1000.0, b / 1000.0


def _column_section_area_m2(variant: "SystemVariant") -> float | None:
    dims = _column_dims_m(variant)
    if dims is None:
        return None
    return dims[0] * dims[1]


# ----------------------------------------------------------------------------
# Section sizing
# ----------------------------------------------------------------------------

def _find_section_for_load(
    query: "RepositoryQueryService",
    base_variant: "SystemVariant",
    n_ed_kN: float,
    f_ck_MPa: float,
    n_max: float = _DEFAULT_N_MAX,
) -> "SystemVariant":
    """Find the smallest variant in the same family that keeps n <= n_max.

    If no variant in the family is large enough, returns the largest available
    (and a warning is logged).
    """
    from .domain_models import ComponentType as CT

    # Minimum concrete area to keep n <= n_max (units: m²)
    a_c_min_m2 = (n_ed_kN * 1000) / (n_max * f_ck_MPa * 1e6)

    base_area = _column_section_area_m2(base_variant) or 0.0
    if base_area >= a_c_min_m2:
        return base_variant   # catalog section is adequate

    # Get all variants in the same family
    try:
        candidates = query.get_variants_for_family(CT.COLUMN, base_variant.family_id)
    except Exception:
        logger.warning("[column_design] Cannot fetch family variants for '%s'; using base.",
                       base_variant.family_id)
        return base_variant

    # Variants in same family with area >= required, sorted smallest first
    viable: list[tuple[float, "SystemVariant"]] = []
    for v in candidates:
        a = _column_section_area_m2(v)
        if a is None:
            continue
        if a >= a_c_min_m2:
            viable.append((a, v))

    if viable:
        viable.sort(key=lambda x: x[0])
        sized = viable[0][1]
        if sized.variant_id != base_variant.variant_id:
            logger.info("[column_design] Upsized column '%s' (A_c=%.4fm²) → '%s' (A_c=%.4fm²) "
                        "for N_Ed=%.0fkN (required A_c≥%.4fm² at n_max=%.2f)",
                        base_variant.variant_id, base_area, sized.variant_id, viable[0][0],
                        n_ed_kN, a_c_min_m2, n_max)
        return sized

    # Nothing in family is large enough — return largest available
    all_with_area = [(a, v) for v in candidates if (a := _column_section_area_m2(v)) is not None]
    if all_with_area:
        all_with_area.sort(key=lambda x: x[0], reverse=True)
        largest = all_with_area[0][1]
        logger.warning("[column_design] No variant in family '%s' meets A_c>=%.4fm² for N_Ed=%.0fkN; "
                       "using largest available '%s' (A_c=%.4fm²). Section may be overstressed.",
                       base_variant.family_id, a_c_min_m2, n_ed_kN,
                       largest.variant_id, all_with_area[0][0])
        return largest

    return base_variant


# ----------------------------------------------------------------------------
# Main: storey-by-storey design
# ----------------------------------------------------------------------------

def design_column_full_height(
    *,
    base_variant: "SystemVariant",
    query: "RepositoryQueryService",
    column_demand_per_storey: "ColumnDemand | None",
    project: "ProjectContext",
    f_ck_MPa: float = _DEFAULT_FCK,
    f_yk_MPa: float = _DEFAULT_FYK,
    c_nom_mm: float = 35.0,
    phi_main_mm: float = 20.0,
    phi_link_mm: float = 8.0,
    n_max: float = _DEFAULT_N_MAX,
) -> dict[str, Any]:
    """Design a multi-storey RC column for embodied carbon assessment.

    Inputs
    ------
    base_variant : starting column variant from the catalog
    query        : repository query service (for finding upsized variants in same family)
    column_demand_per_storey : ColumnDemand for a single storey's tributary load
                               (axial_dead_kn / axial_live_kn are PER STOREY)
    project      : provides storey_count and floor_to_floor_m

    Returns
    -------
    dict with keys:
      success                  : bool
      sized_variant_id         : str  (the variant after upsize, if any)
      concrete_volume_m3_per_m : float  (constant through height — section area)
      rebar_volume_m3_per_m    : float  (height-averaged rebar volume per linear m)
      rho_pct_max              : float  (rebar ratio at the worst storey)
      rho_pct_avg              : float  (height-averaged rebar ratio)
      n_storeys                : int
      storey_results           : list of dicts, one per storey, with N_Ed and As_req
    """
    from edca_code.scripts.code_checks.rc_column import check_rc_column

    out: dict[str, Any] = {
        "success": False,
        "sized_variant_id": base_variant.variant_id,
        "concrete_volume_m3_per_m": _as_float(base_variant.properties.get("concrete_volume")) or 0.0,
        "rebar_volume_m3_per_m":    _as_float(base_variant.properties.get("rebar_volume"))    or 0.0,
        "rho_pct_max": 0.0,
        "rho_pct_avg": 0.0,
        "n_storeys": 0,
        "storey_results": [],
        "error": "",
    }

    # ----- 1. Project parameters -----
    n_storeys = (project.geometry.storey_count if project.geometry else 1) or 1
    ftf       = (project.geometry.floor_to_floor_m if project.geometry else 3.5) or 3.5
    out["n_storeys"] = n_storeys

    # ----- 2. Per-storey single-storey load -----
    if column_demand_per_storey is None:
        out["error"] = "No column demand provided"
        return out

    g_single = _as_float(column_demand_per_storey.axial_dead_kn) or 0.0
    q_single = _as_float(column_demand_per_storey.axial_live_kn) or 0.0
    n_uls_single = 1.35 * g_single + 1.5 * q_single

    if n_uls_single <= 0:
        # Fall back to envelope effects if direct dead/live not populated
        envelope_axial = _as_float(getattr(column_demand_per_storey.envelope.effects, "axial", None))
        if envelope_axial:
            # Envelope value is assumed to be already factored (single storey)
            n_uls_single = abs(envelope_axial) / n_storeys
        else:
            out["error"] = "Column demand has no axial load"
            return out

    # ----- 3. Ground-floor cumulative load and section sizing -----
    n_ed_ground = n_uls_single * n_storeys

    if _is_aci_variant(base_variant):
        # EC2 code checks only — skip
        out["error"] = "ACI variant — EC2 column design engine does not apply"
        return out

    sized_variant = _find_section_for_load(query, base_variant, n_ed_ground, f_ck_MPa, n_max)
    out["sized_variant_id"] = sized_variant.variant_id

    dims = _column_dims_m(sized_variant)
    if dims is None:
        out["error"] = f"Cannot extract dimensions from variant '{sized_variant.variant_id}'"
        return out
    h_m, b_m = dims
    a_c_m2  = h_m * b_m

    # ----- 4. Storey-by-storey code check -----
    # Demand moment: catalog moment_capacity is the available capacity; we'll
    # use it as M_Ed conservatively (assumes the column is designed close to its
    # capacity).  Override with envelope moment_major if available.
    m_envelope = _as_float(getattr(column_demand_per_storey.envelope.effects, "moment_major", None))
    if m_envelope is not None:
        m_ed_kNm = abs(m_envelope)
    else:
        m_ed_kNm = _as_float(sized_variant.properties.get("moment_capacity")) or 0.0

    storey_results = []
    total_rebar_volume_x_height = 0.0
    total_link_volume_x_height  = 0.0
    rho_max = 0.0

    # Storey enumeration: s=1 is GROUND floor (worst load), s=n_storeys is TOP.
    # Each storey carries the cumulative load from itself + all storeys above.
    for s in range(1, n_storeys + 1):
        levels_carried = n_storeys - s + 1   # s=1 → carries n_storeys, s=n → carries 1
        n_ed_s = n_uls_single * levels_carried

        result = check_rc_column(
            h_m=h_m, b_m=b_m, clear_height_m=ftf,
            N_Ed_kN=n_ed_s, M_02_kNm=m_ed_kNm,
            f_ck_MPa=f_ck_MPa, f_yk_MPa=f_yk_MPa,
            c_nom_mm=c_nom_mm, phi_main_mm=phi_main_mm, phi_link_mm=phi_link_mm,
        )

        rebar_per_m = result.get("rebar_volume_m3_per_m") or 0.0
        link_per_m  = result.get("link_rebar_volume_m3_per_m") or 0.0
        rho_s       = result.get("rho_pct") or 0.0
        rho_max     = max(rho_max, rho_s)

        # Each storey contributes (volume/m × ftf) m³ of rebar to the column
        total_rebar_volume_x_height += rebar_per_m * ftf
        total_link_volume_x_height  += link_per_m  * ftf

        storey_results.append({
            "storey":               s,
            "levels_carried":       levels_carried,
            "N_Ed_kN":              n_ed_s,
            "rebar_m3_per_m":       rebar_per_m,
            "link_rebar_m3_per_m":  link_per_m,
            "rho_pct":              rho_s,
            "check_status":         result.get("status", "?"),
        })

    total_height = ftf * n_storeys
    avg_rebar_per_m = total_rebar_volume_x_height / total_height if total_height else 0.0
    avg_link_per_m  = total_link_volume_x_height  / total_height if total_height else 0.0
    avg_rho_pct     = sum(s["rho_pct"] for s in storey_results) / len(storey_results)

    # Combined longitudinal + link rebar per linear metre (design value)
    avg_total_rebar_per_m = avg_rebar_per_m + avg_link_per_m

    # Use the SIZED variant's catalog rebar as a floor (typical-practice minimum)
    # so an upsized column gets a physically consistent rebar quantity.
    sized_catalog_rebar = _as_float(sized_variant.properties.get("rebar_volume")) or 0.0
    rebar_per_m_final   = max(avg_total_rebar_per_m, sized_catalog_rebar)

    out["concrete_volume_m3_per_m"]            = a_c_m2          # m³ per linear m
    out["rebar_volume_m3_per_m"]               = rebar_per_m_final
    out["rebar_volume_m3_per_m_design"]        = avg_total_rebar_per_m  # long + links
    out["long_rebar_volume_m3_per_m_design"]   = avg_rebar_per_m        # longitudinal only
    out["link_rebar_volume_m3_per_m_design"]   = avg_link_per_m         # EC2 9.5.3 links only
    out["sized_catalog_rebar_m3_per_m"]        = sized_catalog_rebar
    out["rho_pct_max"]                    = rho_max
    out["rho_pct_avg"]                    = avg_rho_pct
    out["storey_results"]                 = storey_results
    out["success"]                        = True

    logger.info("[column_design] %s → sized=%s | ftf=%.2fm × %d storeys | "
                "N_Ed ground=%.0fkN | A_c=%.4fm² | rebar avg=%.6f m³/m (ρ_avg=%.2f%%, ρ_max=%.2f%%)",
                base_variant.variant_id, sized_variant.variant_id, ftf, n_storeys,
                n_ed_ground, a_c_m2, avg_rebar_per_m, avg_rho_pct, rho_max)

    return out
