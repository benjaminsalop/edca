"""
EC2 continuous RC beam design check.

Methodology follows the CCIP worked-example coefficients:
  - Span moment:    M_span    = (1.25*g_k*0.09 + 1.5*q_k*0.10) * L²
  - Support moment: M_support = n * 0.106 * L²  (n = governing ULS load)
  - Shear (ext):   V_ext     = n * 0.45 * L
  - Shear (int):   V_int     = n * 0.63 * L  (governing)
  - Lever arm:     z_span    = d*(0.5 + sqrt(0.25 - K/1.134))  (K-formula)
                   z_support = min(z_formula, 0.95d), capped at 0.85d when K>K_bal
  - Links:         A_sw/s from V_Ed at d from face of support

Returns a dict compatible with the post-ranking A_s substitution pipeline.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Eurocode material/safety factors ──────────────────────────────────────────
GAMMA_C = 1.5
GAMMA_S = 1.15
ALPHA_CC = 0.85          # long-term concrete strength reduction
LAMBDA = 0.8             # rectangular stress block depth factor (f_ck ≤ 50)
ETA = 1.0                # effective strength factor (f_ck ≤ 50)
EPSILON_CU = 0.0035      # ultimate concrete strain

# CCIP moment / shear coefficients for continuous beam (two-span)
_COEFF_M_SPAN_G  = 0.09   # coefficient on g_k*L² for span moment
_COEFF_M_SPAN_Q  = 0.10   # coefficient on q_k*L² for span moment
_COEFF_M_SUP     = 0.106  # coefficient on n*L² for support moment
_COEFF_V_EXT     = 0.45   # coefficient on n*L for exterior shear
_COEFF_V_INT     = 0.63   # coefficient on n*L for interior shear (governing)
_K_BAL           = 0.167  # balanced section limit for singly-reinforced beam
_k_bal_warned: set[tuple[float, float]] = set()  # suppress duplicate K>K_bal warnings

# EC2 6.2.2 shear: cot_theta bounds
_COT_THETA_DEFAULT = 2.5   # default strut angle (45° ≤ θ → cot=1; max 2.5)


@dataclass
class RCBeamResult:
    status: str = "OK"
    error: str = ""

    # geometry
    h_m: float = 0.0
    b_m: float = 0.0
    L_m: float = 0.0
    d_m: float = 0.0

    # loads
    g_k: float = 0.0
    q_k: float = 0.0
    n_ULS: float = 0.0

    # moments
    M_span_kNm: float = 0.0
    M_support_kNm: float = 0.0
    K_span: float = 0.0
    K_support: float = 0.0

    # flexural reinforcement
    z_span_m: float = 0.0
    z_support_m: float = 0.0
    As_span_mm2: float = 0.0
    As_support_mm2: float = 0.0   # total tension steel at support (inc. doubly-reinforced contribution)
    As2_mm2: float = 0.0          # compression steel at support (0 if singly-reinforced)
    As_req_mm2: float = 0.0       # governing (max of span / support)
    As_min_mm2: float = 0.0
    As_max_mm2: float = 0.0

    # shear
    V_int_kN: float = 0.0
    V_Ed_kN: float = 0.0          # at d from face
    v_Ed_MPa: float = 0.0
    v_Rd_max_MPa: float = 0.0
    Asw_s_req: float = 0.0        # mm²/mm  (= mm²/mm of link steel)
    Asw_s_min: float = 0.0
    Asw_s_prov: float = 0.0       # provided (chosen from standard spacing)

    # deflection
    L_over_d: float = 0.0
    L_over_d_limit: float = 0.0
    deflection_pass: bool = True

    # material quantities (per-m-run of beam)
    concrete_volume_m3_per_m: float = 0.0
    rebar_volume_m3_per_m: float = 0.0   # longitudinal only
    link_volume_m3_per_m: float = 0.0

    raw: Dict[str, Any] = field(default_factory=dict)


def _fcd(f_ck_MPa: float) -> float:
    return ALPHA_CC * f_ck_MPa / GAMMA_C


def _fyd(f_yk_MPa: float) -> float:
    return f_yk_MPa / GAMMA_S


def _Es() -> float:
    return 200_000.0  # MPa


def _effective_depth(h_m: float, c_nom_mm: float, phi_link_mm: float,
                     phi_main_mm: float) -> float:
    """Return effective depth in metres."""
    d_mm = h_m * 1000 - c_nom_mm - phi_link_mm - phi_main_mm / 2.0
    return d_mm / 1000.0


def _lever_arm_K_formula(K: float, d_m: float) -> float:
    """EC2 z from K: z = d*(0.5 + sqrt(0.25 - K/1.134)), capped at 0.95d."""
    inner = 0.25 - K / 1.134
    if inner < 0:
        inner = 0.0
    z = d_m * (0.5 + math.sqrt(inner))
    return min(z, 0.95 * d_m)


def _As_flexure(M_kNm: float, z_m: float, f_yk_MPa: float) -> float:
    """Required area of tension steel in mm²."""
    fyd = _fyd(f_yk_MPa)
    if z_m <= 0 or fyd <= 0:
        return 0.0
    return (M_kNm * 1e6) / (fyd * z_m * 1000)  # mm²


def _K(M_kNm: float, b_m: float, d_m: float, f_ck_MPa: float) -> float:
    # K = M/(bd²fck) — standard CCIP/EC2 form; the 1.134 in the z formula is derived
    # from fcd = 0.567fck (α_cc=0.85, γ_c=1.5) already absorbed into the coefficient.
    return (M_kNm * 1e6) / (b_m * 1000 * (d_m * 1000) ** 2 * f_ck_MPa)


def _As_min(b_m: float, d_m: float, f_ck_MPa: float, f_yk_MPa: float) -> float:
    """EC2 9.2.1.1 minimum tension reinforcement in mm²."""
    fctm = 0.30 * f_ck_MPa ** (2 / 3)  # for f_ck ≤ 50 MPa
    rho_min = max(0.26 * fctm / f_yk_MPa, 0.0013)
    return rho_min * b_m * 1000 * d_m * 1000  # mm²


def _As_max(b_m: float, h_m: float) -> float:
    """EC2 9.2.1.1 maximum reinforcement area in mm²."""
    return 0.04 * b_m * 1000 * h_m * 1000  # mm²


def _v_Rd_max(f_ck_MPa: float, cot_theta: float = _COT_THETA_DEFAULT) -> float:
    """EC2 6.2.3(3) maximum shear stress v_Rd,max in MPa."""
    nu1 = 0.6 * (1 - f_ck_MPa / 250)
    fcd = _fcd(f_ck_MPa)
    return nu1 * fcd * cot_theta / (1 + cot_theta ** 2)


def _Asw_s_required(V_Ed_kN: float, b_m: float, d_m: float,
                    f_yk_MPa: float, cot_theta: float = _COT_THETA_DEFAULT) -> float:
    """Required A_sw/s in mm²/mm from EC2 6.2.3(3)."""
    fyd = _fyd(f_yk_MPa)
    V_Ed_N = V_Ed_kN * 1000
    # A_sw/s = V_Ed / (z * f_ywd * cot_theta)  — using z ≈ 0.9d
    z_m = 0.9 * d_m
    return V_Ed_N / (z_m * 1000 * fyd * cot_theta)  # mm²/mm


def _Asw_s_min(b_m: float, f_ck_MPa: float, f_yk_MPa: float) -> float:
    """EC2 9.2.2(5) minimum shear reinforcement ratio → A_sw,min/s in mm²/mm."""
    rho_w_min = 0.08 * math.sqrt(f_ck_MPa) / f_yk_MPa
    return rho_w_min * b_m * 1000  # mm²/mm (s in mm, b in mm)


def _deflection_check(L_m: float, d_m: float, rho: float,
                      f_ck_MPa: float) -> tuple[float, float, bool]:
    """
    EC2 7.4.2 span/effective-depth check.
    Returns (actual_L_over_d, limit, pass).
    Basic ratio for simply-supported: 20; for continuous beam end-span: 26.
    Modification factor K_def = 1.3 for end span of continuous beam.
    """
    rho_0 = 1e-3 * math.sqrt(f_ck_MPa)   # reference reinforcement ratio
    K_def = 1.3  # continuous beam (end span)
    if rho <= rho_0:
        factor = K_def * (11 + 1.5 * math.sqrt(f_ck_MPa) * rho_0 / rho
                          + (1 / 12) * math.sqrt(f_ck_MPa) * math.sqrt(rho_0 / rho))
    else:
        factor = K_def * (11 + 1.5 * math.sqrt(f_ck_MPa) * rho_0 / (rho - rho_0)
                          + (1 / 12) * math.sqrt(f_ck_MPa) * math.sqrt(rho / rho_0))
    limit = factor
    actual = L_m / d_m
    return actual, limit, actual <= limit


def _link_rebar_volume(phi_link_mm: float, Asw_s_prov: float,
                       L_m: float, b_m: float) -> float:
    """
    Approximate volume of link steel per m-run of beam (m³/m).
    Uses provided A_sw/s and perimeter of rectangular stirrup.
    """
    perimeter_mm = 2 * (b_m * 1000 + 400)  # approx height for link
    # A_sw/s in mm²/mm; convert to m³ per m-run
    # volume per mm of beam length = A_sw/s [mm²/mm] * perimeter/circumference_of_bar...
    # simpler: density approach via cross-section
    # A_sw is for one pair of legs: A_sw/s * beam_length = total link area * length
    # volume per m of beam = (A_sw/s [mm²/mm]) * 1000mm/m * perimeter_mm / 1e9
    return Asw_s_prov * 1000 * perimeter_mm / 1e9  # m³/m


def check_rc_beam(
    *,
    h_m: float = 0.45,
    b_m: float = 0.30,
    L_m: float = 6.0,
    support_width_m: float = 0.30,
    g_k: float = 30.2,          # kN/m (line load — already includes slab SW)
    q_k: float = 11.5,          # kN/m
    f_ck_MPa: float = 30.0,
    f_yk_MPa: float = 500.0,
    c_nom_mm: float = 35.0,
    phi_main_mm: float = 25.0,
    phi_link_mm: float = 10.0,
    cot_theta: float = _COT_THETA_DEFAULT,
    # Optional: provide n_ULS directly to override EN1990 Eq 6.10
    n_ULS: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Design check for an EC2 continuous RC beam using CCIP worked-example coefficients.

    Parameters
    ----------
    h_m : overall depth (m)
    b_m : width (m)
    L_m : clear span (m)
    g_k : characteristic permanent line load (kN/m)
    q_k : characteristic variable line load (kN/m)
    f_ck_MPa : concrete cylinder strength
    f_yk_MPa : rebar yield strength
    c_nom_mm : nominal cover to face of beam (mm)
    phi_main_mm : main bar diameter (mm)
    phi_link_mm : link diameter (mm)

    Returns
    -------
    dict with keys:
        success, status, As_req_mm2, As_span_mm2, As_support_mm2,
        rebar_volume_m3_per_m, concrete_volume_m3_per_m, ...full RCBeamResult fields
    """
    res = RCBeamResult(h_m=h_m, b_m=b_m, L_m=L_m, g_k=g_k, q_k=q_k)

    try:
        # ── 1. Effective depth ─────────────────────────────────────────────────
        d_m = _effective_depth(h_m, c_nom_mm, phi_link_mm, phi_main_mm)
        res.d_m = d_m

        if d_m <= 0:
            raise ValueError(f"Effective depth d={d_m*1000:.1f}mm ≤ 0; check cover/section.")

        # ── 2. ULS load ────────────────────────────────────────────────────────
        if n_ULS is None:
            n_610  = 1.35 * g_k + 1.5 * q_k
            n_610a = 1.35 * g_k + 1.05 * q_k   # ψ₀=0.7 → 0.7*1.5=1.05
            n_610b = 1.25 * g_k + 1.5 * q_k
            n_ULS  = max(n_610, n_610a, n_610b)
        res.n_ULS = n_ULS

        # ── 3. Design moments ──────────────────────────────────────────────────
        M_span    = (1.25 * g_k * _COEFF_M_SPAN_G + 1.5 * q_k * _COEFF_M_SPAN_Q) * L_m ** 2
        M_support = n_ULS * _COEFF_M_SUP * L_m ** 2
        res.M_span_kNm    = M_span
        res.M_support_kNm = M_support

        # ── 4. K values ────────────────────────────────────────────────────────
        K_span    = _K(M_span,    b_m, d_m, f_ck_MPa)
        K_support = _K(M_support, b_m, d_m, f_ck_MPa)
        res.K_span    = K_span
        res.K_support = K_support

        if K_support > _K_BAL:
            _key = (round(K_support, 3), _K_BAL)
            if _key not in _k_bal_warned:
                logger.warning("K_support=%.3f > K_bal=%.3f — doubly-reinforced design required.", K_support, _K_BAL)
                _k_bal_warned.add(_key)

        # ── 5. Lever arms ──────────────────────────────────────────────────────
        z_span    = _lever_arm_K_formula(K_span, d_m)
        z_bal     = _lever_arm_K_formula(_K_BAL, d_m)
        # At support: use K-formula up to K_bal; beyond that use doubly-reinforced approach
        z_support = _lever_arm_K_formula(min(K_support, _K_BAL), d_m)
        if K_support > 0.10:
            z_support = min(z_support, 0.85 * d_m)
        res.z_span_m    = z_span
        res.z_support_m = z_support

        # ── 6. Required flexural steel ─────────────────────────────────────────
        As_span = _As_flexure(M_span, z_span, f_yk_MPa)

        if K_support <= _K_BAL:
            # Singly-reinforced support
            As_support = _As_flexure(M_support, z_support, f_yk_MPa)
            As2 = 0.0
        else:
            # Doubly-reinforced support (EC2 approach):
            # concrete carries M_bal at z_bal; residual M' carried by steel couple at (d - d2).
            d2_m = (c_nom_mm + phi_link_mm + phi_main_mm / 2.0) / 1000.0  # compression steel depth
            M_bal_kNm = _K_BAL * b_m * 1000 * (d_m * 1000) ** 2 * f_ck_MPa / 1e6
            M_prime_kNm = M_support - M_bal_kNm
            lever_couple_m = d_m - d2_m
            fyd = _fyd(f_yk_MPa)
            As2 = (M_prime_kNm * 1e6) / (fyd * lever_couple_m * 1000)  # compression steel (mm²)
            As_support = _As_flexure(M_bal_kNm, z_bal, f_yk_MPa) + As2   # total tension steel

        As_min = _As_min(b_m, d_m, f_ck_MPa, f_yk_MPa)
        As_max = _As_max(b_m, h_m)

        As_req = max(As_span, As_support, As_min)
        res.As_span_mm2    = As_span
        res.As_support_mm2 = As_support
        res.As2_mm2        = As2
        res.As_req_mm2     = As_req
        res.As_min_mm2     = As_min
        res.As_max_mm2     = As_max

        if As_req > As_max:
            raise ValueError(f"As_req={As_req:.0f}mm² > As_max={As_max:.0f}mm² — increase section.")

        # ── 7. Shear ──────────────────────────────────────────────────────────
        V_int = n_ULS * _COEFF_V_INT * L_m
        # V_Ed at d from face of support
        V_Ed = V_int - n_ULS * (support_width_m / 2 + d_m)
        V_Ed = max(V_Ed, 0.0)

        v_Ed     = V_Ed * 1000 / (b_m * 1000 * d_m * 1000)   # MPa
        v_Rd_max = _v_Rd_max(f_ck_MPa, cot_theta)

        res.V_int_kN    = V_int
        res.V_Ed_kN     = V_Ed
        res.v_Ed_MPa    = v_Ed
        res.v_Rd_max_MPa = v_Rd_max

        if v_Ed > v_Rd_max:
            raise ValueError(f"v_Ed={v_Ed:.3f} > v_Rd,max={v_Rd_max:.3f} MPa — section fails shear.")

        Asw_s_req = _Asw_s_required(V_Ed, b_m, d_m, f_yk_MPa, cot_theta)
        Asw_s_min = _Asw_s_min(b_m, f_ck_MPa, f_yk_MPa)
        Asw_s_prov = max(Asw_s_req, Asw_s_min)

        res.Asw_s_req  = Asw_s_req
        res.Asw_s_min  = Asw_s_min
        res.Asw_s_prov = Asw_s_prov

        # ── 8. Deflection (span/d check) ───────────────────────────────────────
        rho_prov = As_req / (b_m * 1000 * d_m * 1000)
        L_over_d, L_over_d_limit, defl_pass = _deflection_check(L_m, d_m, rho_prov, f_ck_MPa)
        res.L_over_d       = L_over_d
        res.L_over_d_limit = L_over_d_limit
        res.deflection_pass = defl_pass

        if not defl_pass:
            logger.warning("Deflection check FAIL: L/d=%.1f > limit=%.1f", L_over_d, L_over_d_limit)

        # ── 9. Material quantities (per m-run of beam) ────────────────────────
        res.concrete_volume_m3_per_m = h_m * b_m          # m³/m
        # longitudinal rebar: As_req [mm²] tension + As2 [mm²] compression steel → m³/m
        res.rebar_volume_m3_per_m    = (As_req + res.As2_mm2) / 1e6  # mm² → m²/m = m³/m
        res.link_volume_m3_per_m     = _link_rebar_volume(phi_link_mm, Asw_s_prov, L_m, b_m)

        res.status = "OK"

    except Exception as exc:
        res.status = "FAILED"
        res.error  = str(exc)
        logger.error("[rc_beam] check failed: %s", exc)

    out = {
        "success":                  res.status == "OK",
        "status":                   res.status,
        "error":                    res.error,
        "h_m":                      res.h_m,
        "b_m":                      res.b_m,
        "L_m":                      res.L_m,
        "d_m":                      res.d_m,
        "g_k_kNm":                  res.g_k,
        "q_k_kNm":                  res.q_k,
        "n_ULS_kNm":                res.n_ULS,
        "M_span_kNm":               res.M_span_kNm,
        "M_support_kNm":            res.M_support_kNm,
        "K_span":                   res.K_span,
        "K_support":                res.K_support,
        "z_span_m":                 res.z_span_m,
        "z_support_m":              res.z_support_m,
        "As_span_mm2":              res.As_span_mm2,
        "As_support_mm2":           res.As_support_mm2,
        "As2_mm2":                  res.As2_mm2,
        "As_req_mm2":               res.As_req_mm2,
        "As_min_mm2":               res.As_min_mm2,
        "As_max_mm2":               res.As_max_mm2,
        "V_int_kN":                 res.V_int_kN,
        "V_Ed_kN":                  res.V_Ed_kN,
        "v_Ed_MPa":                 res.v_Ed_MPa,
        "v_Rd_max_MPa":             res.v_Rd_max_MPa,
        "Asw_s_req_mm2_per_mm":     res.Asw_s_req,
        "Asw_s_min_mm2_per_mm":     res.Asw_s_min,
        "Asw_s_prov_mm2_per_mm":    res.Asw_s_prov,
        "L_over_d":                 res.L_over_d,
        "L_over_d_limit":           res.L_over_d_limit,
        "deflection_pass":          res.deflection_pass,
        "concrete_volume_m3_per_m": res.concrete_volume_m3_per_m,
        "rebar_volume_m3_per_m":    res.rebar_volume_m3_per_m,
        "link_volume_m3_per_m":     res.link_volume_m3_per_m,
    }
    return out
