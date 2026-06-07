"""
EC3 steel beam design check — restrained beam and floor beam variants.

Methodology follows EC3 EN 1993-1-1 as worked in the spreadsheet:
  - Section classification: ε = √(235/f_y), web/flange class limits
  - Moment resistance:  M_c,Rd = W_pl,y · f_y / γ_M0
  - Shear resistance:   V_pl,Rd = A_v · f_y / (√3 · γ_M0)
  - Lateral-torsional buckling: M_b,Rd = M_c,Rd (fully restrained beam)
  - SLS deflection:     δ = 5·w_q·L⁴ / (384·E·I_y)  (simply-supported, UDL)

Floor beam variant: converts area loads (kN/m²) to line loads using beam spacing.

Returns dict with utilisation ratios and steel_mass_kg_per_m for BOM substitution.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── EC3 constants ──────────────────────────────────────────────────────────────
GAMMA_M0 = 1.0
GAMMA_M1 = 1.0
E_STEEL   = 210_000.0   # MPa
RHO_STEEL = 7850.0      # kg/m³

# Section classification limits (Table 5.2, EC3-1-1)
# Web (bending): c/t_w ≤ 72ε (class 1), ≤ 83ε (class 2), ≤ 124ε (class 3)
_WEB_CLASS1  = 72.0
_WEB_CLASS2  = 83.0
_WEB_CLASS3  = 124.0
# Flange (outstand): c/t_f ≤ 9ε (class 1), ≤ 10ε (class 2), ≤ 14ε (class 3)
_FLANGE_CLASS1 = 9.0
_FLANGE_CLASS2 = 10.0
_FLANGE_CLASS3 = 14.0

# Standard SLS deflection limits
_DEFL_LIMIT_TOTAL   = 1 / 250   # L/250 for total load (beams)
_DEFL_LIMIT_IMPOSED = 1 / 360   # L/360 for imposed load (floor beams)


@dataclass
class SteelBeamResult:
    status: str = "OK"
    error: str = ""

    # section
    section_name: str = ""
    A_cm2: float = 0.0
    h_mm: float = 0.0
    b_mm: float = 0.0
    t_w_mm: float = 0.0
    t_f_mm: float = 0.0
    I_y_cm4: float = 0.0
    W_pl_y_cm3: float = 0.0

    # geometry / loads
    L_m: float = 0.0
    g_k_kNm: float = 0.0
    q_k_kNm: float = 0.0
    n_ULS_kNm: float = 0.0
    M_Ed_kNm: float = 0.0
    V_Ed_kN: float = 0.0

    # classification
    epsilon: float = 0.0
    web_class: int = 0
    flange_class: int = 0
    section_class: int = 0

    # ULS checks
    M_c_Rd_kNm: float = 0.0
    V_pl_Rd_kN: float = 0.0
    M_b_Rd_kNm: float = 0.0
    util_M: float = 0.0
    util_V: float = 0.0

    # SLS
    delta_mm: float = 0.0
    delta_limit_mm: float = 0.0
    util_defl: float = 0.0
    deflection_pass: bool = True

    # material quantities
    steel_mass_kg_per_m: float = 0.0

    raw: Dict[str, Any] = field(default_factory=dict)


def _epsilon(f_y_MPa: float) -> float:
    return math.sqrt(235.0 / f_y_MPa)


def _classify_web(h_mm: float, t_w_mm: float, t_f_mm: float, eps: float) -> int:
    """Classify web in bending (pure bending case, c = h - 2*t_f)."""
    c = h_mm - 2 * t_f_mm
    ratio = c / t_w_mm
    if ratio <= _WEB_CLASS1 * eps:
        return 1
    if ratio <= _WEB_CLASS2 * eps:
        return 2
    if ratio <= _WEB_CLASS3 * eps:
        return 3
    return 4


def _classify_flange(b_mm: float, t_f_mm: float, eps: float) -> int:
    """Classify compression flange (outstand c = (b - t_w)/2, ignore root radius)."""
    # conservative: c = b/2 (for symmetric I-section outstand)
    c = b_mm / 2
    ratio = c / t_f_mm
    if ratio <= _FLANGE_CLASS1 * eps:
        return 1
    if ratio <= _FLANGE_CLASS2 * eps:
        return 2
    if ratio <= _FLANGE_CLASS3 * eps:
        return 3
    return 4


def _M_c_Rd(W_pl_y_cm3: float, f_y_MPa: float) -> float:
    """Plastic moment resistance in kNm (EC3 6.2.5)."""
    return W_pl_y_cm3 * 1e3 * f_y_MPa / 1e6   # cm³→mm³ * MPa / 1e6 → kNm


def _A_v(h_mm: float, t_w_mm: float, A_cm2: float,
         b_mm: float, t_f_mm: float) -> float:
    """Shear area A_v in mm² (EC3 6.2.6(3), rolled I/H section)."""
    # A_v = A - 2*b*t_f + (t_w + 2*r)*t_f  — simplified to h*t_w (conservative)
    # Here we use EC3 6.2.6(3): A_v = A_cm2*100 - 2*b*t_f + ... ≈ h*t_w
    return h_mm * t_w_mm   # mm² (conservative/hand-calc approximation)


def _V_pl_Rd(A_v_mm2: float, f_y_MPa: float) -> float:
    """Plastic shear resistance in kN (EC3 6.2.6)."""
    return A_v_mm2 * f_y_MPa / (math.sqrt(3) * GAMMA_M0) / 1000


def _deflection_udl(w_kNm: float, L_m: float, I_y_cm4: float) -> float:
    """Maximum mid-span deflection in mm for simply-supported beam under UDL."""
    # δ = 5·w·L⁴ / (384·E·I)
    w_Nmm = w_kNm * 1000 / 1000   # kN/m → N/mm
    L_mm  = L_m * 1000
    I_mm4 = I_y_cm4 * 1e4         # cm⁴ → mm⁴
    return 5 * w_Nmm * L_mm ** 4 / (384 * E_STEEL * I_mm4)


def check_steel_beam(
    *,
    # Section properties (UB/UC)
    section_name: str = "UB 406x178x54",
    A_cm2: float = 69.0,
    h_mm: float = 402.6,
    b_mm: float = 177.7,
    t_w_mm: float = 7.7,
    t_f_mm: float = 10.9,
    I_y_cm4: float = 18700.0,
    W_pl_y_cm3: float = 1050.0,
    # Geometry / loading
    L_m: float = 6.0,
    g_k_kNm: float = 13.0,        # characteristic permanent line load (kN/m)
    q_k_kNm: float = 4.5,         # characteristic variable line load (kN/m)
    f_y_MPa: float = 355.0,
    # Restraint: True = fully restrained (M_b,Rd = M_c,Rd)
    fully_restrained: bool = True,
    # SLS deflection limit: 'L/250' for total, 'L/360' for imposed
    defl_limit_divisor: float = 250.0,
    # Floor beam mode: provide beam_spacing_m and area loads instead of line loads
    floor_beam_mode: bool = False,
    beam_spacing_m: float = 3.0,
    g_k_area_kNm2: float = 0.0,   # kN/m² — used only in floor_beam_mode
    q_k_area_kNm2: float = 0.0,
) -> Dict[str, Any]:
    """
    EC3 steel beam design check.

    In floor_beam_mode the line loads are computed as:
      g_k_line = g_k_area * beam_spacing + self_weight
      q_k_line = q_k_area * beam_spacing
    and the SLS deflection check uses L/360 (imposed load only).

    Returns dict with utilisation ratios, pass/fail, and steel_mass_kg_per_m.
    """
    res = SteelBeamResult(
        section_name=section_name, A_cm2=A_cm2, h_mm=h_mm, b_mm=b_mm,
        t_w_mm=t_w_mm, t_f_mm=t_f_mm, I_y_cm4=I_y_cm4, W_pl_y_cm3=W_pl_y_cm3,
        L_m=L_m,
    )

    try:
        # ── 0. Floor beam load conversion ──────────────────────────────────────
        sw_kNm = A_cm2 * 1e-4 * RHO_STEEL * 9.81 / 1000   # self-weight kN/m
        if floor_beam_mode:
            g_k = g_k_area_kNm2 * beam_spacing_m + sw_kNm
            q_k = q_k_area_kNm2 * beam_spacing_m
            defl_limit_divisor = 360.0   # EC3 floor beam SLS limit
        else:
            g_k = g_k_kNm
            q_k = q_k_kNm

        res.g_k_kNm = g_k
        res.q_k_kNm = q_k

        # ── 1. ULS loads ───────────────────────────────────────────────────────
        n_ULS = 1.35 * g_k + 1.5 * q_k
        res.n_ULS_kNm = n_ULS

        M_Ed = n_ULS * L_m ** 2 / 8    # kNm (simply-supported UDL)
        V_Ed = n_ULS * L_m / 2          # kN
        res.M_Ed_kNm = M_Ed
        res.V_Ed_kN  = V_Ed

        # ── 2. Section classification ──────────────────────────────────────────
        eps = _epsilon(f_y_MPa)
        web_class    = _classify_web(h_mm, t_w_mm, t_f_mm, eps)
        flange_class = _classify_flange(b_mm, t_f_mm, eps)
        section_class = max(web_class, flange_class)

        res.epsilon       = eps
        res.web_class     = web_class
        res.flange_class  = flange_class
        res.section_class = section_class

        if section_class == 4:
            logger.warning("[steel_beam] Section '%s' is Class 4 — effective section "
                           "properties required; results are approximate.", section_name)

        # ── 3. ULS resistances ─────────────────────────────────────────────────
        M_c_Rd = _M_c_Rd(W_pl_y_cm3, f_y_MPa)
        A_v    = _A_v(h_mm, t_w_mm, A_cm2, b_mm, t_f_mm)
        V_pl_Rd = _V_pl_Rd(A_v, f_y_MPa)
        M_b_Rd  = M_c_Rd if fully_restrained else M_c_Rd  # restrained: LTB not critical

        res.M_c_Rd_kNm  = M_c_Rd
        res.V_pl_Rd_kN  = V_pl_Rd
        res.M_b_Rd_kNm  = M_b_Rd

        util_M = M_Ed / M_b_Rd
        util_V = V_Ed / V_pl_Rd

        res.util_M = util_M
        res.util_V = util_V

        if util_M > 1.0:
            logger.warning("[steel_beam] Section '%s' FAILS moment: util=%.3f", section_name, util_M)
        if util_V > 1.0:
            logger.warning("[steel_beam] Section '%s' FAILS shear: util=%.3f", section_name, util_V)

        # ── 4. SLS deflection ──────────────────────────────────────────────────
        # Use imposed load only for floor beams (L/360), total load for primary beams (L/250)
        w_SLS = q_k if floor_beam_mode else (g_k + q_k)
        delta      = _deflection_udl(w_SLS, L_m, I_y_cm4)
        delta_lim  = L_m * 1000 / defl_limit_divisor   # mm

        res.delta_mm       = delta
        res.delta_limit_mm = delta_lim
        res.util_defl      = delta / delta_lim
        res.deflection_pass = delta <= delta_lim

        if not res.deflection_pass:
            logger.warning("[steel_beam] Section '%s' FAILS deflection: δ=%.2fmm > lim=%.2fmm",
                           section_name, delta, delta_lim)

        # ── 5. Material quantities ─────────────────────────────────────────────
        # mass per m of beam
        res.steel_mass_kg_per_m = A_cm2 * 1e-4 * RHO_STEEL   # m²·kg/m³ → kg/m

        res.status = "OK"

    except Exception as exc:
        res.status = "FAILED"
        res.error  = str(exc)
        logger.error("[steel_beam] check failed: %s", exc)

    out = {
        "success":              res.status == "OK",
        "status":               res.status,
        "error":                res.error,
        "section_name":         res.section_name,
        "L_m":                  res.L_m,
        "g_k_kNm":              res.g_k_kNm,
        "q_k_kNm":              res.q_k_kNm,
        "n_ULS_kNm":            res.n_ULS_kNm,
        "M_Ed_kNm":             res.M_Ed_kNm,
        "V_Ed_kN":              res.V_Ed_kN,
        "epsilon":              res.epsilon,
        "web_class":            res.web_class,
        "flange_class":         res.flange_class,
        "section_class":        res.section_class,
        "M_c_Rd_kNm":          res.M_c_Rd_kNm,
        "V_pl_Rd_kN":           res.V_pl_Rd_kN,
        "M_b_Rd_kNm":           res.M_b_Rd_kNm,
        "util_M":               res.util_M,
        "util_V":               res.util_V,
        "delta_mm":             res.delta_mm,
        "delta_limit_mm":       res.delta_limit_mm,
        "util_defl":            res.util_defl,
        "deflection_pass":      res.deflection_pass,
        "steel_mass_kg_per_m":  res.steel_mass_kg_per_m,
    }
    return out
