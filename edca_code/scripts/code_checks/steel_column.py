"""
EC3 steel column design check (EN 1993-1-1).

Methodology follows the spreadsheet:
  - Non-dimensional slenderness: λ̄ = (L_cr/i) · √(f_y / (π²E))
  - Buckling curves: b (α=0.34) for y-y axis, c (α=0.49) for z-z axis (rolled UC/H)
  - Reduction factor: φ = 0.5(1 + α(λ̄ - 0.2) + λ̄²); χ = 1/(φ + √(φ² - λ̄²))
  - Buckling resistance: N_b,Rd = χ · A · f_y / γ_M1
  - Interaction (simplified, EC3 6.3.3):
      N/N_b,z,Rd + M_y/M_c,y,Rd + M_z/M_c,z,Rd ≤ 1.0

Returns dict with utilisation ratios and steel_mass_kg_per_m.
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
E_STEEL  = 210_000.0   # MPa
RHO_STEEL = 7850.0     # kg/m³

# Buckling curve imperfection factors (Table 6.2, EC3-1-1)
# Rolled I/H section, f_y ≤ 420 MPa:
#   h/b > 1.2, t_f ≤ 100mm → curve b (y-y), curve c (z-z)
_ALPHA_CURVE = {
    "a0": 0.13,
    "a":  0.21,
    "b":  0.34,
    "c":  0.49,
    "d":  0.76,
}

# Default curves for rolled UC (h/b ≤ 1.2):
_DEFAULT_CURVE_YY = "b"
_DEFAULT_CURVE_ZZ = "c"


@dataclass
class SteelColumnResult:
    status: str = "OK"
    error: str = ""

    section_name: str = ""
    A_cm2: float = 0.0
    i_y_cm: float = 0.0
    i_z_cm: float = 0.0
    W_pl_y_cm3: float = 0.0
    W_pl_z_cm3: float = 0.0

    L_m: float = 0.0
    N_Ed_kN: float = 0.0
    M_y_kNm: float = 0.0
    M_z_kNm: float = 0.0

    lambda_bar_y: float = 0.0
    lambda_bar_z: float = 0.0
    phi_y: float = 0.0
    phi_z: float = 0.0
    chi_y: float = 0.0
    chi_z: float = 0.0

    N_b_y_Rd_kN: float = 0.0
    N_b_z_Rd_kN: float = 0.0
    N_b_Rd_kN: float = 0.0   # governing (min)

    M_c_y_Rd_kNm: float = 0.0
    M_c_z_Rd_kNm: float = 0.0

    util_N_y: float = 0.0
    util_N_z: float = 0.0
    interaction: float = 0.0

    steel_mass_kg_per_m: float = 0.0

    raw: Dict[str, Any] = field(default_factory=dict)


def _lambda_bar(L_cr_m: float, i_cm: float, f_y_MPa: float) -> float:
    """Non-dimensional slenderness λ̄ = (L_cr/i) / (π·√(E/f_y))."""
    i_m = i_cm / 100
    return (L_cr_m / i_m) / (math.pi * math.sqrt(E_STEEL / f_y_MPa))


def _phi(alpha: float, lam_bar: float) -> float:
    """EC3 6.3.1.2: φ = 0.5·(1 + α·(λ̄ - 0.2) + λ̄²)."""
    return 0.5 * (1 + alpha * (lam_bar - 0.2) + lam_bar ** 2)


def _chi(phi: float, lam_bar: float) -> float:
    """EC3 6.3.1.2: χ = 1/(φ + √(φ² - λ̄²)), ≤ 1.0."""
    disc = phi ** 2 - lam_bar ** 2
    if disc < 0:
        disc = 0.0
    return min(1.0, 1.0 / (phi + math.sqrt(disc)))


def _N_b_Rd(chi: float, A_cm2: float, f_y_MPa: float) -> float:
    """Buckling resistance in kN."""
    return chi * A_cm2 * 1e2 * f_y_MPa / (GAMMA_M1 * 1000)   # cm²→mm², MPa, /1000→kN


def _M_c_Rd(W_pl_cm3: float, f_y_MPa: float) -> float:
    """Plastic moment resistance in kNm."""
    return W_pl_cm3 * 1e3 * f_y_MPa / 1e6


def check_steel_column(
    *,
    # Section properties
    section_name: str = "UC 254x254x73",
    A_cm2: float = 93.1,
    i_y_cm: float = 11.1,
    i_z_cm: float = 6.48,
    W_pl_y_cm3: float = 992.0,
    W_pl_z_cm3: float = 465.0,
    # Geometry / loading
    L_m: float = 3.5,
    k_eff: float = 1.0,          # effective length factor (= 1.0 for pin-pin)
    N_Ed_kN: float = 900.0,
    M_y_kNm: float = 35.0,
    M_z_kNm: float = 10.0,
    f_y_MPa: float = 355.0,
    # Buckling curves (defaults for rolled UC/H h/b ≤ 1.2)
    curve_yy: str = _DEFAULT_CURVE_YY,
    curve_zz: str = _DEFAULT_CURVE_ZZ,
) -> Dict[str, Any]:
    """
    EC3 steel column design check.

    Parameters
    ----------
    section_name : section designation (informational)
    A_cm2        : gross cross-sectional area (cm²)
    i_y_cm       : radius of gyration about y-y axis (cm)
    i_z_cm       : radius of gyration about z-z axis (cm)
    W_pl_y_cm3   : plastic section modulus about y-y (cm³)
    W_pl_z_cm3   : plastic section modulus about z-z (cm³)
    L_m          : column height (m)
    k_eff        : effective length factor
    N_Ed_kN      : design axial compression (kN)
    M_y_kNm      : design moment about y-y axis (kNm)
    M_z_kNm      : design moment about z-z axis (kNm)
    f_y_MPa      : yield strength (MPa)
    curve_yy     : buckling curve for y-y axis ('a0','a','b','c','d')
    curve_zz     : buckling curve for z-z axis

    Returns
    -------
    dict with success, interaction utilisation, N_b,Rd values, and steel_mass_kg_per_m
    """
    res = SteelColumnResult(
        section_name=section_name, A_cm2=A_cm2,
        i_y_cm=i_y_cm, i_z_cm=i_z_cm,
        W_pl_y_cm3=W_pl_y_cm3, W_pl_z_cm3=W_pl_z_cm3,
        L_m=L_m, N_Ed_kN=N_Ed_kN, M_y_kNm=M_y_kNm, M_z_kNm=M_z_kNm,
    )

    try:
        L_cr = k_eff * L_m   # critical length

        # ── 1. Non-dimensional slenderness ─────────────────────────────────────
        lam_y = _lambda_bar(L_cr, i_y_cm, f_y_MPa)
        lam_z = _lambda_bar(L_cr, i_z_cm, f_y_MPa)
        res.lambda_bar_y = lam_y
        res.lambda_bar_z = lam_z

        # ── 2. Reduction factors ────────────────────────────────────────────────
        alpha_y = _ALPHA_CURVE.get(curve_yy, 0.34)
        alpha_z = _ALPHA_CURVE.get(curve_zz, 0.49)

        phi_y = _phi(alpha_y, lam_y)
        phi_z = _phi(alpha_z, lam_z)
        res.phi_y = phi_y
        res.phi_z = phi_z

        chi_y = _chi(phi_y, lam_y)
        chi_z = _chi(phi_z, lam_z)
        res.chi_y = chi_y
        res.chi_z = chi_z

        # ── 3. Buckling resistances ────────────────────────────────────────────
        N_b_y = _N_b_Rd(chi_y, A_cm2, f_y_MPa)
        N_b_z = _N_b_Rd(chi_z, A_cm2, f_y_MPa)
        N_b   = min(N_b_y, N_b_z)

        res.N_b_y_Rd_kN = N_b_y
        res.N_b_z_Rd_kN = N_b_z
        res.N_b_Rd_kN   = N_b

        # ── 4. Moment resistances ──────────────────────────────────────────────
        M_c_y = _M_c_Rd(W_pl_y_cm3, f_y_MPa)
        M_c_z = _M_c_Rd(W_pl_z_cm3, f_y_MPa)
        res.M_c_y_Rd_kNm = M_c_y
        res.M_c_z_Rd_kNm = M_c_z

        # ── 5. Utilisation checks ──────────────────────────────────────────────
        util_N_y = N_Ed_kN / N_b_y if N_b_y > 0 else 1e9
        util_N_z = N_Ed_kN / N_b_z if N_b_z > 0 else 1e9
        res.util_N_y = util_N_y
        res.util_N_z = util_N_z

        # Simplified interaction (EC3 6.3.3 — conservative linear):
        # use governing buckling axis for axial term
        interaction = (N_Ed_kN / N_b
                       + M_y_kNm / M_c_y if M_c_y > 0 else 0
                       + M_z_kNm / M_c_z if M_c_z > 0 else 0)
        interaction = (N_Ed_kN / N_b
                       + (M_y_kNm / M_c_y if M_c_y > 0 else 0.0)
                       + (M_z_kNm / M_c_z if M_c_z > 0 else 0.0))
        res.interaction = interaction

        if util_N_y > 1.0:
            logger.warning("[steel_column] '%s' FAILS buckling y-y: util=%.3f", section_name, util_N_y)
        if util_N_z > 1.0:
            logger.warning("[steel_column] '%s' FAILS buckling z-z: util=%.3f", section_name, util_N_z)
        if interaction > 1.0:
            logger.warning("[steel_column] '%s' FAILS interaction: util=%.3f", section_name, interaction)

        # ── 6. Material quantities ─────────────────────────────────────────────
        res.steel_mass_kg_per_m = A_cm2 * 1e-4 * RHO_STEEL   # kg/m

        res.status = "OK"

    except Exception as exc:
        res.status = "FAILED"
        res.error  = str(exc)
        logger.error("[steel_column] check failed: %s", exc)

    out = {
        "success":              res.status == "OK",
        "status":               res.status,
        "error":                res.error,
        "section_name":         res.section_name,
        "L_m":                  res.L_m,
        "N_Ed_kN":              res.N_Ed_kN,
        "M_y_kNm":              res.M_y_kNm,
        "M_z_kNm":              res.M_z_kNm,
        "lambda_bar_y":         res.lambda_bar_y,
        "lambda_bar_z":         res.lambda_bar_z,
        "phi_y":                res.phi_y,
        "phi_z":                res.phi_z,
        "chi_y":                res.chi_y,
        "chi_z":                res.chi_z,
        "N_b_y_Rd_kN":         res.N_b_y_Rd_kN,
        "N_b_z_Rd_kN":         res.N_b_z_Rd_kN,
        "N_b_Rd_kN":            res.N_b_Rd_kN,
        "M_c_y_Rd_kNm":        res.M_c_y_Rd_kNm,
        "M_c_z_Rd_kNm":        res.M_c_z_Rd_kNm,
        "util_N_y":             res.util_N_y,
        "util_N_z":             res.util_N_z,
        "interaction":          res.interaction,
        "steel_mass_kg_per_m":  res.steel_mass_kg_per_m,
    }
    return out
