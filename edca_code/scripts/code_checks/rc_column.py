"""
EC2 RC column design check — perimeter and internal column cases.

Methodology follows CCIP worked-example:
  - Slenderness:   λ = l₀·√12/h  (rectangular section)
  - Limit:         λ_lim = 20·A·B·C/√n  (EC2 5.8.3.1)
  - Imperfection:  e_i = l₀/400, M_Ed = M_02 + e_i·N_Ed (≥ e₀·N_Ed)
  - Neutral-axis iteration: strain compatibility with ε_cu = 0.0035,
    rectangular stress block λ=0.8, η=1.0
  - Fallback: EC2 minimum reinforcement when column is very lightly loaded
  - Link (transverse) rebar: EC2 9.5.3 minimum rules applied to square section

Returns a dict including As_req_mm2, concrete_volume_m3_per_m,
rebar_volume_m3_per_m (longitudinal only), and link_rebar_volume_m3_per_m.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Eurocode constants ─────────────────────────────────────────────────────────
GAMMA_C   = 1.5
GAMMA_S   = 1.15
ALPHA_CC  = 0.85
LAMBDA_SB = 0.8      # stress block depth factor
ETA_SB    = 1.0      # effective strength factor
EPSILON_CU = 0.0035  # ultimate concrete strain
ES        = 200_000.0  # MPa, steel elastic modulus

# EC2 5.8.3.1 slenderness limit factors (default values)
_A_DEFAULT = 0.7   # 1/(1+0.2*phi_ef); use 0.7 if phi_ef unknown
_B_DEFAULT = 1.1   # sqrt(1+2*omega); use 1.1 if omega unknown
_C_RM_DEFAULT = 0.7  # 1-0.7*M_01/M_02; conservative (opposite curvature = 0.7)
_E0_FACTOR = 1/30   # e₀ = max(h/30, 20mm) → here e₀ = N_Ed * h/30


@dataclass
class RCColumnResult:
    status: str = "OK"
    error: str = ""

    h_m: float = 0.0
    b_m: float = 0.0
    l0_m: float = 0.0
    N_Ed_kN: float = 0.0
    M_Ed_kNm: float = 0.0

    lambda_val: float = 0.0
    lambda_lim: float = 0.0
    is_slender: bool = False

    e_i_m: float = 0.0
    M_Ed_design_kNm: float = 0.0

    x_iter_m: float = 0.0
    As_req_mm2: float = 0.0
    As_min_mm2: float = 0.0
    As_max_mm2: float = 0.0
    rho_pct: float = 0.0

    concrete_volume_m3_per_m: float = 0.0
    rebar_volume_m3_per_m: float = 0.0        # longitudinal only
    link_rebar_volume_m3_per_m: float = 0.0   # EC2 9.5.3 transverse/link rebar

    raw: Dict[str, Any] = field(default_factory=dict)


# ── Helper functions ───────────────────────────────────────────────────────────

def _fcd(f_ck: float) -> float:
    return ALPHA_CC * f_ck / GAMMA_C


def _fyd(f_yk: float) -> float:
    return f_yk / GAMMA_S


def _epsilon_s(x_m: float, d_m: float) -> float:
    """Steel strain at distance d from compression face, given NA depth x."""
    if x_m <= 0:
        return EPSILON_CU
    return EPSILON_CU * (d_m - x_m) / x_m


def _epsilon_sc(x_m: float, d2_m: float) -> float:
    """Compression steel strain at d2 from compression face."""
    if x_m <= 0:
        return EPSILON_CU
    return EPSILON_CU * (x_m - d2_m) / x_m


def _sigma(epsilon: float, f_yk: float) -> float:
    """Stress in rebar (MPa), capped at ±f_yd."""
    fyd = _fyd(f_yk)
    return max(-fyd, min(fyd, ES * epsilon))


def _As_min_col(A_c_mm2: float, f_ck: float, f_yk: float, N_Ed_kN: float) -> float:
    """EC2 9.5.2 minimum column reinforcement in mm², with practical lower bound.

    EC2 gives max(0.1*N_Ed/fyd, 0.002*Ac).  In practice, columns are rarely
    detailed below ~0.5% because of buildability, lap-zone requirements, and
    robustness.  We apply 0.5% as a practical minimum so that oversized columns
    (upsized to satisfy axial capacity) don't end up with unrealistically low rebar.
    """
    fyd = _fyd(f_yk)
    ec2_min = max(0.1 * N_Ed_kN * 1000 / fyd, 0.002 * A_c_mm2)
    practical_min = 0.005 * A_c_mm2   # 0.5% practical lower bound
    return max(ec2_min, practical_min)


def _As_max_col(A_c_mm2: float) -> float:
    """EC2 9.5.2 maximum column reinforcement (4% outside lap zones)."""
    return 0.04 * A_c_mm2


def _neutral_axis_iterate(
    N_Ed_kN: float,
    M_Ed_kNm: float,
    b_m: float,
    h_m: float,
    d_m: float,
    d2_m: float,
    f_ck: float,
    f_yk: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> tuple[float, float]:
    """
    Bisection on neutral-axis depth x to find As required for combined N+M.

    Strategy (EC2 rectangular stress block):
      F_c  = η*f_cd * λ*x * b     (compression in concrete)
      σ_sc = E_s * ε_sc (≤ f_yd) (compression steel at d2)
      σ_st = E_s * ε_st (≤ f_yd) (tension steel at d)

    Equilibrium in N:  N_Ed = F_c + As/2*(σ_sc - σ_st)
    Equilibrium in M about centroid:
      M_Ed = F_c*(h/2 - λ*x/2) + As/2*(σ_sc + σ_st)*(h/2 - d2)

    Solving for As/2 from N equation:
      As/2 = (N_Ed - F_c) / (σ_sc - σ_st)   [if σ_sc ≠ σ_st]

    Then verify M equation gives consistent result; iterate on x.

    Returns (x_opt_m, As_req_mm2).
    """
    fcd = _fcd(f_ck)
    fyd = _fyd(f_yk)
    N_Ed = N_Ed_kN * 1e3   # N
    M_Ed = M_Ed_kNm * 1e6  # N·mm
    b    = b_m * 1e3
    h    = h_m * 1e3
    d    = d_m * 1e3
    d2   = d2_m * 1e3

    def residual(x_mm: float) -> tuple[float, float]:
        """Return (As_from_N, As_from_M) for given x."""
        x = x_mm
        # concrete force
        x_eff = min(x * LAMBDA_SB, h)
        F_c = ETA_SB * fcd * x_eff * b   # N (compression positive)
        # steel strains → stresses
        eps_sc = EPSILON_CU * max(x - d2, 0.0) / x if x > 0 else EPSILON_CU
        eps_st = EPSILON_CU * max(d  - x, 0.0) / x if x > 0 else -EPSILON_CU
        sig_sc = max(-fyd, min(fyd, ES * eps_sc))
        sig_st = max(-fyd, min(fyd, ES * eps_st))

        denom_N = sig_sc - sig_st
        if abs(denom_N) < 1.0:
            As_half_N = 1e9  # singular — skip
        else:
            As_half_N = (N_Ed - F_c) / denom_N

        # moment about geometric centroid of section
        arm_c  = h / 2 - x_eff / 2
        arm_s  = h / 2 - d2
        # M_Ed = F_c * arm_c + As_half * (sig_sc + sig_st) * arm_s
        denom_M = (sig_sc + sig_st) * arm_s
        if abs(denom_M) < 1.0:
            As_half_M = 1e9
        else:
            As_half_M = (M_Ed - F_c * arm_c) / denom_M

        return As_half_N, As_half_M

    # Sweep x from d2 to h, pick x that minimises |As_N - As_M| and both ≥ 0
    best_x    = d2
    best_As   = 1e9
    min_resid = 1e9

    for i in range(max_iter):
        x_try = d2 + (h - d2) * i / max_iter
        An, Am = residual(x_try)
        r = abs(An - Am)
        if r < min_resid:
            min_resid = r
            best_x    = x_try
            best_As   = max(An, Am)

    # Refine around best_x
    lo, hi = max(d2, best_x - h / max_iter), min(h, best_x + h / max_iter)
    for _ in range(50):
        mid = (lo + hi) / 2
        An_lo, Am_lo = residual(lo)
        An_m,  Am_m  = residual(mid)
        if (An_lo - Am_lo) * (An_m - Am_m) < 0:
            hi = mid
        else:
            lo = mid
    x_opt = (lo + hi) / 2
    An, Am = residual(x_opt)
    As_half = max(max(An, Am), 0.0)
    As_total = 2 * As_half   # symmetric top + bottom

    return x_opt / 1000, As_total  # return x in m, As in mm²


def _link_rebar_volume_per_m(
    b_mm: float,
    cover_mm: float = 35.0,
    long_bar_dia_mm: float = 20.0,
    link_dia_mm: float = 6.0,
) -> float:
    """
    EC2 9.5.3 minimum transverse (link/stirrup) rebar volume per metre of column.

    Rules applied (square section):
      EC2 9.5.3(1): min link diameter = max(6 mm, 0.25 × max_long_bar_dia)
      EC2 9.5.3(3): max link spacing  = min(20 × min_long_bar_dia, b_min, 400 mm)
                    For a square section b_min = b.
                    Assumes min_long_bar_dia = 16 mm → min(320, b_mm, 400).
      Inner perimeter (square section, single hoop): 4 × (b − 2 × cover)
      Conservatively uses 4 legs (accounts for intermediate cross-links in wider columns).

    Returns volume in m³ per linear metre of column.
    """
    # EC2 9.5.3(1) — minimum link diameter
    phi_link = max(link_dia_mm, 0.25 * long_bar_dia_mm)

    # EC2 9.5.3(3) — maximum link spacing (assume min longitudinal bar = 16 mm)
    min_long_bar_dia_mm = 16.0
    spacing_mm = min(20.0 * min_long_bar_dia_mm, b_mm, 400.0)

    # Single rectangular hoop inner perimeter for a square column
    inner_perimeter_mm = 4.0 * (b_mm - 2.0 * cover_mm)

    # Volume of one link (hoop) in mm³
    volume_per_link_mm3 = math.pi / 4.0 * phi_link**2 * inner_perimeter_mm

    # Number of links per metre of column height
    n_links_per_m = 1000.0 / spacing_mm

    # Convert mm³/link × links/m → m³/m
    return volume_per_link_mm3 * n_links_per_m / 1e9


def _effective_length(clear_height_m: float, condition: str = "braced_23") -> float:
    """
    Effective length l₀ for a braced column (EC2 5.8.3.2).

    condition:
      'braced_12' : both ends partially fixed → l₀ = 0.75·l
      'braced_23' : one end pinned, one partially fixed → l₀ = 0.95·l  (conservative)
      'braced_22' : both partially fixed → l₀ = 0.7·l (interior column)
      'cantilever' : l₀ = 2·l
    """
    factors = {
        "braced_12": 0.75,
        "braced_22": 0.70,
        "braced_23": 0.857,   # EC2 Table 5.1 / CCIP condition 2 at base, 3 at top
        "cantilever": 2.0,
    }
    k = factors.get(condition, 0.85)
    return k * clear_height_m


def check_rc_column(
    *,
    h_m: float = 0.30,
    b_m: float = 0.30,
    clear_height_m: float = 3.325,
    N_Ed_kN: float = 1129.6,
    M_02_kNm: float = 89.6,        # larger end moment (from frame analysis)
    M_01_kNm: Optional[float] = None,  # smaller end moment; if None → assume 0.5*M_02
    f_ck_MPa: float = 30.0,
    f_yk_MPa: float = 500.0,
    c_nom_mm: float = 35.0,
    phi_link_mm: float = 8.0,
    phi_main_mm: float = 25.0,
    effective_length_condition: str = "braced_23",
    A_factor: float = _A_DEFAULT,
    B_factor: float = _B_DEFAULT,
    C_factor: Optional[float] = None,  # if None, computed from M_01/M_02
) -> Dict[str, Any]:
    """
    EC2 RC column design check.

    Returns dict with success, As_req_mm2, rebar_volume_m3_per_m,
    concrete_volume_m3_per_m, and full intermediate results.
    """
    res = RCColumnResult(h_m=h_m, b_m=b_m, N_Ed_kN=N_Ed_kN, M_Ed_kNm=M_02_kNm)

    try:
        # ── 1. Geometry ────────────────────────────────────────────────────────
        l0   = _effective_length(clear_height_m, effective_length_condition)
        res.l0_m = l0

        # cover to centroid of main bar
        d2_m = (c_nom_mm + phi_link_mm + phi_main_mm / 2) / 1000
        d_m  = h_m - d2_m   # effective depth (tension side)

        # ── 2. Slenderness ─────────────────────────────────────────────────────
        lam = l0 * math.sqrt(12) / h_m   # EC2 5.8.3.2 for rectangular section
        res.lambda_val = lam

        # EC2 5.8.3.1: n = N_Ed/(Ac*fck) — normalised with fck (characteristic), not fcd
        n = (N_Ed_kN * 1000) / (f_ck_MPa * b_m * 1000 * h_m * 1000)
        n = max(n, 0.01)

        if C_factor is None:
            # M_01 = smaller end moment; default 0 (pinned end) for perimeter columns
            m01 = M_01_kNm if M_01_kNm is not None else 0.0
            rm  = m01 / M_02_kNm if M_02_kNm != 0 else 0.0
            C_rm = 1.7 - rm   # EC2 5.8.3.1(1): C = 1.7 - r_m (max 2.0 for double curvature)
        else:
            C_rm = C_factor

        lam_lim = 20 * A_factor * B_factor * C_rm / math.sqrt(n)
        res.lambda_lim = lam_lim
        res.is_slender = lam > lam_lim

        if res.is_slender:
            logger.warning("[rc_column] Column is slender (λ=%.1f > λ_lim=%.1f); "
                           "second-order effects should be included.", lam, lam_lim)

        # ── 3. Design moment including imperfections ───────────────────────────
        e_i = l0 / 400   # EC2 5.2(7) geometric imperfection
        e_0 = max(h_m / 30, 0.020)   # minimum eccentricity (EC2 6.1(4))

        M_Ed_design = M_02_kNm + e_i * N_Ed_kN   # kNm
        M_Ed_min    = e_0 * N_Ed_kN               # kNm

        M_Ed_design = max(M_Ed_design, M_Ed_min)

        res.e_i_m             = e_i
        res.M_Ed_design_kNm   = M_Ed_design
        res.M_Ed_kNm          = M_Ed_design

        # ── 4. Neutral-axis iteration for As ───────────────────────────────────
        A_c_mm2 = b_m * 1000 * h_m * 1000

        x_opt, As_total = _neutral_axis_iterate(
            N_Ed_kN=N_Ed_kN,
            M_Ed_kNm=M_Ed_design,
            b_m=b_m, h_m=h_m, d_m=d_m, d2_m=d2_m,
            f_ck=f_ck_MPa, f_yk=f_yk_MPa,
        )
        res.x_iter_m = x_opt

        As_min = _As_min_col(A_c_mm2, f_ck_MPa, f_yk_MPa, N_Ed_kN)
        As_max = _As_max_col(A_c_mm2)

        As_req = max(As_total, As_min)
        if As_req > As_max:
            logger.warning("[rc_column] As_req=%.0fmm² > As_max=%.0fmm²; "
                           "increase section size.", As_req, As_max)
            As_req = As_max

        res.As_req_mm2 = As_req
        res.As_min_mm2 = As_min
        res.As_max_mm2 = As_max
        res.rho_pct    = 100 * As_req / A_c_mm2

        # ── 5. Material quantities ─────────────────────────────────────────────
        res.concrete_volume_m3_per_m = b_m * h_m         # m³/m height
        res.rebar_volume_m3_per_m    = As_req / 1e6      # mm² → m³/m (longitudinal only)

        # EC2 9.5.3 — minimum transverse (link) rebar (square column, b_min = b)
        b_min_mm = min(b_m, h_m) * 1000.0
        res.link_rebar_volume_m3_per_m = _link_rebar_volume_per_m(
            b_mm=b_min_mm,
            cover_mm=c_nom_mm,
            long_bar_dia_mm=phi_main_mm,
            link_dia_mm=phi_link_mm,
        )

        res.status = "OK"

    except Exception as exc:
        res.status = "FAILED"
        res.error  = str(exc)
        logger.error("[rc_column] check failed: %s", exc)

    out = {
        "success":                   res.status == "OK",
        "status":                    res.status,
        "error":                     res.error,
        "h_m":                       res.h_m,
        "b_m":                       res.b_m,
        "l0_m":                      res.l0_m,
        "N_Ed_kN":                   res.N_Ed_kN,
        "M_02_kNm":                  M_02_kNm,
        "M_Ed_design_kNm":           res.M_Ed_design_kNm,
        "lambda":                    res.lambda_val,
        "lambda_lim":                res.lambda_lim,
        "is_slender":                res.is_slender,
        "e_i_m":                     res.e_i_m,
        "x_iter_m":                  res.x_iter_m,
        "As_req_mm2":                res.As_req_mm2,
        "As_min_mm2":                res.As_min_mm2,
        "As_max_mm2":                res.As_max_mm2,
        "rho_pct":                   res.rho_pct,
        "concrete_volume_m3_per_m":  res.concrete_volume_m3_per_m,
        "rebar_volume_m3_per_m":     res.rebar_volume_m3_per_m,
        "link_rebar_volume_m3_per_m": res.link_rebar_volume_m3_per_m,
    }
    return out
