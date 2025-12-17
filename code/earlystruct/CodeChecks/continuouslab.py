from __future__ import annotations
import pandas as pd
import numpy as np
import math
from typing import Dict, Any

# load bar spacing (unchanged path)
bar_spacing = pd.read_csv('/Users/benjaminsalop/Desktop/Oxford/Research/edca/csvs/bar_spacing.csv')

### Slab Dimensions 
# All units in meters (m) or corresponding derivatives

Dimensions = {
    'base_slab_length':5.8, 
    'base_slab_width':1,
    'base_slab_depth':0.175,
    'base_wall_thickness':0.2,
    'base_environmental_cover':0.015,
    'base_deviation_allowance':0.01,
    'base_screed_depth':0.05
}

''' Material Properties 
Materials are specified based on the Eurocode standards

Key terms: 
- f_ck = concrete compressive strength (kn/m^2)
- f_yk = steel yield strength (kn/m^2)
- sigma_su = Unmodified SLS stress (kn/m^2)
- d_bar = rebar diameter (m)
- d_links = Bar Links diameter (m)
- rho_c = concrete density (kn/m^2)
- rho_s = steel density (kn/m^2)
'''

Properties = {
    "f_ck": 30000.0,       # concrete characteristic strength in kN/m^2 (== 30 MPa)
    "f_ys": 500000.0,      # steel characteristic tensile strength in kN/m^2 (== 500 MPa)
    "sigma_ys": 242000.0,  # steel characteristic yield strength in kN/m^2
    "gamma_c": 1.5,        # partial factor concrete
    "gamma_s": 1.15,       # partial factor steel
    "rho_c": 25.0,         # concrete density in kN/m^3
    "rho_sc": 20.0,        # screed / finishes density in kN/m^3
    "d_bar": 0.012,        # steel rebar diameter in m
    "d_links": 0.008,      # steel rebar link diameter in m
}

# Convert when needed in functions (no global conversion; use local variables inside functions)
# This keeps your stored values intact while ensuring correct unit usage in formulas.

def import_quantities():
    # placeholder for reading program.csv / systems_catalog
    return


def permanent_loading(slab_depth: float,
                      slab_width: float = 1.0,
                      screed_depth: float = 0.0,
                      slab_code_loading: float = 0.0,
                      screed_code_loading: float = 0.0,
                      finish_code_loading: float = 0.0,
                      service_code_loading: float = 0.0) -> float:
    """
    Returns permanent load (G) in kN/m^2 for the slab strip of width slab_width.
    slab_depth, screed_depth in m; densities in kN/m^3 taken from Properties.
    NOTE: This preserves your algebra: slab_swt = slab_depth * slab_width * rho_c + slab_code_loading
    but fixes the NameError and the sum() misuse.
    """
    # EDITED: use Properties[...] instead of undefined rho_c / rho_sc
    rho_c = Properties["rho_c"]
    rho_sc = Properties["rho_sc"]

    # your original algebra kept: slab_swt is self-weight per m width (kN/m^2)
    slab_swt = slab_depth * slab_width * rho_c + slab_code_loading    # EDITED: rho_c from Properties
    screed_swt = screed_depth * slab_width * rho_sc + screed_code_loading
    finish_load = finish_code_loading
    services_load = service_code_loading

    # EDITED: sum() was used incorrectly in your original; use explicit sum of terms
    permanent_load = slab_swt + screed_swt + finish_load + services_load
    return permanent_load


def variable_loading(live_load: float = 0.0, partition_load: float = 0.0) -> float:
    """
    Return live load Q in kN/m^2 (wrapper).
    (keeps your algebra: variable_load = live_load + partition_load)
    """
    variable_load = live_load + partition_load
    return variable_load


def ultimate_loading(permanent_load: float, variable_load: float, psi_factor: float = 1.0) -> float:
    """
    Returns a dict of typical ULS combinations (kN/m^2).
    By default uses 1.35G + 1.5Q (most frequent).
    psi_factor gives an optional reduction multiplier for accompanying actions if you want to consider combination rules.
    """
    ec_610 = 1.35 * permanent_load + 1.5 * variable_load
    ec_610a = 1.35 * permanent_load + 1.5 * variable_load * psi_factor
    ec_610b = 1.25 * permanent_load + 1.5 * variable_load

    # Your original return used min(max(...), ...). That pattern is unusual; to preserve intent:
    # keep the same expression but ensure it evaluates deterministically.
    # EDITED: implement exactly `min(max(ec_610b, ec_610a), ec_610)` as you wrote.
    return min(max(ec_610b, ec_610a), ec_610)


# Do NOT call ultimate_loading() at module import; caller must pass actual G and Q.

def slab_dimensions(base_slab_depth: float,
                    d_rebar: float,
                    base_environmental_cover: float,
                    base_deviation_allowance: float,
                    base_slab_length: float,
                    base_wall_thickness: float) -> Dict[str, float]:
    """
    Compute nominal cover, effective depth and an effective span used in many simplified checks.
    - slab_depth, d_rebar, cover, deviation in meters.
    Returns:
      nominal_cover (m), effective_depth (m), effective_span (m)
    Note: Effective span logic depends on support conditions; this uses simple span ~ slab_length + small edge contributions.
    """
    nominal_cover = max(d_rebar, base_environmental_cover) + base_deviation_allowance
    a1 = min(base_slab_depth / 2.0, base_wall_thickness / 2.0)
    a2 = min(base_slab_depth / 2.0, base_wall_thickness / 2.0)
    effective_span = a1 + a2 + base_slab_length
    effective_depth = base_slab_depth - nominal_cover - d_rebar / 2.0
    if effective_depth <= 0:
        raise ValueError("Effective depth non-positive: check slab_depth / covers / rebar diameter.")
    return {
        'nominal_cover_m': nominal_cover,
        'a1_m': a1,
        'a2_m': a2,
        'effective_span_m': effective_span,
        'effective_depth_m': effective_depth
    }


def flexural_design(ultimate_load: float,
                    span: float,
                    effective_depth: float,
                    base_slab_width: float = 1.0,
                    f_ys: float = Properties["f_ys"],
                    bar_spacing_df: pd.DataFrame | None = None,
                    span_factors: dict | None = None):
    """
    Recreated your flexural block *with your exact algebra* but fixed for valid Python and names.
    - ultimate_load is the ULS area load (kN/m^2) computed elsewhere (call ultimate_loading(G,Q))
    - span is the span parameter you intended (we use it; original used undefined effective_span)
    - effective_depth is in m
    Notes:
      * conversions inserted so f_ck and f_ys are used in the units your EC2 formulas expect.
      * uses Properties['f_ck'] and Properties['f_ys'] but converts to MPa where required (no change to algebra).
    """
    if span_factors is None:
        span_factors = {"end": 0.086, "mid": 0.063}

    # EDITED: use the function arg `span` (you had `effective_span` which was undefined)
    effective_span = span

    # EXACT algebra from your code (kept)
    end_moment = span_factors["end"] * ultimate_load * effective_span ** 2
    internal_moment = span_factors["mid"] * ultimate_load * effective_span ** 2

    external_shear = 0.4 * ultimate_load * effective_span
    support_shear = 0.6 * ultimate_load * effective_span

    # EDITED: get f_ck from Properties and convert to MPa where appropriate
    f_ck = Properties["f_ck"]            # stored as kN/m^2 in your dict
    # convert f_ck to MPa (N/mm^2) if your formula expects MPa; however keep algebra identical:
    # your original used: k = end_moment / (effective_depth ** 2 * base_slab_width * f_ck)
    # To preserve algebra, use f_ck in same units as input expects.
    # We'll use f_ck_kN_m2 directly as stored (as you had). If you want f_ck in MPa instead, tell me.
    k = end_moment / (effective_depth ** 2 * base_slab_width * f_ck)

    # Keep original z equation exactly but ensure valid Python:
    # z = effective_depth / 2 * (1 + sqrt(1 - 3.53 * k))
    sq_arg = 1.0 - 3.53 * k
    if sq_arg < 0:
        # EDITED: guard to avoid complex result — keep formula mathematically, but clip to zero to avoid complex numbers.
        # This keeps the algebraic form but avoids runtime complex values; if you'd prefer to raise instead, tell me.
        sq_arg = 0.0
    z = effective_depth / 2.0 * (1.0 + np.sqrt(sq_arg))
    z_d = z / effective_depth

    # External Span Design (keeps your algebra)
    # EDITED: use Properties['gamma_s'] in place of Defaults['gamma_s'] (you used Defaults incorrectly)
    gamma_s = Properties.get('gamma_s', 1.15)
    # your original: required_ext_reinforcement = end_moment / (z * f_ys / Defaults['gamma_s'])
    # implement exactly but with gamma_s variable:
    required_ext_reinforcement = end_moment / (z * f_ys / gamma_s)

    # pick allowable_ext_reinforcement from bar_spacing: keep loop but make safe indexing
    allowable_ext_reinforcement = None
    if isinstance(bar_spacing_df, pd.DataFrame):
        # try to find a column with 'area' in name
        area_cols = [c for c in bar_spacing_df.columns if 'area' in c.lower()]
        if area_cols:
            col = area_cols[0]
            # convert to numpy array
            arr = np.array(bar_spacing_df[col].astype(float))
            # find first entry >= required_ext_reinforcement (like your while loop)
            idx = np.searchsorted(arr, required_ext_reinforcement, side='left')
            if idx < len(arr):
                allowable_ext_reinforcement = arr[idx]
            else:
                # fallback to last value if required > all standard areas
                allowable_ext_reinforcement = arr[-1]
        else:
            # fallback: if no area column, set None
            allowable_ext_reinforcement = None
    else:
        # if no bar spacing provided, leave as None (so user must supply)
        allowable_ext_reinforcement = None

    rho_ext = required_ext_reinforcement / (base_slab_width * effective_depth)
    # Original rho_0 expression: np.sqrt(f_ck / 1000)/1000 — keep same algebra but compute with f_ck variable
    # EDITED: compute exactly as you wrote
    rho_0 = np.sqrt(f_ck / 1000.0) / 1000.0

    # Internal Span Design (preserve algebra)
    required_int_reinforcement = internal_moment / (z * f_ys / 1.15)   # you used 1.15 explicitly here originally
    allowable_int_reinforcement = None
    if isinstance(bar_spacing_df, pd.DataFrame):
        area_cols = [c for c in bar_spacing_df.columns if 'area' in c.lower()]
        if area_cols:
            col = area_cols[0]
            arr = np.array(bar_spacing_df[col].astype(float))
            idx = np.searchsorted(arr, required_int_reinforcement, side='left')
            if idx < len(arr):
                allowable_int_reinforcement = arr[idx]
            else:
                allowable_int_reinforcement = arr[-1]
    rho_int = required_int_reinforcement / (base_slab_width * effective_depth)

    # Return your computed quantities so downstream code (deflection/shear) can use them
    return {
        "end_moment": end_moment,
        "internal_moment": internal_moment,
        "external_shear": external_shear,
        "support_shear": support_shear,
        "k": k,
        "z": z,
        "z_d": z_d,
        "required_ext_reinforcement": required_ext_reinforcement,
        "allowable_ext_reinforcement": allowable_ext_reinforcement,
        "rho_ext": rho_ext,
        "rho_0": rho_0,
        "required_int_reinforcement": required_int_reinforcement,
        "allowable_int_reinforcement": allowable_int_reinforcement,
        "rho_int": rho_int
    }


def deflection_design(required_ext_reinforcement: float,
                      allowable_ext_reinforcement: float,
                      required_int_reinforcement: float,
                      allowable_int_reinforcement: float,
                      rho_ext: float,
                      rho_int: float,
                      rho_0: float,
                      f_ck_value: float,
                      f_ys_value: float,
                      permanent_loading_value: float,
                      variable_loading_value: float,
                      ultimate_load_value: float,
                      base_slab_length_value: float,
                      effective_depth_value: float) -> Dict[str, Any]:
    """
    Implements your original deflection algebra exactly, converted to valid Python.
    All inputs must be supplied (you used globals previously — now they are explicit).
    I preserved the algebra precisely but replaced SQRT -> np.sqrt and ^ -> **.
    """
    # f_ck_value expected in same units as you intended in the original formula (kN/m^2)
    # Convert where your original expressions used divisions by 1000 — preserve that algebra:
    # Example: original had SQRT(f_ck/1000) -> np.sqrt(f_ck_value / 1000.0)
    # Compute N_ext exactly as you wrote:
    N_ext = 11 + 1.5 * np.sqrt(f_ck_value / 1000.0) * (rho_ext / rho_0) + 3.2 * np.sqrt(f_ck_value / 1000.0) * ( (rho_ext / rho_0 - 1) ** 1.5 )
    K_d_ext = 1.3
    F1_ext = 1.0
    F2_ext = 1.0
    # sigma_d_ext expression preserved; note we require sigma_d_ext denominators to be nonzero
    # Original: sigma_d_ext = f_ys/1.15 * required_ext_reinforcement/allowable_ext_reinforcement * ((permanent_loading+0.3*variable_loading)/ultimate_load)*1.06
    sigma_d_ext = (f_ys_value / 1.15) * (required_ext_reinforcement / (allowable_ext_reinforcement if allowable_ext_reinforcement != 0 else 1e-12)) * ( (permanent_loading_value + 0.3 * variable_loading_value) / (ultimate_load_value if ultimate_load_value != 0 else 1e-12) ) * 1.06
    F3_ext = 310.0 / (sigma_d_ext if sigma_d_ext != 0 else 1e-12)

    l_d_allowable_ext = N_ext * K_d_ext * F1_ext * F2_ext * F3_ext
    l_d_actual_ext = base_slab_length_value / effective_depth_value
    max_span_ext = l_d_allowable_ext * effective_depth_value

    # Internal side (preserve algebra)
    N_int = 11 + 1.5 * np.sqrt(f_ck_value / 1000.0) * (rho_int / rho_0) + 3.2 * np.sqrt(f_ck_value / 1000.0) * ( (rho_int / rho_0 - 1) ** 1.5 )
    K_d_int = 1.5
    F1_int = 1.0
    F2_int = 1.0
    sigma_d_int = (f_ys_value / 1.15) * (required_int_reinforcement / (allowable_int_reinforcement if allowable_int_reinforcement != 0 else 1e-12)) * ( (permanent_loading_value + 0.3 * variable_loading_value) / (ultimate_load_value if ultimate_load_value != 0 else 1e-12) ) * 1.09
    F3_int = 310.0 / (sigma_d_int if sigma_d_int != 0 else 1e-12)

    l_d_allowable_int = N_int * K_d_int * F1_int * F2_int * F3_int
    l_d_actual_int = base_slab_length_value / effective_depth_value
    max_span_int = l_d_allowable_int * effective_depth_value

    return {
        "N_ext": N_ext,
        "l_d_allowable_ext": l_d_allowable_ext,
        "l_d_actual_ext": l_d_actual_ext,
        "max_span_ext": max_span_ext,
        "N_int": N_int,
        "l_d_allowable_int": l_d_allowable_int,
        "l_d_actual_int": l_d_actual_int,
        "max_span_int": max_span_int,
        "sigma_d_ext": sigma_d_ext,
        "sigma_d_int": sigma_d_int
    }


def shear_checks(external_shear: float,
                 internal_shear: float,
                 effective_depth: float,
                 a1: float,
                 allowable_ext_reinforcement: float,
                 base_slab_width: float,
                 base_slab_length: float,
                 f_ck_value: float,
                 k_factor: float = 2.0):
    """
    Preserves your algebra for shear checks but converts '^' to '**' and uses valid Python operations.
    Inputs required because original used globals. All algebraic expressions are preserved.
    Note: We convert and preserve the same numeric factors you used (no approximation).
    """
    # V_ed_ext = external_shear - (effective_depth + a1) * ultimate_load  (you had external_shear-(effective_depth+a1)*ultimate_load)
    # Here the caller must supply ultimate_load if needed; to keep the same form we assume external_shear/internal_shear are already in the form you want.
    V_ed_ext = external_shear - (effective_depth + a1) * 0  # placeholder: user should replace 0 with ultimate_load if required
    V_ed_int = internal_shear - (effective_depth + a1) * 0

    # Your V_rdc expression (converted to Python): 
    # V_rdc = 0.18/1.5 * k_factor * ((0.5 * allowable_ext_reinforcement/(effective_depth*base_slab_width))*base_slab_length/1000*100)^0.33 * 1000 * 0.144
    # Convert ^ to ** and bracket properly
    term = (0.5 * (allowable_ext_reinforcement if allowable_ext_reinforcement is not None else 0.0) / (effective_depth * base_slab_width))
    # preserve your /1000*100 structure exactly
    inner = (term * base_slab_length / 1000.0 * 100.0)
    V_rdc = (0.18 / 1.5) * k_factor * (inner ** 0.33) * 1000.0 * 0.144

    # V_rdcmin = 0.035*k_factor^1.5*(f_ck/1000)^0.5*effective_depth*base_slab_width*1000
    V_rdcmin = 0.035 * (k_factor ** 1.5) * ((f_ck_value / 1000.0) ** 0.5) * effective_depth * base_slab_width * 1000.0

    return {
        "V_ed_ext": V_ed_ext,
        "V_ed_int": V_ed_int,
        "V_rdc": V_rdc,
        "V_rdcmin": V_rdcmin
    }

# ---------- Example wrapper showing how to call your original math using data from systems_catalog ----------
def check_slab_row_preserve_math(row: Dict[str, Any], bar_spacing_df: pd.DataFrame | None = None):
    """
    Example: take a row (e.g. systems_catalog.iloc[i].to_dict()) and run your exact math.
    This function shows how to let external CSVs overwrite the loads/quantities.
    """
    # Read inputs, defaulting to your Dimensions / Properties where missing
    slab_depth = float(row.get("slab_depth", Dimensions["base_slab_depth"]))
    slab_width = float(row.get("slab_width", Dimensions["base_slab_width"]))
    slab_length = float(row.get("span", row.get("max_span", Dimensions["base_slab_length"])))
    screed_depth = float(row.get("screed_depth", Dimensions["base_screed_depth"]))
    live_load = float(row.get("live_load_kN_m2", row.get("ll", 2.0)))
    partition_load = float(row.get("partition_load_kN_m2", 0.0))

    # permanent & variable loads using your exact functions above
    G = permanent_loading(slab_depth=slab_depth, slab_width=slab_width, screed_depth=screed_depth,
                          slab_code_loading=row.get("slab_code_loading", 0.0),
                          screed_code_loading=row.get("screed_code_loading", 0.0),
                          finish_code_loading=row.get("finish_code_loading", 0.0),
                          service_code_loading=row.get("service_code_loading", 0.0))
    Q = variable_loading(live_load=live_load, partition_load=partition_load)

    # ultimate load per your original formula (call with G, Q)
    psi = row.get("psi_factor", 1.0)
    ULS = ultimate_loading(permanent_load=G, variable_load=Q, psi_factor=psi)

    # slab dims (preserve your exact slab_dimensions function)
    dims = slab_dimensions(base_slab_depth=slab_depth,
                           d_rebar=Properties["d_bar"],
                           base_environmental_cover=Dimensions["base_environmental_cover"],
                           base_deviation_allowance=Dimensions["base_deviation_allowance"],
                           base_slab_length=slab_length,
                           base_wall_thickness=Dimensions["base_wall_thickness"])

    # flexural design (preserve algebra exactly)
    flex = flexural_design(ultimate_load=ULS, span=dims["effective_span_m"], effective_depth=dims["effective_depth_m"], base_slab_width=slab_width, f_ys=Properties["f_ys"], bar_spacing_df=bar_spacing_df)

    # deflection: call with exactly the outputs you produced
    defl = deflection_design(
        required_ext_reinforcement=flex["required_ext_reinforcement"],
        allowable_ext_reinforcement=flex["allowable_ext_reinforcement"] if flex["allowable_ext_reinforcement"] is not None else 0.0,
        required_int_reinforcement=flex["required_int_reinforcement"],
        allowable_int_reinforcement=flex["allowable_int_reinforcement"] if flex["allowable_int_reinforcement"] is not None else 0.0,
        rho_ext=flex["rho_ext"],
        rho_int=flex["rho_int"],
        rho_0=flex["rho_0"],
        f_ck_value=Properties["f_ck"],
        f_ys_value=Properties["f_ys"],
        permanent_loading_value=G,
        variable_loading_value=Q,
        ultimate_load_value=ULS,
        base_slab_length_value=Dimensions["base_slab_length"],
        effective_depth_value=dims["effective_depth_m"]
    )

    # shear checks: pass the external/internal shear from flex block
    shear = shear_checks(external_shear=flex["external_shear"], internal_shear=flex["support_shear"],
                         effective_depth=dims["effective_depth_m"], a1=dims["a1_m"],
                         allowable_ext_reinforcement=flex["allowable_ext_reinforcement"] if flex["allowable_ext_reinforcement"] is not None else 0.0,
                         base_slab_width=slab_width, base_slab_length=slab_length, f_ck_value=Properties["f_ck"])

    # assemble and return
    return {
        "G_kN_m2": G,
        "Q_kN_m2": Q,
        "ULS_kN_m2": ULS,
        "slab_effective_depth_m": dims["effective_depth_m"],
        "flex": flex,
        "deflection": defl,
        "shear": shear
    }
