from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib as plt
import math as math

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
    "f_ck": 30000.0,       # concrete characteristic strength in kN/m^2
    "f_ys": 500000.0,      # steel characteristic tensile strength in kN/m^2
    "sigma_ys": 242000.0,  # steel characteristic yield strength in kN/m^2
    "gamma_c": 1.5,        # partial factor concrete
    "gamma_s": 1.15,       # partial factor steel
    "rho_c": 25.0,         # concrete density in kN/m^3
    "rho_sc": 20.0,        # screed / finishes density in kN/m^3
    "d_bar": 0.012,        # steel rebar diameter in m
    "d_links": 0.008,      # steel rebar link diameter in m
}


def import_quantities():
    return



def permanent_loading(slab_depth:float, slab_width:float=1.0, screed_depth: float=0.0, 
                      slab_code_loading:float=0, screed_code_loading:float=0, finish_code_loading:float=0, 
                      service_code_loading:float=0):
    
    """
    Returns permanent load (G) in kN/m^2 for the slab strip of width `slab_width`.
    slab_depth, screed_depth in m; densities in kN/m^3.
    """

    slab_swt = slab_depth * slab_width * rho_c + slab_code_loading
    screed_swt = screed_depth * slab_width * rho_sc + screed_code_loading
    finish_load = finish_code_loading
    services_load = service_code_loading
    permanent_load = np.sum(slab_swt, screed_swt, finish_load, services_load)
    return permanent_load

def variable_loading(live_load:float, partition_load:float=0):
    """
    Return live load Q in kN/m^2 (wrapper).
    """

    variable_load = live_load + partition_load
    return variable_load

def ultimate_loading(permanent_load:float, variable_load: float, psi_factor: float):

    """
    Returns a dict of typical ULS combinations (kN/m^2).
    By default uses 1.35G + 1.5Q (most frequent).
    psi_factor gives an optional reduction multiplier for accompanying actions if you want to consider combination rules.
    """
    ec_610 = 1.35*permanent_load + 1.5 * variable_load
    ec_610a = 1.35*permanent_load + 1.5 * variable_load * psi_factor
    ec_610b = 1.25*permanent_load + 1.5 * variable_load
    
    return min(max(ec_610b,ec_610a),ec_610)

def slab_dimensions(base_slab_depth: float, d_rebar: float, 
                    base_environmental_cover: float, base_deviation_allowance: float, 
                    base_slab_length: float, base_wall_thickness: float):
    
    """
    Compute nominal cover, effective depth and an effective span used in many simplified checks.
    - slab_depth, d_rebar, cover, deviation in meters.
    Returns:
      nominal_cover (m), effective_depth (m), effective_span (m)
    Note: Effective span logic depends on support conditions; this uses simple span ~ slab_length + small edge contributions.
    """
    
    nominal_cover = max(d_rebar,base_environmental_cover) + base_deviation_allowance
    a1 = min(base_slab_depth/2, base_wall_thickness/2)
    a2 = min(base_slab_depth/2, base_wall_thickness/2)
    effective_span = a1 + a2 + base_slab_length
    effective_depth = base_slab_depth - nominal_cover - d_rebar/2
    if effective_depth <= 0:
        raise ValueError("Effective depth non-positive: check slab_depth / covers / rebar diameter.")
    return {'nominal_cover_m': nominal_cover, 'a1_m': a1, 'a2_m': a2, 'effective_span_m': effective_span, 'effective_depth_m': effective_depth}

def flexural_design(
        ultimate_load: float, effective_span: float,
        effective_depth: float, base_slab_width: float = 1.0,
        f_ys: float = Properties["f_ys"],
        bar_spacing: pd.DataFrame | None = None,
        span_factors: Dict[str, float] | None = None,
        ):
    
    """
    Simplified flexural design for a 1m strip:
    - w_uls_kN_m2 : ULS distributed load (kN/m^2)
    - span_m, effective_depth_m, slab_width_m in m
    This function:
      * computes characteristic bending moments using continuous-slab factors (default 0.086,0.063)
      * computes required steel area As (mm^2 per meter width)
      * returns rho and required As and optionally nearest bar option from bar_spacing_df if provided.
    Units internally converted to N/mm and mm.
    NOTE: This is a simplified method suitable for preliminary checks. Use full Eurocode formulas for final design.
    """

    if span_factors is None:
        span_factors = {"end": 0.086, "mid": 0.063}

    end_moment = span_factors["end"] * ultimate_load * effective_span ** 2
    internal_moment = span_factors["mid"] * ultimate_load * effective_span ** 2

    external_shear = 0.4 * ultimate_load * effective_span
    support_shear = 0.6 * ultimate_load * effective_span

    k = end_moment / (effective_depth ** 2 * base_slab_width * f_ck)
    z = effective_depth / 2 * (1 + np.sqrt(1-3.53*k))
    z_d = z / effective_depth

    #External Span Design
    required_ext_reinforcement = end_moment / (z * f_ys / Defaults['gamma_s'])
    ext_row_index = 0
    allowable_ext_reinforcement = bar_spacing['area'][0]
    while allowable_ext_reinforcement <= required_ext_reinforcement:
        ext_row_index +=1
        allowable_ext_reinforcement = bar_spacing['area'][ext_row_index]
    rho_ext = required_ext_reinforcement / (base_slab_width * effective_depth)
    rho_0 = np.sqrt(f_ck / 1000)/1000

    #Internal Span Design
    required_int_reinforcement = internal_moment / (z * f_ys / 1.15)
    int_row_index = 0
    allowable_int_reinforcement = 0
    while allowable_int_reinforcement <= required_int_reinforcement:
        int_row_index +=1
        allowable_int_reinforcement = bar_spacing['area'][int_row_index]
    rho_int = required_int_reinforcement / (base_slab_width * effective_depth) 
    return allowable_ext_reinforcement, allowable_int_reinforcement

def deflection_design():
    N_ext = 11+1.5*SQRT(f_ck/1000)*(rho_ext/rho_0)+3.2*SQRT(f_ck/1000)*(rho_ext/rho_0 - 1)^1.5
    K_d_ext = 1.3
    F1_ext = 1
    F2_ext = 1
    F3_ext = 310/sigma_d_ext
    sigma_d_ext = f_ys/1.15*required_ext_reinforcement/allowable_ext_reinforcement*((permanent_loading+0.3*variable_loading)/ultimate_load)*1.06
    l_d_allowable_ext = N_ext * K_d_ext * F1_ext * F2_ext * F3_ext
    l_d_actual_ext = base_slab_length/effective_depth
    max_span_ext = l_d_allowable_ext * effective_depth

    N_int = 11+1.5*SQRT(f_ck/1000)*(rho_int/rho_0)+3.2*SQRT(f_ck/1000)*(rho_int/rho_0 - 1)^1.5
    K_d_int = 1.5
    F1_int = 1
    F2_int = 1
    F3_int = 310/sigma_d_int
    sigma_d_int = f_ys/1.15*required_int_reinforcement/allowable_int_reinforcement*((permanent_loading+0.3*variable_loading)/ultimate_load)*1.09
    l_d_allowable_int = N_int * K_d_int * F1_int * F2_int * F3_int
    l_d_actual_int = base_slab_length/effective_depth
    max_span_int = l_d_allowable_int * effective_depth
    return

def shear_checks():
    k_factor = 2
    V_ed_ext = external_shear-(effective_depth+a1)*ultimate_load
    V_ed_int = internal_shear-(effective_depth+a1)*ultimate_load
    V_rdc = 0.18/1.5 * k_factor * ((0.5 * allowable_ext_reinforcement/(effective_depth*base_slab_width))*base_slab_length/1000*100)^0.33 * 1000 * 0.144
    V_rdcmin = 0.035*k_factor^1.5*(f_ck/1000)^0.5*effective_depth*base_slab_width*1000