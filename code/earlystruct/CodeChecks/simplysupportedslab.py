from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib as plt
import math as math

bar_spacing = pd.read_csv('/Users/benjaminsalop/Desktop/Oxford/Research/edca/csvs/bar_spacing.csv')

### Slab Dimensions 
# All units in meters (m) or corresponding derivatives

base_slab_length = 5.8
base_slab_width = 1
base_slab_depth = 0.175
base_wall_thickness = 0.2
base_environmental_cover = 0.015
base_deviation_allowance = 0.01
base_screed_depth = 0.05

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

f_ck = 30000
f_ys = 500000
sigma_su = 242000
d_rebar = 0.012
d_links = 0.008
rho_c = 25
rho_sc = 20

def import_quantities():
    return

def permanent_loading(slab_depth = base_slab_depth, slab_width = base_slab_width, 
                      wall_thickness = base_wall_thickness, screed_depth = base_screed_depth, 
                      slab_code_loading=0, screed_code_loading=0, finish_code_loading=0, 
                      service_code_loading=0):
    slab_load = slab_depth * slab_width * rho_c + slab_code_loading
    screed_load = screed_depth * slab_width * rho_sc + screed_code_loading
    finish_load = finish_code_loading
    services_load = service_code_loading
    ec_pl = sum(slab_load, screed_load, finish_load, services_load)
    return ec_pl

def variable_loading(ec_ll, ec_pl):
    ec_vl = ec_ll + ec_pl
    return ec_vl

def ultimate_loading():
    ec_pl = permanent_loading(slab_depth = base_slab_depth, slab_width = base_slab_width, 
                      wall_thickness = base_wall_thickness, screed_depth = base_screed_depth, 
                      slab_code_loading=0, screed_code_loading=0, finish_code_loading=0, 
                      service_code_loading=0)
    ec_vl = variable_loading(ec_ll, ec_pl)
    ec_610 = 1.35*ec_pl + 1.5 * ec_vl
    ec_610a = 1.35*ec_pl + 1.5 * ec_vl * ec_vlrf
    ec_610b = 1.25*ec_pl + 1.5 * ec_vl
    
    return min(max(ec_610b,ec_610a),ec_610)

ultimate_load = ultimate_loading()

def slab_dimensions(d_rebar, base_environmental_cover, base_deviation_allowance):
    nominal_cover = max(d_rebar,base_environmental_cover) + base_deviation_allowance
    a1 = min(base_slab_depth/2, base_wall_thickness/2)
    a2 = min(base_slab_depth/2, base_wall_thickness/2)
    effective_span = a1 + a2 + base_slab_length
    effective_depth = base_slab_depth - nominal_cover - d_rebar/2
    return nominal_cover, a1, a2, effective_span, effective_depth

def flexural_design():
    end_moment = 0.086 * ultimate_load * effective_span ** 2
    internal_moment = 0.063 * ultimate_load * effective_span ** 2

    external_shear = 0.4 * ultimate_load * effective_span
    support_shear = 0.6 * ultimate_load * effective_span

    k = end_moment / (effective_depth ** 2 * base_slab_width * f_ck)
    z = effective_depth / 2 * (1 + np.sqrt(1-3.53*k))
    z_d = z / effective_depth

    #External Span Design
    required_ext_reinforcement = end_moment / (z * f_ys / 1.15)
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
    return

