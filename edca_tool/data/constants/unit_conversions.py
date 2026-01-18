import math

'''
This module provides unit conversion factors between Imperial and Metric systems.
It includes functions to convert lengths, pressures, densities, and areas. 
Over time, every conversion factor used in the EDCA Tool should be centralized here for consistency and easy access.
Each function takes a value in an imperial or metric unit and returns the equivalent value in the corresponding unit, where each unit is named by its abbreviated notation.
'''

# Imperial to Metric Conversion Factors

def in_to_mm(inches: float) -> float:
    return inches * 25.4

def ft_to_m(feet: float) -> float:
    return feet * 0.3048

def psi_to_mpa(psi: float) -> float:
    return psi * 0.00689476

def ksi_to_mpa(ksi: float) -> float:
    return ksi * 6.89476

def pcf_to_kgm3(pcf: float) -> float:
    return pcf * 16.0185

def psf_to_nm2(psf: float) -> float:
    return psf * 47.8803

def in2_to_mm2(inches2: float) -> float:
    return inches2 * 645.16

def ft2_to_m2(feet2: float) -> float:
    return feet2 * 0.092903

# Metric to Imperial Conversion Factors

def mm_to_inches(mm: float) -> float:
    return mm / 25.4

def m_to_ft(meters: float) -> float:
    return meters / 0.3048

def mpa_to_psi(mpa: float) -> float:
    return mpa / 0.00689476

def mpa_to_ksi(mpa: float) -> float:
    return mpa / 6.89476

def kgm3_to_pcf(kgm3: float) -> float:
    return kgm3 / 16.0185

def nm2_to_psf(nm2: float) -> float:
    return nm2 / 47.8803

def mm2_to_in2(mm2: float) -> float:
    return mm2 / 645.16

def m2_to_ft2(m2: float) -> float:
    return m2 / 0.092903

# Additional Conversions
def ton_km_to_lb_mile(ton_km: float) -> float:
    return ton_km * 0.000556

def lb_mile_to_ton_km(lb_mile: float) -> float:
    return lb_mile / 0.000556

