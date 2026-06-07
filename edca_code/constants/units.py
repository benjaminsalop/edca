from __future__ import annotations


def in_to_mm(inches: float) -> float:
    return float(inches) * 25.4


def ft_to_m(feet: float) -> float:
    return float(feet) * 0.3048


def psi_to_mpa(psi: float) -> float:
    return float(psi) * 0.00689476


def ksi_to_mpa(ksi: float) -> float:
    return float(ksi) * 6.89476


def pcf_to_kgm3(pcf: float) -> float:
    return float(pcf) * 16.0185


def kgm3_to_kN_m3(kgm3: float) -> float:
    return float(kgm3) * 9.80665 / 1000.0


def pcf_to_kN_m3(pcf: float) -> float:
    return kgm3_to_kN_m3(pcf_to_kgm3(pcf))


def psf_to_nm2(psf: float) -> float:
    return float(psf) * 47.8803


def psf_to_knm2(psf: float) -> float:
    return float(psf) * 0.0478803


def in2_to_mm2(inches2: float) -> float:
    return float(inches2) * 645.16


def ft2_to_m2(feet2: float) -> float:
    return float(feet2) * 0.092903


def mm_to_inches(mm: float) -> float:
    return float(mm) / 25.4


def m_to_ft(meters: float) -> float:
    return float(meters) / 0.3048


def mpa_to_psi(mpa: float) -> float:
    return float(mpa) / 0.00689476


def mpa_to_ksi(mpa: float) -> float:
    return float(mpa) / 6.89476


def kgm3_to_pcf(kgm3: float) -> float:
    return float(kgm3) / 16.0185


def nm2_to_psf(nm2: float) -> float:
    return float(nm2) / 47.8803


def mm2_to_in2(mm2: float) -> float:
    return float(mm2) / 645.16


def m2_to_ft2(m2: float) -> float:
    return float(m2) / 0.092903


# Backward-compatible aliases.
psi_to_MPa = psi_to_mpa
ksi_to_MPa = ksi_to_mpa
