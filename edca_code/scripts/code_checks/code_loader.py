from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from edca_code.constants.units import kgm3_to_kN_m3, ksi_to_mpa, pcf_to_kgm3, psi_to_mpa


@dataclass(slots=True)
class Material:
    material_id: str
    f_ck_MPa: float = 0.0
    f_yk_MPa: float = 0.0
    density_kN_m3: float = 0.0
    gamma_c: float = 1.5
    gamma_s: float = 1.15
    original_units: str = "metric"
    raw: Dict[str, Any] | None = None


def _read_material_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Materials file not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        match = lowered.get(name.strip().lower())
        if match is not None:
            return match
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _detect_units(row: pd.Series) -> str:
    for key in ("unit", "units", "unit_system", "unitsystem"):
        if key in row.index:
            token = str(row[key]).strip().lower()
            if token in {"imperial", "us", "us_customary", "psi", "pcf"}:
                return "imperial"
    return "metric"


def load_material_from_csv(
    path: str | Path,
    material_id: str,
    *,
    id_col: str = "material_id",
    fck_col: str = "concrete_f_ck",
    fyk_col: str = "steel_fy",
    density_col: str = "density",
    gamma_c_col: str = "gamma_c",
    gamma_s_col: str = "gamma_s",
) -> Material:
    df = _read_material_table(path)
    id_name = _find_col(df, id_col)
    if id_name is None:
        raise KeyError(f"Missing id column {id_col!r} in materials file")
    row_match = df[df[id_name].astype(str) == str(material_id)]
    if row_match.empty:
        raise KeyError(f"Material {material_id!r} not found in {path}")
    row = row_match.iloc[0]
    units = _detect_units(row)

    fck_name = _find_col(df, fck_col)
    fyk_name = _find_col(df, fyk_col)
    density_name = _find_col(df, density_col)
    gamma_c_name = _find_col(df, gamma_c_col)
    gamma_s_name = _find_col(df, gamma_s_col)

    fck = _to_float(row[fck_name]) if fck_name else 0.0
    fyk = _to_float(row[fyk_name]) if fyk_name else 0.0
    density = _to_float(row[density_name]) if density_name else 0.0

    if units == "imperial":
        fck = psi_to_mpa(fck)
        fyk = ksi_to_mpa(fyk) if fyk <= 100.0 else psi_to_mpa(fyk)
        density = kgm3_to_kN_m3(pcf_to_kgm3(density))
    else:
        density = kgm3_to_kN_m3(density)

    return Material(
        material_id=str(row[id_name]),
        f_ck_MPa=fck,
        f_yk_MPa=fyk,
        density_kN_m3=density,
        gamma_c=_to_float(row[gamma_c_name], 1.5) if gamma_c_name else 1.5,
        gamma_s=_to_float(row[gamma_s_name], 1.15) if gamma_s_name else 1.15,
        original_units=units,
        raw={str(k): row[k] for k in row.index},
    )
