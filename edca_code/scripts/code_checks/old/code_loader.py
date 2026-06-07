from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import math
import logging
# your existing conversions
from edca_code.constants.units import pcf_to_kgm3, kgm3_to_kN_m3, pcf_to_kN_m3, psi_to_mpa, ksi_to_mpa
from pathlib import Path

# Constants
G = 9.80665  # m/s^2

@dataclass
class Material:
    material_id: str
    f_ck_MPa: float          # MPa (N/mm^2)
    f_yk_MPa: float          # MPa
    density_kN_m3: float     # kN/m^3
    gamma_c: float = 1.5
    gamma_s: float = 1.15
    original_units: str = "metric"   # 'metric' or 'imperial'
    raw: Dict[str, Any] = None       # original CSV row dict for traceability

# Accept many synonyms for the units column
_UNIT_KEYS = ("unit", "units", "unitsystem", "unit_system", "unitstype")

def _detect_unit_flag(row: pd.Series) -> Optional[str]:
    for k in _UNIT_KEYS:
        if k in row.index:
            val = str(row[k]).strip().lower()
            if val in ("metric", "si", "m", "mpa", "kg/m3", "kgm3"):
                return "metric"
            if val in ("imperial", "us", "us_customary", "psi", "pcf"):
                return "imperial"
            # Accept common synonyms
            if val in ("si_units","si_unit"):
                return "metric"
            if val in ("us_units","us_unit"):
                return "imperial"
    return None

def _read_material_table(path: str) -> pd.DataFrame:
    """Read materials table from CSV or Parquet; return all-string DataFrame (deterministic)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Materials file not found: {path}")

    suf = p.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
    elif suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str)
        df = df.fillna("")
    else:
        raise ValueError(f"Unsupported materials file type '{suf}'. Use .csv or .parquet")

    return df

def _resolve_col(df: pd.DataFrame, name: str) -> str:
    """Resolve a column by exact name or case-insensitive/trimmed match."""
    target = str(name).strip().lower()

    for c in df.columns:
        if str(c).strip() == str(name).strip():
            return c

    for c in df.columns:
        if str(c).strip().lower() == target:
            return c

    raise KeyError(f"Expected column '{name}' in materials file. Available: {list(df.columns)}")

def load_material_from_csv(path: str,
                           material_id: str,
                           *,
                           fck_col: str = "concrete_f_ck",
                           fyk_col: str = "steel_fy",
                           fu_col: str = "steel_fu",
                           density_col: str = "density",
                           gamma_c_col: str = "gamma_c",
                           gamma_s_col: str = "gamma_s",
                           id_col: str = "material_id") -> Material:
    """Load material properties from a CSV or Parquet file.

    Canonical outputs:
      - f_ck_MPa       : concrete strength in MPa
      - f_yk_MPa       : steel yield strength in MPa
      - density_kN_m3  : density in kN/m^3

    Conventions:
      - metric concrete_f_ck: MPa if <= 1000, else kN/m^2 -> MPa
      - imperial concrete_f_ck: psi -> MPa
      - metric steel_fy: MPa
      - imperial steel_fy: ksi if <= 50, else psi
      - metric density: kg/m^3
      - imperial density: pcf
    """

    df = _read_material_table(path)

    def _has_col(name: str) -> bool:
        target = str(name).strip().lower()
        return any(str(c).strip().lower() == target for c in df.columns)

    # Required columns
    id_col = _resolve_col(df, id_col)
    density_col = _resolve_col(df, density_col)

    # Optional / family-dependent columns
    fck_col = _resolve_col(df, fck_col) if _has_col(fck_col) else None
    fyk_col = _resolve_col(df, fyk_col) if _has_col(fyk_col) else None
    fu_col = _resolve_col(df, fu_col) if _has_col(fu_col) else None
    gamma_c_col = _resolve_col(df, gamma_c_col) if _has_col(gamma_c_col) else None
    gamma_s_col = _resolve_col(df, gamma_s_col) if _has_col(gamma_s_col) else None

    # Find row by material_id
    material_id_str = str(material_id).strip().lower()
    mask = df[id_col].astype(str).str.strip().str.lower() == material_id_str
    if not mask.any():
        raise KeyError(f"Material '{material_id}' not found in CSV at '{path}' (id_col='{id_col}').")
    row = df.loc[mask].iloc[0]

    unit_flag = _detect_unit_flag(row) or "metric"
    family_val = str(row.get("family", "")).strip().lower()

    def _as_float_or_none(key: str | None):
        if key is None:
            return None
        if key not in row.index:
            return None
        val = row.get(key, "")
        if val == "" or pd.isna(val):
            return None
        try:
            return float(str(val).strip())
        except Exception as e:
            raise ValueError(f"Could not parse numeric value for column '{key}': '{val}'") from e

    raw_fck = _as_float_or_none(fck_col)
    raw_fyk = _as_float_or_none(fyk_col)
    raw_fu = _as_float_or_none(fu_col)
    raw_density = _as_float_or_none(density_col)
    raw_gamma_c = _as_float_or_none(gamma_c_col)
    raw_gamma_s = _as_float_or_none(gamma_s_col)

    kind_text = " ".join([
        str(row.get("material_id", "")),
        str(row.get("family", "")),
        str(row.get("standard_grade", "")),
    ]).strip().lower()

    is_reinforcement_like = any(tok in kind_text for tok in (
        "steel",
        "rebar",
        "reinforcement",
        "pt",
        "prestress",
        "prestressing",
        "fiber",
        "fibre",
        "strand",
        "mesh",
        "wire",
        "tendon",
    ))

    # Convert to canonical units
    if unit_flag == "metric":
        # Concrete strength
        if raw_fck is None:
            if is_reinforcement_like:
                f_ck_MPa = 0.0
            else:
                raise ValueError(f"Missing '{fck_col}' for metric material '{material_id}'.")
        else:
            f_ck_MPa = float(raw_fck) / 1000.0 if float(raw_fck) > 1000.0 else float(raw_fck)

        # Steel strength
        if raw_fyk is None:
            if is_reinforcement_like:
                # optional PT fallback to fu if fy is absent
                if ("pt" in kind_text or "prestress" in kind_text) and raw_fu is not None:
                    f_yk_MPa = float(raw_fu)
                else:
                    raise ValueError(f"Missing '{fyk_col}' for metric material '{material_id}'.")
            else:
                f_yk_MPa = 0.0
        else:
            f_yk_MPa = float(raw_fyk)

        # Density
        if raw_density is None:
            if is_reinforcement_like:
                density_kN_m3 = kgm3_to_kN_m3(7850.0)
            else:
                logging.warning("No density found for '%s' — assuming 2500 kg/m3.", material_id)
                density_kN_m3 = kgm3_to_kN_m3(2500.0)
        else:
            density_kN_m3 = kgm3_to_kN_m3(float(raw_density))

    else:
        # Concrete strength
        if raw_fck is None:
            if is_reinforcement_like:
                f_ck_MPa = 0.0
            else:
                raise ValueError(f"Missing '{fck_col}' for imperial material '{material_id}'.")
        else:
            f_ck_MPa = psi_to_mpa(float(raw_fck))

        # Steel strength
        if raw_fyk is None:
            if is_reinforcement_like:
                if ("pt" in kind_text or "prestress" in kind_text) and raw_fu is not None:
                    v = float(raw_fu)
                    f_yk_MPa = ksi_to_mpa(v) if v <= 50 else psi_to_mpa(v)
                else:
                    raise ValueError(f"Missing '{fyk_col}' for imperial material '{material_id}'.")
            else:
                f_yk_MPa = 0.0
        else:
            v = float(raw_fyk)
            f_yk_MPa = ksi_to_mpa(v) if v <= 50 else psi_to_mpa(v)

        # Density
        if raw_density is None:
            if is_reinforcement_like:
                density_kN_m3 = pcf_to_kN_m3(490.0)
            else:
                density_kN_m3 = pcf_to_kN_m3(150.0)
        else:
            kgm3 = pcf_to_kgm3(float(raw_density))
            density_kN_m3 = kgm3_to_kN_m3(kgm3)

    gamma_c_val = float(raw_gamma_c) if raw_gamma_c is not None else 1.5
    gamma_s_val = float(raw_gamma_s) if raw_gamma_s is not None else 1.15

    return Material(
        material_id=str(row.get(id_col, material_id)),
        f_ck_MPa=float(f_ck_MPa),
        f_yk_MPa=float(f_yk_MPa),
        density_kN_m3=float(density_kN_m3),
        gamma_c=float(gamma_c_val),
        gamma_s=float(gamma_s_val),
        original_units=unit_flag,
        raw=row.to_dict(),
    )