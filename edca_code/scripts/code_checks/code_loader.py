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
    elif suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        # normalize to strings to match existing behavior
        df = df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str)
        df = df.fillna("")
    else:
        raise ValueError(f"Unsupported materials file type '{suf}'. Use .csv or .parquet")

    return df

def _resolve_col(df: pd.DataFrame, name: str) -> str:
    """Resolve a column by exact name or case-insensitive match."""
    if name in df.columns:
        return name
    low = {c.lower(): c for c in df.columns}
    if name.lower() in low:
        return low[name.lower()]
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
    """Load material properties from a CSV file (simple, strict schema).
    Assumptions (explicit):
      - metric rows: concrete_f_ck is given in kN/m^2 (so 4000 -> 4 MPa).
      - imperial rows: concrete_f_ck is given in psi (so 3000 -> psi -> MPa).
      - steel_fy / steel_fu are in MPa (metric) or psi/ksi (imperial).
      - density is kg/m^3 (metric) or pcf (imperial).
    """

    df = _read_material_table(path)

    # Resolve columns (supports case variations)
    id_col = _resolve_col(df, id_col)
    fck_col = _resolve_col(df, fck_col) if fck_col in df.columns or fck_col.lower() in [c.lower() for c in df.columns] else fck_col
    fyk_col = _resolve_col(df, fyk_col) if fyk_col in df.columns or fyk_col.lower() in [c.lower() for c in df.columns] else fyk_col
    fu_col  = _resolve_col(df, fu_col)  if fu_col  in df.columns or fu_col.lower()  in [c.lower() for c in df.columns] else fu_col
    density_col  = _resolve_col(df, density_col)  if density_col  in df.columns or density_col.lower()  in [c.lower() for c in df.columns] else density_col
    gamma_c_col  = _resolve_col(df, gamma_c_col)  if gamma_c_col  in df.columns or gamma_c_col.lower()  in [c.lower() for c in df.columns] else gamma_c_col
    gamma_s_col  = _resolve_col(df, gamma_s_col)  if gamma_s_col  in df.columns or gamma_s_col.lower()  in [c.lower() for c in df.columns] else gamma_s_col

    # case-insensitive match for material_id
    material_id_str = str(material_id).strip().lower()
    if id_col not in df.columns:
        raise KeyError(f"Expected id column '{id_col}' in materials CSV.")
    mask = df[id_col].astype(str).str.strip().str.lower() == material_id_str
    if not mask.any():
        raise KeyError(f"Material '{material_id}' not found in CSV at '{path}' (id_col='{id_col}').")
    row = df.loc[mask].iloc[0]

    # detect unit flag (fallback default metric)
    unit_flag = _detect_unit_flag(row) or "metric"

    # helper to read numeric or None
    def _as_float_or_none(key: str):
        if key not in row.index:
            return None
        val = row.get(key, "")
        if val == "" or pd.isna(val):
            return None
        try:
            return float(str(val).strip())
        except Exception as e:
            raise ValueError(f"Could not parse numeric value for column '{key}': '{val}'") from e

    # read raw CSV numeric values (may be None)
    raw_fck = _as_float_or_none(fck_col)
    raw_fyk = _as_float_or_none(fyk_col)
    raw_fu  = _as_float_or_none(fu_col)
    raw_density = _as_float_or_none(density_col)
    raw_gamma_c = _as_float_or_none(gamma_c_col)
    raw_gamma_s = _as_float_or_none(gamma_s_col)

    # Convert to canonical units:
    # -> f_ck_MPa (MPa), f_yk_MPa (MPa), density_kN_m3 (kN/m^3)
    if unit_flag == "metric":
        # Interpret concrete_f_ck as kN/m^2 (so divide by 1000 to get MPa)
        if raw_fck is None:
            raise ValueError(f"Missing '{fck_col}' for metric material '{material_id}'.")
        # Heuristic: if value looks very large (>1000) we assume it's given in kN/m2 and convert to MPa.
        # If value looks small (<=1000) assume user supplied MPa already (common).
        if float(raw_fck) > 1000.0:
            # value in kN/m2 -> MPa
            f_ck_MPa = float(raw_fck) / 1000.0
        else:
            # value likely already in MPa
            f_ck_MPa = float(raw_fck)
        # steel fy if present: treat as MPa (if missing, set 0.0)
        f_yk_MPa = float(raw_fyk) if raw_fyk is not None and raw_fyk != "" else 0.0
        # density: kg/m3 -> kN/m3
        if raw_density is None or raw_density == "":
            logging.warning("No density found for '%s' — assuming 2500 kg/m3 (concrete default).", material_id)
            density_kN_m3 = kgm3_to_kN_m3(2500.0)
        else:
            # we expect metric density in kg/m3
            density_kN_m3 = kgm3_to_kN_m3(float(raw_density))
    else:
        # imperial row: interpret concrete_f_ck as psi (convert to MPa)
        if raw_fck is None:
            raise ValueError(f"Missing '{fck_col}' for imperial material '{material_id}'.")
        # convert psi -> MPa
        f_ck_MPa = psi_to_mpa(float(raw_fck))
        # steel fy: could be psi or ksi; assume psi if > 1000 else ksi (simple deterministic rule)
        if raw_fyk is None or raw_fyk == "":
            f_yk_MPa = 0.0
        else:
            v = float(raw_fyk)
            if v <= 50:
                # treat as ksi
                f_yk_MPa = ksi_to_mpa(v)
            else:
                # treat as psi
                f_yk_MPa = psi_to_mpa(v)

        # density: pcf -> kN/m3 (use pcf_to_kgm3 then kgm3_to_kN_m3)
        if raw_density is None or raw_density == "":
            # assume typical ~150 pcf
            density_kN_m3 = pcf_to_kN_m3(150.0)
        else:
            # convert pcf -> kg/m3, then to kN/m3
            kgm3 = pcf_to_kgm3(float(raw_density))
            density_kN_m3 = kgm3_to_kN_m3(kgm3)

    # gamma factors fall back to defaults if missing
    gamma_c_val = float(raw_gamma_c) if raw_gamma_c is not None and raw_gamma_c != "" else 1.5
    gamma_s_val = float(raw_gamma_s) if raw_gamma_s is not None and raw_gamma_s != "" else 1.15

    # Build Material dataclass
    mat = Material(
        material_id = str(row.get(id_col, material_id)),
        f_ck_MPa = float(f_ck_MPa),
        f_yk_MPa = float(f_yk_MPa),
        density_kN_m3 = float(density_kN_m3),
        gamma_c = float(gamma_c_val),
        gamma_s = float(gamma_s_val),
        original_units = unit_flag,
        raw = row.to_dict()
    )

    return mat
