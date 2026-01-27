# edca_code/scripts/core/loads.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import itertools
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import numbers
from edca_code.constants.units import load_to_knm2

def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load and parse the control file YAML.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Control file not found: {p}")

    with p.open("r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML control file: {exc}") from exc

    return data or {}

def load_definitions(load_values_path: str, load_combinations_path: str):
    load_values = load_yaml(load_values_path)
    load_combinations = load_yaml(load_combinations_path)
    return load_values, load_combinations

def _is_permanent_action(action: str) -> bool:
    return action in ("D", "G", "Gk", "G_unf", "G_fav")

def _sum_companion_loads(loads: Dict[str, float], companions: List[str]) -> float:
    return sum(loads.get(c, 0.0) for c in companions)


def load_occupancies_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"use": str})
    # normalize column existence
    for c in ("sdl_partition", "sdl", "ll", "unit"):
        if c not in df.columns:
            df[c] = 0.0 if c in ("sdl_partition", "sdl", "sdl") else ""
    return df

def program_to_dataframe(program_mapping: Dict[int, str], default_unit: str = "metric") -> pd.DataFrame:
    """
    Convert control.program (floor -> occupancy_key) to a DataFrame with start_floor,end_floor,use,unit.
    program_mapping assumed e.g. {1: "Office_ASCE", 2: "Office_ASCE", 3: "Res_ASCE"}
    """
    rows = []
    for floor, use in program_mapping.items():
        rows.append({"start_floor": int(floor), "end_floor": int(floor), "use": use, "unit": default_unit})
    return pd.DataFrame(rows)


# -------------------------
# Core aggregation
# -------------------------
def compute_floor_loads(program_mapping: Dict[int, str],
                        occ_df: pd.DataFrame,
                        default_unit: str = "metric") -> Dict[int, Dict[str, Any]]:
    """
    For each floor in program_mapping return a small dict:
      { floor: {"use": <occupancy_key>, "unit": <unit>, "sdl": <sdl_total>, "ll": <ll>} }

    Assumptions:
      - occ_df has a column 'use' matching occupancy keys from the control file program.
      - occ_df has numeric columns 'sdl', 'sdl_partition', 'll' (missing values treated as 0).
      - If occ_df contains a 'unit' column that will be used, otherwise default_unit is used.
    """
    # normalize column name for occupancy key if needed
    if "occupancy_key" in occ_df.columns and "use" not in occ_df.columns:
        occ_df = occ_df.rename(columns={"occupancy_key": "use"})

    # create quick lookup by 'use'
    occ_lookup = occ_df.set_index("use").to_dict(orient="index")

    out: Dict[int, Dict[str, Any]] = {}
    for floor, occ_key in program_mapping.items():
        if occ_key not in occ_lookup:
            raise KeyError(f"Occupancy '{occ_key}' referenced in program (floor {floor}) not found in occupancies CSV")

        row = occ_lookup[occ_key]
        sdl_part = float(row.get("sdl_partition") or 0.0)
        sdl = float(row.get("sdl") or 0.0)
        ll = float(row.get("ll") or 0.0)
        unit = row.get("unit") or default_unit

        out[int(floor)] = {
            "use": occ_key,
            "unit": unit,
            "sdl": sdl + sdl_part,
            "ll": ll,
            # keep raw row if you want quick access later
            "raw": row,
        }
    return out


# -------------------------
# Simple combination calculators
# -------------------------
def simple_asce_lrfd_totals(load_row: Dict[str, Any], load_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Minimal ASCE LRFD: use the 'factors' mapping in ASCE7.LRFD.factors when present,
    or fall back to evaluating each combo expression in load_combinations.yaml if present.

    Returns list of {"name": ..., "total": value_in_same_unit_as_inputs, "factors": {...}}
    """
    results = []
    asce_values = load_values.get("ASCE7", {})
    # preferred: load_values/ASCE7/LRFD/factors  (simple mapping)
    lrfd_factors = asce_values.get("LRFD", {}).get("factors") or asce_values.get("LRFD", {}).get("factors", {})
    if lrfd_factors:
        # multiply named actions
        total = 0.0
        factors = {}
        for action, f in lrfd_factors.items():
            # map 'D' -> dead load (avg_sdl), 'L' -> live load (avg_ll)
            val = 0.0
            if action in ("D", "DL", "G"):
                val = load_row["avg_sdl"]
            elif action in ("L", "LL"):
                val = load_row["avg_ll"]
            else:
                # unknown action (maybe S, W) -> try raw lookup from raw dict
                val = load_row.get("raw", {}).get(action.lower(), 0.0)
            factors[action] = float(f)
            total += float(f) * float(val)
        results.append({"name": "ASCE7_LRFD_simple", "total": total, "factors": factors})
        return results

    # fallback: try to use load_combinations.yaml style (user may supply combo expressions)
    # in that case the loader that calls this function should supply load_combinations YAML separately.
    return results


def simple_en1990_ULS_totals(load_row: Dict[str, Any], load_values: Dict[str, Any], occupancy: str) -> List[Dict[str, Any]]:
    """
    Basic EN1990 Eq. 6.10 style: Gamma for ULS (G and Q), and psi0 accompanying factor for variable actions.
    We'll produce one total per lead variable (if variable actions exist).
    Expects load_values to contain keys like EN1990.gamma.ULS and EN1990.psi.Q.<occupancy>.psi0
    """
    results = []
    en = load_values.get("EN1990", {})
    gamma = en.get("gamma", {}).get("ULS", {})  # e.g. {"G":1.35, "Q":1.5}
    psi_for_occ = en.get("psi", {}).get("Q", {}).get(occupancy, {}) or {}
    psi0 = float(psi_for_occ.get("psi0", 0.0) or 0.0)

    # create a simple loads mapping: D -> avg_sdl, L -> avg_ll
    loads = {"D": float(load_row["avg_sdl"]), "L": float(load_row["avg_ll"])}
    # variable actions are those except dead
    var_actions = [k for k in loads.keys() if k != "D"]
    if not var_actions:
        # trivial: only dead load
        total = float(gamma.get("G", 1.0)) * loads.get("D", 0.0)
        results.append({"name": "EN1990_ULS_only_dead", "total": total, "factors": {"D": float(gamma.get("G", 1.0))}})
        return results

    # For each lead variable, apply gamma.Q to lead and gamma.Q*psi0 to accompanying variables
    for lead in var_actions:
        total = 0.0
        factors = {}
        # dead load
        g_dead = float(gamma.get("G", 1.0))
        factors["D"] = g_dead
        total += g_dead * loads.get("D", 0.0)
        # lead variable
        q_gamma = float(gamma.get("Q", 1.0))
        factors[lead] = q_gamma
        total += q_gamma * loads.get(lead, 0.0)
        # accompaniers
        for acc in var_actions:
            if acc == lead:
                continue
            factors[acc] = q_gamma * psi0
            total += (q_gamma * psi0) * loads.get(acc, 0.0)
        results.append({"name": f"EN1990_ULS_lead_{lead}", "total": total, "factors": factors})

    return results

# -------------------------
# Convenience wrapper
# -------------------------
def compute_and_combo(program_mapping: Dict[int, str],
                      occupancies_csv: str,
                      load_values_yaml: Optional[str] = None,
                      control_unit: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    High-level convenience:
      - program_mapping: {floor: occupancy_key}
      - occupancies_csv: path to occupancy definitions
      - load_values_yaml: (optional) path to load_values.yaml for factors
      - control_unit: optional override for unit convention (e.g., 'imperial' or 'metric')
    Returns: dict keyed by floor -> {use, unit, avg_sdl, avg_ll, en1990: [...], asce: [...]}
    """
    occ_df = load_occupancies_csv(occupancies_csv)
    prog_df = program_to_dataframe(program_mapping, default_unit=(control_unit or "metric"))
    floor_loads = compute_floor_loads(program_mapping, occ_df, default_unit=(control_unit or "metric"))

    load_values = load_yaml(load_values_yaml) if load_values_yaml else {}

    out = {}
    for floor, r in floor_loads.items():
        floor = int(floor)
        unit = r["unit"] or control_unit or "metric"
        record = {
            "use": r["use"],
            "unit": unit,
            "avg_sdl": r["avg_sdl"],
            "avg_ll": r["avg_ll"],
        }
        if load_values:
            # minimal combos:
            # EN1990 only if EN1990 present in load_values
            if "EN1990" in load_values:
                record["EN1990_ULS"] = simple_en1990_ULS_totals(r, load_values, r["use"])
            if "ASCE7" in load_values:
                record["ASCE7_LRFD"] = simple_asce_lrfd_totals(r, load_values)
        out[floor] = record
    return out
