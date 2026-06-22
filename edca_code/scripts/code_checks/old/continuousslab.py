from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from edca_code.scripts.code_checks.code_loader import load_material_from_csv, Material   # <- your new loader
from edca_code.constants.rebar_properties import RebarSpec, METRIC_REBAR_BY_SPEC, IMPERIAL_REBAR_BY_SPEC

logger = logging.getLogger(__name__)

# -------------------------
# Debug helpers (drop-in)
# -------------------------
import os
from typing import Iterable

def _dbg_enabled(explicit: bool | None = None) -> bool:
    """
    Debug is enabled if:
      - explicit=True passed by caller, OR
      - EDCA_DEBUG=1 environment variable, OR
      - logger level is DEBUG.
    """
    if explicit is True:
        return True
    if explicit is False:
        return False
    if str(os.getenv("EDCA_DEBUG", "")).strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    return bool(getattr(logger, "isEnabledFor", lambda *_: False)(logging.DEBUG))

def _dbg_kv(name: str, d: dict, *, explicit: bool | None = None, level: int = logging.DEBUG) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        items = ", ".join([f"{k}={d[k]!r}" for k in sorted(d.keys())])
    except Exception:
        items = str(d)
    logger.log(level, "[debug] %s: %s", name, items)

def _dbg_df(
    name: str,
    df,
    *,
    explicit: bool | None = None,
    max_rows: int = 15,
    cols: list[str] | None = None,
    level: int = logging.DEBUG,
) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        import pandas as pd
        if df is None:
            logger.log(level, "[debug] %s: df=None", name)
            return
        if not isinstance(df, pd.DataFrame):
            logger.log(level, "[debug] %s: not a DataFrame (%s)", name, type(df).__name__)
            return
        logger.log(level, "[debug] %s: shape=%s", name, df.shape)
        logger.log(level, "[debug] %s: cols=%s", name, list(df.columns))
        sub = df
        if cols:
            keep = [c for c in cols if c in sub.columns]
            sub = sub[keep] if keep else sub
        logger.log(level, "[debug] %s head(%d):\n%s", name, max_rows, sub.head(max_rows).to_string(index=False))
    except Exception:
        logger.exception("[debug] Failed dumping df %s", name)

class DesignError(Exception):
    """High-level exception raised when a design step is impossible / invalid."""
    pass

class DesignStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    NEEDS_CUSTOM_REINF = "NEEDS_CUSTOM_REINF"

LOCAL_DEFAULTS = {
    "d_bar_m": 0.012,
    "d_links_m": 0.008,
    "gamma_c": 1.5,
    "gamma_s": 1.15,
    "fallback_span_m": 5.8,
    "fallback_slab_width_m": 1.0
}

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf8") as fh:
        return yaml.safe_load(fh)


def choose_rebar_for_As_required(as_req_mm2_per_m: float,
                                  metric_only: bool = True) -> Optional[RebarSpec]:
    """
    Find the smallest standard rebar spec (area per m) in the rebar databases that
    meets or exceeds as_req_mm2_per_m.

    Returns a RebarSpec instance or None if nothing matches.
    """
    candidates: list[tuple[float, RebarSpec]] = []

    def spec_area_per_m(spec: RebarSpec) -> float:
        # spec.area_mm2 available via RebarSpec API (assumed)
        # compute bars per m from spacing: if spacing unit is 'mm' treat spacing as mm
        try:
            # prefer property/method area_per_m if available, else compute
            area_single = spec.area_mm2
        except Exception:
            # fallback compute from diameter
            d_mm = spec.diameter_mm
            area_single = math.pi * (d_mm / 2.0) ** 2
        # spacing unit handling: many specs use spacing in mm or in

        spacing_val = getattr(spec, "spacing", None)
        spacing_unit = getattr(spec, "spacing_unit", None)
        if spacing_val is None:
            # fallback huge spacing (effectively zero bars per m)
            spacing_m = 1e6
        else:
            try:
                if spacing_unit == "mm":
                    spacing_m = float(spacing_val) / 1000.0
                elif spacing_unit == "in":
                    spacing_m = float(spacing_val) * 0.0254
                else:
                    spacing_m = float(spacing_val) / 1000.0
            except Exception:
                spacing_m = 1e6
        bars_per_m = 1.0 / spacing_m if spacing_m > 0 else 0.0
        return area_single * bars_per_m

    def collect_from_dict(d: dict):
        for spec in d.values():
            try:
                a = spec_area_per_m(spec)
                candidates.append((a, spec))
            except Exception:
                continue

    collect_from_dict(METRIC_REBAR_BY_SPEC)
    if not metric_only:
        collect_from_dict(IMPERIAL_REBAR_BY_SPEC)

    # sort by area ascending
    candidates.sort(key=lambda x: x[0])

    for area, spec in candidates:
        if area >= as_req_mm2_per_m - 1e-9:
            return spec
    # nothing found
    return None
    
def run_code_check_for_typology(systems_catalog: pd.DataFrame,
                                typology_selector: Optional[str | int] = None,
                                material_csv_path: Optional[str] = None,
                                load_combos_yaml: Optional[str] = None,
                                load_values_yaml: Optional[str] = None,
                                program_df: Optional[pd.DataFrame] = None) -> dict:
    """
    High-level wrapper to select a row from systems_catalog and run the slab checks.

    Important: This function DOES NOT hardcode material or dimension values. It will:
      - pick the typology row (as before)
      - allow `program_df` to override row keys
      - pass paths for material CSV and load-combo YAMLs into the math function
    """
    # choose the row (same logic as before)
    if typology_selector is None:
        mask = None
        if 'category' in systems_catalog.columns:
            mask = systems_catalog['category'].str.lower().eq('concrete')
        elif 'type' in systems_catalog.columns:
            mask = systems_catalog['type'].str.lower().str.contains('rc|concrete', na=False)
        if mask is not None and mask.any():
            sel_row = systems_catalog[mask].iloc[0]
        else:
            sel_row = systems_catalog.iloc[0]
    else:
        if isinstance(typology_selector, int):
            sel_row = systems_catalog.iloc[typology_selector]
        else:
            sel_row = systems_catalog[systems_catalog.get('system_id','') == typology_selector].iloc[0]

    row_dict = sel_row.to_dict()

    # allow program_df overrides
    if program_df is not None and 'system_id' in program_df.columns:
        pmatch = program_df[program_df['system_id'] == sel_row.get('system_id')]
        if len(pmatch) > 0:
            row_dict.update(pmatch.iloc[0].dropna().to_dict())

    results = check_slab_row_preserve_math(row_dict,
                                          material_csv_path=material_csv_path,
                                          load_combos_yaml=load_combos_yaml,
                                          load_values_yaml=load_values_yaml)

    # deflection_ok: use row span or fallback
    fallback_span = row_dict.get('span', row_dict.get('max_span', LOCAL_DEFAULTS['fallback_span_m']))
    deflection_ok = results['deflection'].get('max_span_int_m', results['deflection'].get('max_span_int', 0.0)) >= fallback_span

    shear = results['shear']
    shear_ok = (shear.get('V_rdc_kN', 0.0) >= shear.get('V_ed_ext_kN', 0.0)) if shear.get('V_ed_ext_kN') is not None else True

    flex = results['flex']
    flex_ok = (flex.get('allowable_ext_reinforcement') is not None and flex.get('allowable_int_reinforcement') is not None)

    pass_overall = bool(deflection_ok and shear_ok and flex_ok)

    reinforcement = {
        "As_req_ext_mm2_per_m": flex["required_ext_reinforcement"],
        "As_allowable_ext_mm2_per_m": flex["allowable_ext_reinforcement"],
        "rho_ext": flex["rho_ext"],
        "As_req_int_mm2_per_m": flex["required_int_reinforcement"],
        "As_allowable_int_mm2_per_m": flex["allowable_int_reinforcement"],
        "rho_int": flex["rho_int"]
    }

    diagnostics = {
        "G_kN_m2": results["G_kN_m2"],
        "Q_kN_m2": results["Q_kN_m2"],
        "ULS_kN_m2": results["ULS_kN_m2"],
        "slab_effective_depth_m": results["slab_effective_depth_m"],
        "deflection": results["deflection"],
        "shear": results["shear"],
        "flex_details": results["flex"]
    }

    return {
        "system_id": sel_row.get("system_id"),
        "pass": pass_overall,
        "deflection_ok": bool(deflection_ok),
        "shear_ok": bool(shear_ok),
        "flex_ok": bool(flex_ok),
        "reinforcement": reinforcement,
        "diagnostics": diagnostics
    }

def permanent_loading(slab_depth_m: float,
                      slab_width_m: float = 1.0,
                      screed_depth_m: float = 0.0,
                      concrete_density_kN_m3: Optional[float] = None,
                      screed_density_kN_m3: Optional[float] = None,
                      slab_code_loading: float = 0.0,
                      screed_code_loading: float = 0.0,
                      finish_code_loading: float = 0.0,
                      service_code_loading: float = 0.0) -> float:
    """
    Compute permanent load G (kN/m^2) for a 1-m width strip using densities provided.
    densities must be in kN/m^3 (canonical).
    """
    if concrete_density_kN_m3 is None:
        raise DesignError("concrete_density_kN_m3 must be provided (kN/m^3)")
    if screed_density_kN_m3 is None:
        screed_density_kN_m3 = concrete_density_kN_m3  # fallback to same density if screed not provided

    slab_self_weight = slab_depth_m * slab_width_m * concrete_density_kN_m3 + slab_code_loading
    screed_self_weight = screed_depth_m * slab_width_m * screed_density_kN_m3 + screed_code_loading
    return slab_self_weight + screed_self_weight + finish_code_loading + service_code_loading


def variable_loading(live_load_kN_m2: float = 0.0, partition_load_kN_m2: float = 0.0) -> float:
    return live_load_kN_m2 + partition_load_kN_m2


def ultimate_loading(permanent_load_kN_m2: float,
                     variable_load_kN_m2: float,
                     load_combos_yaml: Optional[str] = None,
                     load_values_yaml: Optional[str] = None,
                     combo_name: Optional[str] = None) -> Dict[str, float]:
    combos = {}

    # 1) Try simple 'combos' format (legacy)
    if load_combos_yaml and Path(load_combos_yaml).exists():
        cfg = load_yaml(load_combos_yaml)
        if isinstance(cfg, dict) and 'combos' in cfg:
            for name, expr in (cfg.get('combos') or {}).items():
                gcoef = expr.get('G', expr.get('g', None))
                qcoef = expr.get('Q', expr.get('q', None))
                if gcoef is None or qcoef is None:
                    continue
                combos[name] = gcoef * permanent_load_kN_m2 + qcoef * variable_load_kN_m2

        # 2) Support EN1990-style YAML (your file)
        elif isinstance(cfg, dict) and 'EN1990' in cfg:
            vals = None
            if load_values_yaml and Path(load_values_yaml).exists():
                vals = load_yaml(load_values_yaml)

            psi_root = (vals.get('psi', {}) if isinstance(vals, dict) else {})
            # iterate ULS entries if present
            uls = cfg['EN1990'].get('ULS', {})
            for entry_name, entry in uls.items():
                expr_list = entry.get('expression') or entry.get('alternatives') or []
                total = None
                if isinstance(expr_list, list) and len(expr_list) > 0 and isinstance(expr_list[0], list):
                    alt_vals = []
                    for alt in expr_list:
                        s = 0.0
                        for term in alt:
                            act = term.get('action') or term.get('action'.upper(), None)
                            factor = term.get('factor')
                            if isinstance(factor, str) and vals:
                                try:
                                    parts = factor.split('.')
                                    node = vals
                                    for p in parts:
                                        node = node[p]
                                    factor_val = float(node)
                                except Exception:
                                    factor_val = None
                            else:
                                factor_val = factor
                            if act is None or factor_val is None:
                                continue
                            if act in ('G', 'D'):
                                s += float(factor_val) * permanent_load_kN_m2
                            else:
                                # treat all other actions as variable-type for this simple eval
                                s += float(factor_val) * variable_load_kN_m2
                        alt_vals.append(s)
                    total = max(alt_vals) if alt_vals else None
                else:
                    # single expression: list of action/factor dicts
                    s = 0.0
                    for term in (expr_list or []):
                        act = term.get('action')
                        factor = term.get('factor')
                        factor_val = None
                        if isinstance(factor, str) and vals:
                            # attempt lookup in load_values.yaml then as literal evaluation fallback
                            try:
                                parts = factor.split('.')
                                node = vals
                                for p in parts:
                                    node = node[p]
                                factor_val = float(node)
                            except Exception:
                                try:
                                    factor_val = float(factor)
                                except Exception:
                                    factor_val = None
                        else:
                            factor_val = factor
                        if factor_val is None or act is None:
                            continue
                        if act in ('G', 'D'):
                            s += float(factor_val) * permanent_load_kN_m2
                        else:
                            s += float(factor_val) * variable_load_kN_m2
                    total = s

                if total is not None:
                    combos[entry_name] = total

    # fallback defaults if nothing parsed
    if not combos:
        combos = {
            "EC_6.10": 1.35 * permanent_load_kN_m2 + 1.5 * variable_load_kN_m2,
            "EC_6.10a": 1.35 * permanent_load_kN_m2 + 1.5 * variable_load_kN_m2,
            "EC_6.10b": 1.25 * permanent_load_kN_m2 + 1.5 * variable_load_kN_m2
        }

    if combo_name:
        if combo_name in combos:
            return {combo_name: combos[combo_name]}
        else:
            raise KeyError(f"Requested combo '{combo_name}' not found in computed combos.")

    return combos

def slab_dimensions(slab_depth_m: float,
                    screed_depth_m: float,
                    d_rebar_m: float,
                    cover_m: float,
                    deviation_allowance_m: float,
                    slab_length_m: float,
                    wall_thickness_m: float) -> Dict[str, float]:
    """
    Compute nominal cover, effective depth and an effective span.
    All inputs in meters. Returns a dict with keys:
    nominal_cover_m, a1_m, a2_m, effective_span_m, effective_depth_m
    """
    nominal_cover = max(d_rebar_m, cover_m) + deviation_allowance_m
    a1 = min(slab_depth_m / 2.0, wall_thickness_m / 2.0)
    a2 = a1
    effective_span = a1 + a2 + slab_length_m
    effective_depth = slab_depth_m + screed_depth_m - nominal_cover - d_rebar_m / 2.0
    if effective_depth <= 0:
        raise DesignError("Effective depth non-positive: check slab_depth / covers / rebar diameter.")
    return {
        'nominal_cover_m': nominal_cover,
        'a1_m': a1,
        'a2_m': a2,
        'effective_span_m': effective_span,
        'effective_depth_m': effective_depth
    }


def flexural_design(ultimate_load_kN_m2: float,
                    span_m: float,
                    effective_depth_m: float,
                    material: Material,
                    slab_width_m: float = 1.0,
                    bar_db: Optional[dict] = None,
                    span_factors: Optional[dict] = None) -> dict:
    """
    Perform flexural design and attempt to pick a standard bar pattern from rebar DB.
    """
    if span_factors is None:
        span_factors = {"end": 0.086, "mid": 0.063}

    effective_span = span_m
    end_moment_kNm_per_m = span_factors["end"] * ultimate_load_kN_m2 * effective_span ** 2
    internal_moment_kNm_per_m = span_factors["mid"] * ultimate_load_kN_m2 * effective_span ** 2

    external_shear_kN_per_m = 0.4 * ultimate_load_kN_m2 * effective_span
    support_shear_kN_per_m = 0.6 * ultimate_load_kN_m2 * effective_span

    f_ck_MPa = material.f_ck_MPa
    f_yk_MPa = material.f_yk_MPa
    gamma_s = material.gamma_s or LOCAL_DEFAULTS['gamma_s']
    f_yd_MPa = f_yk_MPa / gamma_s

    M_ed_Nmm = end_moment_kNm_per_m * 1e6
    d_mm = effective_depth_m * 1000.0

    f_ck = f_ck_MPa  # MPa == N/mm2
    b_mm = slab_width_m * 1000.0

    k = M_ed_Nmm / ( (d_mm ** 2) * b_mm * (f_ck_MPa) )  
    sq_arg = 1.0 - 3.53 * k
    if sq_arg < 0:
        raise DesignError(f"Unfeasible geometry/materials: sqrt arg negative (k={k:.6g}). Section inadequate or formula out of range.")
    z_mm = d_mm / 2.0 * (1.0 + math.sqrt(sq_arg))
    if z_mm <= 0:
        raise DesignError("Computed lever arm z non-positive.")

    if f_yk_MPa is None or float(f_yk_MPa) <= 0:
        raise DesignError(f"Invalid steel strength for flexural design: f_yk_MPa={f_yk_MPa}")

    if gamma_s is None or float(gamma_s) <= 0:
        raise DesignError(f"Invalid gamma_s for flexural design: gamma_s={gamma_s}")

    f_yd_MPa = f_yk_MPa / gamma_s

    if f_yd_MPa <= 0:
        raise DesignError(f"Invalid design steel strength: f_yd_MPa={f_yd_MPa}")

    required_ext_As_mm2_per_m = M_ed_Nmm / (z_mm * f_yd_MPa * 1e3)  # f_yd MPa -> N/mm2 ; multiply by 1e3 to get N/mm2 * mm -> N

    selected_spec_ext = choose_rebar_for_As_required(required_ext_As_mm2_per_m, metric_only=True)
    allowable_ext_As_mm2_per_m = None
    if selected_spec_ext is not None:
        area_single_mm2 = selected_spec_ext.area_mm2
        if getattr(selected_spec_ext, "spacing_unit", None) == "mm":
            spacing_m = selected_spec_ext.spacing / 1000.0
        elif getattr(selected_spec_ext, "spacing_unit", None) == "in":
            spacing_m = selected_spec_ext.spacing * 0.0254
        else:
            spacing_m = selected_spec_ext.spacing / 1000.0
        if spacing_m <= 0:
            allowable_ext_As_mm2_per_m = None
        else:
            allowable_ext_As_mm2_per_m = area_single_mm2 * (1.0 / spacing_m)

    rho_ext = (required_ext_As_mm2_per_m * 1e-6) / (slab_width_m * effective_depth_m)

    M_int_Nmm = internal_moment_kNm_per_m * 1e6
    required_int_As_mm2_per_m = M_int_Nmm / (z_mm * f_yd_MPa * 1e3)
    selected_spec_int = choose_rebar_for_As_required(required_int_As_mm2_per_m, metric_only=True)
    allowable_int_As_mm2_per_m = None
    if selected_spec_int is not None:
        area_single_mm2 = selected_spec_int.area_mm2
        if getattr(selected_spec_int, "spacing_unit", None) == "mm":
            spacing_m = selected_spec_int.spacing / 1000.0
        elif getattr(selected_spec_int, "spacing_unit", None) == "in":
            spacing_m = selected_spec_int.spacing * 0.0254
        else:
            spacing_m = selected_spec_int.spacing / 1000.0
        if spacing_m > 0:
            allowable_int_As_mm2_per_m = area_single_mm2 * (1.0 / spacing_m)

    rho_int = (required_int_As_mm2_per_m * 1e-6) / (slab_width_m * effective_depth_m)
    rho_0 = 0.001 * math.sqrt(f_ck_MPa)  

    return {
        "end_moment_kNm_per_m": end_moment_kNm_per_m,
        "internal_moment_kNm_per_m": internal_moment_kNm_per_m,
        "external_shear_kN_per_m": external_shear_kN_per_m,
        "support_shear_kN_per_m": support_shear_kN_per_m,
        "k": k,
        "z_mm": z_mm,
        "required_ext_reinforcement": required_ext_As_mm2_per_m,
        "allowable_ext_reinforcement": allowable_ext_As_mm2_per_m,
        "selected_ext_spec": getattr(selected_spec_ext, "spec", None),
        "rho_ext": rho_ext,
        "rho_0": rho_0,
        "required_int_reinforcement": required_int_As_mm2_per_m,
        "allowable_int_reinforcement": allowable_int_As_mm2_per_m,
        "selected_int_spec": getattr(selected_spec_int, "spec", None),
        "rho_int": rho_int
    }


def deflection_design(required_ext_reinforcement: float,
                      allowable_ext_reinforcement: float,
                      required_int_reinforcement: float,
                      allowable_int_reinforcement: float,
                      rho_ext: float,
                      rho_int: float,
                      rho_0: float,
                      f_ck_value_MPa: float,
                      f_ys_value_MPa: float,
                      permanent_loading_value: float,
                      variable_loading_value: float,
                      ultimate_load_value: float,
                      base_slab_length_value: float,
                      effective_depth_value: float,
                      K_factor: float = 1.0) -> Dict[str, Any]:
    """
    EC2-style limiting span/effective depth check.

    Returns limiting L/d (allowable) and actual L/d and computed max spans (m).
    Implementation follows the EC2 'deemed-to-satisfy' limiting L/d expression:
      if rho <= rho0:
         L/d_lim = K * [11 + 1.5*sqrt(f_ck) * (rho0 / rho) + 3.2*sqrt(f_ck) * ((rho0/rho - 1)**1.5)]
      else:
         L/d_lim = K * [11 + 1.5*sqrt(f_ck) * (rho0/(rho - rho_prime)) + 1/12 * sqrt(f_ck) * sqrt(rho0/rho)]
    For most slab checks you will not have compression reinforcement (rho' approx 0), so we use the first form unless rho>rho0.
    """
    eps = 1e-12
    # ensure f_ck in MPa
    fck = float(f_ck_value_MPa)

    # Avoid zero division; rho_ext/rho_int are expected > 0 if reinforcement found
    rho_ext_val = max(float(rho_ext), eps)
    rho_int_val = max(float(rho_int), eps)
    rho0_val = max(float(rho_0), eps)

    sqrt_fck = math.sqrt(max(fck, 0.0))

    # External span limit
    if rho_ext_val <= rho0_val:
        L_over_d_allowable_ext = K_factor * (11.0 + 1.5 * sqrt_fck * (rho0_val / rho_ext_val)
                                             + 3.2 * sqrt_fck * (max(rho0_val / rho_ext_val - 1.0, 0.0) ** 1.5))
    else:
        # rho' (compression reinforcement) not tracked here; use simplified alternative (rho'~0)
        L_over_d_allowable_ext = K_factor * (11.0 + 1.5 * sqrt_fck * (rho0_val / (rho_ext_val - 0.0))
                                             + (1.0 / 12.0) * sqrt_fck * math.sqrt(rho0_val / rho_ext_val))

    # Internal span limit (same form)
    if rho_int_val <= rho0_val:
        L_over_d_allowable_int = K_factor * (11.0 + 1.5 * sqrt_fck * (rho0_val / rho_int_val)
                                             + 3.2 * sqrt_fck * (max(rho0_val / rho_int_val - 1.0, 0.0) ** 1.5))
    else:
        L_over_d_allowable_int = K_factor * (11.0 + 1.5 * sqrt_fck * (rho0_val / (rho_int_val - 0.0))
                                             + (1.0 / 12.0) * sqrt_fck * math.sqrt(rho0_val / rho_int_val))

    # Actual L/d using base slab length and effective depth (both in meters)
    l_d_actual = float(base_slab_length_value) / float(effective_depth_value)

    max_span_ext_m = L_over_d_allowable_ext * effective_depth_value
    max_span_int_m = L_over_d_allowable_int * effective_depth_value

    return {
        "L_over_d_allowable_ext": L_over_d_allowable_ext,
        "L_over_d_allowable_int": L_over_d_allowable_int,
        "L_over_d_actual": l_d_actual,
        "max_span_ext_m": max_span_ext_m,
        "max_span_int_m": max_span_int_m,
        "rho0": rho0_val,
        "rho_ext": rho_ext_val,
        "rho_int": rho_int_val
    }


def shear_checks(external_shear: float,
                 internal_shear: float,
                 effective_depth_m: float,
                 a1_m: float,
                 allowable_ext_reinforcement_mm2_per_m: float,
                 base_slab_width_m: float,
                 base_slab_length_m: float,
                 f_ck_value_MPa: float,
                 ultimate_load_kN_m2: float,
                 gamma_c: float = None) -> Dict[str, float]:
    """
    Shear checks using Eurocode 2 (EN1992-1-1) style expressions.

    Inputs:
      - external_shear, internal_shear : kN per metre (design shear per m)
      - effective_depth_m, a1_m : meters
      - allowable_ext_reinforcement_mm2_per_m : mm^2 per metre (As provided by selected bar spec)
      - base_slab_width_m : width (m) for the strip considered (usually 1.0)
      - f_ck_value_MPa : concrete f_ck in MPa (N/mm^2)
      - ultimate_load_kN_m2 : chosen ULS load (kN/m^2) — used for V_ed calc if needed
      - gamma_c : partial safety factor for concrete (default from LOCAL_DEFAULTS)

    Returns:
      - dict with V_ed_ext_kN, V_ed_int_kN, V_rd_c_kN, V_rd_min_kN, v_rd_c_N_per_mm2, v_min_N_per_mm2
    """
    eps = 1e-12
    if gamma_c is None:
        gamma_c = LOCAL_DEFAULTS.get("gamma_c", 1.5)

    # Convert dimensions to mm where EC2 expects mm
    d_mm = effective_depth_m * 1000.0
    b_mm = base_slab_width_m * 1000.0

    # k factor: k = 1 + sqrt(200 / d_mm) limited to 2.0 (200 is mm in EC2).
    k_factor = 1.0 + math.sqrt(max(200.0 / max(d_mm, eps), 0.0))
    k_factor = min(2.0, k_factor)

    # Convert allowable reinforcement (mm2 per m) to area per unit width for 1m strip.
    # For a 1m strip Asl (mm2) = allowable_ext_reinforcement_mm2_per_m (this already is mm2 over 1m width).
    Asl_mm2 = max(0.0, allowable_ext_reinforcement_mm2_per_m)

    # rho_l = Asl / (b * d)  (all in mm)
    rho_l = Asl_mm2 / (max(b_mm * d_mm, eps))

    # EC2 constants
    CRd_c = 0.18 / float(gamma_c)   # dimensionless

    # compute v_rd,c (N/mm^2): CRd,c * k * (100 * rho_l * f_ck)^(1/3)
    inner = max(0.0, 100.0 * rho_l * max(f_ck_value_MPa, 0.0))
    v_rd_c_N_per_mm2 = CRd_c * k_factor * (inner ** (1.0 / 3.0))

    # v_min (N/mm^2): 0.035 * k^(3/2) * sqrt(f_ck)
    v_min_N_per_mm2 = 0.035 * (k_factor ** 1.5) * math.sqrt(max(f_ck_value_MPa, 0.0))

    # use the maximum (EC2: V_Rd,c = max(v_rd_c, v_min) * b * d)
    v_rd_N_per_mm2 = max(v_rd_c_N_per_mm2, v_min_N_per_mm2)

    # Convert to kN (for the 1m strip): V_rd = v_rd (N/mm2) * b_mm * d_mm (mm^2) -> N ; /1000 -> kN
    V_rdc_kN = v_rd_N_per_mm2 * b_mm * d_mm / 1000.0
    V_rdc_c_kN = v_rd_c_N_per_mm2 * b_mm * d_mm / 1000.0
    V_rdc_min_kN = v_min_N_per_mm2 * b_mm * d_mm / 1000.0

    # Design shear V_ed: prefer caller provided external/internal shear (kN per m).
    # If those are intended as line forces already (kN per m) we leave as-is.
    V_ed_ext_kN = float(external_shear)
    V_ed_int_kN = float(internal_shear)

    return {
        "V_ed_ext_kN": V_ed_ext_kN,
        "V_ed_int_kN": V_ed_int_kN,
        "v_rd_c_N_per_mm2": v_rd_c_N_per_mm2,
        "v_min_N_per_mm2": v_min_N_per_mm2,
        "v_rd_N_per_mm2": v_rd_N_per_mm2,
        "V_rdc_kN": V_rdc_kN,
        "V_rdc_c_kN": V_rdc_c_kN,
        "V_rdc_min_kN": V_rdc_min_kN,
        "k_factor": k_factor,
        "rho_l": rho_l
    }

# ---------- Example wrapper showing how to call your original math using data from systems_catalog ----------
def check_slab_row_preserve_math(row: Dict[str, Any],
                                 bar_spacing_df: pd.DataFrame | None = None,
                                 material_csv_path: str | None = None,
                                 load_combos_yaml: str | None = None,
                                 load_values_yaml: str | None = None) -> Dict[str, Any]:
    """
    Main workhorse. Expects `row` to contain keys for slab dimensions and material id.
    Pulls material via material_csv_path and computes loads, ULS combos, flex/defl/shear.
    Returns a dict with G, Q, chosen ULS, flex, deflection and shear results.
    """

    # -----------------------------
    # Robust parsing helpers
    # -----------------------------
    def _is_missing(v: Any) -> bool:
        try:
            return v is None or (isinstance(v, str) and v.strip() == "") or pd.isna(v)
        except Exception:
            return v is None or (isinstance(v, str) and v.strip() == "")

    def _f(key: str, default: float) -> float:
        v = row.get(key, default)
        if _is_missing(v):
            v = default
        try:
            return float(v)
        except Exception:
            return float(default)

    def _f2(key1: str, key2: str, default: float) -> float:
        v = row.get(key1, None)
        if _is_missing(v):
            v = row.get(key2, default)
        if _is_missing(v):
            v = default
        try:
            return float(v)
        except Exception:
            return float(default)

    # read geometry from row (caller should provide these; fallbacks are minimal)
    slab_depth_m = _f2("slab_depth", "depth", 0.175)
    slab_width_m = _f("slab_width", LOCAL_DEFAULTS["fallback_slab_width_m"])
    slab_length_m = _f2("span", "max_span", LOCAL_DEFAULTS["fallback_span_m"])
    screed_depth_m = _f("screed_depth", 0.0)
    live_load_kN_m2 = _f2("live_load_kN_m2", "ll", 2.0)
    partition_load_kN_m2 = _f("partition_load_kN_m2", 0.0)
    cover_m = _f2("cover_m", "nominal_cover_m", 0.015)
    deviation_allowance_m = _f2("deviation_allowance_m", "deviation_m", 0.01)
    wall_thickness_m = _f("wall_thickness_m", 0.2)
    d_rebar_m = _f("d_bar_m", LOCAL_DEFAULTS["d_bar_m"])

    concrete_material_id = row.get("material_concrete_id") or row.get("material_id")
    rebar_material_id = row.get("material_rebar_id")
    screed_material_id = row.get("material_screed_id")

    concrete_material_id = row.get("material_concrete_id") or row.get("material_id")
    rebar_material_id = row.get("material_rebar_id")
    screed_material_id = row.get("material_screed_id")

    if material_csv_path:
        if concrete_material_id:
            concrete_mat = load_material_from_csv(material_csv_path, concrete_material_id)
        else:
            raise DesignError("Concrete material not provided for slab check.")

        if rebar_material_id:
            rebar_mat = load_material_from_csv(material_csv_path, rebar_material_id)
        else:
            raise DesignError("Rebar material not provided for slab check.")

        if screed_material_id:
            screed_mat = load_material_from_csv(material_csv_path, screed_material_id)
        else:
            screed_mat = concrete_mat

        # Combined RC design material:
        # concrete strength from concrete, steel strength from rebar
        mat = Material(
            material_id=f"{concrete_mat.material_id}+{rebar_mat.material_id}",
            f_ck_MPa=concrete_mat.f_ck_MPa,
            f_yk_MPa=rebar_mat.f_yk_MPa,
            density_kN_m3=concrete_mat.density_kN_m3,
            gamma_c=concrete_mat.gamma_c,
            gamma_s=rebar_mat.gamma_s,
            original_units=concrete_mat.original_units,
            raw={
                "concrete_material_id": concrete_mat.material_id,
                "rebar_material_id": rebar_mat.material_id,
                "screed_material_id": screed_mat.material_id,
            },
        )

    else:
        logger.warning("Material CSV path not provided; using fallback defaults.")
        concrete_mat = Material(
            material_id=str(concrete_material_id or "fallback_concrete"),
            f_ck_MPa=float(row.get("concrete_f_ck", 30.0)),
            f_yk_MPa=0.0,
            density_kN_m3=float(row.get("density", 25.0)),
            gamma_c=float(row.get("gamma_c", LOCAL_DEFAULTS["gamma_c"])),
            gamma_s=float(row.get("gamma_s", LOCAL_DEFAULTS["gamma_s"])),
            original_units=str(row.get("unit", "metric")),
            raw=row,
        )

        rebar_mat = Material(
            material_id=str(rebar_material_id or "fallback_rebar"),
            f_ck_MPa=0.0,
            f_yk_MPa=float(row.get("steel_fy", 500.0)),
            density_kN_m3=float(row.get("density", 78.5)),
            gamma_c=float(row.get("gamma_c", LOCAL_DEFAULTS["gamma_c"])),
            gamma_s=float(row.get("gamma_s", LOCAL_DEFAULTS["gamma_s"])),
            original_units=str(row.get("unit", "metric")),
            raw=row,
        )

    screed_mat = concrete_mat

    mat = Material(
        material_id=f"{concrete_mat.material_id}+{rebar_mat.material_id}",
        f_ck_MPa=concrete_mat.f_ck_MPa,
        f_yk_MPa=rebar_mat.f_yk_MPa,
        density_kN_m3=concrete_mat.density_kN_m3,
        gamma_c=concrete_mat.gamma_c,
        gamma_s=rebar_mat.gamma_s,
        original_units=concrete_mat.original_units,
        raw=row,
    )

    design_mat = Material(
        material_id=f"{concrete_mat.material_id}+{rebar_mat.material_id}",
        f_ck_MPa=concrete_mat.f_ck_MPa,
        f_yk_MPa=rebar_mat.f_yk_MPa,
        density_kN_m3=concrete_mat.density_kN_m3,
        gamma_c=concrete_mat.gamma_c,
        gamma_s=rebar_mat.gamma_s,
        original_units=concrete_mat.original_units,
        raw={
            "concrete_material_id": concrete_mat.material_id,
            "rebar_material_id": rebar_mat.material_id,
            "screed_material_id": screed_mat.material_id,
        },
)
    
    # permanent & variable loads
    G_kN_m2 = permanent_loading(
                                slab_depth_m=slab_depth_m,
                                slab_width_m=slab_width_m,
                                screed_depth_m=screed_depth_m,
                                concrete_density_kN_m3=concrete_mat.density_kN_m3,
                                screed_density_kN_m3=screed_mat.density_kN_m3,
                                slab_code_loading=_f("slab_code_loading", 0.0),
                                screed_code_loading=_f("screed_code_loading", 0.0),
                                finish_code_loading=_f("finish_code_loading", 0.0),
                                service_code_loading=_f("service_code_loading", 0.0))

    Q_kN_m2 = variable_loading(live_load_kN_m2=live_load_kN_m2, partition_load_kN_m2=partition_load_kN_m2)

    # ultimate load per your original formula (call with G, Q)
    psi = _f("psi_factor", 1.0)
    combos = ultimate_loading(permanent_load_kN_m2=G_kN_m2,
                              variable_load_kN_m2=Q_kN_m2,
                              load_combos_yaml=load_combos_yaml,
                              load_values_yaml=load_values_yaml)

    chosen_combo_name, chosen_ULS_kN_m2 = max(combos.items(), key=lambda it: it[1])

    dims = slab_dimensions(slab_depth_m=slab_depth_m,
                           screed_depth_m=screed_depth_m,
                           d_rebar_m=d_rebar_m,
                           cover_m=cover_m,
                           deviation_allowance_m=deviation_allowance_m,
                           slab_length_m=slab_length_m,
                           wall_thickness_m=wall_thickness_m)

    flex = flexural_design(ultimate_load_kN_m2=chosen_ULS_kN_m2,
                           span_m=dims["effective_span_m"],
                           effective_depth_m=dims["effective_depth_m"],
                           material=design_mat,
                           slab_width_m=slab_width_m,
                           bar_db=None)

    defl = deflection_design(required_ext_reinforcement=flex["required_ext_reinforcement"],
                             allowable_ext_reinforcement=flex["allowable_ext_reinforcement"] if flex["allowable_ext_reinforcement"] is not None else 0.0,
                             required_int_reinforcement=flex["required_int_reinforcement"],
                             allowable_int_reinforcement=flex["allowable_int_reinforcement"] if flex["allowable_int_reinforcement"] is not None else 0.0,
                             rho_ext=flex["rho_ext"],
                             rho_int=flex["rho_int"],
                             rho_0=flex["rho_0"],
                             f_ck_value_MPa=design_mat.f_ck_MPa,
                             f_ys_value_MPa=design_mat.f_yk_MPa,
                             permanent_loading_value=G_kN_m2,
                             variable_loading_value=Q_kN_m2,
                             ultimate_load_value=chosen_ULS_kN_m2,
                             base_slab_length_value=slab_length_m,
                             effective_depth_value=dims["effective_depth_m"])

    shear = shear_checks(external_shear=flex["external_shear_kN_per_m"],
                        internal_shear=flex["support_shear_kN_per_m"],
                        effective_depth_m=dims["effective_depth_m"],
                        a1_m=dims["a1_m"],
                        allowable_ext_reinforcement_mm2_per_m=flex.get("allowable_ext_reinforcement", 0.0) or 0.0,
                        base_slab_width_m=slab_width_m,
                        base_slab_length_m=slab_length_m,
                        f_ck_value_MPa=design_mat.f_ck_MPa,
                        ultimate_load_kN_m2=chosen_ULS_kN_m2,
                        gamma_c=design_mat.gamma_c)


    return {
        "G_kN_m2": G_kN_m2,
        "Q_kN_m2": Q_kN_m2,
        "ULS_kN_m2": chosen_ULS_kN_m2,
        "ULS_combo_name": chosen_combo_name,
        "slab_effective_depth_m": dims["effective_depth_m"],
        "flex": flex,
        "deflection": defl,
        "shear": shear
    }
