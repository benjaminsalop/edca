from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
import json

logger = logging.getLogger(__name__)

Number = Union[int, float]

# -------------------------
# Debug helpers (drop-in)
# -------------------------
import os
from typing import Iterable

def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        s = str(value).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _resolve_partition_load(
    cf: Any,
    floor: int,
    occ_token: str,
    occ_resolved: str,
    occ_row: pd.Series,
    partition_col: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Resolve any partition load that should be added to SDL before factored loads
    and load combinations are evaluated.

    Priority:
      1) partition-like column in occupancies.csv for the current occupancy row
      2) control-file attribute on cf (scalar / per-floor list / dict)
      3) default to 0.0
    """
    if partition_col and partition_col in occ_row.index:
        v = _coerce_optional_float(occ_row.get(partition_col))
        if v is not None:
            return float(v), f"occupancies.csv:{partition_col}"

    candidate_attrs = (
        "partition_load",
        "partition_loads",
        "partition_sdl",
        "partition_sdls",
        "partition_dead_load",
        "partition_dead_loads",
        "partition_dl",
        "partition_dls",
    )

    lookup_keys = (
        floor,
        str(floor),
        occ_resolved,
        occ_resolved.lower(),
        occ_token,
        str(occ_token).lower(),
        "default",
        "DEFAULT",
    )

    for attr in candidate_attrs:
        if not hasattr(cf, attr):
            continue

        raw = getattr(cf, attr)

        v = _coerce_optional_float(raw)
        if v is not None:
            return float(v), f"cf.{attr}"

        if isinstance(raw, dict):
            for key in lookup_keys:
                if key in raw:
                    v = _coerce_optional_float(raw[key])
                    if v is not None:
                        return float(v), f"cf.{attr}[{key!r}]"
            continue

        if isinstance(raw, (list, tuple)) and 0 <= int(floor) < len(raw):
            v = _coerce_optional_float(raw[int(floor)])
            if v is not None:
                return float(v), f"cf.{attr}[{floor}]"

    return 0.0, "none"

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

# -------------------------
# YAML helpers
# -------------------------
def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must parse to dict: {p}")
    return data

def resolve_psi_key(
    occupancy_key: str,
    load_values: Dict[str, Any],
    *,
    unit: Optional[str] = None,
    code_standard: Optional[str] = None,
    default_base: str = "office",) -> Tuple[str, str]:
    """
    Map an occupancy (e.g. 'Laboratory_EC') to an available psi.Q key in load_values.yaml.

    Returns (psi_key, strategy_used).
    """

    # pull the available psi keys
    psi_root = None
    if isinstance(load_values.get("EN1990"), dict) and isinstance(load_values.get("EN1990", {}).get("psi"), dict):
        psi_root = load_values["EN1990"]["psi"]
    elif isinstance(load_values.get("psi"), dict):
        psi_root = load_values["psi"]

    psiQ = (psi_root or {}).get("Q", {}) if isinstance(psi_root, dict) else {}
    if not isinstance(psiQ, dict) or not psiQ:
        # if psi is missing, just fall back
        return f"{default_base}_EC", "no_psi_in_yaml_fallback"

    available = set(psiQ.keys())

    occ = str(occupancy_key).strip()
    occ_l = occ.lower()

    # choose suffix preference
    cs = (code_standard or "").lower()
    u = (unit or "").lower()
    prefer_ec = ("euro" in cs) or (u in {"metric", "si"})
    suf = "_EC" if prefer_ec else "_ASCE"

    def norm_key(s: str) -> str:
        return s.strip()

    # Try direct candidates first (case-insensitive match)
    candidates = [
        occ,
        occ_l,
        # if it isn't suffixed, append suffix
        (occ + suf) if not re.search(r"_(EC|ASCE)$", occ, flags=re.IGNORECASE) else occ,
        (occ_l + suf.lower()) if not re.search(r"_(ec|asce)$", occ_l) else occ_l,
    ]

    # If already suffixed (Laboratory_EC), also try base + suffix in lowercase
    base = re.sub(r"_(EC|ASCE)$", "", occ, flags=re.IGNORECASE)
    base_l = base.lower()
    candidates += [
        f"{base}{suf}",
        f"{base_l}{suf.lower()}",
    ]

    # Match against available keys with casefold
    avail_casefold = {k.lower(): k for k in available}
    for c in candidates:
        k = avail_casefold.get(norm_key(c).lower())
        if k:
            return k, "direct_or_casefold"

    # Synonym mapping to the *known* EN1990 categories in your YAML
    # (edit these mappings as you like)
    synonym_base_map = {
        # offices / work areas
        "lab": "office",
        "laboratory": "office",
        "research": "office",
        "clinic": "office",

        # people congregate
        "lecture": "assembly",
        "auditorium": "assembly",
        "theatre": "assembly",
        "classroom": "assembly",
        "teaching": "assembly",

        # storage
        "warehouse": "storage",
        "archive": "storage",
        "plant": "storage",

        # retail
        "retail": "shopping",
        "shop": "shopping",

        # parking
        "garage": "parking",
        "carpark": "parking",
        "parking": "parking",

        # residential
        "hotel": "residential",
        "dorm": "residential",
        "residential": "residential",
    }

    # heuristic: if occupancy contains any synonym token
    for token, mapped in synonym_base_map.items():
        if token in base_l:
            key_try = f"{mapped}{suf}"
            k = avail_casefold.get(key_try.lower())
            if k:
                return k, f"synonym:{token}->{mapped}"

    # final fallback: default category
    fallback = f"{default_base}{suf}"
    k = avail_casefold.get(fallback.lower())
    if k:
        return k, f"fallback_default:{default_base}"

    # if even that doesn't exist, return ANY key (deterministic) so you still run
    any_key = sorted(list(available))[0]
    return any_key, f"fallback_any:{any_key}"


def get_by_path(d: Dict[str, Any], path: str) -> Any:
    """
    Resolve dotted paths like 'EN1990.gamma.ULS.Q' into nested dicts.

    DEFENSIVE: if a key isn't found, try a case-insensitive match among dict keys.
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"Missing path '{path}' at '{key}' (parent not a dict)")

        if key in cur:
            cur = cur[key]
            continue

        # case-insensitive fallback (fixes Assembly_EC vs assembly_EC)
        key_l = key.lower()
        match = None
        for k in cur.keys():
            if isinstance(k, str) and k.lower() == key_l:
                match = k
                break

        if match is None:
            raise KeyError(f"Missing path '{path}' at '{key}'")

        cur = cur[match]

    return cur



NUM_RE = re.compile(r"^\s*[+-]?(\d+(\.\d*)?|\.\d+)\s*$")


def eval_factor_expr(expr: Union[str, Number], values: Dict[str, Any], occupancy_key: str) -> float:
    """
    Evaluate expressions like:
      'EN1990.gamma.ULS.Q * EN1990.psi.Q.{occupancy}.psi0'
    by multiplying pieces separated by '*'.
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    s = str(expr).strip().replace("{occupancy}", occupancy_key)

    parts = [p.strip() for p in s.split("*") if p.strip()]
    out = 1.0
    for part in parts:
        if NUM_RE.match(part):
            out *= float(part)
        else:
            out *= float(get_by_path(values, part))
    return float(out)


def inject_aliases(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your combos refer to EN1990.psi..., but your values YAML stores psi at top-level.
    Create an alias so EN1990.psi... resolves properly.
    """
    if "EN1990" in values and "psi" in values and isinstance(values["EN1990"], dict):
        values["EN1990"].setdefault("psi", values["psi"])
    return values

def resolve_occupancy_key(
    occ_token: str,
    occ_lookup: Dict[str, object],
    *,
    unit: Optional[str] = None,
    code_standard: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve a program occupancy token like 'assembly' into the key used in occupancies.csv
    like 'assembly_EC' or 'assembly_ASCE'.

    Returns (resolved_key, strategy_used).

    Strategy:
      1) if token exists exactly, use it
      2) if token already has suffix _EC/_ASCE and exists, use it
      3) append preferred suffix based on code_standard/unit and try
      4) try the other suffix
      5) finally try raw token again (case/strip normalizations are handled by caller)
    """
    occ = str(occ_token).strip()

    if occ in occ_lookup:
        return occ, "exact"

    # detect already-suffixed
    upper = occ.upper()
    if upper.endswith("_EC") or upper.endswith("_ASCE"):
        if occ in occ_lookup:
            return occ, "already_suffixed_exact"
        # sometimes case differs (e.g., Assembly_EC vs assembly_EC)
        for k in occ_lookup.keys():
            if k.lower() == occ.lower():
                return k, "already_suffixed_casefold"

    # choose preferred suffix
    cs = (code_standard or "").lower()
    u = (unit or "").lower()

    prefer_ec = ("euro" in cs) or (u in {"metric", "si"})
    preferred = "_EC" if prefer_ec else "_ASCE"
    alternate = "_ASCE" if preferred == "_EC" else "_EC"

    # helper: add suffix only if not present
    def with_suffix(base: str, suf: str) -> str:
        b = base
        if b.upper().endswith("_EC") or b.upper().endswith("_ASCE"):
            return b
        return f"{b}{suf}"

    cand1 = with_suffix(occ, preferred)
    if cand1 in occ_lookup:
        return cand1, f"preferred_suffix{preferred}"

    # casefold match for cand1
    for k in occ_lookup.keys():
        if k.lower() == cand1.lower():
            return k, f"preferred_suffix_casefold{preferred}"

    cand2 = with_suffix(occ, alternate)
    if cand2 in occ_lookup:
        return cand2, f"alternate_suffix{alternate}"

    for k in occ_lookup.keys():
        if k.lower() == cand2.lower():
            return k, f"alternate_suffix_casefold{alternate}"

    # last-ditch: try raw token casefold
    for k in occ_lookup.keys():
        if k.lower() == occ.lower():
            return k, "raw_casefold"

    raise KeyError(
        f"Occupancy '{occ_token}' not found. Tried: "
        f"{occ!r}, {cand1!r}, {cand2!r}. "
        f"Available examples: {list(sorted(list(occ_lookup.keys())))[:15]}..."
    )

# -------------------------
# Combo evaluation
# -------------------------
def compute_en1990_uls_combos(
    raw_sdl: float,
    raw_ll: float,
    occupancy_key: str,
    load_values: Dict[str, Any],
    load_combos: Dict[str, Any],
    *,
    debug: bool | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {name, total, factors, terms}
    Adds 'terms' for traceability.
    """
    out: List[Dict[str, Any]] = []
    uls = (load_combos.get("EN1990", {}) or {}).get("ULS", {}) or {}
    if not isinstance(uls, dict):
        return out

    def load_for_action(action: str) -> float:
        if action in ("G", "G_unf"):
            return float(raw_sdl)
        if action == "G_fav":
            return 0.0
        if action == "Q_lead":
            return float(raw_ll)
        if action == "Q_acc":
            return 0.0
        return 0.0

    gamma_g = float(get_by_path(load_values, "EN1990.gamma.ULS.G"))
    gamma_q = float(get_by_path(load_values, "EN1990.gamma.ULS.Q"))

    psi0 = None
    try:
        psi0 = float(get_by_path(load_values, f"EN1990.psi.Q.{occupancy_key}.psi0"))
    except Exception:
        psi0 = None

    for combo_name, spec in uls.items():
        expr_list = spec.get("expression") if isinstance(spec, dict) else None
        if not isinstance(expr_list, list):
            continue

        total = 0.0
        terms = []
        for term in expr_list:
            if not isinstance(term, dict):
                continue
            action = str(term.get("action", "")).strip()
            factor_expr = term.get("factor", 1.0)
            f = eval_factor_expr(factor_expr, load_values, occupancy_key)
            q = load_for_action(action)
            total += q * f
            terms.append({"action": action, "q": q, "factor": f, "contrib": q * f})

        out.append({
            "name": str(combo_name),
            "total": float(total),
            "factors": {"DL": gamma_g, "LL": gamma_q, **({"psi0": psi0} if psi0 is not None else {})},
            "terms": terms,
        })

    if _dbg_enabled(debug):
        # show a compact ranking + governing combo
        ranked = sorted(out, key=lambda d: float(d.get("total", 0.0)), reverse=True)
        gov = ranked[0] if ranked else None
        _dbg_kv("loads.EN1990.uls.summary", {
            "occupancy_key": occupancy_key,
            "raw_sdl": float(raw_sdl),
            "raw_ll": float(raw_ll),
            "n_combos": len(out),
            "governing": (gov.get("name") if gov else None),
            "governing_total": (gov.get("total") if gov else None),
        }, explicit=True, level=logging.INFO)
        if gov:
            logger.debug("[debug] loads.EN1990.governing.terms=%s", gov.get("terms"))

    return out


def compute_asce7_lrfd_combos(
    raw_sdl: float,
    raw_ll: float,
    load_combos: Dict[str, Any],
    *,
    debug: bool | None = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lrfd = ((load_combos.get("ASCE7", {}) or {}).get("LRFD", {}) or {})
    if not isinstance(lrfd, dict):
        return out

    def load_for_action(action: str) -> float:
        if action == "D":
            return float(raw_sdl)
        if action == "L":
            return float(raw_ll)
        return 0.0

    for name, spec in lrfd.items():
        if not isinstance(spec, dict):
            continue
        alts = spec.get("alternatives")
        if not isinstance(alts, list):
            continue

        best_total = None
        best = None
        for alt in alts:
            expr_list = alt.get("expression") if isinstance(alt, dict) else None
            if not isinstance(expr_list, list):
                continue
            total = 0.0
            factors: Dict[str, float] = {}
            terms = []
            for term in expr_list:
                if not isinstance(term, dict):
                    continue
                action = str(term.get("action", "")).strip()
                f = float(term.get("factor", 1.0))
                q = load_for_action(action)
                total += q * f
                if action in ("D", "L"):
                    factors[action] = f
                terms.append({"action": action, "q": q, "factor": f, "contrib": q * f})

            if best_total is None or total > best_total:
                best_total = float(total)
                best = {"name": str(name), "total": float(total), "factors": factors, "terms": terms}

        if best is not None:
            out.append(best)

    if _dbg_enabled(debug):
        ranked = sorted(out, key=lambda d: float(d.get("total", 0.0)), reverse=True)
        gov = ranked[0] if ranked else None
        _dbg_kv("loads.ASCE7.lrfd.summary", {
            "raw_sdl": float(raw_sdl),
            "raw_ll": float(raw_ll),
            "n_combos": len(out),
            "governing": (gov.get("name") if gov else None),
            "governing_total": (gov.get("total") if gov else None),
        }, explicit=True, level=logging.INFO)
        if gov:
            logger.debug("[debug] loads.ASCE7.governing.terms=%s", gov.get("terms"))

    return out


def resolve_span_values(cf, args, logger: Optional[logging.Logger] = None) -> List[float]:
    """
    Resolve the list of span values (in metres) to evaluate.

    Priority:
      1) CLI overrides (--span-min/--span-max/--span-step/--no-sweep)
      2) Control file spans list (cf.spans) already parsed/expanded by parse.parse_spans
      3) Fallback to cf.ideal_column_spacing (single value) or 6.0 m
    """
    log = logger or globals().get("logger") or logging.getLogger("spans")

    # 1) CLI overrides take precedence if either bound is provided
    span_min = getattr(args, "span_min", None)
    span_max = getattr(args, "span_max", None)
    span_step = float(getattr(args, "span_step", 0.5) or 0.5)
    no_sweep = bool(getattr(args, "no_sweep", False))

    if span_min is not None or span_max is not None:
        # If only one bound provided, treat it as a single span
        if span_min is None:
            span_min = float(span_max)
        if span_max is None:
            span_max = float(span_min)

        mn = float(span_min)
        mx = float(span_max)
        if mx < mn:
            mn, mx = mx, mn

        if no_sweep or abs(mx - mn) < 1e-9:
            spans = [round(mn, 6)]
            log.info("[span] Span override: evaluating single span %.3f m (--no-sweep or equal bounds).", spans[0])
            return spans

        if span_step <= 0:
            span_step = 0.5
            log.warning("[span] Invalid --span-step; defaulting to %.2f m.", span_step)

        spans: List[float] = []
        x = mn
        # inclusive sweep with tolerance
        while x <= mx + 1e-9:
            spans.append(round(x, 6))
            x += span_step

        log.info("[span] Span override: sweeping %.3f..%.3f m step=%.3f (%d values).", mn, mx, span_step, len(spans))
        return spans

    # 2) Control file spans
    cf_spans = getattr(cf, "spans", None)
    if isinstance(cf_spans, (list, tuple)) and len(cf_spans) > 0:
        spans = [float(v) for v in cf_spans]
        if no_sweep:
            spans = [min(spans)]
            log.info("[span] Control-file spans present but --no-sweep specified; using min span %.3f m.", spans[0])
        return spans

    # 3) Fallbacks
    ics = getattr(cf, "ideal_column_spacing", None)
    if ics is not None:
        try:
            spans = [float(ics)]
            log.warning("[span] No spans specified; falling back to IDEAL_COLUMN_SPACING=%.3f m.", spans[0])
            return spans
        except Exception:
            pass

    log.warning("[span] No spans specified; falling back to default span 6.0 m.")
    return [6.0]


# -------------------------
# Public API
# -------------------------
def build_load_context(
    cf: Any,
    occupancies_csv: Union[str, Path],
    load_values_yaml: Optional[Union[str, Path]] = None,
    load_combinations_yaml: Optional[Union[str, Path]] = None,
    *,
    debug: bool = False,
) -> Tuple[Dict[str, float], Dict[str, List[int]], pd.DataFrame]:
    """
    Returns:
      required_loads: dict with max_sdl/max_ll (+ factored maxima)
      floors_by_case: occupancy -> list of floors
      loads_df_floor: per-floor rows, including combo lists and factored outputs
    """
    occ_path = Path(occupancies_csv)

    # Auto-locate YAMLs next to occupancies.csv unless explicitly passed
    if load_values_yaml is None:
        load_values_yaml = occ_path.with_name("load_values.yaml")
    if load_combinations_yaml is None:
        load_combinations_yaml = occ_path.with_name("load_combinations.yaml")

    load_values = inject_aliases(read_yaml(load_values_yaml))
    load_combos = read_yaml(load_combinations_yaml)

    use_factored_loads = bool(getattr(cf, "factored_loads", True))

    occ_df = pd.read_csv(occ_path)

    # Normalize columns
    colmap = {c: c.strip().lower() for c in occ_df.columns}
    occ_df = occ_df.rename(columns=colmap)

    def pick_col(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in occ_df.columns:
                return c
        return None

    occ_col = pick_col(["occupancy", "load_case", "case", "space_type", "program", "use"])
    sdl_col = pick_col(["sdl", "raw_sdl", "dead_load", "dl", "gk"])
    ll_col  = pick_col(["ll", "raw_ll", "live_load", "imposed_load", "qk"])

    if occ_col is None or sdl_col is None or ll_col is None:
        raise ValueError(
            "occupancies.csv must include occupancy + SDL + LL columns. "
            f"Found columns: {list(occ_df.columns)}"
        )

    # Optional: explicit psi key column (otherwise infer as f'{occupancy}_EC')
    psi_col = pick_col(["psi_key", "eurocode_key", "ec_key"])

    # Optional: partition load column to be added to SDL before factored loads
    # and governing combinations are evaluated.
    partition_col = pick_col([
        "sdl_partition",
        "partition_load",
        "partition_sdl",
        "partition_dead_load",
        "partition_dl",
        "partitions",
        "moveable_partitions",
    ])

    occ_df[occ_col] = occ_df[occ_col].astype(str).str.strip()
    occ_df[sdl_col] = pd.to_numeric(occ_df[sdl_col], errors="coerce").fillna(0.0)
    occ_df[ll_col]  = pd.to_numeric(occ_df[ll_col], errors="coerce").fillna(0.0)
    if partition_col is not None:
        occ_df[partition_col] = pd.to_numeric(occ_df[partition_col], errors="coerce").fillna(0.0)

    occ_lookup = {r[occ_col]: r for _, r in occ_df.iterrows()}

    # Floor program mapping
    floor_to_occ: Dict[int, str] = {}
    if getattr(cf, "program", None):
        floor_to_occ.update({int(k): str(v) for k, v in dict(cf.program).items()})

    program_default = str(getattr(cf, "program_default", "office"))
    num_floors = getattr(cf, "num_floors", None)
    if num_floors is None:
        # infer from program map / area map
        if floor_to_occ:
            num_floors = max(floor_to_occ.keys()) + 1
        else:
            area_map = getattr(cf, "area_per_floor", {}) or {}
            num_floors = max(area_map.keys()) + 1 if area_map else 1

    code_standard = str(getattr(cf, "code_standard", "Default"))
    use_euro = "euro" in code_standard.lower()
    combo_col = "EN1990_ULS" if use_euro else "ASCE7_LRFD"

    rows: List[Dict[str, Any]] = []
    floors_by_case: Dict[str, List[int]] = {}

    max_raw_sdl = 0.0
    max_raw_ll = 0.0
    max_factored_sdl = 0.0
    max_factored_ll = 0.0
    max_governing_factored_total = 0.0
    max_design_sdl = 0.0
    max_design_ll = 0.0
    max_design_total = 0.0
    max_characteristic_total = 0.0
    

    for f in range(int(num_floors)):
        occ_token = floor_to_occ.get(f, program_default)

        resolved_key, strategy = resolve_occupancy_key(
            occ_token,
            occ_lookup,
            unit=str(getattr(cf, "unit", "")),
            code_standard=str(getattr(cf, "code_standard", "")),
        )

        if resolved_key != str(occ_token):
            logger.info(f"[loads] Resolved occupancy '{occ_token}' -> '{resolved_key}' ({strategy})")

        r = occ_lookup[resolved_key]
        occ = resolved_key

        base_sdl = float(r[sdl_col])
        partition_sdl, partition_source = _resolve_partition_load(
            cf,
            f,
            str(occ_token),
            str(occ),
            r,
            partition_col=partition_col,
        )
        raw_sdl = float(base_sdl + partition_sdl)
        raw_ll = float(r[ll_col])

        if partition_sdl:
            logger.info(
                "[loads] Added partition load %.3f to SDL for floor %s (%s via %s)",
                partition_sdl,
                f,
                occ,
                partition_source,
            )

        floors_by_case.setdefault(str(occ), []).append(int(f))

        # Determine Eurocode psi lookup key
        psi_key, psi_strategy = resolve_psi_key(
            occ,  # or occ_token, but use the resolved occupancy key you ended up with
            load_values,
            unit=str(getattr(cf, "unit", "")),
            code_standard=str(getattr(cf, "code_standard", "")),
        )

        if psi_strategy != "direct_or_casefold":
            logger.info(f"[loads] psi key resolved for '{occ}': {psi_key} ({psi_strategy})")

        if use_euro:
            combos = compute_en1990_uls_combos(raw_sdl, raw_ll, psi_key, load_values, load_combos, debug=debug)
            gamma_g = float(get_by_path(load_values, "EN1990.gamma.ULS.G"))
            gamma_q = float(get_by_path(load_values, "EN1990.gamma.ULS.Q"))
        else:
            combos = compute_asce7_lrfd_combos(raw_sdl, raw_ll, load_combos, debug=debug)
            # For ASCE, gamma values are combo-dependent; leave None here
            gamma_g = None
            gamma_q = None

        # Maintain your legacy outputs
        factored_sdl = (gamma_g * raw_sdl) if gamma_g is not None else raw_sdl
        factored_ll  = (gamma_q * raw_ll)  if gamma_q is not None else raw_ll

        characteristic_sdl = float(raw_sdl)
        characteristic_ll = float(raw_ll)
        characteristic_total = float(raw_sdl + raw_ll)

        design_sdl = float(factored_sdl) if use_factored_loads else characteristic_sdl
        design_ll = float(factored_ll) if use_factored_loads else characteristic_ll
        design_total = float(best_total) if use_factored_loads else characteristic_total

        # NEW: governing trace details
        best = max(combos, key=lambda x: float(x.get("total", 0.0))) if combos else None
        best_total = float(best["total"]) if best else float(raw_sdl + raw_ll)
        best_name = (best.get("name") if isinstance(best, dict) else None)
        best_terms = (best.get("terms") if isinstance(best, dict) else None)

        rows.append({
            "floor": int(f),
            "occupancy": str(occ),
            "base_sdl": float(base_sdl),
            "partition_sdl": float(partition_sdl),

            "SDL": characteristic_sdl,
            "LL": characteristic_ll,

            "characteristic_sdl": characteristic_sdl,
            "characteristic_ll": characteristic_ll,
            "characteristic_total": characteristic_total,

            "factored_sdl": float(factored_sdl),
            "factored_ll": float(factored_ll),
            "factored_total": float(best_total),

            "design_sdl": design_sdl,
            "design_ll": design_ll,
            "design_total": design_total,
            "loads_mode": "factored" if use_factored_loads else "characteristic",

            combo_col: combos,
            "combo": best_name,
            "combo_terms": json.dumps(best_terms, default=str) if best_terms is not None else None,
            "gamma_g": gamma_g,
            "gamma_q": gamma_q,
            "unit": str(getattr(cf, "unit", "metric")).strip().lower(),
        })

        max_raw_sdl = max(max_raw_sdl, characteristic_sdl)
        max_raw_ll = max(max_raw_ll, characteristic_ll)

        max_factored_sdl = max(max_factored_sdl, float(factored_sdl))
        max_factored_ll = max(max_factored_ll, float(factored_ll))
        max_governing_factored_total = max(max_governing_factored_total, float(best_total))

        max_design_sdl = max(max_design_sdl, design_sdl)
        max_design_ll = max(max_design_ll, design_ll)
        max_design_total = max(max_design_total, design_total)

        max_characteristic_total = max(max_characteristic_total, characteristic_total)

    loads_df_floor = pd.DataFrame(rows)

    required_loads = {
        "max_sdl": float(max_design_sdl),
        "max_ll": float(max_design_ll),
        "max_total": float(max_design_total),

        "max_characteristic_sdl": float(max_raw_sdl),
        "max_characteristic_ll": float(max_raw_ll),
        "max_characteristic_total": float(max_characteristic_total),

        "max_factored_sdl": float(max_factored_sdl),
        "max_factored_ll": float(max_factored_ll),
        "max_factored_total": float(max_governing_factored_total),

        "loads_mode": "factored" if use_factored_loads else "characteristic",
    }

    return required_loads, floors_by_case, loads_df_floor
