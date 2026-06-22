from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import yaml

from edca_code.scripts.core.domain_models import GeometryContext, ProjectContext
from edca_code.scripts.core.exceptions import ValidationError


PathLike = Union[str, Path]


def parameters_from_control_file(path: PathLike) -> Dict[str, Any]:
    """Load a YAML control file into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Control file not found: {p}")

    with p.open("r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValidationError(f"Error parsing YAML control file: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValidationError(
            f"Control file must parse to a mapping/dict, got: {type(data).__name__}"
        )
    return data


def parse_program(program_data: Any) -> Dict[int, str]:
    """
    Convert PROGRAM entries into a per-floor occupancy map.

    Accepted forms:
      - [{"floors": [0], "occupancy": "office"}]
      - [{"floors": [1, 3], "occupancy": "residential"}]
    """
    floor_occupancy: Dict[int, str] = {}
    if not program_data:
        return floor_occupancy

    if not isinstance(program_data, list):
        raise ValidationError(f"PROGRAM must be a list of entries, got: {type(program_data).__name__}")

    for entry in program_data:
        if not isinstance(entry, dict):
            raise ValidationError(f"PROGRAM entries must be dicts, got: {entry!r}")
        if "floors" not in entry or "occupancy" not in entry:
            raise ValidationError(f"PROGRAM entry missing 'floors' or 'occupancy': {entry!r}")

        floors = entry["floors"]
        occupancy = str(entry["occupancy"]).strip()
        if not occupancy:
            raise ValidationError(f"PROGRAM occupancy cannot be blank: {entry!r}")
        if not isinstance(floors, (list, tuple)):
            raise ValidationError(f"'floors' must be a 1- or 2-element list, got: {floors!r}")

        floors_int = [int(f) for f in floors]
        if len(floors_int) == 1:
            floor_occupancy[floors_int[0]] = occupancy
        elif len(floors_int) == 2:
            start, end = floors_int
            if end < start:
                start, end = end, start
            for f in range(start, end + 1):
                floor_occupancy[f] = occupancy
        else:
            raise ValidationError(f"Invalid floors spec (need 1 or 2 ints): {floors!r}")

    return floor_occupancy


def parse_floor_plate_area(floor_plate: Any) -> float:
    """
    Resolve the representative floor area from FLOOR_PLATE.

    If mode=dims and both length and width are present, returns length*width.
    Otherwise returns area_per_floor.
    """
    if not isinstance(floor_plate, dict):
        return 0.0

    mode = str(floor_plate.get("mode", "area")).strip().lower()
    if mode.startswith("dim"):
        length = _coerce_float(floor_plate.get("length"))
        width = _coerce_float(floor_plate.get("width"))
        if length is not None and width is not None:
            return length * width

    area_per_floor = _coerce_float(floor_plate.get("area_per_floor"))
    return area_per_floor or 0.0


def parse_spans(spans_block: Any) -> List[float]:
    """
    Produce the list of spans to consider based on SPANS.

    Supports:
      - scalar shorthand: SPANS: 9
      - list mode: {mode: list, list: [6, 7.5, 9]}
      - range mode: {mode: range, range: [6, 9], sweep: true, step: 0.5}
    """

    def _as_float_list(value: Any) -> List[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, str):
            v = _coerce_float(value)
            return [v] if v is not None else []
        if isinstance(value, (list, tuple)):
            out: List[float] = []
            for item in value:
                v = _coerce_float(item)
                if v is not None:
                    out.append(v)
            return out
        return []

    if not isinstance(spans_block, dict):
        return _as_float_list(spans_block)

    mode = str(spans_block.get("mode", "range")).strip().lower()
    if mode == "list":
        vals = spans_block.get("list")
        if vals is None:
            vals = spans_block.get("values")
        if vals is None:
            vals = spans_block.get("range")
        return _as_float_list(vals)

    r = spans_block.get("range")
    if r is None:
        r = spans_block.get("values")
    if r is None:
        r = spans_block.get("list")

    if isinstance(r, (int, float, str)):
        return _as_float_list(r)
    if not isinstance(r, (list, tuple)):
        return []
    if len(r) == 1:
        return _as_float_list(r[0])
    if len(r) != 2:
        return _as_float_list(r)

    mn = _coerce_float(r[0])
    mx = _coerce_float(r[1])
    if mn is None or mx is None:
        return []
    if mx < mn:
        mn, mx = mx, mn

    sweep = _coerce_bool(spans_block.get("sweep"), default=False)
    step = _coerce_float(spans_block.get("step")) or 1.0
    if step <= 0:
        raise ValidationError("SPANS.step must be positive")

    if abs(mx - mn) < 1e-9:
        return [round(mn, 6)]
    if not sweep:
        return [round(mn, 6), round(mx, 6)]

    out: List[float] = []
    x = mn
    while x <= mx + 1e-9:
        out.append(round(x, 6))
        x += step
    return out


def build_floor_area_lookup(
    area_per_floor: float,
    floors_by_case: Dict[str, Iterable[int]],
) -> Tuple[Dict[int, float], float]:
    """Build a dict floor->area for all floors mentioned in floors_by_case."""
    floor_area_lookup: Dict[int, float] = {}
    for floors in (floors_by_case or {}).values():
        for floor in floors:
            floor_area_lookup[int(floor)] = float(area_per_floor)
    return floor_area_lookup, float(area_per_floor)


@dataclass(slots=True)
class ControlFile:
    """
    Backward-compatible parsed control-file wrapper.

    This keeps the legacy attribute-style access used by the current pipeline,
    while also offering conversion into the new ProjectContext model.
    """

    _raw: Dict[str, Any] = field(default_factory=dict)
    use_csv: bool = False
    data_dir: str = "inputs"
    project_name: str = ""
    location: str = ""
    unit: str = "metric"
    code_standard: Optional[str] = None
    floor_plate: Dict[str, Any] = field(default_factory=dict)
    area_per_floor: float = 0.0
    floor_to_floor_height: Optional[float] = None
    num_floors: Optional[int] = None
    ideal_column_spacing: Optional[float] = None
    depth_limit_enabled: bool = False
    depth_limit: Optional[float] = None
    fire_resistance_period: int = 60
    spans_block: Dict[str, Any] = field(default_factory=dict)
    spans: List[float] = field(default_factory=list)
    edge_cantilever_max: Optional[float] = None
    one_way_irregular: bool = False
    one_way_slab_min_span: Optional[float] = None
    one_way_beam_min_span: Optional[float] = None
    one_way_orientation: Optional[str] = None
    program_default: str = "office"
    program: Dict[int, str] = field(default_factory=dict)
    results_scope: str = "PER_SYSTEM"
    notes: str = ""
    factored_loads: bool = True
    columns_by_floor: bool = True
    moment_frame_columns: bool = False
    steel_beam_sls_checks: bool = True
    steel_beam_max_span_depth_ratio: Optional[float] = None
    steel_secondary_beam_max_span_depth_ratio: Optional[float] = None
    steel_beam_moment_capacity_factor: float = 1.0
    steel_primary_beam_moment_capacity_factor: float = 1.0
    steel_secondary_beam_moment_capacity_factor: float = 1.0
    steel_beam_include_self_weight: bool = True
    rc_beam_span_depth_checks: bool = True
    rc_beam_max_span_depth_ratio: Optional[float] = None
    rc_secondary_beam_max_span_depth_ratio: Optional[float] = None
    fire_resistance_minima: bool = True
    load_basis_by_category: Dict[str, str] = field(default_factory=dict)
    typology_id: Optional[str] = None
    region: Optional[str] = None
    code_family: Optional[str] = None
    occupancy_id: Optional[str] = None
    material_set: Optional[str] = None
    design_options_extra: Dict[str, Any] = field(default_factory=dict)
    analysis_options: Dict[str, Any] = field(default_factory=dict)
    system_overrides: Dict[str, Any] = field(default_factory=dict)
    load_overrides: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, inputs: Mapping[str, Any]):
        raw = dict(inputs)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "use_csv", _coerce_bool(raw.get("USE_CSV"), default=False))
        object.__setattr__(self, "data_dir", str(raw.get("DATA_DIR", "inputs")))

        object.__setattr__(self, "project_name", str(raw.get("PROJECT_NAME", "")))
        object.__setattr__(self, "location", str(raw.get("LOCATION", "")))
        object.__setattr__(self, "unit", str(raw.get("UNIT", "metric")))
        object.__setattr__(self, "code_standard", _strip_or_none(raw.get("CODE_STANDARD")))
        object.__setattr__(self, "typology_id", _strip_or_none(raw.get("TYPOLOGY")) or _strip_or_none(raw.get("TYPOLOGY_ID")))
        object.__setattr__(self, "region", _strip_or_none(raw.get("REGION")))
        object.__setattr__(self, "code_family", _normalize_code_family(raw.get("CODE_FAMILY")))
        object.__setattr__(self, "occupancy_id", _strip_or_none(raw.get("OCCUPANCY")) or _strip_or_none(raw.get("OCCUPANCY_ID")))
        object.__setattr__(self, "material_set", _strip_or_none(raw.get("MATERIAL_SET")) or _strip_or_none(raw.get("MATERIAL_SET_ID")))

        floor_plate = raw.get("FLOOR_PLATE", {}) or {}
        object.__setattr__(self, "floor_plate", floor_plate)
        object.__setattr__(self, "area_per_floor", parse_floor_plate_area(floor_plate))

        object.__setattr__(self, "floor_to_floor_height", _coerce_float(raw.get("FLOOR_TO_FLOOR_HEIGHT")))
        object.__setattr__(self, "num_floors", _coerce_int(raw.get("NUM_FLOORS")))
        object.__setattr__(self, "ideal_column_spacing", _coerce_float(raw.get("IDEAL_COLUMN_SPACING")))

        object.__setattr__(self, "depth_limit_enabled", _coerce_bool(raw.get("DEPTH_LIMIT_ENABLED"), default=False))
        object.__setattr__(self, "depth_limit", _coerce_float(raw.get("DEPTH_LIMIT")))
        object.__setattr__(self, "fire_resistance_period", int(_coerce_float(raw.get("FIRE_RESISTANCE_PERIOD")) or 60))

        spans_block = raw.get("SPANS", {}) or {}
        object.__setattr__(self, "spans_block", spans_block)
        object.__setattr__(self, "spans", parse_spans(spans_block))
        object.__setattr__(self, "edge_cantilever_max", _coerce_float(spans_block.get("edge_cantilever_max")))

        object.__setattr__(self, "one_way_irregular", _coerce_bool(raw.get("ONE_WAY_IRREGULAR"), default=False))
        object.__setattr__(self, "one_way_slab_min_span", _coerce_float(raw.get("ONE_WAY_SLAB_MIN_SPAN")))
        object.__setattr__(self, "one_way_beam_min_span", _coerce_float(raw.get("ONE_WAY_BEAM_MIN_SPAN")))
        object.__setattr__(self, "one_way_orientation", _strip_or_none(raw.get("ONE_WAY_ORIENTATION")))

        object.__setattr__(self, "program_default", str(raw.get("PROGRAM_DEFAULT", "office")))
        object.__setattr__(self, "program", parse_program(raw.get("PROGRAM", [])))

        object.__setattr__(self, "results_scope", str(raw.get("RESULTS_SCOPE", "PER_SYSTEM")))
        object.__setattr__(self, "notes", str(raw.get("NOTES", "")))
        object.__setattr__(self, "factored_loads", _coerce_bool(raw.get("FACTORED_LOADS"), default=True))
        object.__setattr__(self, "columns_by_floor", _coerce_bool(raw.get("COLUMNS_BY_FLOOR"), default=True))
        object.__setattr__(self, "moment_frame_columns", _coerce_bool(raw.get("MOMENT_FRAME_COLUMNS"), default=False))
        object.__setattr__(self, "steel_beam_sls_checks", _coerce_bool(raw.get("STEEL_BEAM_SLS_CHECKS"), default=True))
        object.__setattr__(self, "steel_beam_max_span_depth_ratio", _coerce_float(raw.get("STEEL_BEAM_MAX_SPAN_DEPTH_RATIO")) or 16.0)
        object.__setattr__(self, "steel_secondary_beam_max_span_depth_ratio", _coerce_float(raw.get("STEEL_SECONDARY_BEAM_MAX_SPAN_DEPTH_RATIO")) or 16.0)
        _steel_moment_factor = _coerce_float(raw.get("STEEL_BEAM_MOMENT_CAPACITY_FACTOR")) or 1.0
        object.__setattr__(self, "steel_beam_moment_capacity_factor", _steel_moment_factor)
        object.__setattr__(
            self,
            "steel_primary_beam_moment_capacity_factor",
            _coerce_float(raw.get("STEEL_PRIMARY_BEAM_MOMENT_CAPACITY_FACTOR")) or _steel_moment_factor,
        )
        object.__setattr__(
            self,
            "steel_secondary_beam_moment_capacity_factor",
            _coerce_float(raw.get("STEEL_SECONDARY_BEAM_MOMENT_CAPACITY_FACTOR")) or _steel_moment_factor,
        )
        object.__setattr__(self, "steel_beam_include_self_weight", _coerce_bool(raw.get("STEEL_BEAM_INCLUDE_SELF_WEIGHT"), default=True))
        object.__setattr__(self, "rc_beam_span_depth_checks", _coerce_bool(raw.get("RC_BEAM_SPAN_DEPTH_CHECKS"), default=True))
        object.__setattr__(self, "rc_beam_max_span_depth_ratio", _coerce_float(raw.get("RC_BEAM_MAX_SPAN_DEPTH_RATIO")) or 12.0)
        object.__setattr__(self, "rc_secondary_beam_max_span_depth_ratio", _coerce_float(raw.get("RC_SECONDARY_BEAM_MAX_SPAN_DEPTH_RATIO")) or 12.0)
        object.__setattr__(self, "fire_resistance_minima", _coerce_bool(raw.get("FIRE_RESISTANCE_MINIMA"), default=True))

        # Per-category load basis: precast/timber → unfactored; CIP/composite/PT → factored
        _default_basis = {
            "cast_in_place": "factored",
            "composite":     "factored",
            "pt":            "factored",
            "precast":       "unfactored",
            "timber":        "unfactored",
            "other":         "factored",
        }
        raw_basis = _mapping_or_empty(raw.get("LOAD_BASIS_BY_CATEGORY"))
        merged_basis = {**_default_basis, **{k.lower(): str(v).lower() for k, v in raw_basis.items()}}
        object.__setattr__(self, "load_basis_by_category", merged_basis)

        object.__setattr__(self, "design_options_extra", _mapping_or_empty(raw.get("DESIGN_OPTIONS")))
        object.__setattr__(self, "analysis_options", _mapping_or_empty(raw.get("ANALYSIS_OPTIONS")))
        object.__setattr__(self, "system_overrides", _mapping_or_empty(raw.get("SYSTEM_OVERRIDES")))
        object.__setattr__(self, "load_overrides", _mapping_or_empty(raw.get("LOAD_OVERRIDES")))

        # ── Imperial → metric unit conversion ─────────────────────────────────
        # All downstream code works in SI (metres, m², m³).  When the control
        # file declares unit=imperial, every dimensional field read from the YAML
        # is in US customary units (feet / ft²) and must be converted here so
        # that no conversion logic is needed elsewhere.
        if str(raw.get("UNIT", "metric")).strip().lower() == "imperial":
            _FT   = 0.3048          # 1 ft  = 0.3048 m
            _FT2  = 0.3048 ** 2     # 1 ft² = 0.092903 m²
            _cvt  = lambda v: v * _FT  if v is not None else None   # length
            _cvta = lambda v: v * _FT2 if v is not None else None   # area

            object.__setattr__(self, "spans",
                               [s * _FT for s in self.spans])
            object.__setattr__(self, "area_per_floor",
                               self.area_per_floor * _FT2)
            object.__setattr__(self, "floor_to_floor_height",
                               _cvt(self.floor_to_floor_height))
            object.__setattr__(self, "ideal_column_spacing",
                               _cvt(self.ideal_column_spacing))
            object.__setattr__(self, "depth_limit",
                               _cvt(self.depth_limit))
            object.__setattr__(self, "one_way_slab_min_span",
                               _cvt(self.one_way_slab_min_span))
            object.__setattr__(self, "one_way_beam_min_span",
                               _cvt(self.one_way_beam_min_span))
            object.__setattr__(self, "edge_cantilever_max",
                               _cvt(self.edge_cantilever_max))

    @classmethod
    def from_path(cls, path: PathLike) -> "ControlFile":
        return cls(parameters_from_control_file(path))

    def to_project_context(self) -> ProjectContext:
        region = _infer_region(self.region, self.unit, self.location)
        code_family = self.code_family or _infer_code_family(self.code_standard, self.unit)

        occupancy_id = self.occupancy_id or (self.program_default.strip() if self.program_default else None)
        if self.one_way_irregular and self.one_way_beam_min_span is not None and self.one_way_slab_min_span is not None:
            span_x = float(self.one_way_beam_min_span)
            span_y = float(self.one_way_slab_min_span)
        else:
            span_x = self._preferred_span()
            span_y = self._preferred_span()
        bay_area = (span_x * span_y) if (span_x is not None and span_y is not None) else None
        geometry = GeometryContext(
            span_x_m=span_x,
            span_y_m=span_y,
            floor_to_floor_m=self.floor_to_floor_height,
            storey_count=self.num_floors,
            bay_area_m2=bay_area,
        )

        design_options = {
            "unit": self.unit,
            "use_csv": self.use_csv,
            "data_dir": self.data_dir,
            "depth_limit_enabled": self.depth_limit_enabled,
            "depth_limit": self.depth_limit,
            "edge_cantilever_max": self.edge_cantilever_max,
            "one_way_irregular": self.one_way_irregular,
            "one_way_slab_min_span": self.one_way_slab_min_span,
            "one_way_beam_min_span": self.one_way_beam_min_span,
            "one_way_orientation": self.one_way_orientation,
            "results_scope": self.results_scope,
            "notes": self.notes,
            "factored_loads": self.factored_loads,
            "load_basis_by_category": dict(self.load_basis_by_category),
            "program": dict(self.program),
            "spans": list(self.spans),
            "area_per_floor": self.area_per_floor,
        }
        design_options.update(self.design_options_extra)

        overrides = {
            "control_file": dict(self._raw),
            "material_set": self.material_set,
            "analysis_options": dict(self.analysis_options),
            "system_overrides": dict(self.system_overrides),
        }
        overrides.update(_load_overrides_from_analysis_options(self.analysis_options))
        overrides.update(self.load_overrides)

        return ProjectContext(
            project_id=self.project_name or None,
            region=region,
            code_family=code_family,
            typology_id=self.typology_id,
            occupancy_id=occupancy_id,
            geometry=geometry,
            design_options=design_options,
            overrides=overrides,
        )

    def _preferred_span(self) -> Optional[float]:
        if self.spans:
            if len(self.spans) == 1:
                return float(self.spans[0])
            return float(sum(self.spans) / len(self.spans))
        if self.ideal_column_spacing is not None:
            return float(self.ideal_column_spacing)
        return None

    def _bay_area(self) -> Optional[float]:
        span = self._preferred_span()
        if span is None:
            return None
        return float(span * span)


def control_file_to_project_context(control_file: Union[ControlFile, Mapping[str, Any], PathLike]) -> ProjectContext:
    if isinstance(control_file, ControlFile):
        return control_file.to_project_context()
    if isinstance(control_file, (str, Path)):
        return ControlFile.from_path(control_file).to_project_context()
    return ControlFile(control_file).to_project_context()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    f = _coerce_float(value)
    return int(f) if f is not None else None


def _strip_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_code_family(value: Any) -> Optional[str]:
    s = _strip_or_none(value)
    if not s:
        return None
    lowered = s.lower()
    if lowered in {"ec", "eurocode", "en", "en1990"}:
        return "EC"
    if lowered in {"asce", "asce7", "asce 7", "us"}:
        return "ASCE"
    return None


def _load_overrides_from_analysis_options(analysis_options: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(analysis_options, Mapping):
        return {}
    nested = analysis_options.get("load_overrides")
    if isinstance(nested, Mapping):
        return dict(nested)
    return {
        key: analysis_options[key]
        for key in ("dead_kpa", "live_kpa", "partition_kpa", "wind_kpa", "seismic_kpa")
        if key in analysis_options
    }


def _infer_region(region: Optional[str], unit: str, location: str) -> str:
    if region:
        r = region.strip().upper()
        if r in {"UK", "US", "EU"}:
            return r
    unit_lower = unit.strip().lower()
    if unit_lower in {"imperial", "us"}:
        return "US"
    location_lower = location.lower()
    if any(token in location_lower for token in ["england", "scotland", "wales", "oxford", "london", "uk", "united kingdom"]):
        return "UK"
    return "EU"


def _infer_code_family(code_standard: Optional[str], unit: str) -> str:
    if code_standard:
        s = code_standard.strip().lower()
        if "asce" in s or "aci" in s or "aisc" in s or "ibc" in s:
            return "ASCE"
        if "euro" in s or s == "ec":
            return "EC"
    if unit.strip().lower() in {"imperial", "us"}:
        return "ASCE"
    return "EC"
