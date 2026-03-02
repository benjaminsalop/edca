# edca_code/scripts/core/parse.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

def parameters_from_control_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse the YAML control file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Control file not found: {p}")

    with p.open("r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML control file: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Control file must parse to a mapping/dict, got: {type(data).__name__}")
    return data


def parse_program(program_data: Any) -> Dict[int, str]:
    """
    Convert PROGRAM list into a per-floor occupancy map.

    Expected entry form (as in your sample YAML):
      - floors: [0] or [1,3]
        occupancy: laboratory
    """
    floor_occupancy: Dict[int, str] = {}
    if not program_data:
        return floor_occupancy

    if not isinstance(program_data, list):
        raise ValueError(f"PROGRAM must be a list of entries, got: {type(program_data).__name__}")

    for entry in program_data:
        if not isinstance(entry, dict):
            raise ValueError(f"PROGRAM entries must be dicts, got: {entry!r}")

        if "floors" not in entry or "occupancy" not in entry:
            raise ValueError(f"PROGRAM entry missing 'floors' or 'occupancy': {entry!r}")

        floors = entry["floors"]
        occupancy = str(entry["occupancy"]).strip()

        if not isinstance(floors, (list, tuple)):
            raise ValueError(f"'floors' must be a 1- or 2-element list, got: {floors!r}")

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
            raise ValueError(f"Invalid floors spec (need 1 or 2 ints): {floors!r}")

    return floor_occupancy

def parse_floor_plate_area(floor_plate: Any) -> float:
    """
    Returns area_per_floor from FLOOR_PLATE.
    If mode=dims and length+width provided, uses length*width.
    Otherwise uses area_per_floor.
    """
    if not isinstance(floor_plate, dict):
        return 0.0

    mode = str(floor_plate.get("mode", "area")).strip().lower()

    if mode.startswith("dim"):
        length = floor_plate.get("length")
        width = floor_plate.get("width")
        if length is not None and width is not None:
            try:
                return float(length) * float(width)
            except Exception:
                logger.debug("Failed length*width computation; falling back to area_per_floor", exc_info=True)

    # default / fallback
    apf = floor_plate.get("area_per_floor")
    try:
        return float(apf) if apf is not None else 0.0
    except Exception:
        return 0.0

def parse_spans(spans_block: Any) -> List[float]:
    """
    Produce the list of spans to consider based on SPANS block.
    Supports:
       mode: range with optional sweep+step
       mode: list with explicit list
    """
    def _as_float_list(x: Any) -> List[float]:
        if x is None:
            return []
        if isinstance(x, (int, float)):
            return [float(x)]
        if isinstance(x, str):
            try:
                return [float(x)]
            except Exception:
                return []
        if isinstance(x, (list, tuple)):
            out: List[float] = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    pass
            return out
        return []

    # Allow SPANS: 9 (scalar) as a shorthand
    if not isinstance(spans_block, dict):
        return _as_float_list(spans_block)

    mode = str(spans_block.get("mode", "range")).strip().lower()

    if mode == "list":
        # Accept list:, values:, or (legacy/mistyped) range: in list-mode.
        vals = spans_block.get("list", None)
        if vals is None:
            vals = spans_block.get("values", None)
        if vals is None:
            vals = spans_block.get("range", None)
        return _as_float_list(vals)

     # default: range
    r = spans_block.get("range", None)
    if r is None:
        r = spans_block.get("values", None)
    if r is None:
        r = spans_block.get("list", None)

    # Allow range: 9 or range: [9] meaning “single span”
    if isinstance(r, (int, float, str)):
        return _as_float_list(r)
    if isinstance(r, (list, tuple)):
        if len(r) == 1:
            return _as_float_list(r[0])
        if len(r) != 2:
            # If someone puts more than 2 entries, treat it as an explicit list.
            return _as_float_list(r)
    else:
        return []

    mn, mx = float(r[0]), float(r[1])
    sweep = bool(spans_block.get("sweep", False))
    step = spans_block.get("step", 1.0)
    step = float(step) if step else 1.0

    # If bounds are equal (or sweep is off), treat as single span
    if abs(mx - mn) < 1e-9:
        return [round(mn, 6)]
    if not sweep:
        return [round(mn, 6), round(mx, 6)]

    # inclusive sweep with floating tolerance
    out: List[float] = []
    x = mn
    while x <= mx + 1e-9:
        out.append(round(x, 6))
        x += step
    return out

def build_floor_area_lookup(
    area_per_floor: float,
    floors_by_case: Dict[str, Iterable[int]]) -> Tuple[Dict[int, float], float]:
    """Build a dict floor->area for all floors mentioned in floors_by_case."""
    floor_area_lookup: Dict[int, float] = {}
    for _case, floors in (floors_by_case or {}).items():
        for f in floors:
            floor_area_lookup[int(f)] = float(area_per_floor)
    return floor_area_lookup, float(area_per_floor)

class ControlFile:
    """Lightweight holder for control file values."""

    def __init__(self, inputs: Dict[str, Any]):
        # Raw
        self._raw = dict(inputs)

        # Mode / sources
        self.use_csv = bool(inputs.get("USE_CSV", False))
        self.data_dir = inputs.get("DATA_DIR", "inputs")

        # Project data
        self.project_name = inputs.get("PROJECT_NAME", "")
        self.location = inputs.get("LOCATION", "")
        self.unit = inputs.get("UNIT", "metric")

        self.floor_plate = inputs.get("FLOOR_PLATE", {}) or {}
        self.area_per_floor = parse_floor_plate_area(self.floor_plate)

        self.floor_to_floor_height = inputs.get("FLOOR_TO_FLOOR_HEIGHT", None)
        self.num_floors = inputs.get("NUM_FLOORS", None)
        self.ideal_column_spacing = inputs.get("IDEAL_COLUMN_SPACING", None)

        self.depth_limit_enabled = bool(inputs.get("DEPTH_LIMIT_ENABLED", False))
        self.depth_limit = inputs.get("DEPTH_LIMIT", None)

        # Spans
        self.spans_block = inputs.get("SPANS", {}) or {}
        self.spans = parse_spans(self.spans_block)
        self.edge_cantilever_max = self.spans_block.get("edge_cantilever_max", None)

        # One-way slab logic
        self.one_way_irregular = bool(inputs.get("ONE_WAY_IRREGULAR", False))
        self.one_way_slab_min_span = inputs.get("ONE_WAY_SLAB_MIN_SPAN", None)
        self.one_way_beam_min_span = inputs.get("ONE_WAY_BEAM_MIN_SPAN", None)
        self.one_way_orientation = inputs.get("ONE_WAY_ORIENTATION", None)

        # Program
        self.program_default = inputs.get("PROGRAM_DEFAULT", "office")
        self.program = parse_program(inputs.get("PROGRAM", []))

        # Reporting
        self.results_scope = inputs.get("RESULTS_SCOPE", "PER_SYSTEM")
        self.notes = inputs.get("NOTES", "")

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "ControlFile":
        return cls(parameters_from_control_file(path))