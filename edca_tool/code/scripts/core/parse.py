# src/edca_tool/core/parse.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml

def parameters_from_control_file(path: str) -> Dict[str, Any]:
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

def parse_program(program_data) -> Dict[int, str]:
    """
    Convert PROGRAM block into a per-floor occupancy map.

    Supported input forms for each entry in program_data:
      - dict with keys "floors" and "occupancy":
            {"floors": [3], "occupancy": "Office_US"}
            {"floors": [1, 5], "occupancy": "Office_US"}  # inclusive range
      - string like "1-5: Office_US" or "3:Office_US" (if your YAML stores PROGRAM as a list of strings)

    Returns:
        mapping floor_number -> occupancy label

    Raises:
        ValueError on malformed entries.
    """

    floor_occupancy: Dict[int, str] = {}
    if not program_data:
        return floor_occupancy

    for entry in program_data:
        # make sure entry is a mapping with expected keys
        if not isinstance(entry, dict):
            raise ValueError(f"PROGRAM entries must be dicts, got: {entry!r}")

        floors = entry["floors"]
        occupancy = entry["occupancy"]

        if not isinstance(floors, (list, tuple)):
            raise ValueError(f"'floors' must be a 1- or 2-element list, got: {floors!r}")

        # coerce to ints
        floors_int = [int(f) for f in floors]

        if len(floors_int) == 1:
            floor_occupancy[floors_int[0]] = occupancy
        elif len(floors_int) == 2:
            start, end = floors_int
            for floor in range(start, end + 1):
                floor_occupancy[floor] = occupancy
        else:
            raise ValueError(f"Invalid floors spec: {floors}")

    return floor_occupancy

class ControlFile:
    """
    Lightweight holder for control file values.

    Usage:
        cf = ControlFile.from_path("/path/to/control_file.yaml")
    """

    def __init__(self, inputs: Dict[str, Any]):
        # General Info
        self.data_dir = inputs["DATA_DIR"]
        self.project_name = inputs["PROJECT_NAME"]
        self.location = inputs["LOCATION"]
        self.unit = inputs["UNIT"]
        self.program = parse_program(inputs.get("PROGRAM", []))
        self.notes = inputs["NOTES"]

        # Building Parameters
        self.floor_to_floor_height = inputs['FLOOR_TO_FLOOR_HEIGHT']
        self.num_floors = inputs['NUM_FLOORS']
        self.ideal_column_spacing = inputs['IDEAL_COLUMN_SPACING']
        self.depth_limit_enabled = inputs['DEPTH_LIMIT_ENABLED']
        self.depth_limit = inputs['DEPTH_LIMIT']
        
        # Slab Parameters
        self.one_way_irregular = inputs['ONE_WAY_IRREGULAR']
        self.one_way_slab_min_span = inputs['ONE_WAY_SLAB_MIN_SPAN']
        self.one_way_beam_min_span = inputs['ONE_WAY_BEAM_MIN_SPAN']
        self.one_way_orientation = inputs['ONE_WAY_ORIENTATION']
        self.program_default = inputs['PROGRAM_DEFAULT']
        self.results_scope = inputs['RESULTS_SCOPE']

        # Miscellaneous Parameters
        self.floor_plate = inputs['FLOOR_PLATE']
        self.spans = inputs['SPANS']

        # keep raw dict if you want access to other keys
        self._raw = dict(inputs)
        
    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "ControlFile":
        inputs = parameters_from_control_file(path)
        return cls(inputs)
