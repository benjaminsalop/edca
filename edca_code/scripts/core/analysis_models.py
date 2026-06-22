from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domain_models import EvaluatedLoadCombination, LoadPathMethod


@dataclass(slots=True)
class ForceEffects:
    axial: float | None = None
    shear_major: float | None = None
    shear_minor: float | None = None
    moment_major: float | None = None
    moment_minor: float | None = None
    torsion: float | None = None


@dataclass(slots=True)
class DemandEnvelope:
    governing_combination_id: str | None = None
    effects: ForceEffects = field(default_factory=ForceEffects)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FloorPanelDemand:
    tributary_area_m2: float | None = None
    unfactored_area_load_kpa: float | None = None
    factored_area_load_kpa: float | None = None
    deflection_limit_ratio: float | None = None
    vibration_category: str | None = None
    envelope: DemandEnvelope = field(default_factory=DemandEnvelope)


@dataclass(slots=True)
class BeamDemand:
    role: str
    span_m: float | None = None
    tributary_width_m: float | None = None
    unfactored_line_load_kn_per_m: float | None = None
    factored_line_load_kn_per_m: float | None = None
    point_loads_kn: list[float] = field(default_factory=list)
    envelope: DemandEnvelope = field(default_factory=DemandEnvelope)


@dataclass(slots=True)
class ColumnDemand:
    storey: int | None = None
    tributary_area_m2: float | None = None
    axial_dead_kn: float | None = None
    axial_live_kn: float | None = None
    effective_length_m: float | None = None
    envelope: DemandEnvelope = field(default_factory=DemandEnvelope)


@dataclass(slots=True)
class WallDemand:
    storey: int | None = None
    tributary_width_m: float | None = None
    axial_kn: float | None = None
    in_plane_shear_kn: float | None = None
    out_of_plane_pressure_kpa: float | None = None
    envelope: DemandEnvelope = field(default_factory=DemandEnvelope)


@dataclass(slots=True)
class StoreyForces:
    storey: int
    elevation_m: float | None = None
    seismic_base_share_kn: float | None = None
    wind_shear_kn: float | None = None
    drift_ratio: float | None = None
    torsional_moment_knm: float | None = None


@dataclass(slots=True)
class LateralDemand:
    total_base_shear_kn: float | None = None
    overturning_moment_knm: float | None = None
    accidental_torsion_eccentricity_m: float | None = None
    storey_forces: list[StoreyForces] = field(default_factory=list)
    envelope: DemandEnvelope = field(default_factory=DemandEnvelope)


@dataclass(slots=True)
class TorsionalDemand:
    source: str
    design_eccentricity_m: float | None = None
    torsional_moment_knm: float | None = None
    amplification_factor: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssemblyAnalysisInput:
    candidate_id: str
    load_path_method: LoadPathMethod
    load_combinations: list[EvaluatedLoadCombination] = field(default_factory=list)
    floor: FloorPanelDemand | None = None
    primary_beam: BeamDemand | None = None
    secondary_beam: BeamDemand | None = None
    column: ColumnDemand | None = None
    wall: WallDemand | None = None
    lateral: LateralDemand | None = None
    torsion: TorsionalDemand | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
