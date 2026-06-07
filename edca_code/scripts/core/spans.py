"""Compatibility wrapper for the legacy dataframe span-sweep pipeline."""

from __future__ import annotations

from edca_code.scripts.core.old import carbon as _carbon
from edca_code.scripts.core.old import rank as _rank
from edca_code.scripts.core.old import spans as _spans
from edca_code.scripts.core.old import systems as _systems
from edca_code.scripts.core.old import takeoff as _takeoff
from edca_code.scripts.core.old import utils as _utils


_spans.systems_mod = _systems
_spans.takeoff_mod = _takeoff
_spans.carbon_mod = _carbon
_spans.rank_mod = _rank
_spans.filter_systems = _systems.filter_systems
_spans.infer_type = _utils.infer_type
_spans.reorder_output_columns = _utils.reorder_output_columns
_spans.ensure_dir = _utils.ensure_dir

resolve_span_values = _spans.resolve_span_values
compute_candidates_for_span = _spans.compute_candidates_for_span
aggregate_span_results = _spans.aggregate_span_results
run_span_sweep = _spans.run_span_sweep
expand_winners_and_write_materials_per_floor = (
    _spans.expand_winners_and_write_materials_per_floor
)

__all__ = [
    "resolve_span_values",
    "compute_candidates_for_span",
    "aggregate_span_results",
    "run_span_sweep",
    "expand_winners_and_write_materials_per_floor",
]
