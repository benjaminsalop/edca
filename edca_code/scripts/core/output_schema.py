from __future__ import annotations

ASSEMBLY_RESULT_COLUMNS = [
    "candidate_id",
    "passed",
    "floor_family_id",
    "floor_variant_id",
    "primary_beam_family_id",
    "primary_beam_variant_id",
    "secondary_beam_family_id",
    "secondary_beam_variant_id",
    "column_family_id",
    "column_variant_id",
    "lateral_family_id",
    "lateral_variant_id",
    "total_cost",
    "total_embodied_carbon",
    "total_penalty",
]

COMPONENT_RESULT_COLUMNS = [
    "candidate_id",
    "component",
    "family_id",
    "variant_id",
    "passed",
    "utilization_max",
    "selected_section",
    "cost",
    "embodied_carbon",
]
