"""structural_class.py — Human-readable structural classification labels.

Classification uses the per-component category fields
(floor_category, beam_category, column_category, lateral_category) that come
from the family metadata.  material_family is used only as a fallback when
the category field is absent.

Label format:  "<Floor label> · <Frame label>"

Examples
--------
  Precast Hollowcore · Steel Frame
  Composite Deck · Steel Frame
  CIP Flat Plate · RC Frame
  CIP Waffle Slab · RC Frame
  CLT Floor · Timber Frame
  CLT Floor · Steel Frame
"""
from __future__ import annotations

from typing import Any

from .domain_models import ComponentType, SystemFamily

# re-exported so callers only need one import
__all__ = [
    "infer_structural_class",
    "build_assembly_label",
    "structural_sort_key",
]

# ---------------------------------------------------------------------------
# Floor-category → short display name
# ---------------------------------------------------------------------------
_FLOOR_CATEGORY: dict[str, str] = {
    "cast_in_place": "CIP",
    "precast": "Precast",
    "composite": "Composite",
    "timber": "Timber",
    "clt": "CLT",
    "lvl": "LVL",
}

# Floor-type → short display name (appended after category when informative)
_FLOOR_TYPE: dict[str, str] = {
    "solid_slab": "Solid Slab",
    "two_way_slab": "2-Way Slab",
    "flat_plate": "Flat Plate",
    "flat_slab": "Flat Slab",
    "flat_slab_drop_panel": "Drop-Panel Slab",
    "joist_slab": "Joist Slab",
    "waffle_slab": "Waffle Slab",
    "hollowcore": "Hollowcore",
    "hollow_core": "Hollowcore",
    "double_tee": "Double Tee",
    "beam_block": "Beam & Block",
    "beam_and_block": "Beam & Block",
    "solid_plank": "Solid Plank",
    "plank": "Solid Plank",
    "thermal_floor": "Thermal Floor",
    "composite_deck": "Composite Deck",
    "clt_floor": "CLT",
    "lvl_panel": "LVL Panel",
    "pt_flat_slab": "PT Flat Slab",
    "pt_slab": "PT Flat Slab",
}

# ---------------------------------------------------------------------------
# Frame-component category → frame label
# ---------------------------------------------------------------------------
_COLUMN_CATEGORY: dict[str, str] = {
    "steel_section": "Steel",
    "steel": "Steel",
    "rc_column": "RC",
    "cast_in_place": "RC",
    "precast_column": "Precast RC",
    "precast": "Precast RC",
    "timber_column": "Timber",
    "timber": "Timber",
    "composite_column": "Composite",
    "glulam": "Glulam",
    "clt": "CLT",
}

_BEAM_CATEGORY: dict[str, str] = {
    "steel_section": "Steel",
    "steel": "Steel",
    "rc_beam": "RC",
    "cast_in_place": "RC",
    "precast_beam": "Precast RC",
    "precast": "Precast RC",
    "timber_beam": "Timber",
    "timber": "Timber",
    "glulam": "Glulam",
    "composite_beam": "Composite Steel",
    "composite": "Composite Steel",
}

_MATERIAL_FAMILY: dict[str, str] = {
    "steel": "Steel",
    "concrete": "RC",
    "precast": "Precast RC",
    "timber": "Timber",
    "hybrid": "Hybrid",
    "cast_in_place": "RC",
    "composite": "Composite Steel",
}

# ---------------------------------------------------------------------------
# Sort-key groups — used by the visualiser to cluster bars
# ---------------------------------------------------------------------------
_GROUP_ORDER: list[str] = ["Timber", "Composite", "Precast", "CIP"]


def infer_structural_class(
    floor_family: SystemFamily | None,
    primary_beam_family: SystemFamily | None,
    column_family: SystemFamily | None,
) -> str:
    """Return a human-readable structural class label for an assembly.

    Uses *_category metadata fields, falling back to material_family.
    """
    floor_label = _floor_label(floor_family)
    frame_label = _frame_label(primary_beam_family, column_family)

    if frame_label:
        return f"{floor_label} · {frame_label} Frame"
    return floor_label


def structural_sort_key(label: str) -> tuple[int, str]:
    """Sort key for structural class labels — groups then alpha within group."""
    for rank, prefix in enumerate(_GROUP_ORDER):
        if label.startswith(prefix):
            return (rank, label)
    return (len(_GROUP_ORDER), label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_label(family: SystemFamily | None) -> str:
    if family is None:
        return "Unknown Floor"
    meta = family.metadata

    category = _norm(meta.get("floor_category")) or _norm(family.material_family)
    floor_type = _norm(meta.get("floor_type"))

    cat_label = _FLOOR_CATEGORY.get(category or "", "")
    type_label = _FLOOR_TYPE.get(floor_type or "", "")

    # Avoid redundancy: "Composite Composite Deck" → "Composite Deck"
    if type_label and cat_label and type_label.lower().startswith(cat_label.lower()):
        return type_label
    if type_label and cat_label:
        return f"{cat_label} {type_label}"
    return type_label or cat_label or family.family_id


def _frame_label(
    beam_family: SystemFamily | None,
    column_family: SystemFamily | None,
) -> str:
    """Return the frame material label (e.g. 'Steel', 'RC', 'Steel Beam + RC Col').

    When beam and column are the same material → short label ("Steel Frame").
    When they differ → hybrid label ("Steel Beam + RC Col Frame").
    """
    col_label = ""
    if column_family is not None:
        cat = _norm(column_family.metadata.get("column_category"))
        col_label = _COLUMN_CATEGORY.get(cat or "", "")
        if not col_label:
            col_label = _MATERIAL_FAMILY.get(_norm(column_family.material_family) or "", "")

    beam_label = ""
    if beam_family is not None:
        cat = _norm(beam_family.metadata.get("beam_category"))
        beam_label = _BEAM_CATEGORY.get(cat or "", "")
        if not beam_label:
            beam_label = _MATERIAL_FAMILY.get(_norm(beam_family.material_family) or "", "")

    if col_label and beam_label and col_label != beam_label:
        return f"{beam_label} Beam + {col_label} Col"
    return col_label or beam_label or ""


def build_assembly_label(
    floor_family: SystemFamily | None,
    primary_beam_family: SystemFamily | None,
    column_family: SystemFamily | None,
    secondary_beam_family: SystemFamily | None = None,
) -> str:
    """Verbose comma-separated assembly label for CSV output.

    Examples
    --------
    "CLT Floor, Steel Primary Beams, Steel Secondary Beams, RC Columns"
    "Precast Hollowcore Floor, Steel Primary Beams, RC Columns"
    "CIP Flat Plate Floor, RC Columns"
    """
    parts: list[str] = []
    parts.append(_verbose_floor_label(floor_family))

    if primary_beam_family is not None:
        mat = _material_label_from_family(primary_beam_family, "beam")
        parts.append(f"{mat} Primary Beams")

    if secondary_beam_family is not None:
        mat = _material_label_from_family(secondary_beam_family, "beam")
        parts.append(f"{mat} Secondary Beams")

    if column_family is not None:
        mat = _material_label_from_family(column_family, "column")
        parts.append(f"{mat} Columns")

    return ", ".join(parts)


def _verbose_floor_label(family: SystemFamily | None) -> str:
    """Return floor label with 'Floor' suffix, e.g. 'CLT Floor', 'Precast Hollowcore Floor'."""
    short = _floor_label(family)
    if not short or short.lower().endswith("floor"):
        return short or "Unknown Floor"
    return f"{short} Floor"


def _material_label_from_family(family: Any, component: str) -> str:
    """Return material short-name ('Steel', 'RC', 'Timber' …) for beam or column family."""
    if family is None:
        return "Unknown"
    meta: dict[str, Any] = getattr(family, "metadata", {}) or {}
    mat_fam = _norm(getattr(family, "material_family", None))
    if component == "beam":
        cat = _norm(meta.get("beam_category"))
        label = _BEAM_CATEGORY.get(cat or "", "")
        if not label:
            label = _MATERIAL_FAMILY.get(mat_fam or "", "")
    elif component == "column":
        cat = _norm(meta.get("column_category"))
        label = _COLUMN_CATEGORY.get(cat or "", "")
        if not label:
            label = _MATERIAL_FAMILY.get(mat_fam or "", "")
    else:
        label = _MATERIAL_FAMILY.get(mat_fam or "", "")
    return label or "Unknown"


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s and s not in {"none", "nan", "na"} else None
