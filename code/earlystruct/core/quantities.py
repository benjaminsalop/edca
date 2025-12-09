# earlystruct/core/quantities.py
from __future__ import annotations

def totals_from_intensity(per_m2: dict, area_m2: float) -> dict:
    """
    Multiply per-m² intensities by total area.

    Accepts keys like:
      - concrete_m3_per_m2, concrete_volume, ...
      - steel_m3_per_m2, steel_volume, ...
      - timber_m3_per_m2, timber_volume, ...

    Returns:
      - concrete_m3, steel_m3, timber_m3, depth_m
    """
    def _first(d: dict, keys, default: float = 0.0) -> float:
        for k in keys:
            if k in d:
                v = d.get(k)
                if v is None or v == "":
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
        return float(default)

    conc_per_m2   = _first(per_m2, ["concrete_m3_per_m2", "concrete_volume", "concrete_m3", "concrete"], 0.0)
    steel_per_m2  = _first(per_m2, ["steel_m3_per_m2", "steel_volume", "steel_m3", "steel"], 0.0)
    timber_per_m2 = _first(per_m2, ["timber_m3_per_m2", "timber_volume", "timber_m3", "timber"], 0.0)

    depth_m = float(per_m2.get("depth", 0.0) or 0.0)

    return dict(
        concrete_m3 = conc_per_m2   * area_m2,
        steel_m3    = steel_per_m2  * area_m2,
        timber_m3   = timber_per_m2 * area_m2,
        depth_m     = depth_m,
    )


# NEW: simple parametric frame model for concrete beams/columns
def frame_concrete_intensities_per_m2(
    span_x_m: float,
    span_y_m: float,
    floor_height_m: float,
    slab_type: str = "two_way",   # "two_way" or "one_way"
    beam_width_m: float = 0.20,   # ~8 in
    beam_slenderness: float = 20.0,  # L/d limit
    beam_depth_max_m: float = 0.80,  # max allowed beam depth
    column_width_m: float = 0.30, # ~12 in
    column_depth_m: float = 0.30, # ~12 in
) -> dict:
    """
    Very simple frame model to get BEAM and COLUMN concrete volumes per m².

    Returns:
      {
        "concrete_volume_beams_m3_per_m2": ...,
        "concrete_volume_columns_m3_per_m2": ...,
        "beam_depth_m": ...,
        "beam_depth_required_m": ...,
        "beam_ok": True/False,
      }

    Assumptions:
      - One column "belongs" to each bay (each column shared by 4 bays in a grid).
      - Beams along the perimeter only.
      - Beam depth d ~ L / beam_slenderness, with an upper cap beam_depth_max_m.
      - For slab_type="one_way", beams are only sized on the major span direction;
        for "two_way", use the max(span_x, span_y).
    """
    area = float(span_x_m) * float(span_y_m)
    if area <= 0.0 or floor_height_m <= 0.0:
        return {
            "concrete_volume_beams_m3_per_m2": 0.0,
            "concrete_volume_columns_m3_per_m2": 0.0,
            "beam_depth_m": 0.0,
            "beam_depth_required_m": 0.0,
            "beam_ok": False,
        }

    # Columns: 1 column "per bay" (each shared by 4 bays)
    column_area = column_width_m * column_depth_m
    v_columns_m3_per_m2 = (column_area * floor_height_m) / area

    # Beam governing span
    if str(slab_type).lower().startswith("one"):
        # For one-way, beams in one direction only; size by that span
        L_beam = max(span_x_m, span_y_m)
    else:
        # Two-way: perimeter beams sized by max span
        L_beam = max(span_x_m, span_y_m)

    # Required depth from slenderness
    if beam_slenderness <= 0.0:
        beam_slenderness = 20.0

    d_beam_req = L_beam / beam_slenderness
    beam_ok = d_beam_req <= beam_depth_max_m

    d_beam_use = min(d_beam_req, beam_depth_max_m)
    beam_area_section = beam_width_m * d_beam_use

    # Perimeter beams
    perimeter = 2.0 * (span_x_m + span_y_m)
    v_beams_m3_per_m2 = (beam_area_section * perimeter) / area

    return {
        "concrete_volume_beams_m3_per_m2": v_beams_m3_per_m2,
        "concrete_volume_columns_m3_per_m2": v_columns_m3_per_m2,
        "beam_depth_m": d_beam_use,
        "beam_depth_required_m": d_beam_req,
        "beam_ok": beam_ok,
    }
