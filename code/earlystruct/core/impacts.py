from __future__ import annotations

def _get(series, key, default=None):
    v = series.get(key) if hasattr(series, 'get') else series[key] if key in series else None
    return default if (v is None or v == '') else v


def calc_impacts_cost(qty: dict, mat_ids: dict, mats_by_id: dict):
    """
    Return (carbon_total_kg, cost_total, breakdown dict).

    Quantities (qty) can contain:
      - concrete_m3 (total)  OR
      - concrete_slab_m3, concrete_beams_m3, concrete_columns_m3

      - steel_kg (kg) or steel_m3 (m^3) if density available
      - timber_m3 (m^3)

    breakdown will include:
      - 'concrete_slab', 'concrete_beams', 'concrete_columns' where possible
      - otherwise a single 'concrete'
      - 'steel', 'timber'
    """
    conc_id  = mat_ids.get('material_concrete_id') or ''
    steel_id = mat_ids.get('material_steel_id') or ''
    timber_id= mat_ids.get('material_timber_id') or ''
    pt_id    = mat_ids.get('material_pt_id') or ''  # currently no qty; placeholder

    breakdown = {}
    carbon_total = 0.0
    cost_total   = 0.0

    # ---------- Concrete (slab / beams / columns) ----------
    if conc_id and conc_id in mats_by_id:
        m = mats_by_id[conc_id]
        dens = float(_get(m, 'density_kg_per_m3', 2400) or 2400)

        def _concrete_impacts(vol_m3: float):
            if vol_m3 <= 0.0:
                return 0.0, 0.0, 0.0
            mass_kg = vol_m3 * dens
            a1a3 = (float(_get(m, 'ec_a1a3_per_m3', 0.0) or 0.0)) * vol_m3
            a4   = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) \
                   * (mass_kg / 1000.0) * float(_get(m, 'transport_km', 0.0) or 0.0)
            a5   = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * mass_kg
            cost = (float(_get(m, 'cost_per_m3', 0.0) or 0.0)) * vol_m3
            carbon = a1a3 + a4 + a5
            return mass_kg, carbon, cost

        slab_v   = float(qty.get('concrete_slab_m3', 0.0) or 0.0)
        beams_v  = float(qty.get('concrete_beams_m3', 0.0) or 0.0)
        cols_v   = float(qty.get('concrete_columns_m3', 0.0) or 0.0)
        total_v  = float(qty.get('concrete_m3', 0.0) or 0.0)

        # If we have per-component volumes, use them
        if slab_v > 0.0 or beams_v > 0.0 or cols_v > 0.0:
            if slab_v > 0.0:
                slab_mass, slab_carbon, slab_cost = _concrete_impacts(slab_v)
                carbon_total += slab_carbon
                cost_total   += slab_cost
                breakdown['concrete_slab'] = {
                    'vol_m3': slab_v,
                    'mass_kg': slab_mass,
                    'carbon_kg': slab_carbon,
                    'cost': slab_cost,
                }

            if beams_v > 0.0:
                beam_mass, beam_carbon, beam_cost = _concrete_impacts(beams_v)
                carbon_total += beam_carbon
                cost_total   += beam_cost
                breakdown['concrete_beams'] = {
                    'vol_m3': beams_v,
                    'mass_kg': beam_mass,
                    'carbon_kg': beam_carbon,
                    'cost': beam_cost,
                }

            if cols_v > 0.0:
                col_mass, col_carbon, col_cost = _concrete_impacts(cols_v)
                carbon_total += col_carbon
                cost_total   += col_cost
                breakdown['concrete_columns'] = {
                    'vol_m3': cols_v,
                    'mass_kg': col_mass,
                    'carbon_kg': col_carbon,
                    'cost': col_cost,
                }

            # If concrete_m3 is larger than sum of parts, treat the remainder as generic concrete
            other_v = total_v - (slab_v + beams_v + cols_v)
            if other_v > 1e-9:
                o_mass, o_carbon, o_cost = _concrete_impacts(other_v)
                carbon_total += o_carbon
                cost_total   += o_cost
                breakdown['concrete_other'] = {
                    'vol_m3': other_v,
                    'mass_kg': o_mass,
                    'carbon_kg': o_carbon,
                    'cost': o_cost,
                }

        else:
            # No per-component split → fall back to treating all concrete as one bucket
            vol = total_v
            if vol > 0.0:
                mass_kg, carbon_conc, cost = _concrete_impacts(vol)
                carbon_total += carbon_conc
                cost_total   += cost
                breakdown['concrete'] = {
                    'vol_m3': vol,
                    'mass_kg': mass_kg,
                    'carbon_kg': carbon_conc,
                    'cost': cost,
                }

    # ---------- Steel (structural) ----------
    if steel_id and steel_id in mats_by_id:
        m = mats_by_id[steel_id]
        steel_kg = float(qty.get('steel_kg', 0.0) or 0.0)
        if steel_kg <= 0.0:
            # Convert from m3 if provided
            steel_m3 = float(qty.get('steel_m3', 0.0) or 0.0)
            dens = float(_get(m, 'density_kg_per_m3', 7850) or 7850)
            steel_kg = steel_m3 * dens
        if steel_kg > 0.0:
            a1a3 = (float(_get(m, 'ec_a1a3_per_kg', 0.0) or 0.0)) * steel_kg
            a4 = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) \
                 * (steel_kg/1000.0) * float(_get(m,'transport_km',0.0) or 0.0)
            a5 = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * steel_kg
            cost = (float(_get(m, 'cost_per_kg', 0.0) or 0.0)) * steel_kg
            carbon_steel = a1a3 + a4 + a5
            carbon_total += carbon_steel
            cost_total   += cost
            breakdown['steel'] = {
                'mass_kg': steel_kg,
                'carbon_kg': carbon_steel,
                'cost': cost,
            }

    # ---------- Timber ----------
    if timber_id and timber_id in mats_by_id:
        m = mats_by_id[timber_id]
        vol = float(qty.get('timber_m3', 0.0) or 0.0)
        if vol > 0.0:
            dens = float(_get(m, 'density_kg_per_m3', 500) or 500)
            mass_kg = vol * dens
            a1a3 = (float(_get(m, 'ec_a1a3_per_m3', 0.0) or 0.0)) * vol
            a4 = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) \
                 * (mass_kg/1000.0) * float(_get(m,'transport_km',0.0) or 0.0)
            a5 = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * mass_kg
            cost = (float(_get(m, 'cost_per_m3', 0.0) or 0.0)) * vol
            carbon_timber = a1a3 + a4 + a5
            carbon_total += carbon_timber
            cost_total   += cost
            breakdown['timber'] = {
                'vol_m3': vol,
                'mass_kg': mass_kg,
                'carbon_kg': carbon_timber,
                'cost': cost,
            }

    # PT steel placeholder: add when quantities available

    return carbon_total, cost_total, breakdown