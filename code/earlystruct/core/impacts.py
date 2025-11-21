
from __future__ import annotations

def _get(series, key, default=None):
    v = series.get(key) if hasattr(series, 'get') else series[key] if key in series else None
    return default if (v is None or v == '') else v

def calc_impacts_cost(qty: dict, mat_ids: dict, mats_by_id: dict):
    """Return (carbon_total_kg, cost_total, breakdown dict).

    Accepts quantities in:
      - concrete_m3 (m^3)
      - timber_m3 (m^3)
      - steel_kg (kg) or steel_m3 (m^3) if density available
    """
    conc_id = mat_ids.get('material_concrete_id') or ''
    steel_id= mat_ids.get('material_steel_id') or ''
    timber_id=mat_ids.get('material_timber_id') or ''
    pt_id   = mat_ids.get('material_pt_id') or ''  # currently no qty; placeholder

    breakdown = {}
    carbon_total = 0.0
    cost_total   = 0.0

    # Concrete
    if conc_id and conc_id in mats_by_id:
        m = mats_by_id[conc_id]
        vol = float(qty.get('concrete_m3', 0.0) or 0.0)
        dens = float(_get(m, 'density_kg_per_m3', 2400) or 2400)
        mass_kg = vol * dens
        a1a3 = (float(_get(m, 'ec_a1a3_per_m3', 0.0) or 0.0)) * vol
        a4 = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) * (mass_kg/1000.0) * float(_get(m,'transport_km',0.0) or 0.0)
        a5 = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * mass_kg
        cost = (float(_get(m, 'cost_per_m3', 0.0) or 0.0)) * vol
        carbon_conc = a1a3 + a4 + a5
        carbon_total += carbon_conc; cost_total += cost
        breakdown['concrete'] = {'vol_m3':vol, 'mass_kg':mass_kg, 'carbon_kg':carbon_conc, 'cost':cost}

    # Steel (structural)
    if steel_id and steel_id in mats_by_id:
        m = mats_by_id[steel_id]
        steel_kg = float(qty.get('steel_kg', 0.0) or 0.0)
        if steel_kg <= 0.0:
            # Convert from m3 if provided
            steel_m3 = float(qty.get('steel_m3', 0.0) or 0.0)
            dens = float(_get(m, 'density_kg_per_m3', 7850) or 7850)
            steel_kg = steel_m3 * dens
        a1a3 = (float(_get(m, 'ec_a1a3_per_kg', 0.0) or 0.0)) * steel_kg
        a4 = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) * (steel_kg/1000.0) * float(_get(m,'transport_km',0.0) or 0.0)
        a5 = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * steel_kg
        cost = (float(_get(m, 'cost_per_kg', 0.0) or 0.0)) * steel_kg
        carbon_steel = a1a3 + a4 + a5
        carbon_total += carbon_steel; cost_total += cost
        breakdown['steel'] = {'mass_kg':steel_kg, 'carbon_kg':carbon_steel, 'cost':cost}

    # Timber
    if timber_id and timber_id in mats_by_id:
        m = mats_by_id[timber_id]
        vol = float(qty.get('timber_m3', 0.0) or 0.0)
        dens = float(_get(m, 'density_kg_per_m3', 500) or 500)
        mass_kg = vol * dens
        a1a3 = (float(_get(m, 'ec_a1a3_per_m3', 0.0) or 0.0)) * vol
        a4 = (float(_get(m, 'ec_a4_per_ton_km', 0.0) or 0.0)) * (mass_kg/1000.0) * float(_get(m,'transport_km',0.0) or 0.0)
        a5 = (float(_get(m, 'ec_a5_per_kg', 0.0) or 0.0)) * mass_kg
        cost = (float(_get(m, 'cost_per_m3', 0.0) or 0.0)) * vol
        carbon_timber = a1a3 + a4 + a5
        carbon_total += carbon_timber; cost_total += cost
        breakdown['timber'] = {'vol_m3':vol, 'mass_kg':mass_kg, 'carbon_kg':carbon_timber, 'cost':cost}

    # PT steel placeholder: add when quantities available

    return carbon_total, cost_total, breakdown
