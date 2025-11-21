from __future__ import annotations
import pandas as pd
from .units import load_to_knm2

def required_loads_per_block(program_df: pd.DataFrame, occ_df: pd.DataFrame):
    prog_key = 'use' if 'use' in program_df.columns else 'occupancy_key'
    if 'use' not in occ_df.columns:
        if 'occupancy_key' in occ_df.columns:
            occ_df = occ_df.rename(columns={'occupancy_key':'use'})
        else:
            raise ValueError("occupancies.csv must include 'use' column.")

    df = program_df.merge(occ_df, left_on=prog_key, right_on='use', how='left', suffixes=('','_occ'))

    # Friendly error if any occupancy is missing
    if df['use'].isna().any():
        missing = df[df['use'].isna()][[prog_key]].to_dict('records')
        raise ValueError(f"Program rows missing occupancy matches: {missing}")

    rows = []
    for _, r in df.iterrows():
        unit = r.get('unit','metric')

        # Defensive conversion for floors
        sf = pd.to_numeric(r.get('start_floor'), errors='coerce')
        ef = pd.to_numeric(r.get('end_floor'), errors='coerce')
        if pd.isna(sf) or pd.isna(ef):
            raise ValueError(f"Program row has blank/invalid floors: {r[['start_floor','end_floor',prog_key]].to_dict()}")

        sdl_part = load_to_knm2(r.get('sdl_partition',0.0), unit)
        sdl_extra= load_to_knm2(r.get('sdl',0.0), unit)
        ll      = load_to_knm2(r.get('ll',0.0), unit)

        rows.append({
            "start_floor": int(sf),
            "end_floor": int(ef),
            "use": r['use'],
            "req_sdl_non_swt_knm2": sdl_part + sdl_extra,
            "req_ll_knm2": ll,
            "code": r.get('code',''),
            "notes": r.get('notes','')
        })
    return pd.DataFrame(rows)
