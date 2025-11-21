from __future__ import annotations
import pandas as pd

def carbon_first_then_cost(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df['feasible']].copy()
    if ok.empty:
        return ok
    return ok.sort_values(['carbon_per_m2','cost_per_m2','depth_m','span_m'])

def pareto_min_carbon_cost(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df['feasible']].copy().sort_values('carbon_per_m2')
    pareto_idx = []
    best_cost = float('inf')
    for i, r in ok.iterrows():
        if r['cost_per_m2'] <= best_cost:
            pareto_idx.append(i)
            best_cost = r['cost_per_m2']
    ok['on_pareto'] = ok.index.isin(pareto_idx)
    return ok
