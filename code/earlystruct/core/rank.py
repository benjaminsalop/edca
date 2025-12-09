from __future__ import annotations
import pandas as pd

def carbon_first_then_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by carbon_per_m2, then cost_per_m2, then depth_m, then span_m.

    If any rows have feasible==True, only those are considered.
    If none are feasible (or 'feasible' column is missing), fall back to all rows.
    """
    if "feasible" in df.columns:
        ok = df[df["feasible"]].copy()
        if ok.empty:
            ok = df.copy()
    else:
        ok = df.copy()

    if ok.empty:
        return ok

    return ok.sort_values(['carbon_per_m2','cost_per_m2','depth_m','span_m'])


def pareto_min_carbon_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple Pareto front in carbon vs cost.

    If any rows have feasible==True, only those are considered.
    If none are feasible (or 'feasible' column is missing), fall back to all rows.
    """
    if "feasible" in df.columns:
        ok = df[df["feasible"]].copy()
        if ok.empty:
            ok = df.copy()
    else:
        ok = df.copy()

    if ok.empty:
        ok['on_pareto'] = False
        return ok

    ok = ok.sort_values('carbon_per_m2')
    pareto_idx = []
    best_cost = float('inf')
    for i, r in ok.iterrows():
        if r['cost_per_m2'] <= best_cost:
            pareto_idx.append(i)
            best_cost = r['cost_per_m2']
    ok['on_pareto'] = ok.index.isin(pareto_idx)
    return ok