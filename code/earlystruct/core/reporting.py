# earlystruct/core/reporting.py
from __future__ import annotations
import math
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Typology bucketing ----------
def infer_typology(row: pd.Series) -> str:
    c = (str(row.get("category","")) + " " + str(row.get("type",""))).lower()
    if "timber" in c or "clt" in c or "glulam" in c:
        return "Timber"
    if "composite" in c:
        return "Composite"
    if "steel" in c and "deck" in c:
        return "Composite"  # composite steel deck slabs
    if "steel" in c:
        return "Steel"
    if "concrete" in c or "precast" in c or "pt" in c:
        return "Concrete"
    return "Other"

def add_typology(df: pd.DataFrame) -> pd.DataFrame:
    if "typology" not in df.columns:
        df = df.copy()
        df["typology"] = df.apply(infer_typology, axis=1)
    return df

# ---------- Best-per-group selection ----------
def best_per_group(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """
    Among feasible rows, pick the lowest value of `metric` for each group in `group_col`.
    Example group_col: "typology" or "type".
    """
    if df.empty:
        return pd.DataFrame(columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"])

    if group_col.lower() == "typology":
        df = add_typology(df)
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"])

    filt = df[(df["feasible"] == True) & df[metric].notna() & (df[metric] > 0)]
    if filt.empty:
        return pd.DataFrame(columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"])

    idx = filt.groupby(group_col, as_index=False)[metric].idxmin()
    out = filt.loc[idx].copy()
    cols = [group_col, "system_id", "system_name", "category", "type", "manufacturer",
            metric, "depth_m", "span_m"]
    keep = [c for c in cols if c in out.columns]
    return out[keep].sort_values(metric, ascending=True).reset_index(drop=True)

def best_per_typology(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return best_per_group(df, metric, "typology")

def best_per_type(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return best_per_group(df, metric, "type")

# ---------- Pareto (carbon vs cost) ----------
def pareto_mask(df: pd.DataFrame, x_col="carbon_per_m2", y_col="cost_per_m2") -> pd.Series:
    """
    True where a point is Pareto-optimal (minimize both x and y).
    """
    vals = df[[x_col, y_col]].values
    keep = [True] * len(vals)
    for i, (x, y) in enumerate(vals):
        if not math.isfinite(x) or not math.isfinite(y):
            keep[i] = False
            continue
        for j, (x2, y2) in enumerate(vals):
            if j == i:
                continue
            if math.isfinite(x2) and math.isfinite(y2):
                # strictly better in at least one, no worse in the other
                if (x2 <= x and y2 < y) or (x2 < x and y2 <= y):
                    keep[i] = False
                    break
    return pd.Series(keep, index=df.index)

# ---------- Plots ----------
def plot_pareto(df: pd.DataFrame, title="Carbon vs Cost (Pareto)"):
    """
    Scatter plot of carbon_per_m2 vs cost_per_m2 and mark Pareto points.
    One plot only. Uses matplotlib (no custom colors).
    """
    if df.empty:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
        ax.set_ylabel("Cost (per m²)")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return

    feas = df[(df["feasible"] == True)].copy()
    feas = feas[feas["carbon_per_m2"].notna() & feas["cost_per_m2"].notna()]
    if feas.empty:
        feas = df.copy()

    mask = pareto_mask(feas, "carbon_per_m2", "cost_per_m2")
    pareto = feas[mask]
    nonpareto = feas[~mask]

    fig, ax = plt.subplots()
    ax.scatter(nonpareto["carbon_per_m2"], nonpareto["cost_per_m2"], label="Candidates")
    ax.scatter(pareto["carbon_per_m2"], pareto["cost_per_m2"], marker="s", label="Pareto")
    ax.set_title(title)
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_ylabel("Cost (per m²)")
    ax.legend()
    plt.show()

def plot_best_typology_carbon(df: pd.DataFrame, title="Lowest-carbon option by typology"):
    best = best_per_typology(df, "carbon_per_m2")
    fig, ax = plt.subplots()
    if best.empty:
        ax.set_title(title)
        ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
        ax.text(0.5, 0.5, "No feasible options", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return
    ax.barh(best["typology"], best["carbon_per_m2"])
    for i, v in enumerate(best["carbon_per_m2"]):
        ax.text(v, i, f" {v:.0f}", va="center")
    ax.set_title(title)
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_ylabel("Typology")
    plt.tight_layout()
    plt.show()

def plot_best_type_carbon(df: pd.DataFrame, title="Lowest-carbon option by type"):
    best = best_per_type(df, "carbon_per_m2")
    fig, ax = plt.subplots()
    if best.empty:
        ax.set_title(title)
        ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
        ax.text(0.5, 0.5, "No feasible options", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return
    ax.barh(best["type"], best["carbon_per_m2"])
    for i, v in enumerate(best["carbon_per_m2"]):
        ax.text(v, i, f" {v:.0f}", va="center")
    ax.set_title(title)
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_ylabel("Type")
    plt.tight_layout()
    plt.show()

# ---------- Simple table helper ----------
def tables_best_by_typology(df: pd.DataFrame):
    carb = best_per_typology(df, "carbon_per_m2")
    cost = best_per_typology(df, "cost_per_m2")
    return carb, cost
