# edca_code/scripts/core/reporting.py
from __future__ import annotations
import math
from typing import Optional, Sequence, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------
# Utilities
# ------------------------
def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce to numeric, keeping NaN where coercion fails."""
    return pd.to_numeric(series, errors="coerce")


def add_typology(df: pd.DataFrame, typology_col: str = "typology") -> pd.DataFrame:
    """
    Add a simple typology column inferred from 'category' and 'type' if not present.
    Rules (simple): Timber, Composite, Steel, Concrete, Other.
    """
    if typology_col in df.columns:
        return df

    def _infer(row):
        c = f"{row.get('category','')} {row.get('type','')}".lower()
        if any(k in c for k in ("timber", "clt", "glulam")):
            return "Timber"
        if "composite" in c:
            return "Composite"
        if "steel" in c and "deck" in c:
            return "Composite"
        if "steel" in c:
            return "Steel"
        if any(k in c for k in ("concrete", "precast", "pt", "post-tension")):
            return "Concrete"
        return "Other"

    df = df.copy()
    df[typology_col] = df.apply(_infer, axis=1)
    return df


def _best_row_by_metric(df: pd.DataFrame, metric: str, feasible_col: Optional[str] = "feasible") -> Optional[pd.Series]:
    """
    Return the single best row (Series) selected by metric ascending.
    Prefer rows marked feasible; otherwise take the best numeric metric row.
    """
    if df is None or df.empty:
        return None
    d = df.copy()

    # coerce metric
    d["_metric"] = _to_numeric_safe(d.get(metric))

    # apply feasible mask if present and truthy (tolerant)
    mask_feas = None
    if feasible_col in d.columns:
        s = d[feasible_col]
        if pd.api.types.is_bool_dtype(s):
            mask_feas = s.fillna(False)
        elif pd.api.types.is_numeric_dtype(s):
            mask_feas = s.fillna(0).astype(bool)
        else:
            mask_feas = s.astype(str).str.strip().str.upper().isin({"Y", "YES", "TRUE", "T", "1"})

    if mask_feas is not None and mask_feas.any():
        cand = d[mask_feas & d["_metric"].notna()].copy()
        if not cand.empty:
            return cand.sort_values("_metric", ascending=True).iloc[0]

    # fallback: any numeric metric
    cand2 = d[d["_metric"].notna()].copy()
    if cand2.empty:
        return d.iloc[0]  # last resort: first row
    return cand2.sort_values("_metric", ascending=True).iloc[0]


# ------------------------
# Printing summary
# ------------------------
def print_summary(df: pd.DataFrame,
                  metric: str = "carbon_per_m2",
                  area_m2: Optional[float] = None,
                  feasible_col: Optional[str] = "feasible") -> None:
    """
    Print a short, human-friendly summary for the best candidate (by metric, preferring feasible).
    - df: results DataFrame
    - metric: column used to choose the 'best' candidate (default carbon_per_m2)
    - area_m2: analysed floor area; if provided, prints totals (kgCO2 / currency)
    """
    best = _best_row_by_metric(df, metric, feasible_col=feasible_col)
    if best is None:
        print("No data available for summary.")
        return

    def _safe(k, default=0.0):
        v = best.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        try:
            return float(v)
        except Exception:
            return default

    print("\n=== Summary (best candidate) ===")
    print(f"System variant: {best.get('system_variant', best.get('system_id', 'n/a'))}")
    print(f"Family / Type:  {best.get('system_family', best.get('type', 'n/a'))} / {best.get('type', '')}")
    print(f"Typology:       {best.get('typology', '')}")
    print(f"Span (m):       {best.get('max_span', best.get('span_m', 'n/a'))}")
    print(f"Depth (mm):     {best.get('slab_depth', best.get('depth_m', 'n/a'))}")
    c_per_m2 = _safe("carbon_per_m2", 0.0)
    cost_per_m2 = _safe("cost_per_m2", 0.0)
    print(f"Carbon / m²:    {c_per_m2:,.1f} kgCO₂e")
    print(f"Cost / m²:      {cost_per_m2:,.2f}")

    if area_m2 is not None and area_m2 > 0:
        print(f"Analysed area:  {area_m2:,.1f} m²")
        print(f"Total carbon:   {c_per_m2 * area_m2:,.0f} kgCO₂e")
        print(f"Total cost:     {cost_per_m2 * area_m2:,.2f}")
    print("================================\n")


# ------------------------
# Plots
# ------------------------
def plot_pareto(df: pd.DataFrame,
                carbon_col: str = "carbon_per_m2",
                cost_col: str = "cost_per_m2",
                title: str = "Carbon vs Cost (Pareto)",
                savefig: Optional[str] = None,
                show: bool = True) -> plt.Figure:
    """
    Scatter carbon vs cost and highlight Pareto front (minimize both).
    Returns matplotlib.Figure.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    D = df.copy()
    D[carbon_col] = _to_numeric_safe(D.get(carbon_col))
    D[cost_col] = _to_numeric_safe(D.get(cost_col))
    D = D.dropna(subset=[carbon_col, cost_col])

    if D.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric carbon / cost", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    # compute Pareto mask (simple O(n^2) but fine for typical candidate counts)
    vals = D[[carbon_col, cost_col]].values
    n = len(vals)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        xi, yi = vals[i]
        for j in range(n):
            if j == i:
                continue
            xj, yj = vals[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                pareto[i] = False
                break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(D.loc[~pareto, carbon_col], D.loc[~pareto, cost_col], label="Candidates")
    ax.scatter(D.loc[pareto, carbon_col], D.loc[pareto, cost_col], marker="s", label="Pareto")
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_ylabel("Cost (per m²)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)
    if show:
        plt.show()
    return fig


def plot_best_typology_bar(df: pd.DataFrame,
                           carbon_col: str = "carbon_per_m2",
                           typology_col: str = "typology",
                           order: Optional[Sequence[str]] = ("Timber", "Concrete", "Composite", "Steel", "Other"),
                           title: str = "Lowest-carbon option per typology",
                           savefig: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
    """
    For each typology, plot the minimum carbon_col value observed.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    D = add_typology(df, typology_col)
    D[carbon_col] = _to_numeric_safe(D.get(carbon_col))
    grouped = D.groupby(typology_col)[carbon_col].min().reindex(order).dropna()
    if grouped.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No typologies found", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    labels = list(grouped.index)
    vals = grouped.values

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(labels))))
    ax.barh(labels, vals)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.0f}", va="center")
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)
    if show:
        plt.show()
    return fig


def plot_carbon_vs_span(df: pd.DataFrame,
                        span_col: str = "max_span",
                        carbon_col: str = "carbon_per_m2",
                        typology_col: str = "typology",
                        min_span: Optional[float] = None,
                        title: str = "Embodied carbon vs span",
                        savefig: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    Scatter carbon_per_m2 vs span (span_col). Optionally filter by min_span.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    D = add_typology(df, typology_col)
    D["_span"] = _to_numeric_safe(D.get(span_col))
    D["_carbon"] = _to_numeric_safe(D.get(carbon_col))
    D = D.dropna(subset=["_span", "_carbon"])
    if min_span is not None:
        D = D[D["_span"] >= float(min_span)]
    if D.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric span/carbon rows", ha="center", va="center", transform=ax.transAxes)
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        return fig

    fig, ax = plt.subplots(figsize=(9, 6))
    for typ in sorted(D[typology_col].unique()):
        sub = D[D[typology_col] == typ]
        ax.scatter(sub["_span"], sub["_carbon"], label=str(typ), s=30, alpha=0.85, edgecolors="none")
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    ax.legend(title="Typology", loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)
    if show:
        plt.show()
    return fig

def plot_lowest_rows_by_group(df: pd.DataFrame, group_col: str, out_path: str | Path,
                              variant_col: str = "system_variant",
                              carbon_col: str = "carbon_total_kgCO2"):
    """
    Plot each row in df (typically lowest_per_type) as a bar with height equal to carbon_col,
    and group / legend by group_col (e.g. 'type' or 'category').

    - df: DataFrame containing at least variant_col, carbon_col and group_col (or merges will be attempted).
    - group_col: column name to group/colour legend by.
    - out_path: path to write PNG file.
    """
    out_path = Path(out_path)
    df = df.copy()

    # ensure the columns exist
    if variant_col not in df.columns:
        df[variant_col] = df.get("system_variant", df.index.astype(str))

    if carbon_col not in df.columns:
        # try a few fallbacks
        for alt in ("carbon_per_m2", "carbon_total_kgCO2", "carbon_total"):
            if alt in df.columns:
                carbon_col = alt
                break

    # coerce numeric carbon, replace NaN with 0 for plotting
    df[carbon_col] = pd.to_numeric(df[carbon_col], errors="coerce").fillna(0.0)

    # ensure group column exists
    if group_col not in df.columns:
        df[group_col] = df.get("type", df.get("category", "Unknown"))

    # order rows by group then by carbon for nicer visual grouping
    df["_group_order"] = df[group_col].astype(str) + "___" + df[carbon_col].astype(str)
    df = df.sort_values(by=[group_col, carbon_col], ascending=[True, False]).reset_index(drop=True)

    labels = df[variant_col].astype(str).tolist()
    values = df[carbon_col].values
    groups = df[group_col].astype(str).tolist()
    unique_groups = sorted(list(dict.fromkeys(groups)))  # preserve order of first occurrence

    x_positions = np.arange(len(labels))
    fig_width = max(8, len(labels) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # plot each group in sequence so matplotlib cycles colours automatically
    for grp in unique_groups:
        mask = [g == grp for g in groups]
        if not any(mask):
            continue
        xs = x_positions[mask]
        ys = values[mask]
        # call bar without explicit color so default cycle is used
        ax.bar(xs, ys, label=grp)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(f"{carbon_col} (kgCO₂e)")
    ax.set_title(f"Carbon per variant (grouped by {group_col})")
    # legend placed to the right
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


# ------------------------
# Convenience: create small dashboard (prints + plots)
# ------------------------
def generate_simple_report(df: pd.DataFrame,
                           area_m2: Optional[float] = None,
                           metric: str = "carbon_per_m2",
                           save_dir: Optional[str] = None,
                           show: bool = True) -> Dict[str, Any]:
    """
    Produce a short textual summary and three plots (pareto, typology bar, carbon vs span).
    Returns a dict with keys: 'summary_row', 'figures' (mapping fig_name -> matplotlib.Figure).
    If save_dir is provided the figs are also saved as PNG files in that directory.
    """
    out = {}
    print_summary(df, metric, area_m2)
    figs = {}

    figs['pareto'] = plot_pareto(df, show=show, savefig=(f"{save_dir}/pareto.png" if save_dir else None))
    figs['typology'] = plot_best_typology_bar(df, show=show, savefig=(f"{save_dir}/typology.png" if save_dir else None))
    figs['span'] = plot_carbon_vs_span(df, show=show, savefig=(f"{save_dir}/span.png" if save_dir else None))

    # best row included in output
    out['summary_row'] = _best_row_by_metric(df, metric)
    out['figures'] = figs
    return out
