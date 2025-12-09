# earlystruct/core/reporting.py
from __future__ import annotations

import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Typology bucketing ----------

def infer_typology(row: pd.Series) -> str:
    c = (str(row.get("category", "")) + " " + str(row.get("type", ""))).lower()
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

def _safe_label(val) -> str:
    """
    Convert a possibly-list/array-like value to a plain string label.
    """
    # None/NaN
    try:
        if val is None:
            return ""
    except Exception:
        pass
    # numpy scalar
    if isinstance(val, (np.generic, np.number)):
        return str(val)
    # iterable but not string: join items
    if isinstance(val, (list, tuple, set, np.ndarray)):
        try:
            return ", ".join(str(x) for x in val)
        except Exception:
            return str(val)
    # fallback
    return str(val)

# ---------- Best-per-group selection ----------

def best_per_group(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """
    Among feasible rows, pick the lowest value of `metric` for each group in `group_col`.
    Tolerant to a variety of 'feasible' representations and falls back to metric-only
    selection if no feasible rows are found.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    df2 = df.copy()

    # ensure typology when requested
    if group_col.lower() == "typology":
        df2 = add_typology(df2)

    if group_col not in df2.columns:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    # --- build tolerant feasible mask ---
    def _feasible_mask(series: pd.Series) -> pd.Series:
        if series is None:
            return pd.Series([False] * len(df2), index=df2.index)
        if pd.api.types.is_bool_dtype(series):
            return series.astype(bool).fillna(False)
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(0) != 0
        s = series.astype(str).str.strip().str.upper().fillna("")
        return s.isin({"Y", "YES", "TRUE", "T", "1"})

    mask_feas = _feasible_mask(df2.get("feasible")) if "feasible" in df2.columns else pd.Series(
        [False] * len(df2), index=df2.index
    )

    # --- coerce metric to numeric ---
    if metric in df2.columns:
        df2["_metric_numeric"] = pd.to_numeric(df2[metric], errors="coerce")
    else:
        df2["_metric_numeric"] = pd.Series([float("nan")] * len(df2), index=df2.index)

    mask_metric_ok = df2["_metric_numeric"].notna() & (df2["_metric_numeric"] > 0)

    # first try: feasible AND metric_ok
    filt = df2[mask_feas & mask_metric_ok].copy()

    # fallback: if nothing matched, use metric_ok alone
    if filt.empty:
        filt = df2[mask_metric_ok].copy()

    if filt.empty:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    # Now pick the min metric per group robustly
    try:
        # compute idx of min numeric metric per group
        idx_series = filt.groupby(group_col, as_index=False)["_metric_numeric"].idxmin()
        # idx_series may include NaN — drop those then convert to ints
        idx_valid = idx_series.dropna().astype(int).tolist()
        out = filt.loc[idx_valid].copy()
    except Exception:
        # last-resort safe: iterative pick
        rows = []
        for g, sub in filt.groupby(group_col):
            sub_sorted = sub.sort_values("_metric_numeric", ascending=True)
            rows.append(sub_sorted.iloc[0])
        if not rows:
            return pd.DataFrame(
                columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
            )
        out = pd.DataFrame(rows)

    cols = [
        group_col,
        "system_id",
        "system_name",
        "category",
        "type",
        "manufacturer",
        metric,
        "depth_m",
        "span_m",
    ]
    keep = [c for c in cols if c in out.columns]

    # sort by numeric metric if present
    if "_metric_numeric" in out.columns:
        out = out.sort_values("_metric_numeric", ascending=True)

    return out[keep].reset_index(drop=True)

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
    Robust to mixed-type and missing columns.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title)
        ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
        ax.set_ylabel("Cost (per m²)")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return

    df2 = df.copy()

    # coerce numeric columns safely
    for c in ("carbon_per_m2", "cost_per_m2"):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # build feasible mask tolerant
    if "feasible" in df2.columns:
        if pd.api.types.is_bool_dtype(df2["feasible"]):
            mask_feas = df2["feasible"].astype(bool)
        elif pd.api.types.is_numeric_dtype(df2["feasible"]):
            mask_feas = df2["feasible"].fillna(0) != 0
        else:
            mask_feas = df2["feasible"].astype(str).str.strip().str.upper().isin({"Y", "YES", "TRUE", "T", "1"})
        feas = df2[mask_feas & df2["carbon_per_m2"].notna() & df2["cost_per_m2"].notna()].copy()
    else:
        feas = df2[df2["carbon_per_m2"].notna() & df2["cost_per_m2"].notna()].copy()

    # fallback to everything with numeric columns if none feasible
    if feas.empty:
        feas = df2[df2["carbon_per_m2"].notna() & df2["cost_per_m2"].notna()].copy()

    if feas.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title)
        ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
        ax.set_ylabel("Cost (per m²)")
        ax.text(0.5, 0.5, "No numeric carbon/cost data", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return

    mask = pareto_mask(feas, "carbon_per_m2", "cost_per_m2")
    pareto = feas[mask]
    nonpareto = feas[~mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(nonpareto["carbon_per_m2"], nonpareto["cost_per_m2"], label="Candidates")
    if not pareto.empty:
        ax.scatter(pareto["carbon_per_m2"], pareto["cost_per_m2"], marker="s", label="Pareto")
    ax.set_title(title)
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_ylabel("Cost (per m²)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_best_typology_carbon(df, carbon_col="carbon_per_m2", title="Lowest-carbon option per typology"):
    import pandas as pd, matplotlib.pyplot as plt, numpy as np

    if df is None or len(df) == 0:
        print("Empty df")
        return

    D = df.copy()
    D = add_typology(D)
    D["_carbon"] = pd.to_numeric(D.get(carbon_col), errors="coerce")
    D = D[D["_carbon"].notna()]

    if D.empty:
        print("No numeric carbon data")
        return

    # Optional: remove absurd outliers ( > 1e4 kgCO2e/m2 are almost certainly wrong)
    # Comment out the next line if you want to include everything.
    D = D[D["_carbon"].abs() < 1e4]

    grouped = D.groupby("typology")["_carbon"].min().reindex(["Timber","Concrete","Composite","Steel","Other"]).dropna()

    if grouped.empty:
        print("No typologies found")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.8*len(grouped))))
    y_labels = grouped.index.astype(str).tolist()
    vals = grouped.values
    ax.barh(y_labels, vals)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.0f}", va="center")
    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_best_type_carbon(df, carbon_col="carbon_per_m2", title="Lowest-carbon option per type"):
    """
    One horizontal bar per 'type' (the column named 'type' in your catalog/results).
    Each bar shows the minimum carbon_per_m2 observed for that type.
    """
    import pandas as pd, matplotlib.pyplot as plt, numpy as np

    if df is None or len(df) == 0:
        print("Empty df")
        return

    D = df.copy()
    # Ensure 'type' exists
    if "type" not in D.columns and "typology" in D.columns:
        D["type"] = D["typology"]

    D["_carbon"] = pd.to_numeric(D.get(carbon_col), errors="coerce")
    D = D[D["_carbon"].notna()]

    if D.empty:
        print("No numeric carbon data")
        return

    # Optional: remove absurd outliers ( > 1e5 kgCO2e/m2 are almost certainly wrong)
    D = D[D["_carbon"].abs() < 1e5]

    grouped = D.groupby("type")["_carbon"].min()
    if grouped.empty:
        print("No 'type' groups found")
        return

    # Sort by value ascending for readability
    grouped = grouped.sort_values(ascending=True)

    # Prepare labels and values
    labels = grouped.index.astype(str).tolist()
    vals = grouped.values

    # Limit number of types visualized if huge (but show all by default)
    max_display = 50
    if len(labels) > max_display:
        labels = labels[:max_display]
        vals = vals[:max_display]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    ax.barh(labels, vals)
    for i, v in enumerate(vals):
        try:
            ax.text(v, i, f" {float(v):.0f}", va="center")
        except Exception:
            ax.text(0, i, " n/a", va="center")

    ax.set_xlabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_carbon_vs_span(
    df: pd.DataFrame,
    span_col: str = "span_slab_dir_m",
    carbon_col: str = "carbon_per_m2",
    min_span: float | None = None,
    only_feasible: bool = True,
    title: str = "Embodied carbon vs span",
):
    """
    Scatter plot: embodied carbon (kgCO2e / m²) vs span (m).

    Parameters
    - df: DataFrame with rows for systems (expects columns: span_col, carbon_col).
    - span_col: column to use for span (e.g. 'span_slab_dir_m' or 'span_input_m').
    - carbon_col: embodied carbon per m2 column.
    - min_span: if provided, only plot rows with span >= min_span.
    - only_feasible: if True prefer rows where feasible is truthy; if none, fall back
                     to any row with numeric carbon/span.
    - title: plot title.
    """
    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt

    if df is None or df.empty:
        fig, ax = _plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        _plt.show()
        return

    df2 = df.copy()

    # Ensure typology column exists
    if "typology" not in df2.columns:
        df2 = add_typology(df2)

    # Coerce numeric columns
    df2["_span_numeric"] = _pd.to_numeric(df2.get(span_col), errors="coerce")
    df2["_carbon_numeric"] = _pd.to_numeric(df2.get(carbon_col), errors="coerce")

    # Build feasible truthy mask (tolerant)
    def _truthy_mask(series: pd.Series) -> pd.Series:
        if series is None:
            return _pd.Series([False] * len(df2), index=df2.index)
        if _pd.api.types.is_bool_dtype(series):
            return series.astype(bool).fillna(False)
        if _pd.api.types.is_numeric_dtype(series):
            return series.fillna(0) != 0
        s = series.astype(str).str.strip().str.upper().fillna("")
        return s.isin({"Y", "YES", "TRUE", "T", "1"})

    mask_numeric = df2["_span_numeric"].notna() & df2["_carbon_numeric"].notna()
    if min_span is not None:
        mask_numeric = mask_numeric & (df2["_span_numeric"] >= float(min_span))

    if only_feasible and "feasible" in df2.columns:
        mask_feas = _truthy_mask(df2["feasible"])
        mask = mask_feas & mask_numeric
        # if none pass feasibility, fall back to numeric-only
        if mask.sum() == 0:
            mask = mask_numeric
    else:
        mask = mask_numeric

    plot_df = df2.loc[mask].copy()
    if plot_df.empty:
        fig, ax = _plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No rows match filters (feasible/min_span/numeric).", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        _plt.show()
        return

    # Define typology order and color mapping (use matplotlib defaults if available)
    typologies = ["Concrete", "Composite", "Steel", "Timber", "Other"]
    # get unique typologies present
    present = [t for t in typologies if t in plot_df["typology"].unique()]
    # choose a color cycle from matplotlib (do not hardcode many custom colors)
    colors = _plt.rcParams.get("axes.prop_cycle").by_key().get("color", None)
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(typologies)}

    fig, ax = _plt.subplots(figsize=(10, 6))

    # Scatter each typology separately for clear legend
    for typ in sorted(plot_df["typology"].unique()):
        sub = plot_df[plot_df["typology"] == typ]
        if sub.empty:
            continue
        ax.scatter(
            sub["_span_numeric"],
            sub["_carbon_numeric"],
            label=str(typ),
            alpha=0.85,
            edgecolors="none",
            s=30,
            c=color_map.get(typ, "gray"),
        )

    ax.set_title(title)
    ax.set_xlabel(f"Span (m) — column: {span_col}")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(title="Typology", loc="best")
    _plt.tight_layout()
    _plt.show()

def plot_carbon_vs_span_grid(
    df,
    span_min_global=6.6,
    span_max_global=None,
    step=0.25,
    carbon_col="carbon_per_m2",
    typology_col="typology",
    only_feasible=False,
    max_points=300000,
    title="Embodied carbon vs span — full expanded grid (v3)",
):
    """
    For each span in a global grid, include ALL systems whose allowed interval
    [row_min, row_max] contains that span. If a row has no row_max, treat it
    as allowed from row_min up to the global span_max (so it appears at every
    span >= row_min).
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        print("Empty df")
        return

    D = df.copy()

    # ensure typology exists
    if typology_col not in D.columns:
        D = add_typology(D)

    # infer row_min & row_max
    min_candidates = ["span_slab_dir_m","span_input_m","span_m","span_x_m","span_y_m"]
    max_candidates = ["max_span","max_span_m","span_max"]

    D["_row_min"] = pd.NA
    for c in min_candidates:
        if c in D.columns:
            vals = pd.to_numeric(D[c], errors="coerce")
            D["_row_min"] = vals.where(D["_row_min"].isna(), D["_row_min"])
    D["_row_min"] = pd.to_numeric(D["_row_min"], errors="coerce")

    D["_row_max"] = pd.NA
    for c in max_candidates:
        if c in D.columns:
            vals = pd.to_numeric(D[c], errors="coerce")
            D["_row_max"] = vals.where(D["_row_max"].isna(), D["_row_max"])
    D["_row_max"] = pd.to_numeric(D["_row_max"], errors="coerce")

    D["_carbon"] = pd.to_numeric(D.get(carbon_col), errors="coerce")

    # feasible mask
    def truthy_mask(s):
        if s is None:
            return pd.Series([False]*len(D), index=D.index)
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0) != 0
        ss = s.astype(str).str.strip().str.upper().fillna("")
        return ss.isin({"Y","YES","TRUE","T","1"})

    mask_numeric = D["_row_min"].notna() & D["_carbon"].notna()
    if only_feasible and "feasible" in D.columns:
        mask = truthy_mask(D["feasible"]) & mask_numeric
        if mask.sum() == 0:
            mask = mask_numeric
    else:
        mask = mask_numeric
    D = D.loc[mask].copy()
    if D.empty:
        print("No rows after filtering.")
        return

    # infer global max if needed
    if span_max_global is None:
        row_max_vals = D["_row_max"].dropna()
        if len(row_max_vals) > 0:
            span_max_global = max(row_max_vals.max(), D["_row_min"].max())
        else:
            span_max_global = float(D["_row_min"].max()) + 6.0

    span_min_global = float(span_min_global)
    span_max_global = float(span_max_global)
    grid = np.arange(span_min_global, span_max_global + step*0.499, step)
    if len(grid) == 0:
        raise ValueError("Grid empty. Check bounds/step.")

    grid_arr = np.asarray(grid)

    spans_out = []
    carb_out = []
    typ_out = []
    sys_out = []
    total = 0

    # IMPORTANT CHANGE: if row_max is NaN -> treat row_max = span_max_global (unbounded upward)
    for i, row in D.iterrows():
        rmin = row["_row_min"]
        rmax = row["_row_max"]
        if pd.isna(rmin):
            continue
        if pd.isna(rmax):
            rmax_eff = span_max_global
        else:
            rmax_eff = rmax

        if rmax_eff < rmin:
            # invalid interval -> single point at rmin
            spans = np.array([rmin])
        else:
            maskg = (grid_arr >= (rmin - 1e-9)) & (grid_arr <= (rmax_eff + 1e-9))
            if not maskg.any():
                spans = np.array([rmin])
            else:
                spans = grid_arr[maskg]

        n = spans.size
        if n == 0:
            continue

        spans_out.append(spans)
        carb_out.append(np.full(n, row["_carbon"]))
        typ_out.extend([row.get(typology_col, "Other")] * n)
        sys_out.extend([row.get("system_id", str(i))] * n)
        total += n
        if total >= max_points:
            break

    if total == 0:
        print("No expanded points generated.")
        return

    spans_arr = np.concatenate(spans_out)
    carb_arr = np.concatenate(carb_out)
    typ_arr = np.array(typ_out[: len(spans_arr)])
    sys_arr = np.array(sys_out[: len(spans_arr)])

    plot_df = pd.DataFrame({"span": spans_arr, "carbon": carb_arr, "typology": typ_arr, "system_id": sys_arr})
    plot_df = plot_df[plot_df["carbon"].notna()]

    # Plot
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["C0","C1","C2","C3","C4"])
    typ_order = ["Concrete","Composite","Steel","Timber","Other"]
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(typ_order)}

    fig, ax = plt.subplots(figsize=(12,6))
    for typ in sorted(plot_df["typology"].unique()):
        sub = plot_df[plot_df["typology"] == typ]
        ax.scatter(sub["span"], sub["carbon"], s=18, alpha=0.7, label=str(typ), c=color_map.get(typ,"gray"), edgecolors="none")

    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(title="Typology", loc="best")
    plt.tight_layout()
    plt.show()

def plot_carbon_vs_span_grid_types(
    df,
    span_min_global=6.6,
    span_max_global=None,
    step=0.25,
    carbon_col="carbon_per_m2",
    type_col="type",
    only_feasible=False,
    max_points=300000,
    title="Embodied carbon vs span — full expanded grid (colored by type)",
):
    """
    Global-span grid scatter: for each span in the global grid, include every system
    whose allowed span interval contains that span. Color points by the `type` column.
    If a row has no row_max value, treat it as allowed up to span_max_global.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        print("Empty df")
        return

    D = df.copy()

    # ensure 'type' exists (if not, fall back to typology)
    if type_col not in D.columns and "typology" in D.columns:
        D[type_col] = D["typology"]

    # infer row_min & row_max
    min_candidates = ["span_slab_dir_m","span_input_m","span_m","span_x_m","span_y_m"]
    max_candidates = ["max_span","max_span_m","span_max"]

    D["_row_min"] = pd.NA
    for c in min_candidates:
        if c in D.columns:
            vals = pd.to_numeric(D[c], errors="coerce")
            D["_row_min"] = vals.where(D["_row_min"].isna(), D["_row_min"])
    D["_row_min"] = pd.to_numeric(D["_row_min"], errors="coerce")

    D["_row_max"] = pd.NA
    for c in max_candidates:
        if c in D.columns:
            vals = pd.to_numeric(D[c], errors="coerce")
            D["_row_max"] = vals.where(D["_row_max"].isna(), D["_row_max"])
    D["_row_max"] = pd.to_numeric(D["_row_max"], errors="coerce")

    D["_carbon"] = pd.to_numeric(D.get(carbon_col), errors="coerce")

    # feasible mask tolerant
    def truthy_mask(s):
        if s is None:
            return pd.Series([False]*len(D), index=D.index)
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0) != 0
        ss = s.astype(str).str.strip().str.upper().fillna("")
        return ss.isin({"Y","YES","TRUE","T","1"})

    mask_numeric = D["_row_min"].notna() & D["_carbon"].notna()
    if only_feasible and "feasible" in D.columns:
        mask = truthy_mask(D["feasible"]) & mask_numeric
        if mask.sum() == 0:
            mask = mask_numeric
    else:
        mask = mask_numeric
    D = D.loc[mask].copy()
    if D.empty:
        print("No rows after filtering.")
        return

    # infer global max if not given
    if span_max_global is None:
        row_max_vals = D["_row_max"].dropna()
        if len(row_max_vals) > 0:
            span_max_global = max(row_max_vals.max(), D["_row_min"].max())
        else:
            span_max_global = float(D["_row_min"].max()) + 6.0

    span_min_global = float(span_min_global)
    span_max_global = float(span_max_global)
    grid = np.arange(span_min_global, span_max_global + step*0.499, step)
    if len(grid) == 0:
        raise ValueError("Grid empty. Check bounds/step.")

    grid_arr = np.asarray(grid)

    spans_out = []
    carb_out = []
    type_out = []
    sys_out = []
    total = 0

    # If row_max is NaN -> treat as unbounded up to span_max_global
    for i, row in D.iterrows():
        rmin = row["_row_min"]
        if pd.isna(rmin):
            continue
        rmax = row["_row_max"] if not pd.isna(row["_row_max"]) else span_max_global
        if rmax < rmin:
            spans = np.array([rmin])
        else:
            maskg = (grid_arr >= (rmin - 1e-9)) & (grid_arr <= (rmax + 1e-9))
            if not maskg.any():
                spans = np.array([rmin])
            else:
                spans = grid_arr[maskg]

        n = spans.size
        if n == 0:
            continue

        spans_out.append(spans)
        carb_out.append(np.full(n, row["_carbon"]))
        # use the requested type column; fall back to stringified category if missing
        tval = row.get(type_col, None)
        if pd.isna(tval):
            tval = str(row.get("category", "Unknown"))
        type_out.extend([tval] * n)
        sys_out.extend([row.get("system_id", str(i))] * n)

        total += n
        if total >= max_points:
            break

    if total == 0:
        print("No expanded points generated.")
        return

    spans_arr = np.concatenate(spans_out)
    carb_arr = np.concatenate(carb_out)
    type_arr = np.array(type_out[: len(spans_arr)])
    sys_arr = np.array(sys_out[: len(spans_arr)])

    plot_df = pd.DataFrame({"span": spans_arr, "carbon": carb_arr, "type": type_arr, "system_id": sys_arr})
    plot_df = plot_df[plot_df["carbon"].notna()]

    # Build color mapping for types using matplotlib cycle
    unique_types = list(pd.unique(plot_df["type"]))
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", None)
    if not colors:
        colors = ["C%d" % i for i in range(10)]
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each type separately to make legend usable
    for t in unique_types:
        sub = plot_df[plot_df["type"] == t]
        if sub.empty:
            continue
        ax.scatter(sub["span"], sub["carbon"], s=18, alpha=0.75, label=str(t), c=color_map.get(t, "gray"), edgecolors="none")

    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Embodied carbon (kgCO₂e / m²)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    # If too many types, shrink legend or place outside
    if len(unique_types) <= 20:
        ax.legend(title="Type", loc="best")
    else:
        ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
        plt.subplots_adjust(right=0.75)
    plt.tight_layout()
    plt.show()



# ---------- Tabular helpers ----------

def cheapest_span_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each 'type', return the single row (span + system) that gives
    the minimum total building cost among feasible candidates.
    """
    if df.empty:
        return df.copy()

    feas = df[df["feasible"] == True].copy()
    if feas.empty:
        # fall back to everything if nothing is strictly feasible
        feas = df.copy()

    # sort so the cheapest rows per type come first
    feas = feas.sort_values(["type", "cost_total"], ascending=[True, True])

    # keep the first row per type
    best = feas.groupby("type", as_index=False).first()
    return best


def tables_best_by_typology(df: pd.DataFrame):
    """
    Convenience helper: returns (best_by_carbon, best_by_cost) per typology.
    """
    carb = best_per_typology(df, "carbon_per_m2")
    cost = best_per_typology(df, "cost_per_m2")
    return carb, cost


# ---------- Verbose CLI summary ----------


def _get_ctl_str(ctl: Dict[str, Any], key: str, default: str = "") -> str:
    v = ctl.get(key)
    if v is None or v == "":
        return default
    return str(v)


def _get_ctl_float(ctl: Dict[str, Any], key: str, default: float) -> float:
    v = ctl.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _safe(col: Any, default: float = 0.0) -> float:
    """Convert a scalar or missing value to float, safely."""
    if col is None or (isinstance(col, float) and math.isnan(col)):
        return default
    try:
        return float(col)
    except Exception:
        return default


def _pct(part: float, whole: float) -> float:
    if whole <= 0.0:
        return 0.0
    return 100.0 * part / whole


def print_verbose_summary(df: pd.DataFrame, project: Dict[str, Any] | None, ctl: Dict[str, Any] | None):
    """
    Verbose console summary:

      - project basics
      - frame (beam/column) penalty knobs
      - best option (by carbon_per_m2, then cost_per_m2), with split:
          * floor vs beams vs columns carbon and cost (per m² and totals)
          * slab vs beam vs column concrete volumes
    """
    project = project or {}
    ctl = ctl or {}

    print("\n================ VERBOSE SUMMARY ================")

    # ---------- Project basics ----------
    proj_name = project.get("project_name") or project.get("PROJECT_NAME") or ""
    location  = project.get("location") or project.get("LOCATION") or ""
    unit_flag = (
        project.get("UNIT")
        or project.get("unit")
        or ctl.get("UNIT")
        or ctl.get("unit")
        or "metric"
    )

    print(f"Project:   {proj_name}")
    print(f"Location:  {location}")
    print(f"Units:     {unit_flag}")

    area_raw = project.get("floor_area_per_floor") or project.get("FLOOR_AREA_PER_FLOOR") or ""
    floors   = project.get("num_floors") or project.get("NUM_FLOORS") or ""
    print(f"Floor area per floor: {area_raw}")
    print(f"Number of floors:     {floors}")

    # ---------- Frame (beam/column) penalties ----------
    frame_enabled = bool(ctl.get("FRAME_PENALTIES_ENABLED_BOOL", True))
    print("\nFrame (beam/column) penalties:", "ENABLED" if frame_enabled else "DISABLED")

    frame_beam_slenderness = _get_ctl_float(ctl, "FRAME_BEAM_SLENDERNESS", 20.0)
    beam_width_raw         = _get_ctl_str(ctl, "FRAME_BEAM_WIDTH",    "default (≈0.20 m)")
    col_width_raw          = _get_ctl_str(ctl, "FRAME_COLUMN_WIDTH",  "default (≈0.30 m)")
    col_depth_raw          = _get_ctl_str(ctl, "FRAME_COLUMN_DEPTH",  "default (≈0.30 m)")
    beam_depth_max_raw     = _get_ctl_str(ctl, "FRAME_BEAM_DEPTH_MAX","default (≈0.80 m)")

    print(f"  Beam slenderness L/d:     {frame_beam_slenderness}")
    print(f"  Beam width (control units):    {beam_width_raw}")
    print(f"  Column width (control units):  {col_width_raw}")
    print(f"  Column depth (control units):  {col_depth_raw}")
    print(f"  Max beam depth (control units): {beam_depth_max_raw}")

    if df is None or df.empty:
        print("\nNo results to summarise (DataFrame is empty).")
        print("=================================================\n")
        return

    # ---------- Choose rows to treat as "feasible" for summary ----------
    if "feasible" in df.columns:
        dff = df[df["feasible"]].copy()
        if dff.empty:
            print("\nNOTE: No systems are marked as feasible under current checks.")
            print("      Showing best available option (ignoring the feasible flag).")
            dff = df.copy()
    else:
        dff = df.copy()

    if dff.empty:
        print("\nNo rows available to summarise.")
        print("=================================================\n")
        return

    # ---------- Choose a "best" option for detailed split ----------
    dff_sorted = dff.sort_values(["carbon_per_m2", "cost_per_m2", "depth_m", "span_m"])
    best = dff_sorted.iloc[0]

    print("\nBest option (by carbon_per_m2, then cost_per_m2):")
    print(f"  System:     {best.get('system_id', '')}  ({best.get('system_name', '')})")
    print(f"  Category:   {best.get('category', '')}, type: {best.get('type', '')}")
    print(f"  Span (slab dir):  {best.get('span_slab_dir_m', best.get('span_input_m', 0.0)):.3f} m")
    print(f"  Span (beam dir):  {best.get('span_beam_dir_m', best.get('span_m', 0.0)):.3f} m")
    print(f"  Structural depth: {best.get('depth_m', 0.0):.3f} m")
    print(f"  Slab depth:       {best.get('slab_depth_m', 0.0):.3f} m")
    print(f"  Beam depth:       {best.get('beam_depth_m', 0.0):.3f} m")

    area_m2          = _safe(best.get("area_m2"), 0.0)
    carbon_total_per_m2 = _safe(best.get("carbon_per_m2"), 0.0)
    cost_total_per_m2   = _safe(best.get("cost_per_m2"), 0.0)
    carbon_total_kg     = _safe(best.get("carbon_total_kg"), 0.0)
    cost_total          = _safe(best.get("cost_total"), 0.0)

    # Frame (beams + columns) per m² from DataFrame
    carbon_beams_per_m2 = _safe(best.get("carbon_beams_per_m2"), 0.0)
    carbon_cols_per_m2  = _safe(best.get("carbon_columns_per_m2"), 0.0)
    cost_beams_per_m2   = _safe(best.get("cost_beams_per_m2"), 0.0)
    cost_cols_per_m2    = _safe(best.get("cost_columns_per_m2"), 0.0)

    carbon_frame_per_m2 = carbon_beams_per_m2 + carbon_cols_per_m2
    cost_frame_per_m2   = cost_beams_per_m2 + cost_cols_per_m2

    # DEFINE slab as "everything that is not beams+columns"
    carbon_slab_per_m2 = max(0.0, carbon_total_per_m2 - carbon_frame_per_m2)
    cost_slab_per_m2   = max(0.0, cost_total_per_m2 - cost_frame_per_m2)

    carbon_slab_kg  = carbon_slab_per_m2  * area_m2
    carbon_beams_kg = carbon_beams_per_m2 * area_m2
    carbon_cols_kg  = carbon_cols_per_m2  * area_m2

    cost_slab   = cost_slab_per_m2   * area_m2
    cost_beams  = cost_beams_per_m2  * area_m2
    cost_cols   = cost_cols_per_m2   * area_m2

    # ---------- Carbon split ----------
    print("\nCarbon split (per m²):")
    print(f"  Floor (slab only):   {carbon_slab_per_m2:10.3f} kgCO2/m²")
    print(f"  Beams:               {carbon_beams_per_m2:10.3f} kgCO2/m²")
    print(f"  Columns:             {carbon_cols_per_m2:10.3f} kgCO2/m²")
    print(f"  Frame (beams+cols):  {carbon_frame_per_m2:10.3f} kgCO2/m²")
    print(f"  TOTAL (used in optimisation): {carbon_total_per_m2:10.3f} kgCO2/m²")

    print("\nCarbon split (totals over analysed area):")
    print(f"  Analysed area:       {area_m2:.1f} m²")
    print(f"  Floor (slab only):   {carbon_slab_kg:10.1f} kgCO2 "
          f"({ _pct(carbon_slab_kg, carbon_total_kg):5.1f}% of total)")
    print(f"  Beams:               {carbon_beams_kg:10.1f} kgCO2 "
          f"({ _pct(carbon_beams_kg, carbon_total_kg):5.1f}% of total)")
    print(f"  Columns:             {carbon_cols_kg:10.1f} kgCO2 "
          f"({ _pct(carbon_cols_kg, carbon_total_kg):5.1f}% of total)")
    print(f"  TOTAL:               {carbon_total_kg:10.1f} kgCO2 (slab + beams + columns)")

    # ---------- Cost split ----------
    print("\nCost split (per m²):")
    print(f"  Floor (slab only):   {cost_slab_per_m2:10.2f} currency/m²")
    print(f"  Beams:               {cost_beams_per_m2:10.2f} currency/m²")
    print(f"  Columns:             {cost_cols_per_m2:10.2f} currency/m²")
    print(f"  Frame (beams+cols):  {cost_frame_per_m2:10.2f} currency/m²")
    print(f"  TOTAL (used in optimisation): {cost_total_per_m2:10.2f} currency/m²")

    print("\nCost split (totals over analysed area):")
    print(f"  Floor (slab only):   {cost_slab:10.2f} "
          f"({ _pct(cost_slab, cost_total):5.1f}% of total)")
    print(f"  Beams:               {cost_beams:10.2f} "
          f"({ _pct(cost_beams, cost_total):5.1f}% of total)")
    print(f"  Columns:             {cost_cols:10.2f} "
          f"({ _pct(cost_cols, cost_total):5.1f}% of total)")
    print(f"  TOTAL:               {cost_total:10.2f} (slab + beams + columns)")

    # ---------- Concrete volume split ----------
    conc_slab_m3  = _safe(best.get("concrete_slab_m3"), 0.0)
    conc_beams_m3 = _safe(best.get("concrete_beams_m3"), 0.0)
    conc_cols_m3  = _safe(best.get("concrete_columns_m3"), 0.0)
    conc_total_m3 = conc_slab_m3 + conc_beams_m3 + conc_cols_m3

    print("\nConcrete volume split (totals):")
    print(f"  Slab:                {conc_slab_m3:10.3f} m³ "
          f"({ _pct(conc_slab_m3, conc_total_m3):5.1f}% of concrete)")
    print(f"  Beams:               {conc_beams_m3:10.3f} m³ "
          f"({ _pct(conc_beams_m3, conc_total_m3):5.1f}% of concrete)")
    print(f"  Columns:             {conc_cols_m3:10.3f} m³ "
          f"({ _pct(conc_cols_m3, conc_total_m3):5.1f}% of concrete)")
    print(f"  TOTAL concrete:      {conc_total_m3:10.3f} m³")

    print("\n(Optimisation uses TOTAL carbon_per_m2 and cost_per_m2;")
    print(" this summary simply splits out the slab vs beam/column contributions.)")
    print("=================================================\n")