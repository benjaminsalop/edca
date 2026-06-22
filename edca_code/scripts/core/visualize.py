"""visualize.py — Embodied-carbon comparison charts for EDCA assemblies.

Produces horizontal stacked bar charts with one bar per structural class,
stacked by (component × material) — e.g. "Floor: Concrete", "Beam: Struct. Steel".

When per-component-material columns are absent the chart falls back to
per-component stacking for backwards compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .structural_class import _GROUP_ORDER, structural_sort_key

# ---------------------------------------------------------------------------
# Legacy per-component colour palette (fallback when comp×mat cols absent)
# ---------------------------------------------------------------------------
_COMPONENT_COLORS: dict[str, str] = {
    "floor":          "#E07B39",
    "primary_beam":   "#4A7EB5",
    "secondary_beam": "#7EB5E0",
    "column":         "#5C5C5C",
    "lateral":        "#A63A3A",
}

_COMPONENT_LABELS: dict[str, str] = {
    "floor":          "Floor",
    "primary_beam":   "Primary Beam",
    "secondary_beam": "Secondary Beam",
    "column":         "Column",
    "lateral":        "Lateral System",
}

# ---------------------------------------------------------------------------
# Per-(component × material) colour palette
# ---------------------------------------------------------------------------
# Key: (df_column_prefix, material_key)
# Component prefixes match assembly_summary.py output column names.
_COMP_MAT_COLORS: dict[tuple[str, str], str] = {
    # Floor — orange family
    ("floor", "concrete"):         "#D4692A",
    ("floor", "rebar"):            "#A84020",
    ("floor", "structural_steel"): "#F0955A",
    ("floor", "pt"):               "#E8B878",
    ("floor", "timber"):           "#C8A855",
    ("floor", "screed"):           "#F5CC98",
    # Primary beam — blue family
    ("beam", "concrete"):         "#3D6FA0",
    ("beam", "rebar"):            "#264870",
    ("beam", "structural_steel"): "#1565C0",
    ("beam", "pt"):               "#5590C0",
    ("beam", "timber"):           "#4098C8",
    ("beam", "screed"):           "#7ABCE0",
    # Secondary beam — light blue
    ("sec_beam", "concrete"):         "#6AAAD5",
    ("sec_beam", "rebar"):            "#4888B8",
    ("sec_beam", "structural_steel"): "#3880AA",
    ("sec_beam", "pt"):               "#88B8D5",
    ("sec_beam", "timber"):           "#A0CCE0",
    ("sec_beam", "screed"):           "#C0DEF0",
    # Column — grey family
    ("column", "concrete"):         "#888888",
    ("column", "rebar"):            "#505050",
    ("column", "structural_steel"): "#282828",
    ("column", "pt"):               "#A0A0A0",
    ("column", "timber"):           "#989898",
    ("column", "screed"):           "#C8C8C8",
    # Lateral — red/brick family
    ("lateral", "concrete"):         "#A03030",
    ("lateral", "rebar"):            "#702020",
    ("lateral", "structural_steel"): "#C03828",
    ("lateral", "pt"):               "#B06055",
    ("lateral", "timber"):           "#C07868",
    ("lateral", "screed"):           "#C89898",
}

_COMP_DISPLAY: dict[str, str] = {
    "floor":    "Floor",
    "beam":     "Prim. Beam",
    "sec_beam": "Sec. Beam",
    "column":   "Column",
    "lateral":  "Lateral",
}

_MAT_DISPLAY: dict[str, str] = {
    "concrete":         "Concrete",
    "rebar":            "Rebar",
    "structural_steel": "Struct. Steel",
    "pt":               "Post-Tension",
    "timber":           "Timber",
    "screed":           "Screed",
}

_COMP_ORDER = ["floor", "beam", "sec_beam", "column", "lateral"]
_MAT_ORDER  = ["concrete", "rebar", "structural_steel", "pt", "timber", "screed"]

# Broad group → background strip colour
_GROUP_COLORS: dict[str, str] = {
    "Timber":    "#F5F0E8",
    "Composite": "#EBF0F5",
    "Precast":   "#F5F5F5",
    "CIP":       "#F0F5EB",
}

# Legacy material palette (used by fallback material-breakdown path)
_MATERIAL_COLORS: dict[str, str] = {
    "concrete":        "#8C7B6B",
    "rebar":           "#C0392B",
    "structural_steel": "#2980B9",
    "pt":              "#8E44AD",
    "timber":          "#27AE60",
    "screed":          "#F39C12",
}

_MATERIAL_LABELS: dict[str, str] = {
    "concrete":         "Concrete",
    "rebar":            "Rebar",
    "structural_steel": "Structural Steel",
    "pt":               "Post-Tension",
    "timber":           "Timber",
    "screed":           "Screed / Topping",
}


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _comp_mat_segments(
    df: pd.DataFrame,
    exclude_lateral: bool = False,
) -> list[tuple[str, str, str]]:
    """Return (col_name, hex_color, legend_label) for each non-zero comp×mat segment."""
    result = []
    for comp in _COMP_ORDER:
        if exclude_lateral and comp == "lateral":
            continue
        for mat in _MAT_ORDER:
            col = f"{comp}_carbon_{mat}_per_m2"
            if col not in df.columns:
                continue
            if df[col].fillna(0.0).abs().sum() < 1e-6:
                continue
            color = _COMP_MAT_COLORS.get((comp, mat), "#CCCCCC")
            label = f"{_COMP_DISPLAY.get(comp, comp)}: {_MAT_DISPLAY.get(mat, mat)}"
            result.append((col, color, label))
    return result


def _lateral_carbon_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith("lateral_carbon_") and c.endswith("_per_m2")]


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------

def plot_structural_class_comparison(
    df_best: pd.DataFrame,
    *,
    title: str = "Superstructure Options — Embodied Carbon Comparison",
    subtitle: str | None = None,
    carbon_col: str = "total_embodied_carbon_per_m2",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show_values: bool = True,
    show_breakdown_table: bool = True,
    exclude_lateral: bool = False,
) -> plt.Figure:
    """Draw a horizontal stacked bar chart for structural class comparison.

    Bars are stacked by (component × material) when those columns are present,
    otherwise falls back to per-component stacking.

    Parameters
    ----------
    exclude_lateral:
        If True, omit lateral-system carbon from every bar.
    """
    if df_best is None or df_best.empty:
        raise ValueError("df_best is empty — nothing to plot.")

    df = _prepare_assembly_plot_df(df_best)
    df[carbon_col] = pd.to_numeric(df[carbon_col], errors="coerce")
    df = df.dropna(subset=[carbon_col, "structural_class"])
    df = df.sort_values(carbon_col, ascending=True).reset_index(drop=True)

    # Alias for legacy column name used in some DataFrames
    if "primary_beam_carbon_per_m2" not in df.columns and "beam_carbon_per_m2" in df.columns:
        df["primary_beam_carbon_per_m2"] = df["beam_carbon_per_m2"]

    # Decide stacking mode
    cm_segs = _comp_mat_segments(df, exclude_lateral=exclude_lateral)
    if cm_segs:
        seg_cols   = [s[0] for s in cm_segs]
        seg_colors = [s[1] for s in cm_segs]
        seg_labels = [s[2] for s in cm_segs]
        numeric_cols = [c for c in seg_cols if c in df.columns]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df["_plot_total"] = df[seg_cols].sum(axis=1)
        # For the breakdown table: aggregate back to per-component totals
        _comp_totals_for_table = _aggregate_comp_totals(df, exclude_lateral)
    else:
        # Legacy per-component stacking
        components = [c for c in ["floor", "primary_beam", "secondary_beam", "column", "lateral"]
                      if f"{c}_carbon_per_m2" in df.columns]
        if exclude_lateral:
            components = [c for c in components if c != "lateral"]
        seg_cols   = [f"{c}_carbon_per_m2" for c in components]
        seg_colors = [_COMPONENT_COLORS.get(c, "#AAAAAA") for c in components]
        seg_labels = [_COMPONENT_LABELS.get(c, c) for c in components]
        existing = [c for c in seg_cols if c in df.columns]
        df[existing] = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df["_plot_total"] = df[seg_cols].sum(axis=1)
        _comp_totals_for_table = None  # table uses seg_cols directly

    n_rows   = len(df)
    if show_breakdown_table and n_rows > 24:
        show_breakdown_table = False
    bar_height = 0.62
    row_height = 0.62
    chart_height = max(5.0, n_rows * row_height + 1.6)

    if figsize is None:
        figsize = (13.0, chart_height + (2.5 if show_breakdown_table else 0))

    fig = plt.figure(figsize=figsize, facecolor="white")

    if show_breakdown_table:
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[chart_height, 2.0],
            hspace=0.08,
            left=0.34, right=0.86, top=0.90, bottom=0.05,
        )
        ax = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis("off")
    else:
        ax = fig.add_axes([0.34, 0.09, 0.52, 0.80])
        ax_table = None

    _draw_group_strips(ax, df, n_rows, row_height)

    y_positions = np.arange(n_rows)
    pos_lefts = np.zeros(n_rows)
    neg_lefts = np.zeros(n_rows)
    bar_handles: list[mpatches.Patch] = []

    for col, color, label in zip(seg_cols, seg_colors, seg_labels):
        values = df[col].values
        lefts = np.where(values >= 0, pos_lefts, neg_lefts)
        ax.barh(y_positions, values, left=lefts, height=bar_height,
                color=color, edgecolor="white", linewidth=0.3)
        pos_lefts += np.where(values >= 0, values, 0.0)
        neg_lefts += np.where(values < 0, values, 0.0)
        bar_handles.append(mpatches.Patch(color=color, label=label))

    visible_totals = pos_lefts + neg_lefts
    df["_plot_total"] = visible_totals
    df["_carbon_display"] = np.rint(visible_totals).astype(int)
    totals_for_axis = np.concatenate([pos_lefts, neg_lefts, df["_plot_total"].values])
    xmin = min(0.0, float(np.nanmin(totals_for_axis)))
    xmax = max(0.0, float(np.nanmax(totals_for_axis)))
    pad = max((xmax - xmin) * 0.08, 5.0)

    if show_values:
        for i, label in enumerate(df["_carbon_display"]):
            total = float(visible_totals[i])
            x = total + (pad * 0.12 if total >= 0 else -pad * 0.12)
            ax.text(x, i, f"{label:,}",
                    va="center", ha="left" if total >= 0 else "right", fontsize=8.2,
                    fontweight="bold", color="#333333")

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["_plot_label"], fontsize=7.4)
    ax.invert_yaxis()
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Embodied carbon (kgCO2e / m2 GFA)", fontsize=9, labelpad=6)
    ax.set_xlim(xmin - pad, xmax + pad * 1.6)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.set_title(
        title + (f"\n{subtitle}" if subtitle else ""),
        fontsize=12,
        fontweight="bold",
        pad=12,
        loc="left",
        linespacing=1.35,
    )

    # Legend — wrap into multiple columns to handle many segments
    n_legend_cols = 1 if len(bar_handles) > 10 else 2
    ax.legend(handles=bar_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.2,
              framealpha=0.95, edgecolor="#CCCCCC", ncol=n_legend_cols, title="Component / material",
              title_fontsize=7.6)

    if show_breakdown_table and ax_table is not None:
        if _comp_totals_for_table is not None:
            _draw_comp_totals_table(ax_table, df, _comp_totals_for_table, exclude_lateral)
        else:
            _draw_breakdown_table(ax_table, df,
                                  [s.replace("_carbon_per_m2", "") for s in seg_cols], seg_cols)

    if "span_x_m" in df.columns:
        span_vals = df["span_x_m"].dropna()
        if not span_vals.empty:
            ax.text(0.01, 0.99, f"Span: {span_vals.iloc[-1]:.1f} m",
                    transform=ax.transAxes, fontsize=7.5, color="#888888", va="top", ha="left")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return fig


def plot_material_breakdown_comparison(
    df_assemblies: pd.DataFrame,
    *,
    title: str = "Superstructure Options — Embodied Carbon by Material (kgCO₂e/m² GFA)",
    subtitle: str | None = None,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show_values: bool = True,
    exclude_lateral: bool = False,
) -> plt.Figure:
    """Horizontal stacked bar chart stacked by (component × material).

    Parameters
    ----------
    exclude_lateral:
        If True, omit lateral-system carbon from every bar.
    """
    if df_assemblies is None or df_assemblies.empty:
        raise ValueError("df_assemblies is empty — nothing to plot.")

    df = _prepare_assembly_plot_df(df_assemblies)
    df["total_embodied_carbon_per_m2"] = pd.to_numeric(
        df["total_embodied_carbon_per_m2"], errors="coerce"
    )
    df = df.dropna(subset=["total_embodied_carbon_per_m2", "structural_class"])
    df = df.sort_values("total_embodied_carbon_per_m2", ascending=True).reset_index(drop=True)

    materials = [m for m in _MAT_ORDER if f"mat_{m}_per_m2" in df.columns]
    if exclude_lateral:
        for mat in materials:
            lat_col = f"lateral_carbon_{mat}_per_m2"
            if lat_col in df.columns:
                df[f"mat_{mat}_per_m2"] = pd.to_numeric(df[f"mat_{mat}_per_m2"], errors="coerce").fillna(0.0) - pd.to_numeric(df[lat_col], errors="coerce").fillna(0.0)
    seg_cols = [f"mat_{m}_per_m2" for m in materials if pd.to_numeric(df[f"mat_{m}_per_m2"], errors="coerce").fillna(0.0).abs().sum() > 1e-6]
    if not seg_cols:
        cm_segs = _comp_mat_segments(df, exclude_lateral=exclude_lateral)
        seg_cols = [s[0] for s in cm_segs]
        seg_colors = [s[1] for s in cm_segs]
        seg_labels = [s[2] for s in cm_segs]
    else:
        seg_colors = [_MATERIAL_COLORS.get(c.replace("mat_", "").replace("_per_m2", ""), "#AAAAAA") for c in seg_cols]
        seg_labels = [_MATERIAL_LABELS.get(c.replace("mat_", "").replace("_per_m2", ""), c) for c in seg_cols]
    df[seg_cols] = df[seg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n_rows = len(df)
    bar_height = 0.62
    row_height = 0.62
    chart_height = max(4.0, n_rows * row_height + 1.5)
    if figsize is None:
        figsize = (13.5, chart_height)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.subplots_adjust(left=0.34, right=0.84, top=0.90, bottom=0.08)

    _draw_group_strips(ax, df, n_rows, row_height)

    y_positions = np.arange(n_rows)
    pos_lefts = np.zeros(n_rows)
    neg_lefts = np.zeros(n_rows)
    bar_handles: list[mpatches.Patch] = []

    for col, color, label in zip(seg_cols, seg_colors, seg_labels):
        values = df[col].values
        lefts = np.where(values >= 0, pos_lefts, neg_lefts)
        ax.barh(y_positions, values, left=lefts, height=bar_height,
                color=color, edgecolor="white", linewidth=0.3)
        pos_lefts += np.where(values >= 0, values, 0.0)
        neg_lefts += np.where(values < 0, values, 0.0)
        bar_handles.append(mpatches.Patch(color=color, label=label))

    totals = df[seg_cols].sum(axis=1).values
    xmin = min(0.0, float(np.nanmin(np.concatenate([neg_lefts, totals]))))
    xmax = max(0.0, float(np.nanmax(np.concatenate([pos_lefts, totals]))))
    pad = max((xmax - xmin) * 0.08, 5.0)

    if show_values:
        for i, total in enumerate(totals):
            ax.text(total + (pad * 0.12 if total >= 0 else -pad * 0.12), i, f"{total:.0f}",
                    va="center", ha="left" if total >= 0 else "right", fontsize=8.2,
                    fontweight="bold", color="#333333")

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["_plot_label"], fontsize=7.4)
    ax.invert_yaxis()
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Embodied carbon (kgCO2e / m2 GFA)", fontsize=9, labelpad=6)
    ax.set_xlim(xmin - pad, xmax + pad * 1.6)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.set_title(
        title + (f"\n{subtitle}" if subtitle else ""),
        fontsize=12,
        fontweight="bold",
        pad=12,
        loc="left",
        linespacing=1.35,
    )

    ax.legend(handles=bar_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.4,
              framealpha=0.95, edgecolor="#CCCCCC", title="Material", title_fontsize=7.8)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_comp_totals(
    df: pd.DataFrame, exclude_lateral: bool
) -> list[tuple[str, str, str]]:
    """Return (df_col, color, label) list of per-component aggregate columns for the table."""
    comps = [c for c in _COMP_ORDER if not (exclude_lateral and c == "lateral")]
    result = []
    for comp in comps:
        mat_cols = [f"{comp}_carbon_{m}_per_m2" for m in _MAT_ORDER
                    if f"{comp}_carbon_{m}_per_m2" in df.columns]
        if not mat_cols:
            continue
        agg_col = f"_agg_{comp}"
        df[agg_col] = df[mat_cols].fillna(0.0).sum(axis=1)
        color = _COMPONENT_COLORS.get(
            "primary_beam" if comp == "beam" else
            "secondary_beam" if comp == "sec_beam" else comp,
            "#AAAAAA"
        )
        label = _COMP_DISPLAY.get(comp, comp)
        result.append((agg_col, color, label))
    return result


def _draw_group_strips(ax: plt.Axes, df: pd.DataFrame, n_rows: int, row_height: float) -> None:
    if "structural_class" not in df.columns:
        return
    groups = [_class_to_group(c) for c in df["structural_class"]]
    prev_group = None
    stripe = False
    for i, group in enumerate(groups):
        if group != prev_group:
            stripe = not stripe
            prev_group = group
        if stripe:
            color = _GROUP_COLORS.get(group, "#F8F8F8")
            ax.axhspan(i - 0.5, i + 0.5, color=color, alpha=0.6, linewidth=0, zorder=0)


def _class_to_group(label: str) -> str:
    for group in _GROUP_ORDER:
        if label.startswith(group):
            return group
    return "Other"


def _draw_comp_totals_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    comp_totals: list[tuple[str, str, str]],
    exclude_lateral: bool,
) -> None:
    """Render a component-level summary table below the bars (comp×mat mode)."""
    ax.axis("off")
    col_labels = [t[2] for t in comp_totals] + ["TOTAL"]
    row_labels  = list(df["structural_class"])
    header_colors = [t[1] for t in comp_totals] + ["#444444"]

    table_cols = [t[0] for t in comp_totals] + ["_plot_total"]
    cell_data = df[table_cols].round(0).astype(int).astype(str).values.tolist()

    cell_colors = [["#FAFAFA"] * len(col_labels) for _ in row_labels]
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="upper center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    tbl.scale(1, 1.3)
    for j, hc in enumerate(header_colors):
        cell = tbl[(0, j)]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(row_labels) + 1):
        try:
            tbl[(i, -1)].set_width(0.28)
        except KeyError:
            pass


def _draw_breakdown_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    components: list[str],
    comp_cols: list[str],
) -> None:
    """Render per-component carbon breakdown table (legacy comp-stacking mode)."""
    ax.axis("off")
    col_labels = [_COMPONENT_LABELS.get(c, c) for c in components] + ["TOTAL"]
    row_labels  = list(df["structural_class"])
    table_cols = comp_cols + ["total_embodied_carbon_per_m2"]
    cell_data = df[table_cols].round(0).astype(int).astype(str).values.tolist()
    header_colors = [_COMPONENT_COLORS.get(c, "#888888") for c in components] + ["#444444"]
    cell_colors = [["#FAFAFA"] * len(col_labels) for _ in row_labels]
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="upper center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)
    for j, hc in enumerate(header_colors):
        cell = tbl[(0, j)]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(row_labels) + 1):
        try:
            tbl[(i, -1)].set_width(0.25)
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# Assembly comparison filters and sensitivity figures
# ---------------------------------------------------------------------------

_VERIFIED_FEASIBLE_RULES: tuple[dict[str, Any], ...] = (
    # Manual workbook coverage: EC2 solid/ribbed/flat/CIP slab and RC members.
    {"floor_category": "cast_in_place", "beam_material": "concrete", "column_material": "concrete"},
    {"floor_type": "flat_slab", "column_material": "concrete"},
    {"floor_type": "flat_slab_drop_panel", "column_material": "concrete"},
    {"floor_type": "pt_slab", "column_material": "concrete"},
    # Manual workbook coverage: starter EC3 steel beam, floor beam, and column.
    {"floor_type": "composite_deck", "beam_material": "steel", "column_material": "steel"},
    # Useful steel-frame alternatives where the floor is a verified/standard deck/slab family.
    {"floor_type": "hollowcore", "beam_material": "steel", "column_material": "steel"},
    {"floor_type": "solid_plank", "beam_material": "steel", "column_material": "steel"},
    # Emerging low-carbon timber/hybrid cases to keep visible in the selected set.
    {"floor_category": "timber", "beam_material": "timber"},
    {"floor_category": "timber", "beam_material": "steel", "column_material": "steel"},
)


def filter_verified_feasible_assemblies(df_assemblies: pd.DataFrame) -> pd.DataFrame:
    """Return the default verification-style subset of assembly typologies."""
    if df_assemblies is None or df_assemblies.empty:
        return pd.DataFrame()
    df = df_assemblies.copy()
    mask = pd.Series(False, index=df.index)
    for rule in _VERIFIED_FEASIBLE_RULES:
        m = pd.Series(True, index=df.index)
        for col, expected in rule.items():
            if col not in df.columns:
                m &= False
                continue
            m &= df[col].fillna("").astype(str).str.lower().eq(str(expected).lower())
        mask |= m
    return df[mask].copy()


def write_verified_feasible_comparison_chart(
    df_assemblies: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Verified Feasible Structural Typologies — Embodied Carbon",
) -> Path | None:
    """Write a polished comparison chart for the verification-style typology subset."""
    subset = filter_verified_feasible_assemblies(df_assemblies)
    if subset.empty:
        return None
    out_path = Path(out_path)
    plot_structural_class_comparison(
        subset,
        title=title,
        subtitle="Subset aligned with the manual EC2/EC3 verification workbook coverage",
        out_path=out_path,
        show_breakdown_table=False,
    )
    return out_path


def write_selected_typology_comparison_charts(
    df_assemblies: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write selected verified + emerging typology comparison charts."""
    out_dir = Path(out_dir)
    subset = filter_verified_feasible_assemblies(df_assemblies)
    if subset.empty:
        return {}
    title = "Selected Structural Typologies — Verified + Emerging Options"
    subtitle = "Manual EC2/EC3 verification set plus low-carbon timber and timber/steel emerging options"
    outputs: dict[str, Path] = {}
    specs = [
        ("comparison_chart_verified_feasible.png", False, plot_structural_class_comparison, title),
        ("comparison_chart_verified_feasible_no_lateral.png", True, plot_structural_class_comparison, title + " (Excluding Lateral)"),
        ("material_breakdown_chart_verified_feasible.png", False, plot_material_breakdown_comparison, title + " — Material Breakdown"),
        ("material_breakdown_chart_verified_feasible_no_lateral.png", True, plot_material_breakdown_comparison, title + " — Material Breakdown (Excluding Lateral)"),
    ]
    for filename, exclude_lateral, plotter, chart_title in specs:
        path = out_dir / filename
        kwargs = {
            "title": chart_title,
            "subtitle": subtitle + (" | excl. lateral system" if exclude_lateral else ""),
            "out_path": path,
            "exclude_lateral": exclude_lateral,
        }
        if plotter is plot_structural_class_comparison:
            kwargs["show_breakdown_table"] = False
        plotter(subset, **kwargs)
        outputs[path.stem] = path
    return outputs


def write_assembly_sensitivity_figure_bundle(
    df_assemblies: pd.DataFrame,
    out_dir: str | Path,
    *,
    component_candidates_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write assembly and component sensitivity-style figures from current run outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if df_assemblies is not None and not df_assemblies.empty:
        asm = _prepare_assembly_plot_df(df_assemblies)
        _write_plot(paths, "all_assemblies_span_vs_carbon_by_governing_material", out_dir,
                    lambda p: _plot_assembly_scatter(asm, p))
        _write_plot(paths, "verified_feasible_boxplot_by_typology", out_dir,
                    lambda p: _plot_assembly_boxplot(filter_verified_feasible_assemblies(asm), p))
        _write_plot(paths, "verified_feasible_stacked_components", out_dir,
                    lambda p: _plot_verified_component_stacked(filter_verified_feasible_assemblies(asm), p))
        _write_plot(paths, "assembly_material_share_stacked", out_dir,
                    lambda p: _plot_assembly_material_share(asm, p))

    if component_candidates_df is not None and not component_candidates_df.empty:
        comp = component_candidates_df.copy()
        _write_plot(paths, "floor_span_load_carbon_envelope", out_dir,
                    lambda p: _plot_floor_span_load_envelope(comp, p))
        _write_plot(paths, "component_carbon_by_depth", out_dir,
                    lambda p: _plot_component_depth_sensitivity(comp, p))
        _write_plot(paths, "component_pass_rate_by_system_type", out_dir,
                    lambda p: _plot_component_pass_rate(comp, p))
    return paths


def _prepare_assembly_plot_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "assembly_family" not in df.columns:
        df["assembly_family"] = df.apply(_assembly_family_from_row, axis=1)
    if "governing_material" not in df.columns:
        mat_cols = [c for c in df.columns if c.startswith("mat_") and c.endswith("_per_m2")]
        if mat_cols:
            df["governing_material"] = df[mat_cols].abs().idxmax(axis=1).str.replace("mat_", "", regex=False).str.replace("_per_m2", "", regex=False).map(_pretty_label)
        else:
            df["governing_material"] = "Unknown"
    if "demand_span_m" not in df.columns:
        df["demand_span_m"] = df.get("span_x_m", np.nan)
    if "demand_factored_load_kpa" not in df.columns:
        df["demand_factored_load_kpa"] = df.get("total_load", np.nan)
    if "floor_overall_depth" not in df.columns:
        df["floor_overall_depth"] = np.nan
    df["_plot_label"] = df["structural_class"].astype(str).map(_compact_assembly_label)
    return df


def _assembly_family_from_row(row: pd.Series) -> str:
    floor = _pretty_label(row.get("floor_type", row.get("floor_category", "Floor")))
    col = _pretty_label(row.get("column_material", "Column"))
    beam_raw = row.get("beam_material")
    if pd.isna(beam_raw) or str(beam_raw).lower() in {"", "none", "nan"}:
        return f"{floor} + {col} Columns"
    beam = _pretty_label(beam_raw)
    return f"{floor} + {beam}/{col}"


def _compact_assembly_label(label: str) -> str:
    text = str(label)
    replacements = {
        "Primary Beams": "Prim.",
        "Secondary Beams": "Sec.",
        "Columns": "Cols",
        "Floor,": "Floor,",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 3:
        return "\n".join([parts[0], ", ".join(parts[1:])])
    return text


def _write_plot(paths: dict[str, Path], name: str, out_dir: Path, plotter: Any) -> None:
    path = out_dir / f"{name}.png"
    try:
        fig = plotter(path)
        if fig is not None:
            paths[name] = path
    except Exception:
        plt.close("all")


def _plot_assembly_scatter(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    if df.empty:
        return None
    d = df.copy()
    d["total_embodied_carbon_per_m2"] = pd.to_numeric(d["total_embodied_carbon_per_m2"], errors="coerce")
    d["demand_span_m"] = pd.to_numeric(d.get("demand_span_m"), errors="coerce")
    if d["demand_span_m"].isna().all():
        d["demand_span_m"] = d["rank_carbon"] if "rank_carbon" in d.columns else np.arange(len(d)) + 1
        x_label = "Assembly rank (span not available in this output)"
    else:
        x_label = "Grid span (m)"
    d = d.dropna(subset=["demand_span_m", "total_embodied_carbon_per_m2"])
    if d.empty:
        return None
    if d["demand_span_m"].nunique(dropna=True) <= 1 and len(d) > 1:
        base = float(d["demand_span_m"].iloc[0])
        d["demand_span_m"] = base + np.linspace(-0.06, 0.06, len(d))
        x_label = f"Grid span (m), jittered around {base:g} m for current single-span run"
    fig, ax = plt.subplots(figsize=(10.8, 6.4), facecolor="white")
    palette = _category_palette(d["governing_material"])
    for mat, grp in d.groupby("governing_material", dropna=False):
        ax.scatter(grp["demand_span_m"], grp["total_embodied_carbon_per_m2"], s=54, alpha=0.78,
                   color=palette.get(str(mat), "#777777"), edgecolor="white", linewidth=0.35, label=str(mat))
    _style_component_ax(ax, "All Assembly Options — Carbon vs Span", x_label=x_label, y_label="Embodied carbon (kgCO2e / m2 GFA)")
    ax.legend(loc="best", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC", title="Governing material", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_assembly_boxplot(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    if df.empty or "assembly_family" not in df.columns:
        return None
    d = df.copy()
    d["total_embodied_carbon_per_m2"] = pd.to_numeric(d["total_embodied_carbon_per_m2"], errors="coerce")
    groups = [(k, g["total_embodied_carbon_per_m2"].dropna().values) for k, g in d.groupby("assembly_family") if len(g)]
    if not groups:
        return None
    groups = sorted(groups, key=lambda item: np.nanmedian(item[1]))
    labels, data = zip(*groups)
    fig, ax = plt.subplots(figsize=(11.0, max(4.8, 0.5 * len(labels) + 1.8)), facecolor="white")
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=labels, showfliers=True)
    palette = _category_palette(pd.Series(labels))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(palette[str(label)])
        patch.set_alpha(0.70)
    _style_component_ax(ax, "Verified Feasible Typologies — Carbon Distribution", x_label="Embodied carbon (kgCO2e / m2 GFA)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_verified_component_stacked(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    if df.empty:
        return None
    return _plot_component_total_stacked(df.sort_values("total_embodied_carbon_per_m2"), out_path, "Verified Feasible Typologies — Component Carbon")


def _plot_component_total_stacked(df: pd.DataFrame, out_path: Path, title: str) -> plt.Figure | None:
    cols = [
        ("floor_carbon_per_m2", "#D4692A", "Floor"),
        ("beam_carbon_per_m2", "#3D6FA0", "Primary beam"),
        ("secondary_beam_carbon_per_m2", "#6AAAD5", "Secondary beam"),
        ("column_carbon_per_m2", "#5C5C5C", "Column"),
        ("lateral_carbon_per_m2", "#A63A3A", "Lateral"),
    ]
    cols = [(c, color, label) for c, color, label in cols if c in df.columns]
    if not cols:
        return None
    d = df.copy().head(24)
    for c, _, _ in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    labels = d["assembly_family"] if "assembly_family" in d.columns else d["structural_class"].map(_compact_assembly_label)
    fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.48 * len(d) + 1.8)), facecolor="white")
    y = np.arange(len(d))
    pos = np.zeros(len(d))
    neg = np.zeros(len(d))
    handles = []
    for c, color, label in cols:
        vals = d[c].values
        left = np.where(vals >= 0, pos, neg)
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.35, height=0.62)
        pos += np.where(vals >= 0, vals, 0)
        neg += np.where(vals < 0, vals, 0)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    _style_component_ax(ax, title, x_label="Embodied carbon (kgCO2e / m2 GFA)")
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_assembly_material_share(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    if df.empty:
        return None
    d = df.sort_values("total_embodied_carbon_per_m2").head(24).copy()
    cols = [f"mat_{m}_per_m2" for m in _MAT_ORDER if f"mat_{m}_per_m2" in d.columns]
    if not cols:
        return None
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    denom = d[cols].sum(axis=1).replace(0, np.nan)
    d[cols] = d[cols].div(denom, axis=0).fillna(0.0) * 100
    labels = d["assembly_family"] if "assembly_family" in d.columns else d["structural_class"].map(_compact_assembly_label)
    fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.48 * len(d) + 1.8)), facecolor="white")
    y = np.arange(len(d))
    left = np.zeros(len(d))
    handles = []
    for col in cols:
        mat = col.replace("mat_", "").replace("_per_m2", "")
        vals = d[col].values
        ax.barh(y, vals, left=left, height=0.62, color=_MATERIAL_COLORS.get(mat, "#AAAAAA"), edgecolor="white", linewidth=0.35)
        left += vals
        handles.append(mpatches.Patch(color=_MATERIAL_COLORS.get(mat, "#AAAAAA"), label=_MATERIAL_LABELS.get(mat, _pretty_label(mat))))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_component_ax(ax, "Assembly Material Shares — Lowest Carbon Options", x_label="Share of absolute carbon contribution (%)")
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_floor_span_load_envelope(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    d = df[df.get("component_type", df.get("component", "")).astype(str).str.lower().eq("floor")].copy()
    if d.empty or "max_span" not in d.columns:
        return None
    d["max_span"] = pd.to_numeric(d["max_span"], errors="coerce")
    d["_load"] = _first_numeric_series(d, ["total_capacity", "sdl_total", "ll", "load_capacity"])
    d["carbon_per_m2"] = pd.to_numeric(d.get("carbon_per_m2"), errors="coerce")
    d = d.dropna(subset=["max_span", "_load", "carbon_per_m2"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 6.4), facecolor="white")
    palette = _category_palette(d["type"] if "type" in d.columns else pd.Series(["Floor"] * len(d)))
    for typ, grp in d.groupby("type" if "type" in d.columns else lambda _: "Floor"):
        ax.scatter(grp["max_span"], grp["_load"], s=np.clip(grp["carbon_per_m2"].abs(), 12, 140), alpha=0.55,
                   color=palette.get(str(typ), "#777777"), edgecolor="white", linewidth=0.25, label=_pretty_label(typ))
    _style_component_ax(ax, "Floor System Sensitivity — Span, Load and Carbon", x_label="Maximum span (m)", y_label="Catalog load / capacity")
    if len(palette) <= 14:
        ax.legend(loc="best", fontsize=7.5, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_component_depth_sensitivity(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    d = df.copy()
    d["carbon_per_m2"] = pd.to_numeric(d.get("carbon_per_m2"), errors="coerce")
    d["_depth"] = _first_numeric_series(d, ["overall_depth", "slab_depth", "beam_depth", "column_depth", "wall_thickness"])
    d = d.dropna(subset=["_depth", "carbon_per_m2"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 6.4), facecolor="white")
    color_col = "component_type" if "component_type" in d.columns else "component"
    palette = _category_palette(d[color_col])
    for comp, grp in d.groupby(color_col):
        ax.scatter(grp["_depth"], grp["carbon_per_m2"], s=28, alpha=0.62, color=palette.get(str(comp), "#777777"),
                   edgecolor="white", linewidth=0.25, label=_pretty_label(comp))
    _style_component_ax(ax, "Component Sensitivity — Depth vs Carbon", x_label="Depth / thickness (catalog units)", y_label="Embodied carbon (kgCO2e / m2 GFA)")
    ax.legend(loc="best", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_component_pass_rate(df: pd.DataFrame, out_path: Path) -> plt.Figure | None:
    if "pass_overall" not in df.columns:
        return None
    d = df.copy()
    group_col = "component_type" if "component_type" in d.columns else "component"
    d["_pass"] = d["pass_overall"].astype(str).str.lower().isin(["true", "1", "yes"])
    s = d.groupby([group_col, "typology" if "typology" in d.columns else group_col])["_pass"].mean().reset_index()
    s["label"] = s[group_col].astype(str).map(_pretty_label) + " / " + s[s.columns[1]].astype(str).map(_pretty_label)
    s = s.sort_values("_pass")
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.34 * len(s) + 1.8)), facecolor="white")
    ax.barh(np.arange(len(s)), s["_pass"] * 100, color="#6E8FB5", edgecolor="white", linewidth=0.4)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s["label"], fontsize=7.5)
    _style_component_ax(ax, "Feasibility Sensitivity — Pass Rate by Component Typology", x_label="Pass rate (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Component-catalog figure bundle
# ---------------------------------------------------------------------------

_COMPONENT_TITLES: dict[str, str] = {
    "floor": "Floor Systems",
    "beam": "Beam Systems",
    "column": "Column Systems",
    "lateral": "Lateral Systems",
    "wall": "Wall Systems",
    "foundation": "Foundation Systems",
    "cladding": "Cladding Systems",
}

_COMPONENT_MEASURE_LABELS: dict[str, str] = {
    "floor": "Span (m)",
    "beam": "Moment / load capacity",
    "column": "Axial capacity (kN)",
    "lateral": "Maximum storeys",
    "wall": "Maximum storeys",
    "foundation": "Demand",
    "cladding": "Demand",
}


def write_component_figure_bundle(
    component_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    component: str,
    title_prefix: str | None = None,
    top_n: int = 18,
) -> dict[str, Path]:
    """Write floor-style diagnostic figures for a component catalogue.

    The bundle intentionally mirrors the legacy floor-only folder naming where
    the available columns make sense, but works for beams, columns, laterals and
    future component catalogues with the same summary schema.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if component_df is None or component_df.empty:
        return {}

    df = _prepare_component_plot_df(component_df, component)
    if df.empty:
        return {}

    title_prefix = title_prefix or _COMPONENT_TITLES.get(component, component.replace("_", " ").title())
    paths: dict[str, Path] = {}

    def _save(name: str, plotter: Any) -> None:
        path = out_dir / f"{name}.png"
        try:
            fig = plotter(path)
            if fig is not None:
                paths[name] = path
        except Exception:
            # Plot generation is intentionally best-effort. The caller can log
            # the returned paths while one missing chart should not sink a run.
            plt.close("all")

    _save(
        "bar_best_per_type",
        lambda path: _plot_component_best_bar(
            _best_component_rows(df, "_plot_type"),
            group_col="_plot_type",
            title=f"{title_prefix} — Lowest Carbon per Type",
            out_path=path,
        ),
    )
    _save(
        "bar_best_per_typology",
        lambda path: _plot_component_best_bar(
            _best_component_rows(df, "typology"),
            group_col="typology",
            title=f"{title_prefix} — Lowest Carbon per Typology",
            out_path=path,
        ),
    )
    _save(
        "scatter_successful_span_vs_carbon_by_type",
        lambda path: _plot_component_scatter(
            df[df["_success"]],
            x_col="_x_metric",
            color_col="_plot_type",
            title=f"{title_prefix} — Passing Options",
            x_label=_component_x_label(component),
            out_path=path,
        ),
    )
    _save(
        "scatter_all_span_vs_carbon_by_type",
        lambda path: _plot_component_scatter(
            df,
            x_col="_x_metric",
            color_col="_plot_type",
            title=f"{title_prefix} — All Options",
            x_label=_component_x_label(component),
            out_path=path,
        ),
    )
    _save(
        "span_vs_carbon",
        lambda path: _plot_component_scatter(
            df[df["_success"]],
            x_col="_x_metric",
            color_col="typology",
            title=f"{title_prefix} — Carbon vs Demand",
            x_label=_component_x_label(component),
            out_path=path,
        ),
    )
    if "_total_load" in df.columns and df["_total_load"].notna().any():
        _save(
            "span_vs_total_load",
            lambda path: _plot_component_scatter(
                df[df["_success"]],
                x_col="_x_metric",
                y_col="_total_load",
                color_col="_plot_type",
                title=f"{title_prefix} — Demand Envelope",
                x_label=_component_x_label(component),
                y_label="Total load / capacity demand",
                out_path=path,
            ),
        )
    if "_depth_metric" in df.columns and df["_depth_metric"].notna().any():
        _save(
            "depth_vs_carbon",
            lambda path: _plot_component_scatter(
                df[df["_success"]],
                x_col="_depth_metric",
                color_col="_plot_type",
                title=f"{title_prefix} — Depth vs Carbon",
                x_label=_component_depth_label(component),
                out_path=path,
            ),
        )
    _save(
        "carbon_distribution_by_type",
        lambda path: _plot_component_distribution(
            df[df["_success"]],
            group_col="_plot_type",
            title=f"{title_prefix} — Carbon Distribution by Type",
            out_path=path,
        ),
    )
    _save(
        "carbon_breakdown_top",
        lambda path: _plot_component_material_stacked(
            df[df["_success"]].sort_values("carbon_per_m2").head(top_n),
            title=f"{title_prefix} — Carbon Breakdown, Top {top_n}",
            out_path=path,
        ),
    )
    _save(
        "carbon_composition_stacked",
        lambda path: _plot_component_material_stacked(
            _best_component_rows(df[df["_success"]], "_plot_type"),
            title=f"{title_prefix} — Best Type Material Composition",
            out_path=path,
        ),
    )
    _save(
        "carbon_composition_share",
        lambda path: _plot_component_material_share(
            _best_component_rows(df[df["_success"]], "_plot_type"),
            title=f"{title_prefix} — Best Type Material Share",
            out_path=path,
        ),
    )

    for group_col in ("system_family", "manufacturer"):
        if group_col in df.columns and df[group_col].notna().any():
            slug = group_col
            best = _best_component_rows(df[df["_success"]], group_col).head(top_n)
            _save(
                f"lowest_per_{slug}_span_vs_carbon",
                lambda path, best=best, group_col=group_col: _plot_component_scatter(
                    best,
                    x_col="_x_metric",
                    color_col=group_col,
                    title=f"{title_prefix} — Lowest Carbon per {group_col.replace('_', ' ').title()}",
                    x_label=_component_x_label(component),
                    out_path=path,
                    label_points=True,
                ),
            )

    return paths


def _prepare_component_plot_df(df_in: pd.DataFrame, component: str) -> pd.DataFrame:
    df = df_in.copy()
    if "carbon_per_m2" not in df.columns and "carbon_total_per_m2" in df.columns:
        df["carbon_per_m2"] = df["carbon_total_per_m2"]
    if "carbon_per_m2" not in df.columns:
        return pd.DataFrame()
    df["carbon_per_m2"] = pd.to_numeric(df["carbon_per_m2"], errors="coerce")
    df = df.dropna(subset=["carbon_per_m2"]).reset_index(drop=True)
    if df.empty:
        return df

    for col in (
        "type", "typology", "system_family", "manufacturer", "pass_overall",
        "max_span", "span", "demand_span_m", "demand_N_kN", "demand_storeys",
        "total_load", "demand_total_unfactored_kpa", "demand_line_load_kNm",
        "load_capacity", "moment_capacity", "axial_capacity", "maximum_story_count",
        "beam_depth", "column_depth",
        "column_width", "wall_thickness", "overall_depth", "slab_depth",
    ):
        if col in df.columns and col not in ("type", "typology", "system_family", "manufacturer", "pass_overall"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("type", "typology", "system_family", "manufacturer", "category"):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).map(_pretty_label)
    if component in {"beam", "column"} and df["type"].nunique(dropna=True) > 35 and df["category"].nunique(dropna=True) > 1:
        df["_plot_type"] = df["category"]
    else:
        df["_plot_type"] = df["type"]

    if "pass_overall" in df.columns:
        df["_success"] = df["pass_overall"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "_success" in df.columns:
        df["_success"] = df["_success"].astype(bool)
    else:
        df["_success"] = True

    df["_x_metric"] = _first_numeric_series(
        df,
        _component_x_candidates(component),
    )
    if df["_x_metric"].isna().all():
        df["_x_metric"] = np.arange(len(df), dtype=float)

    df["_total_load"] = _first_numeric_series(
        df,
        ["total_load", "demand_total_unfactored_kpa", "demand_line_load_kNm", "load_capacity", "axial_capacity"],
    )
    df["_depth_metric"] = _component_depth_series(df, component)
    return df


def _component_x_candidates(component: str) -> list[str]:
    if component == "floor":
        return ["max_span", "span", "demand_span_m"]
    if component == "beam":
        return ["moment_capacity", "load_capacity", "max_span", "span", "beam_length", "demand_span_m"]
    if component == "column":
        return ["axial_capacity", "demand_N_kN", "demand_storeys", "demand_span_m"]
    if component in {"lateral", "wall"}:
        return ["maximum_story_count", "demand_storeys", "demand_span_m"]
    return ["demand_span_m", "max_span", "span"]


def _component_x_label(component: str) -> str:
    return _COMPONENT_MEASURE_LABELS.get(component, "Demand")


def _component_depth_label(component: str) -> str:
    if component == "beam":
        return "Beam depth (m)"
    if component == "column":
        return "Column max dimension (m)"
    if component in {"lateral", "wall"}:
        return "Wall thickness / frame depth"
    return "Overall depth (m)"


def _component_depth_series(df: pd.DataFrame, component: str) -> pd.Series:
    if component == "beam" and "beam_depth" in df.columns:
        return df["beam_depth"]
    if component == "column":
        cols = [c for c in ("column_depth", "column_width") if c in df.columns]
        if cols:
            return df[cols].max(axis=1)
    if component in {"lateral", "wall"}:
        return _first_numeric_series(df, ["wall_thickness", "frame_depth"])
    return _first_numeric_series(df, ["overall_depth", "slab_depth"])


def _first_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.fillna(vals)
    return out


def _best_component_rows(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["carbon_per_m2"]).copy()
    if d.empty:
        return d
    idx = d.groupby(group_col, dropna=False)["carbon_per_m2"].idxmin()
    return d.loc[idx].sort_values("carbon_per_m2").reset_index(drop=True)


def _pretty_label(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "Unknown"
    return text.replace("_", " ").replace("-", " ").title()


def _category_palette(values: pd.Series) -> dict[str, str]:
    vals = list(dict.fromkeys(values.fillna("Unknown").astype(str)))
    cmap_names = ["tab20", "Set2", "Dark2"]
    colors: list[str] = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N))
    return {v: colors[i % len(colors)] for i, v in enumerate(vals)}


def _style_component_ax(ax: plt.Axes, title: str, x_label: str | None = None, y_label: str | None = None) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    if x_label:
        ax.set_xlabel(x_label, fontsize=9)
    if y_label:
        ax.set_ylabel(y_label, fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _plot_component_best_bar(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    out_path: Path,
) -> plt.Figure | None:
    if df is None or df.empty or group_col not in df.columns:
        return None
    d = df.sort_values("carbon_per_m2", ascending=True).copy()
    height = max(4.0, 0.45 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, height), facecolor="white")
    colors = [_MATERIAL_COLORS.get(_dominant_material(row), "#6E8FB5") for _, row in d.iterrows()]
    y = np.arange(len(d))
    ax.barh(y, d["carbon_per_m2"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(d[group_col], fontsize=8)
    ax.invert_yaxis()
    for i, val in enumerate(d["carbon_per_m2"]):
        xoff = (d["carbon_per_m2"].max() - d["carbon_per_m2"].min() or 1) * 0.015
        ha = "left" if val >= 0 else "right"
        ax.text(val + (xoff if val >= 0 else -xoff), i, f"{val:.0f}", va="center", ha=ha, fontsize=8, fontweight="bold")
    _style_component_ax(ax, title, y_label=None, x_label="Embodied Carbon (kgCO2e / m2 GFA)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_component_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    title: str,
    x_label: str,
    out_path: Path,
    y_col: str = "carbon_per_m2",
    y_label: str = "Embodied Carbon (kgCO2e / m2 GFA)",
    color_col: str = "type",
    label_points: bool = False,
) -> plt.Figure | None:
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    palette = _category_palette(d[color_col] if color_col in d.columns else pd.Series(["All"] * len(d)))
    for key, grp in d.groupby(color_col if color_col in d.columns else lambda _: "All", dropna=False):
        label = str(key)
        ax.scatter(
            grp[x_col],
            grp[y_col],
            s=34,
            alpha=0.72,
            color=palette.get(label, "#6E8FB5"),
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        if label_points:
            for _, row in grp.head(10).iterrows():
                txt = str(row.get("system_family", row.get("manufacturer", "")))[:28]
                ax.annotate(txt, (row[x_col], row[y_col]), fontsize=6.5, alpha=0.72, xytext=(3, 3), textcoords="offset points")
    _style_component_ax(ax, title, x_label=x_label, y_label=y_label)
    if len(palette) <= 14:
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_component_distribution(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    out_path: Path,
) -> plt.Figure | None:
    if df is None or df.empty or group_col not in df.columns:
        return None
    d = df.dropna(subset=["carbon_per_m2"]).copy()
    groups = [(k, g["carbon_per_m2"].values) for k, g in d.groupby(group_col) if len(g)]
    if not groups:
        return None
    groups = sorted(groups, key=lambda item: np.nanmedian(item[1]))
    labels, data = zip(*groups)
    height = max(4.5, 0.4 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, height), facecolor="white")
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=labels, showfliers=False)
    palette = _category_palette(pd.Series(labels))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(palette[str(label)])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#555555")
    for med in bp["medians"]:
        med.set_color("#222222")
        med.set_linewidth(1.2)
    _style_component_ax(ax, title, x_label="Embodied Carbon (kgCO2e / m2 GFA)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _material_cols(df: pd.DataFrame) -> list[str]:
    cols = [f"carbon_{m}_per_m2" for m in ("concrete", "structural_steel", "rebar", "pt", "timber", "screed")]
    # carbon_steel_per_m2 is a legacy roll-up in some outputs. Use it only
    # when the disaggregated structural-steel/rebar columns are unavailable.
    if "carbon_structural_steel_per_m2" not in df.columns and "carbon_rebar_per_m2" not in df.columns:
        cols += ["carbon_steel_per_m2"]
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").fillna(0.0).abs().sum() > 1e-9]


def _material_label(col: str) -> str:
    key = col.replace("carbon_", "").replace("_per_m2", "")
    if key == "steel":
        key = "structural_steel"
    return _MATERIAL_LABELS.get(key, _pretty_label(key))


def _material_color(col: str) -> str:
    key = col.replace("carbon_", "").replace("_per_m2", "")
    if key == "steel":
        key = "structural_steel"
    return _MATERIAL_COLORS.get(key, "#AAAAAA")


def _dominant_material(row: pd.Series) -> str:
    vals = {}
    for col in _material_cols(pd.DataFrame([row])):
        vals[col] = abs(float(row.get(col, 0.0) or 0.0))
    if not vals:
        return "structural_steel"
    key = max(vals, key=vals.get).replace("carbon_", "").replace("_per_m2", "")
    return "structural_steel" if key == "steel" else key


def _component_row_label(row: pd.Series) -> str:
    for col in ("system_family", "system_variant", "type", "typology"):
        val = row.get(col)
        if pd.notna(val) and str(val):
            return str(val)[:52]
    return "Option"


def _plot_component_material_stacked(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> plt.Figure | None:
    if df is None or df.empty:
        return None
    d = df.copy().reset_index(drop=True)
    cols = _material_cols(d)
    if not cols:
        return None
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = [_component_row_label(row) for _, row in d.iterrows()]
    height = max(4.5, 0.45 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, height), facecolor="white")
    y = np.arange(len(d))
    pos_left = np.zeros(len(d))
    neg_left = np.zeros(len(d))
    handles = []
    for col in cols:
        values = d[col].values
        left = np.where(values >= 0, pos_left, neg_left)
        ax.barh(y, values, left=left, height=0.58, color=_material_color(col), edgecolor="white", linewidth=0.35)
        pos_left += np.where(values >= 0, values, 0)
        neg_left += np.where(values < 0, values, 0)
        handles.append(mpatches.Patch(color=_material_color(col), label=_material_label(col)))
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    _style_component_ax(ax, title, x_label="Embodied Carbon (kgCO2e / m2 GFA)")
    ax.legend(handles=handles, loc="best", fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC", ncol=min(3, len(handles)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _plot_component_material_share(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> plt.Figure | None:
    if df is None or df.empty:
        return None
    d = df.copy().reset_index(drop=True)
    cols = _material_cols(d)
    if not cols:
        return None
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    denom = d[cols].sum(axis=1).replace(0, np.nan)
    d[cols] = d[cols].div(denom, axis=0).fillna(0.0) * 100.0
    labels = [_component_row_label(row) for _, row in d.iterrows()]
    height = max(4.5, 0.45 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, height), facecolor="white")
    y = np.arange(len(d))
    left = np.zeros(len(d))
    handles = []
    for col in cols:
        values = d[col].values
        ax.barh(y, values, left=left, height=0.58, color=_material_color(col), edgecolor="white", linewidth=0.35)
        left += values
        handles.append(mpatches.Patch(color=_material_color(col), label=_material_label(col)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_component_ax(ax, title, x_label="Share of absolute carbon contribution (%)")
    ax.legend(handles=handles, loc="best", fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC", ncol=min(3, len(handles)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig
