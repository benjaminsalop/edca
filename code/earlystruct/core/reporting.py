# earlystruct/core/reporting.py
from __future__ import annotations

import math
import sys

import pandas as pd
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


# ---------- Best-per-group selection ----------

def best_per_group(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """
    Among feasible rows, pick the lowest value of `metric` for each group in `group_col`.
    Example group_col: "typology" or "type".
    """
    if df.empty:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    if group_col.lower() == "typology":
        df = add_typology(df)
    if group_col not in df.columns:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    filt = df[(df["feasible"] == True) & df[metric].notna() & (df[metric] > 0)]
    if filt.empty:
        return pd.DataFrame(
            columns=[group_col, "system_id", "system_name", metric, "depth_m", "span_m"]
        )

    idx = filt.groupby(group_col, as_index=False)[metric].idxmin()
    out = filt.loc[idx].copy()
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

def print_verbose_summary(df: pd.DataFrame, project: dict, ctl: dict, file=sys.stdout) -> None:
    """
    Pretty text summary for CLI usage.

    Expects df to contain:
      - feasibility + basic meta: feasible, type, system_id, system_name, category, manufacturer
      - spans: span_slab_dir_m, span_beam_dir_m (or span_input_m as a fallback)
      - geometry: depth_m, area_m2
      - quantities: concrete_m3, steel_m3, timber_m3
      - impacts: cost_total, cost_per_m2, carbon_total_kg, carbon_per_m2
    """
    if df.empty:
        print("No candidate systems generated.", file=file)
        return

    UNIT = ctl.get("UNIT", "metric").lower()
    is_imperial = UNIT.startswith("imp") or ("ft" in UNIT)

    # Project basics (these keys may vary in your control file)
    proj_name = project.get("PROJECT_NAME", "Unnamed project")
    floor_area_m2 = project.get("FLOOR_AREA_M2") or project.get("FLOOR_AREA") or None

    print("=== Parametric Floor Design Summary ===", file=file)
    print(f"Project: {proj_name}", file=file)
    print(f"Unit system: {UNIT}", file=file)

    if floor_area_m2:
        try:
            fa = float(floor_area_m2)
            print(f"Nominal floorplate area: {fa:.1f} m²", file=file)
        except Exception:
            pass

    # Settings: span sweep + one-way
    print("\n-- Span sweep --", file=file)
    if ctl.get("SPAN_SWEEP_FROM_MIN_BOOL", False):
        print("  Mode: sweep from minimum span", file=file)
        step_raw = ctl.get("SPAN_SWEEP_STEP")
        print(f"  Step in project units: {step_raw or '(default)'}", file=file)
    else:
        print("  Using explicit spans from control / CLI", file=file)

    print("\n-- One-way irregular slabs --", file=file)
    if ctl.get("ONE_WAY_IRREGULAR_BOOL", False):
        slab_min = ctl.get("ONE_WAY_SLAB_MIN_SPAN")
        beam_min = ctl.get("ONE_WAY_BEAM_MIN_SPAN")
        print("  Enabled", file=file)
        if slab_min:
            print(f"  Slab direction min span: {slab_min} ({UNIT})", file=file)
        if beam_min:
            print(f"  Beam direction min span: {beam_min} ({UNIT})", file=file)
    else:
        print("  Disabled", file=file)

    print("\n-- Best option per floor type (sorted by total cost) --", file=file)

    best = cheapest_span_by_type(df)
    if best.empty:
        print("  No feasible systems found.", file=file)
        return

    for _, row in best.sort_values("cost_total").iterrows():
        t = row.get("type", "")
        sys_id = row.get("system_id", "")
        area = row.get("area_m2", float("nan"))

        # Explicit slab / beam spans, falling back to base span if needed
        span_slab = row.get("span_slab_dir_m", row.get("span_input_m"))
        span_beam = row.get("span_beam_dir_m", row.get("span_input_m"))

        depth_m = row.get("depth_m", float("nan"))
        conc = row.get("concrete_m3", 0.0)
        steel = row.get("steel_m3", 0.0)
        timber = row.get("timber_m3", 0.0)

        carbon_total = row.get("carbon_total_kg", float("nan"))
        carbon_per_m2 = row.get("carbon_per_m2", float("nan"))
        cost_total = row.get("cost_total", float("nan"))
        cost_per_m2 = row.get("cost_per_m2", float("nan"))

        print(f"\n[Type: {t}]", file=file)
        print(f"  System: {sys_id}", file=file)

        if is_imperial:
            FT_PER_M = 3.28084
            print(
                "  Spans (slab × beam): "
                f"{span_slab * FT_PER_M:.2f} ft × {span_beam * FT_PER_M:.2f} ft",
                file=file,
            )
        else:
            print(
                "  Spans (slab × beam): "
                f"{span_slab:.2f} m × {span_beam:.2f} m",
                file=file,
            )

        if not pd.isna(depth_m):
            print(f"  Floor depth: {depth_m * 1000:.0f} mm", file=file)

        if not pd.isna(area):
            print(f"  Building floor area covered: {area:.0f} m²", file=file)

        print(
            "  Materials: "
            f"concrete {conc:.1f} m³, steel {steel:.1f} m³, timber {timber:.1f} m³",
            file=file,
        )

        if not pd.isna(cost_total):
            print(
                f"  Cost: {cost_total:,.0f} total "
                f"({cost_per_m2:,.1f} per m²)",
                file=file,
            )

        if not pd.isna(carbon_total):
            print(
                f"  Carbon: {carbon_total / 1000:,.1f} tCO₂e "
                f"({carbon_per_m2:,.1f} kgCO₂e/m²)",
                file=file,
            )