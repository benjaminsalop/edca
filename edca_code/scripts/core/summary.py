#!/usr/bin/env python3
"""
summary.py

Searches for all files named `summary_ranked_all.csv` under one or more root paths,
then writes two combined outputs:

 - total_summary_all.csv       : outer union (concat), exact-duplicate rows removed
 - compatible_summary_all.csv  : intersection based on a key column (default: system_variant).
 - summary_sources_provenance.csv : provenance (source dir, path, row count)

Usage:
    python summary.py --root . --root inputs --out outputs/reporting

Options:
    --root ROOT        Root path(s) to search for summary_ranked_all.csv (can be passed multiple times).
    --out OUTDIR       Output directory (default: outputs/reporting)
    --key KEY          Key column for compatibility intersection (default: system_variant).
                       To use a composite key, provide comma-separated column names, e.g. "system_variant,span"
    --skip-plot        Do not try to create diagnostic plot (span vs carbon_per_m2)
    --verbose / --quiet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import sys

def find_summary_files(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        if not root.exists():
            logging.warning("Root does not exist: %s", root)
            continue
        for p in root.rglob("summary_ranked_all.csv"):
            found.append(p.resolve())
    return sorted(set(found))

def read_and_tag(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    df["_source_path"] = str(p)
    df["_source_dir"] = str(p.parent.name)
    return df

def write_csv(df: pd.DataFrame, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False)

def compute_total_union(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    total = pd.concat(dfs, ignore_index=True, sort=False)
    total = total.drop_duplicates()
    return total

def compute_compatible_intersection(dfs: list[pd.DataFrame], key_cols: list[str]) -> pd.DataFrame:
    """
    If key_cols is a single column: keep rows whose key value appears in every source file.
    If key_cols is multiple: keep rows where the composite tuple appears in every file.
    """
    if not key_cols:
        raise ValueError("key_cols must be provided")

    # Convert each df to set of key tuples
    tuple_sets = []
    for d in dfs:
        # drop NA in key columns for counting presence
        subset = d.dropna(subset=key_cols)
        tuples = set(tuple(row[c] for c in key_cols) for _, row in subset.iterrows())
        tuple_sets.append(tuples)

    # intersection across sets
    common = set.intersection(*tuple_sets) if tuple_sets else set()
    if not common:
        # no common tuples -> return empty df with total columns
        logging.info("No common keys across all sources; returning empty compatible set.")
        return pd.DataFrame(columns=dfs[0].columns if dfs else [])

    # filter total union by presence in common
    total = compute_total_union(dfs)
    def in_common(row):
        return tuple(row[c] for c in key_cols) in common
    compat = total[ total.apply(in_common, axis=1) ].copy()
    return compat

def try_plot_span_vs_carbon(df: pd.DataFrame, out_fp: Path) -> bool:
    # create scatter plot if numeric columns exist
    if not {"span", "carbon_per_m2"}.issubset(set(df.columns)):
        logging.info("Skipping plot: required columns 'span' and 'carbon_per_m2' not both present.")
        return False
    df2 = df.copy()
    df2["span"] = pd.to_numeric(df2["span"], errors="coerce")
    df2["carbon_per_m2"] = pd.to_numeric(df2["carbon_per_m2"], errors="coerce")
    df2 = df2.dropna(subset=["span", "carbon_per_m2"])
    if df2.empty:
        logging.info("Skipping plot: no numeric span/carbon data after coercion.")
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5))
        for src, g in df2.groupby("_source_dir"):
            ax.scatter(g["span"], g["carbon_per_m2"], s=18, alpha=0.7, label=str(src))
        ax.set_xlabel("span (m)")
        ax.set_ylabel("carbon_per_m2 (kgCO2e/m2)")
        ax.set_title("Span vs Carbon (combined)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fp, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logging.info("Wrote diagnostic figure: %s", out_fp)
        return True
    except Exception:
        logging.exception("Failed to create diagnostic figure.")
        return False

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Combine summary_ranked_all.csv files across folders")
    parser.add_argument("--root", "-r", action="append", default=["."],
                        help="Root path(s) to search (can be provided multiple times). Default='.'")
    parser.add_argument("--out", "-o", default="outputs/reporting",
                        help="Output folder (default: outputs/reporting)")
    parser.add_argument("--key", "-k", default="system_variant",
                        help="Key column for intersection (comma separated for composite key). Default: system_variant")
    parser.add_argument("--skip-plot", action="store_true", help="Skip creating the diagnostic plot")
    parser.add_argument("--quiet", action="store_true", help="Be quiet")
    args = parser.parse_args(argv)

    # configure logging
    level = logging.DEBUG if not args.quiet else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    roots = [Path(r) for r in args.root]
    out_dir = Path(args.out)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"

    logging.info("Searching roots: %s", ", ".join(str(p) for p in roots))
    found = find_summary_files(roots)
    if not found:
        logging.error("No summary_ranked_all.csv files found under the roots provided.")
        sys.exit(2)

    logging.info("Found %d file(s):", len(found))
    for p in found:
        logging.info(" - %s", p)

    dfs = []
    for p in found:
        try:
            df = read_and_tag(p)
            dfs.append(df)
        except Exception:
            logging.exception("Failed to read %s (skipping)", p)

    if not dfs:
        logging.error("No readable summary files.")
        sys.exit(3)

    total_df = compute_total_union(dfs)
    total_fp = tables_dir / "total_summary_all.csv"
    write_csv(total_df, total_fp)
    logging.info("Wrote total summary (union) -> %s (rows=%d)", total_fp, len(total_df))

    # parse key (allow composite)
    key_cols = [c.strip() for c in args.key.split(",") if c.strip()]
    for kc in key_cols:
        if all(kc in d.columns for d in dfs):
            continue
        # If a key column is missing in any DF, we still allow; compute_compatible_intersection will drop NA
    compat_df = compute_compatible_intersection(dfs, key_cols)
    compat_fp = tables_dir / "compatible_summary_all.csv"
    write_csv(compat_df, compat_fp)
    logging.info("Wrote compatible summary (intersection on %s) -> %s (rows=%d)", ",".join(key_cols), compat_fp, len(compat_df))

    # provenance table
    prov = pd.DataFrame([{"source_dir": d["_source_dir"].iloc[0], "path": d["_source_path"].iloc[0], "n_rows": len(d)} for d in dfs])
    prov_fp = tables_dir / "summary_sources_provenance.csv"
    write_csv(prov, prov_fp)
    logging.info("Wrote provenance -> %s", prov_fp)

    # optional diagnostic plot
    if not args.skip_plot:
        plotted = try_plot_span_vs_carbon(total_df, figs_dir / "span_vs_carbon_combined.png")
        if not plotted:
            logging.info("No diagnostic figure produced.")

    # print summary JSON to stdout for automation
    out = {
        "found_files": [str(p) for p in found],
        "total_summary_all": str(total_fp),
        "compatible_summary_all": str(compat_fp),
        "provenance": str(prov_fp),
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
