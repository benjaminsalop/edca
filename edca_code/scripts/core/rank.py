# edca_code/scripts/core/rank.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import logging
log = logging.getLogger(__name__)

def rank_candidates_by_carbon(candidates_df: pd.DataFrame,
                              carbon_total_col: str = "carbon_total_kgCO2",
                              group_by: List[str] | None = None,
                              take_lowest_per_group: bool = True) -> pd.DataFrame:
    """
    Rank candidate rows by carbon_total_kgCO2 (ascending). Optionally group and take the lowest
    carbon per group, then return the ranked set.

    candidates_df should contain at least:
      - unique candidate id column(s)
      - carbon_total_kgCO2 or the column name provided

    Parameters:
      - group_by: list of columns to group by (e.g., ['system_family','type','span'])
      - take_lowest_per_group: if True, return only the lowest-carbon row per group (if group_by)
    Returns:
      - DataFrame sorted by carbon_total_kgCO2 (and filtered if grouping)
    """
    df = candidates_df.copy()
    if carbon_total_col not in df.columns:
        raise KeyError(f"carbon total column '{carbon_total_col}' not found in candidates_df")

    df = df.sort_values(by=carbon_total_col, ascending=True).reset_index(drop=True)

    if group_by:
        # take first (lowest carbon) row in each group
        grouped = df.groupby(group_by, as_index=False).first()
        return grouped.sort_values(by=carbon_total_col, ascending=True).reset_index(drop=True)

    return df


def find_lowest_by_type_and_brand(
    candidates_df: pd.DataFrame,
    type_col: str = "system_family",
    brand_col: str = "system_variant",
    carbon_col: str = "carbon_total_kgCO2",
    fallback_col: str | None = None,
):
    """
    For each (type_col, brand_col) group return the row with lowest carbon_col.
    Groups where carbon_col is all NaN are reported and skipped (or you can
    enable fallback by specifying fallback_col).
    """
    import logging
    log = logging.getLogger(__name__)

    grp = candidates_df.groupby([type_col, brand_col])

    # Series of row-labels (index of candidates_df) that are argmin per group
    idx = grp[carbon_col].idxmin()

    # idx is a Series indexed by (type, brand); values are original row labels or NaN
    missing_groups = idx[idx.isna()].index.tolist()
    if missing_groups:
        log.warning(
            "Found %d groups where %s is all-NaN. Examples: %s",
            len(missing_groups),
            carbon_col,
            missing_groups[:10],
        )

    # drop NaN labels before using .loc
    idx_nonan = idx.dropna()
    if idx_nonan.empty:
        log.warning("No groups with valid %s found. Returning empty DataFrame.", carbon_col)
        return pd.DataFrame(columns=candidates_df.columns)

    # idx_nonan values are the row labels in candidates_df; preserve order if needed
    out = candidates_df.loc[idx_nonan.values].reset_index(drop=True)

    # If you want to return missing groups too:
    out.attrs["missing_groups_all_nan"] = missing_groups
    return out



def rank_and_export_summary(candidates_df: pd.DataFrame,
                            group_by_type: bool = True,
                            type_col: str = "system_family",
                            brand_col: str = "system_variant",
                            carbon_col: str = "carbon_total_kgCO2") -> Dict[str, pd.DataFrame]:
    """
    Produce a small set of summary tables:
      - 'ranked_all': all candidates sorted by carbon
      - 'lowest_per_brand': lowest carbon row per (type, brand)
      - 'lowest_per_type': lowest carbon row per type (across brands)
    """
    ranked_all = rank_candidates_by_carbon(candidates_df, carbon_total_col=carbon_col, group_by=None)
    lowest_per_brand = find_lowest_by_type_and_brand(candidates_df, type_col=type_col, brand_col=brand_col, carbon_col=carbon_col)
    if group_by_type:
        idx_type = lowest_per_brand.groupby(type_col)[carbon_col].idxmin()
        lowest_per_type = lowest_per_brand.loc[idx_type].reset_index(drop=True)
    else:
        lowest_per_type = lowest_per_brand

    return {
        "ranked_all": ranked_all,
        "lowest_per_brand": lowest_per_brand,
        "lowest_per_type": lowest_per_type,
    }
