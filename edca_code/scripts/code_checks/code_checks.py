from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from edca_code.scripts.code_checks import continuousslab
from edca_code.scripts.code_checks.continuousslab import DesignError

logger = logging.getLogger("code_checks")


def run_code_checks_on_candidates(
    candidates_df: pd.DataFrame,
    *,
    material_csv_path: Optional[str] = None,
    load_combos_yaml: Optional[str] = None,
    load_values_yaml: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Default code checks: run continuousslab.check_slab_row_preserve_math for each row.

    Expected candidate columns (best-effort):
      - system_variant
      - slab_depth
      - span or max_span
      - screed_depth (optional)
      - ll (optional)
      - sdl_total or sdl (optional)
      - material_concrete_id (used as material_id)
    """
    results: List[Dict[str, Any]] = []

    if candidates_df is None or candidates_df.empty:
        return results

    for _, row in candidates_df.iterrows():
        sv = str(row.get("system_variant", "") or "")
        try:
            # Build row dict for continuousslab
            r = row.to_dict()

            # map material_id expected by continuousslab
            if "material_id" not in r or not r.get("material_id"):
                if r.get("material_concrete_id"):
                    r["material_id"] = r["material_concrete_id"]

            # map load keys (optional; continuousslab uses ll by default)
            if "live_load_kN_m2" not in r and r.get("ll") is not None:
                r["live_load_kN_m2"] = r["ll"]

            # (optional) could include sdl_total as partition/dead loads if you want later

            out = continuousslab.check_slab_row_preserve_math(
                r,
                material_csv_path=material_csv_path,
                load_combos_yaml=load_combos_yaml,
                load_values_yaml=load_values_yaml,
            )

            # Decide a simple success flag (customize later)
            results.append({
                "system_variant": sv,
                "success": True,
                "codecheck_family": "continuousslab",
                "ULS_kN_m2": out.get("ULS_kN_m2"),
                "ULS_combo_name": out.get("ULS_combo_name"),
                # keep the full nested output so reporting can expand later if desired
                "code_outputs": out,
            })

        except DesignError as e:
            logger.warning("[codechecks] design fail for %s: %s", sv, e)
            results.append({"system_variant": sv, "success": False, "error": str(e), "codecheck_family": "continuousslab"})
        except Exception as e:
            logger.exception("[codechecks] unexpected error for %s", sv)
            results.append({"system_variant": sv, "success": False, "error": str(e), "codecheck_family": "continuousslab"})


    return results
