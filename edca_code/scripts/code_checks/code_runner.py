import logging
from pathlib import Path
import json
import pandas as pd
from typing import Any

logger = logging.getLogger("code_checks")

# -------------------------
# Debug helpers (drop-in)
# -------------------------
import os
from typing import Iterable

def _dbg_enabled(explicit: bool | None = None) -> bool:
    """
    Debug is enabled if:
      - explicit=True passed by caller, OR
      - EDCA_DEBUG=1 environment variable, OR
      - logger level is DEBUG.
    """
    if explicit is True:
        return True
    if explicit is False:
        return False
    if str(os.getenv("EDCA_DEBUG", "")).strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    return bool(getattr(logger, "isEnabledFor", lambda *_: False)(logging.DEBUG))

def _dbg_kv(name: str, d: dict, *, explicit: bool | None = None, level: int = logging.DEBUG) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        items = ", ".join([f"{k}={d[k]!r}" for k in sorted(d.keys())])
    except Exception:
        items = str(d)
    logger.log(level, "[debug] %s: %s", name, items)

def _dbg_df(
    name: str,
    df,
    *,
    explicit: bool | None = None,
    max_rows: int = 15,
    cols: list[str] | None = None,
    level: int = logging.DEBUG,
) -> None:
    if not _dbg_enabled(explicit):
        return
    try:
        import pandas as pd
        if df is None:
            logger.log(level, "[debug] %s: df=None", name)
            return
        if not isinstance(df, pd.DataFrame):
            logger.log(level, "[debug] %s: not a DataFrame (%s)", name, type(df).__name__)
            return
        logger.log(level, "[debug] %s: shape=%s", name, df.shape)
        logger.log(level, "[debug] %s: cols=%s", name, list(df.columns))
        sub = df
        if cols:
            keep = [c for c in cols if c in sub.columns]
            sub = sub[keep] if keep else sub
        logger.log(level, "[debug] %s head(%d):\n%s", name, max_rows, sub.head(max_rows).to_string(index=False))
    except Exception:
        logger.exception("[debug] Failed dumping df %s", name)


def run_code_checks_if_requested(candidates_df, out_dir, run_flag, **kwargs):
    if not run_flag:
        return pd.DataFrame()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate the code_checks entrypoint
    try:
        from edca_code.scripts.code_checks import code_checks  # preferred location
    except Exception:
        try:
            from edca_code.scripts.code_checks import code_checks  # fallback
        except Exception:
            logger.warning("[codechecks] requested but no code_checks module found; skipping.")
            return pd.DataFrame()

    # Run checks
    try:
        if hasattr(code_checks, "run_code_checks_on_candidates"):
            # Pull known kwargs (and accept common aliases)
            material_csv_path = (
                kwargs.get("material_csv_path")
                or kwargs.get("materials")
                or kwargs.get("materials_path")
                or kwargs.get("materials_csv")
            )
            load_combos_yaml = kwargs.get("load_combos_yaml")
            load_values_yaml = kwargs.get("load_values_yaml")

            # Optional debug passthrough: dump the inputs used by the checker
            debug_inputs = bool(
                kwargs.get("debug_inputs")
                or kwargs.get("codechecks_debug_inputs")
                or kwargs.get("codechecks_debug")
            )
            debug_only_on_fail = bool(
                kwargs.get("debug_only_on_fail", True)
                if "debug_only_on_fail" in kwargs
                else kwargs.get("codechecks_debug_only_on_fail", True)
            )
            debug_max_rows = int(kwargs.get("debug_max_rows", kwargs.get("codechecks_debug_max_rows", 50)) or 50)

            results: Any = code_checks.run_code_checks_on_candidates(
                candidates_df,
                material_csv_path=material_csv_path,
                load_combos_yaml=load_combos_yaml,
                load_values_yaml=load_values_yaml,
                debug_inputs=debug_inputs,
                debug_only_on_fail=debug_only_on_fail,
                debug_max_rows=debug_max_rows,
            )
        elif hasattr(code_checks, "run"):
            results = code_checks.run(candidates_df)
        else:
            logger.warning("[codechecks] module found but no runnable entrypoint; skipping.")
            return pd.DataFrame()
    except Exception:
        logger.exception("[codechecks] Error running code checks; skipping")
        return pd.DataFrame()

    # Normalize results into list-of-dicts
    if results is None:
        logger.warning("[codechecks] code checks returned None; skipping")
        return pd.DataFrame()
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, (list, tuple)):
        logger.warning("[codechecks] unexpected results type: %s; skipping", type(results).__name__)
        return pd.DataFrame()

    verbose_lines = []
    num_success = 0
    num_fail = 0

    for r in results:
        if not isinstance(r, dict):
            # keep something rather than crashing
            r = {"success": False, "error": f"non-dict result: {type(r).__name__}", "raw": str(r)}

        succ = bool(r.get("success", False))
        num_success += int(succ)
        num_fail += int(not succ)

        try:
            verbose_lines.append(json.dumps(r, default=str))
        except Exception:
            verbose_lines.append(json.dumps({"success": False, "error": "json serialize failed", "raw": str(r)}))

    # Write verbose output
    verbose_path = out_dir / "codechecks_verbose.txt"
    try:
        with verbose_path.open("w", encoding="utf-8") as vf:
            vf.write("Code checks verbose report\n")
            vf.write(f"Success: {num_success}\n")
            vf.write(f"Fail: {num_fail}\n")
            vf.write(f"Total: {len(results)}\n\n")
            vf.write("\n".join(verbose_lines))
            vf.write("\n")
        logger.info("[codechecks] Wrote verbose code checks report to %s", verbose_path)
    except Exception:
        logger.exception("[codechecks] Failed to write codechecks_verbose.txt")

    logger.info(
        "[codechecks] Code checks completed: success=%d, fail=%d, total=%d",
        num_success, num_fail, len(results)
    )

    # Convert to DataFrame for merge
    df_results = pd.DataFrame(results)
    if df_results.empty:
        logger.warning("[codechecks] results DataFrame is empty; skipping merge output")
        return pd.DataFrame()
    if "system_variant" not in df_results.columns:
        logger.warning("[codechecks] results missing system_variant; cannot merge into candidates")
        return pd.DataFrame()

    # Make sure system_variant is string for merges
    df_results["system_variant"] = df_results["system_variant"].astype(str)

    return df_results
