from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from edca_code.scripts.code_checks.code_checks import (
    run_building_code_check,
    run_code_checks_on_candidates,
    run_code_checks_on_components,
)
from edca_code.scripts.code_checks.design_inputs import BuildingSystemDesignInput, ComponentDesignInput
from edca_code.scripts.core.design_results import ComponentDesignResult

logger = logging.getLogger("code_checks")


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _serialize_results(results: Iterable[ComponentDesignResult]) -> List[dict[str, Any]]:
    payload: List[dict[str, Any]] = []
    for result in results:
        if hasattr(result, "to_dict"):
            payload.append(result.to_dict())
        elif is_dataclass(result):
            payload.append(asdict(result))
        else:
            payload.append(getattr(result, "__dict__", {"result": str(result)}))
    return payload


def _records_from_legacy_input(value: Any) -> tuple[list[dict[str, Any]] | None, bool]:
    if value is None:
        return None, False
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return value.to_dict(orient="records"), True
    if isinstance(value, dict):
        return [value], True
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value, True
    return None, False


def _results_to_legacy_dataframe(
    records: list[dict[str, Any]],
    results: Iterable[ComponentDesignResult],
):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record, result in zip(records, results):
        checks = getattr(result, "checks", []) or []
        warnings = getattr(result, "warnings", []) or []
        rows.append(
            {
                "system_variant": record.get("system_variant") or record.get("variant_id") or getattr(result, "variant_id", None),
                "codecheck_component": getattr(result, "component", None),
                "codecheck_family": getattr(result, "family_id", None),
                "codecheck_variant": getattr(result, "variant_id", None),
                "codecheck_passed": bool(getattr(result, "passed", False)),
                "codecheck_utilization_max": getattr(result, "utilization_max", None),
                "codecheck_n_checks": len(checks),
                "codecheck_n_warnings": len(warnings),
                "codecheck_warnings": json.dumps([asdict(w) if is_dataclass(w) else str(w) for w in warnings]),
                "codecheck_details_json": json.dumps(asdict(result) if is_dataclass(result) else getattr(result, "__dict__", {}), default=str),
            }
        )
    return pd.DataFrame(rows)


def run_code_checks_if_requested(
    components_or_candidates: Optional[Iterable[Any]] = None,
    out_dir: str | Path = ".",
    run_flag: bool = False,
    *,
    candidates_df: Any = None,
    building_input: Optional[BuildingSystemDesignInput] = None,
    use_legacy_candidate_wrapper: bool = False,
    **kwargs: Any,
) -> Any:
    if not run_flag:
        return []

    out = _ensure_dir(out_dir)

    source = candidates_df if candidates_df is not None else components_or_candidates
    legacy_records, is_legacy = _records_from_legacy_input(source)

    if use_legacy_candidate_wrapper or is_legacy:
        records = legacy_records if legacy_records is not None else list(source or [])
        results = run_code_checks_on_candidates(records, **kwargs)
    else:
        results = run_code_checks_on_components(source or [], **kwargs)

    with (out / "component_code_checks.json").open("w", encoding="utf-8") as fh:
        json.dump(_serialize_results(results), fh, indent=2)

    if building_input is not None:
        building_check = run_building_code_check(building_input, **kwargs)
        if building_check is not None:
            with (out / "building_code_check.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    building_check.to_dict()
                    if hasattr(building_check, "to_dict")
                    else asdict(building_check)
                    if is_dataclass(building_check)
                    else getattr(building_check, "__dict__", {}),
                    fh,
                    indent=2,
                )

    if is_legacy and legacy_records is not None:
        return _results_to_legacy_dataframe(legacy_records, results)

    return results
