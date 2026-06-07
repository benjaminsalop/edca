# edca_code/scripts/code_checks/__init__.py
"""
Adapter package to expose canonical code_checks API for run_edca.py.
"""

__all__ = [
    "run_code_check_for_typology",
    "check_rc_beam",
    "check_rc_column",
    "check_steel_beam",
    "check_steel_column",
]

try:
    from .continuousslab import run_code_check_for_typology  # type: ignore
except Exception:
    def run_code_check_for_typology(*args, **kwargs):
        raise RuntimeError("continuousslab submodule unavailable — cannot run run_code_check_for_typology")

try:
    from .rc_beam import check_rc_beam  # type: ignore
except Exception:
    def check_rc_beam(*args, **kwargs):
        raise RuntimeError("rc_beam submodule unavailable")

try:
    from .rc_column import check_rc_column  # type: ignore
except Exception:
    def check_rc_column(*args, **kwargs):
        raise RuntimeError("rc_column submodule unavailable")

try:
    from .steel_beam import check_steel_beam  # type: ignore
except Exception:
    def check_steel_beam(*args, **kwargs):
        raise RuntimeError("steel_beam submodule unavailable")

try:
    from .steel_column import check_steel_column  # type: ignore
except Exception:
    def check_steel_column(*args, **kwargs):
        raise RuntimeError("steel_column submodule unavailable")
