# edca_code/scripts/code_checks/__init__.py
"""
Adapter package to expose canonical code_checks API for run_edca.py.
If continuousslab is present under this package, re-export its run_code_check_for_typology.
"""

__all__ = ["run_code_check_for_typology"]

try:
    # import the submodule in the same package
    from .continuousslab import run_code_check_for_typology  # type: ignore
except Exception:
    # fallback stub that raises a clear error if called
    def run_code_check_for_typology(*args, **kwargs):
        raise RuntimeError("continuousslab submodule unavailable — cannot run run_code_check_for_typology")
