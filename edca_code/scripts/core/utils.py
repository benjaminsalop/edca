from pathlib import Path

def data_path(base_dir: str | Path, *parts: str) -> Path:
    """
    Join DATA_DIR with subpaths safely.
    Example:
        data_path(cf.data_dir, "materials", "materials.csv")
    """
    return Path(base_dir).joinpath(*parts)