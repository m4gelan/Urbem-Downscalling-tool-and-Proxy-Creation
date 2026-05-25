from pathlib import Path

_PKG = Path(__file__).resolve().parent


def project_root() -> Path:
    return _PKG.parent


def package_dir() -> Path:
    return _PKG


def config_dir() -> Path:
    return _PKG / "config"


def runs_dir() -> Path:
    return config_dir() / "run"
