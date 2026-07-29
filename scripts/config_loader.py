"""Helpers for loading data and runtime settings from the ``config/`` directory.

Paths are resolved relative to this file, so the scripts can be launched from
any working directory.
"""
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(name):
    """Load and return the parsed JSON contents of ``config/<name>``."""
    with open(CONFIG_DIR / name, "r") as file:
        return json.load(file)


def load_available_dims():
    """Return the available-objects catalog ``{obj_name: [dx, dy, dz, shape]}``."""
    return load_config("available_dims.json")


def load_object_parameters():
    """Return the per-object-type prompt parameters table."""
    return load_config("object_parameters.json")
