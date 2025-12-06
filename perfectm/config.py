\"\"\"Configuration loader for PERFECT-M.

This module loads a single JSON file (config.json by default) and exposes
its content as a simple, attribute-accessible object called ``cfg``.

All other modules import and use ``cfg`` to obtain paths, parameters, and
runtime options, so that no hard-coded values are scattered across the code.
\"\"\"

import json
import os
from pathlib import Path
from typing import Any, Dict


class _ConfigView:
    \"\"\"Simple wrapper around a dictionary to allow attribute-style access.

    Example:
        from perfectm.config import cfg
        dem_path = cfg.paths.dem
        threshold = cfg.hydrology.flood_threshold_mm
    \"\"\"

    def __init__(self, data: Dict[str, Any]) -> None:
        # Store the underlying dictionary with all configuration entries.
        self._data = data

    def __getattr__(self, item: str) -> Any:
        \"\"\"Return attributes by mapping them to dictionary keys.

        If the value is a dictionary, wrap it again in _ConfigView so that
        nested keys can also be accessed via attributes.
        \"\"\"
        if item in self._data:
            value = self._data[item]
            if isinstance(value, dict):
                # Wrap nested dicts in another _ConfigView for recursive access.
                return _ConfigView(value)
            return value
        # If the key does not exist, raise AttributeError so that normal
        # Python behaviour is preserved.
        raise AttributeError(f\"No such config key: {item}\")

    def __getitem__(self, item: str) -> Any:
        \"\"\"Allow dictionary-style access as well (cfg['paths']).\"\"\"
        return self._data[item]

    def as_dict(self) -> Dict[str, Any]:
        \"\"\"Return the underlying dictionary (for debugging or export).\"\"\"
        return self._data


def _load_config() -> _ConfigView:
    \"\"\"Load the configuration from JSON.

    Priority:
      1. Environment variable ``PERFECTM_CONFIG`` if set.
      2. Default file ``config.json`` in the current working directory.

    The resulting dictionary is wrapped in a _ConfigView object.
    \"\"\"
    # Check environment variable first; if not present, use default file name.
    env_path = os.environ.get(\"PERFECTM_CONFIG\", \"config.json\")

    # Resolve the path to an absolute path.
    config_path = Path(env_path).expanduser().resolve()

    # Ensure the configuration file actually exists.
    if not config_path.is_file():
        raise FileNotFoundError(f\"Config file not found: {config_path}\")

    # Open and parse the JSON configuration file.
    with config_path.open(\"r\", encoding=\"utf-8\") as f:
        data = json.load(f)

    # Wrap the loaded dictionary into _ConfigView for convenient access.
    return _ConfigView(data)


# Global configuration object used by all modules.
cfg = _load_config()


def load_config(path: str) -> _ConfigView:
    \"\"\"Manually load configuration from a given JSON file.

    This function is rarely needed during normal runs, but it is useful
    for testing or for tools that have to work with multiple config files.
    \"\"\"
    # Turn the string path into a Path object and resolve it to an absolute path.
    p = Path(path).expanduser().resolve()

    # Read the JSON content from the file.
    with p.open(\"r\", encoding=\"utf-8\") as f:
        data = json.load(f)

    # Return another _ConfigView instance based on this file.
    return _ConfigView(data)
