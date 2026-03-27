"""
Runtime settings for BLDFM.

These globals are set at runtime by the CLI, interface, or user scripts.
For YAML-based configuration, see config_parser.py.

Path variables delegate to ``abltk.config`` with ``BLDFM_*`` env var overrides.
"""

import os
from pathlib import Path

import abltk.config as _abltk_config

# Initialize abltk with BLDFM's project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_abltk_config.init(_PROJECT_ROOT)

# BLDFM env vars as fallback over abltk defaults
_BLDFM_ENV_MAP = {
    "data_dir": "BLDFM_DATA_DIR",
    "cache_dir": "BLDFM_CACHE_DIR",
    "config_dir": "BLDFM_CONFIG_DIR",
    "output_dir": "BLDFM_OUTPUT_DIR",
    "log_dir": "BLDFM_LOG_DIR",
}


def _resolve(key, abltk_val):
    return os.environ.get(_BLDFM_ENV_MAP[key]) or abltk_val


DATA_DIR = _resolve("data_dir", _abltk_config.DATA_DIR)
CACHE_DIR = _resolve("cache_dir", _abltk_config.CACHE_DIR)
CONFIG_DIR = _resolve("config_dir", _abltk_config.CONFIG_DIR)
OUTPUT_DIR = _resolve("output_dir", _abltk_config.OUTPUT_DIR)
LOG_DIR = _resolve("log_dir", _abltk_config.LOG_DIR)

# --- Runtime globals (not path-related) ---

# Number of threads for numba parallel computation.
# NUM_THREADS=1 equivalent to serial
NUM_THREADS = 1

# Number of worker processes for multi-tower/multi-time parallelism.
MAX_WORKERS = 1

# Enable disk caching of Green's functions.
USE_CACHE = False
