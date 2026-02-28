"""
Runtime settings for BLDFM.

These globals are set at runtime by the CLI, interface, or user scripts.
For YAML-based configuration, see config_parser.py.
"""

# Number of threads for numba parallel computation.
# NUM_THREADS=1 equivalent to serial
NUM_THREADS = 1

# Number of worker processes for multi-tower/multi-time parallelism.
MAX_WORKERS = 1

# Enable disk caching of Green's functions.
USE_CACHE = False

# Default output directory.
OUTPUT_DIR = "./output"
