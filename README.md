<p align="center">
  <a href="https://github.com/SchlutowSM2Group/BLFDM">
  <img alt="BLDFM Logo" src="docs/source/_static/logo.png" width=150px>
  </a>
</p>

<h2 align="center">Boundary Layer Dispersion and Footprint Model (BLDFM)</h2>

<p align="center">
<a href="https://github.com/SchlutowSM2Group/BLFDM/actions/workflows/ci.yml">
<img alt="GitHub Actions: CI" src="https://img.shields.io/github/actions/workflow/status/SchlutowSM2Group/BLFDM/ci.yml?logo=github&label=ci">
</a>
<a href="https://www.gnu.org/licenses/gpl-3.0">
<img alt="License: MIT" src="https://img.shields.io/badge/License-GPLv3-blue.svg">
</a>
<a href="https://github.com/psf/black">
<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</a>
</p>

The Boundary Layer Dispersion and Footprint Model (BLDFM) is a novel atmospheric dispersion and footprint model designed for microscale applications in the atmospheric boundary layer under various turbulent regimes. It numerically solves the three-dimensional steady-state advection-diffusion equation in Eulerian form with flux boundary conditions.

---

<p align="center">
  <a href="https://schlutowsm2group.github.io/BLDFM/" style="font-size: 1.5em; font-weight: bold;">
    Read the documentation here
  </a>
</p>

Refer to the documentation for details on available APIs and how to create custom experiments. The documentation provides guidance on configuring the model, running example scripts, and extending the functionality for your specific use cases.

<!-- --- -->

## Features

- **Numerical Solver**: Solves the steady-state advection-diffusion equation using Fourier transforms and the linear shooting method.
- **Fast and Robust**: Utilizes the Fast Fourier Transform and Exponential Integrator Method for computational efficiency.
- **Atmospheric Stability**: Computes vertical profiles of mean wind and eddy diffusivity using Monin-Obukhov Similarity Theory.
- **Config-driven Workflows**: YAML configuration files and a CLI (`bldfm run config.yaml`) for reproducible simulations.
- **Multi-tower & Timeseries**: Run footprints for multiple towers across time-varying meteorology.
- **Parallel Execution**: Distribute tower and timestep solves across CPU cores.
- **NetCDF I/O**: Save and load multi-tower results in CF-1.8 compliant NetCDF format.
- **Plotting Library**: Footprint fields, contour comparisons, convergence plots, vertical profiles and slices, map overlays, and interactive Plotly plots.
- **Caching**: Disk-based caching of Green's function results to avoid redundant solves.
- **Validation**: Tested against analytical solutions with relative differences of less than 0.1‰ under typical conditions.
- **Comparison**: Demonstrates general agreement with the Kormann and Meixner footprint model, highlighting differences in turbulent mixing.

<!-- --- -->

## Requirements

See [`pyproject.toml`](https://github.com/SchlutowSM2Group/BLFDM/blob/main/pyproject.toml).

<!-- --- -->

## Usage

### Installation

Install the package using pip:

```bash
# BLDFM root directory
$ pip install -e .
```

### Running Example Scripts

BLDFM examples are organised in three tiers:

- **`examples/`** — Config-driven, high-level interface (start here)
- **`examples/low_level/`** — Direct API calls, for power users
- **`runs/manuscript/`** — Paper reproduction scripts (interface and low-level)

```bash
# Config-driven examples (recommended)
$ python examples/minimal_example.py
$ python examples/footprint_example.py

# Or use the CLI with a YAML config
$ bldfm run examples/configs/multitower.yaml --plot

# Low-level API examples
$ python examples/low_level/minimal_example.py

# Manuscript figure reproduction
$ python runs/manuscript/generate_all.py
```

Refer to the documentation for details on available APIs and how to create custom experiments.

### Optional dependencies
Additional dependencies for documentation or testing can be found under `optional-dependencies` in `pyproject.toml`. These dependencies can be installed via
```bash
# BLDFM root directory
$ pip install '.[dev]'
```

### Parallelization

Thread-level parallelism (BLAS/numba) and process-level parallelism (multi-tower/timeseries) are configured via YAML:

```yaml
parallel:
  num_threads: 4    # BLAS/numba threads per worker
  max_workers: 2    # parallel workers for multi-tower/timeseries
  use_cache: true   # cache solver results to skip duplicate met conditions
```

Or set threads programmatically:

```python
from bldfm import config
config.NUM_THREADS = 4
```

## License
This project is licensed under the GNU License. See the [LICENSE file](https://github.com/SchlutowSM2Group/BLDFM/blob/main/LICENSE) for details.

## Contributions
Contributions are welcome! Refer to the open issues for tasks that require attention. Submit changes, improvements, or bug fixes via pull requests from a fork or branch.
