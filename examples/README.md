# BLDFM Examples

This directory contains instructional examples for the BLDFM framework,
organised in two tiers of increasing complexity. For production simulation
scripts that reproduce manuscript figures, see [`runs/manuscript/`](../runs/manuscript/).

## Config-driven examples (recommended starting point)

These scripts use the high-level YAML/dataclass workflow: `load_config()`,
`run_bldfm_single()`, and the plotting library. Each script has a matching
YAML file in `configs/`.

| Script | Description |
|--------|-------------|
| `minimal_example.py` | Neutral BL concentration and flux fields |
| `minimal_example_3d.py` | 3D output with vertical slices |
| `3d_plume.py` | Point-source plume with horizontal and vertical slices |
| `footprint_example.py` | Flux footprint with percentile contours |
| `parallel_example.py` | High-resolution solve with multi-threaded FFT |
| `multitower_example.py` | Multiple towers over a synthetic timeseries |
| `visualization_example.py` | Optional-dependency features: map tiles, land cover, wind rose, interactive plots |

```bash
python examples/minimal_example.py
```

## Low-level API examples

These scripts call `vertical_profiles()`, `steady_state_transport_solver()`,
and other solver-level functions directly. Use these to understand the
internals or to build custom workflows outside the config system.

| Script | Description |
|--------|-------------|
| `low_level/minimal_example.py` | Basic neutral BL solve |
| `low_level/minimal_example_3d.py` | 3D output |
| `low_level/3d_plume.py` | Point-source plume |
| `low_level/footprint_example.py` | Flux footprint calculation |
| `low_level/parallel_example.py` | Parallel execution |
| `low_level/plot_profiles.py` | Vertical profiles of wind and eddy diffusivity under MOST |
| `low_level/point_measurement_example.py` | Point measurement via footprint convolution |

```bash
python examples/low_level/minimal_example.py
```

## Prerequisites

All examples save output to `plots/` in the repository root. Create it if
it does not exist:

```bash
mkdir -p plots
```

Install BLDFM in development mode:

```bash
pip install -e .
```

Optional dependencies for `visualization_example.py`:

```bash
pip install contextily owslib windrose plotly
```
