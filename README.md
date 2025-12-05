# PERFECT-M

Parallel Environmental Runoff and Flood Evaluation for Computational Terrain (PERFECT-M) is a
modular, HPC-ready hydrological modelling framework for surface runoff and flood-risk assessment.

The model operates on:

- a **Digital Elevation Model (DEM)**,
- a **Curve Number (CN) map** derived from land cover,
- a **radar precipitation** (or rainfall) dataset with time dimension,

and computes:

- **surface runoff accumulation**, and
- a simple **flood-risk indicator** based on a configurable runoff threshold.

PERFECT-M supports both **GeoTIFF** and **CF-compliant NetCDF** as input and output, uses
**MPI (mpi4py)** for parallelisation, and optionally exploits **GPU acceleration** via CuPy.

All configuration is controlled by a **single JSON file**: `config.json`.

---

## 1. Model description

### 1.1 Conceptual model

PERFECT-M implements a simplified, raster-based hydrological model:

1. **Inputs**
   - DEM: 2D grid of elevations.
   - CN map: 2D grid of Curve Number values (0–100).
   - Radar precipitation: 3D array `P(t, y, x)` of rainfall intensity or accumulated rainfall.

2. **Runoff calculation (per cell)**
   - For each grid cell, a very simple SCS Curve Number–style relationship is used:

     - Potential maximum retention:
       \[
       S = \alpha \cdot \left(\frac{1000}{CN} - 10\right)
       \]

     - Initial abstraction:
       \[
       I_a = \lambda \cdot S
       \]

     - For each time step \( t \), with rainfall depth \( P_t \):
       - If \( P_t \le I_a \), runoff is zero.
       - Else:
         \[
         Q_t = \frac{(P_t - I_a)^2}{P_t - I_a + S}
         \]

   - The model aggregates the runoff over time to obtain a total accumulated runoff:
     \[
     Q_{\mathrm{acc}}(y,x) = \sum_t Q_t(y,x)
     \]

3. **Flood-risk indicator**
   - A user-defined threshold \( Q_{\mathrm{thr}} \) (in mm) is set in `config.json`.
   - A binary flood-risk map is generated:
     - 1 where \( Q_{\mathrm{acc}} \ge Q_{\mathrm{thr}} \),
     - 0 elsewhere.

4. **Flow direction (optional)**
   - A D8 flow direction grid can be computed from the DEM using the *richdem* library if available.
   - This can be used to derive drainage paths or to post-process the runoff fields.

The model is intentionally kept simple and transparent, and is meant as a building block that can be
extended with more sophisticated hydrological components.

---

## 2. Project structure

```text
perfectm_project/
├─ README.md
├─ config.json              # single configuration file
├─ main.py                  # CLI entry point
├─ perfectm/
│  ├─ __init__.py
│  ├─ config.py             # JSON config loader
│  ├─ netcdf_utils.py       # CF-compliant NetCDF I/O helpers
│  ├─ raster.py             # GeoTIFF / NetCDF grid loading utilities
│  ├─ hydrology.py          # MPI-parallel runoff & flood-risk computation (+ optional GPU)
│  ├─ downloads/
│  │   └─ __init__.py       # placeholder for download scripts
│  └─ pipeline/
│      ├─ __init__.py
│      └─ workflow.py       # high-level orchestration
├─ data/
│  ├─ dem/
│  ├─ cn/
│  ├─ radar/
│  ├─ output/
│  └─ work/
```

You are expected to place your own input data under `data/`, and configure their paths in `config.json`.

---

## 3. Installation

### 3.1 Python environment

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # on Linux / macOS
# or:
venv\Scripts\activate       # on Windows
```

### 3.2 Dependencies

Install the core dependencies:

```bash
pip install numpy scipy rasterio netCDF4 mpi4py matplotlib
```

Optional but recommended:

- `richdem` for D8 flow directions:
  ```bash
  pip install richdem
  ```

- `cupy-cuda11x` (or the appropriate CuPy wheel for your CUDA version) for GPU acceleration:
  ```bash
  pip install cupy-cuda11x
  ```

Make sure an MPI implementation is installed on your system, e.g.:

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

---

## 4. Configuration (config.json)

The entire model configuration is controlled via a single JSON file at the project root.

Below is an example `config.json`:

```json
{
  "project": {
    "name": "PERFECT-M Example Domain",
    "region": "ExampleRegion"
  },
  "paths": {
    "dem": "data/dem/dem.tif",
    "cn_map": "data/cn/cn.nc",
    "radar": "data/radar/radar.nc",
    "runoff_output": "data/output/runoff.nc",
    "flood_risk_output": "data/output/flood_risk.nc",
    "d8_flowdir": "data/output/flowdir.tif",
    "work_dir": "data/work"
  },
  "variables": {
    "dem_var": null,
    "cn_var": "cn",
    "radar_var": "precipitation",
    "time_dim": "time"
  },
  "io": {
    "dem_format": "auto",
    "cn_format": "auto",
    "radar_format": "auto",
    "runoff_format": "netcdf",
    "flood_risk_format": "netcdf"
  },
  "hydrology": {
    "alpha_retention": 25.4,
    "lambda_initial_abstraction": 0.2,
    "flood_threshold_mm": 50.0,
    "apply_sea_mask": false
  },
  "mpi": {
    "enabled": true
  },
  "runtime": {
    "use_gpu": false,
    "use_progress_bars": true,
    "plot_intermediate": false,
    "random_seed": 42
  }
}
```

### 4.1 Key sections

- `project`: metadata for logging / provenance.
- `paths`: absolute or relative paths to input and output files.
- `variables`:
  - `cn_var`: name of the CN variable in the CN NetCDF file.
  - `radar_var`: name of the precipitation variable in the radar NetCDF file.
- `io`:
  - `*_format`: `auto`, `netcdf`, or `tiff`.
- `hydrology`:
  - `alpha_retention`: controls S in mm (typical SCS uses 25.4).
  - `lambda_initial_abstraction`: initial abstraction factor (0.2 is common).
  - `flood_threshold_mm`: threshold for binary flood-risk flag.
- `runtime`:
  - `use_gpu`: enable CuPy acceleration if installed.
  - `use_progress_bars`: enable tqdm progress bars for time loops.
  - `plot_intermediate`: if true, simple plots are generated during debugging.

---

## 5. Running the model

### 5.1 Serial execution

```bash
python main.py --config config.json
```

This will:

1. Read `config.json`.
2. Load DEM, CN, and radar data.
3. Compute runoff in serial (single MPI rank).
4. Produce runoff and flood-risk NetCDF outputs in `data/output/`.

### 5.2 Parallel execution with MPI

```bash
mpirun -n 8 python main.py --config config.json
```

- Rank 0 reads the global arrays and scatters subdomains to all ranks.
- Each rank computes runoff for its own rows.
- Results are gathered back on rank 0 and written to disk.

The MPI parallelisation is over the **y-dimension** (rows) of the grid, and is suitable for
regional domains and small clusters.

---

## 6. GPU acceleration

If `runtime.use_gpu` is set to `true` in `config.json` and CuPy is installed, the core runoff
calculation is executed on the GPU:

- Arrays are transferred to the GPU as CuPy arrays.
- The runoff calculation loop is implemented with vectorised CuPy operations.
- Results are transferred back to host memory as NumPy arrays.

If GPU support is not available, the code automatically falls back to NumPy on the CPU.

---

## 7. Example use case

Suppose you have:

- a DEM for your study area in `data/dem/dem.tif`,
- a Curve Number map `cn.nc` (variable `cn`) in `data/cn/`,
- a radar precipitation cube `radar.nc` (variable `precipitation`) in `data/radar/`.

1. Edit `config.json` to match your paths and variable names.
2. Run in parallel:

   ```bash
   mpirun -n 4 python main.py --config config.json
   ```

3. Inspect the outputs:

   - `data/output/runoff.nc`: accumulated runoff (mm).
   - `data/output/flood_risk.nc`: binary flood-risk indicator (0/1).

You can open these NetCDF files with tools like:

- `ncdump`, `ncview`,
- Python (`xarray`, `netCDF4`),
- GIS software with NetCDF support.

---

## 8. Notes and extensions

- The current model is intentionally simple and is meant as a **template** for more complex
  hydrological workflows.
- You can:
  - plug in more sophisticated runoff schemes,
  - add routing along a river network,
  - couple with atmospheric models or AI nowcasting systems,
  - integrate Dask for alternative parallelisation strategies.

Contributions and extensions are welcome.
