\"\"\"Raster loading utilities for PERFECT-M.

This module provides helper functions to load gridded data from either:

- GeoTIFF files using rasterio, or
- CF-compliant NetCDF files using :mod:`perfectm.netcdf_utils`.

For simplicity, each loader returns the *full* grid in memory. MPI-based
parallelisation and domain decomposition are handled at a higher level
(e.g. by broadcasting and scattering these arrays).
\"\"\"

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.plot import show

from .config import cfg
from .netcdf_utils import read_cf_grid


def _load_geotiff(path: str, plot: bool = False, title: str | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    \"\"\"Load a single-band GeoTIFF file.

    Parameters
    ----------
    path:
        Path to the GeoTIFF file.
    plot:
        If True and ``cfg.runtime.plot_intermediate`` is also True,
        a simple image is shown using rasterio's plotting utilities.
    title:
        Optional plot title.

    Returns
    -------
    data:
        2D numpy array of shape (ny, nx).
    profile:
        Rasterio profile dictionary containing metadata.
    \"\"\"
    # Open the raster file with rasterio.
    with rasterio.open(path) as src:
        # Read the first band into a 2D array.
        data = src.read(1)
        # Copy the metadata profile (CRS, transform, etc.).
        profile = src.profile

    # Optionally show a quick-look plot for debugging.
    if plot and getattr(cfg.runtime, \"plot_intermediate\", False):
        show(data, title=title or os.path.basename(path))

    # Return the array and profile.
    return data, profile


def _load_netcdf(path: str, var_name: str | None = None, plot: bool = False, title: str | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    \"\"\"Load a CF-compliant NetCDF grid.

    Parameters
    ----------
    path:
        Path to the NetCDF file.
    var_name:
        Optional explicit variable name to read.
    plot:
        If True and ``cfg.runtime.plot_intermediate`` is also True,
        a quick-look image for the last time step (if any) is shown.
    title:
        Optional plot title.

    Returns
    -------
    data:
        2D numpy array of shape (ny, nx); if the variable is 3D, the last
        time slice is extracted.
    meta:
        Dictionary with 'coords' and 'meta' entries describing the dataset.
    \"\"\"
    # Use the helper from netcdf_utils to read the data and coordinates.
    data, coords, meta = read_cf_grid(path, var_name=var_name)

    # If the data has a time dimension (3D array), take the last time slice.
    if data.ndim == 3:
        data2d = data[-1, :, :]
    else:
        data2d = data

    # Optionally produce a simple plot for debugging.
    if plot and getattr(cfg.runtime, \"plot_intermediate\", False):
        import matplotlib.pyplot as plt

        plt.imshow(data2d, origin=\"lower\")
        plt.title(title or f\"{os.path.basename(path)} ({meta['var_name']})\")
        plt.colorbar()
        plt.show()

    # Return the 2D array and a metadata dictionary.
    return data2d, {\"coords\": coords, \"meta\": meta}


def load_grid(path: str, var_name: str | None = None, plot: bool = False, title: str | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    \"\"\"Load a grid from either GeoTIFF or NetCDF.

    This function inspects the file extension and dispatches to the
    appropriate loader function.

    Parameters
    ----------
    path:
        Path to the file.
    var_name:
        Optional variable name for NetCDF files.
    plot:
        If True and plotting is enabled in the configuration, a quick-look
        image is shown.
    title:
        Optional title for plotting.

    Returns
    -------
    data:
        2D numpy array.
    meta:
        A dictionary that either contains rasterio's profile (for GeoTIFF)
        or the NetCDF coordinates/metadata structure.
    \"\"\"
    # Extract the file extension to distinguish between formats.
    ext = os.path.splitext(path)[1].lower()

    # Dispatch based on extension.
    if ext in [\".tif\", \".tiff\"]:
        return _load_geotiff(path, plot=plot, title=title)
    elif ext == \".nc\":
        return _load_netcdf(path, var_name=var_name, plot=plot, title=title)
    else:
        # If the extension is unknown, raise a descriptive error.
        raise ValueError(f\"Unsupported file extension for {path}\")


def load_dem() -> Tuple[np.ndarray, Dict[str, Any]]:
    \"\"\"Load the DEM as configured in cfg.paths.dem.

    Returns
    -------
    dem:
        2D array containing elevation values.
    meta:
        Metadata dictionary (rasterio profile or NetCDF meta).
    \"\"\"
    # Read the DEM path from the config.
    dem_path = cfg.paths.dem
    # Delegate actual loading to `load_grid`.
    return load_grid(dem_path, var_name=getattr(cfg.variables, \"dem_var\", None), title=\"DEM\")


def load_cn_map() -> Tuple[np.ndarray, Dict[str, Any]]:
    \"\"\"Load the Curve Number (CN) map from cfg.paths.cn_map.

    Returns
    -------
    cn:
        2D array containing CN values (typically in the range 0–100).
    meta:
        Metadata dictionary.
    \"\"\"
    cn_path = cfg.paths.cn_map
    cn_var = getattr(cfg.variables, \"cn_var\", None)
    return load_grid(cn_path, var_name=cn_var, title=\"Curve Number\")


def load_radar_full() -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    \"\"\"Load the full 3D radar precipitation dataset from NetCDF.

    This function assumes that the radar file is stored as NetCDF with
    a time dimension. It returns the full 3D array `P(t, y, x)`.

    Returns
    -------
    data:
        3D NumPy array with shape (nt, ny, nx) or 2D if no time dimension.
    coords:
        Coordinate dictionary as returned by :func:`read_cf_grid`.
    meta:
        Metadata dictionary.
    \"\"\"
    radar_path = cfg.paths.radar

    # Read through netcdf_utils directly so we preserve the time dimension.
    data, coords, meta = read_cf_grid(radar_path, var_name=getattr(cfg.variables, \"radar_var\", None))

    # Ensure data is at least 3D (time, y, x) by adding a fake time axis if needed.
    if data.ndim == 2:
        data = data[None, :, :]

    return data, coords, meta
