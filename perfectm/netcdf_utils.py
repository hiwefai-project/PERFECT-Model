\"\"\"NetCDF utilities for PERFECT-M.

This module provides helper functions to read and write CF-compliant
NetCDF files in a way that is convenient for the model.

The goal is not to implement the full CF standard, but to ensure that
the produced files are easy to consume with tools like xarray, Panoply,
ncview, and GIS software that understands CF metadata.
\"\"\"

from typing import Any, Dict, Tuple

import numpy as np
from netCDF4 import Dataset

CF_CONVENTIONS = \"CF-1.8\"  # CF version string used for output files.


def _infer_data_variable(ds: Dataset, var_name: str | None = None):
    \"\"\"Pick a suitable variable from the dataset.

    Parameters
    ----------
    ds:
        An open netCDF4 Dataset object.
    var_name:
        Optional explicit name of the variable that should be read.

    Returns
    -------
    var:
        The chosen netCDF variable object.

    Notes
    -----
    If ``var_name`` is provided and present in the dataset, it is returned.
    Otherwise a simple heuristic is used: first we look for typical names
    like \"runoff\", \"precipitation\", \"rainfall\", \"cn\", \"elevation\".
    If none of these exist, the first non-coordinate variable with at least
    2 dimensions is chosen.
    \"\"\"
    # If caller provided a variable name explicitly, trust that.
    if var_name is not None:
        return ds.variables[var_name]

    # List of preferred variable names commonly used in hydrology/radar data.
    preferred_names = [\"runoff\", \"precipitation\", \"rainfall\", \"rainfall_rate\", \"cn\", \"elevation\"]

    # Loop over the preferred names and return the first that is present.
    for name in preferred_names:
        if name in ds.variables:
            return ds.variables[name]

    # If nothing was found, fall back to the first non-coordinate variable.
    coord_names = {\"lon\", \"lat\", \"x\", \"y\", \"time\"}
    for name, var in ds.variables.items():
        if len(var.dimensions) >= 2 and name not in coord_names:
            return var

    # If we reach this point, we have no suitable variable.
    raise ValueError(\"No suitable data variable found in NetCDF file.\")


def read_cf_grid(path: str, var_name: str | None = None) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    \"\"\"Read a CF-compliant grid from a NetCDF file.

    Parameters
    ----------
    path:
        Path to the NetCDF file.
    var_name:
        Optional name of the variable to read. If None, a heuristic is used.

    Returns
    -------
    data:
        Numpy array containing the variable values. Shape is (ny, nx) or
        (nt, ny, nx).
    coords:
        Dictionary with coordinate arrays, containing keys:
          - 'x': 1D array of x coordinates.
          - 'y': 1D array of y coordinates.
          - optionally 'time': 1D time array if present.
    meta:
        Dictionary with metadata such as variable attributes and dimension names.
    \"\"\"
    # Open the dataset for reading.
    ds = Dataset(path, \"r\")

    # Select the data variable, possibly via heuristic.
    var = _infer_data_variable(ds, var_name=var_name)

    # Read the entire variable into a NumPy array.
    data = np.array(var[...])

    # Extract the list of dimension names associated with this variable.
    dims = var.dimensions  # e.g. ('y', 'x') or ('time', 'y', 'x')

    # Ensure that we have at least two dimensions (y, x).
    if len(dims) < 2:
        raise ValueError(f\"Variable {var.name} has less than 2 dimensions: {dims}\")

    # Convention: last two dimensions correspond to y and x.
    y_dim = dims[-2]
    x_dim = dims[-1]

    # Extract 1D coordinate arrays for x and y.
    x = np.array(ds.variables[x_dim][:])
    y = np.array(ds.variables[y_dim][:])

    # Pack coordinates into a dictionary for convenience.
    coords: Dict[str, Any] = {\"x\": x, \"y\": y}

    # Optionally, read a time coordinate if present.
    if \"time\" in dims and \"time\" in ds.variables:
        time = np.array(ds.variables[\"time\"][:])
        coords[\"time\"] = time

    # Collect metadata: variable name, dimensions, variable attrs, global attrs.
    meta: Dict[str, Any] = {
        \"var_name\": var.name,
        \"dims\": dims,
        \"attrs\": dict(var.__dict__),
        \"global_attrs\": dict(ds.__dict__),
        \"x_dim\": x_dim,
        \"y_dim\": y_dim,
    }

    # Close the dataset to free resources.
    ds.close()

    # Return data, coordinates, and metadata.
    return data, coords, meta


def write_cf_grid(
    path: str,
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    var_name: str = \"runoff\",
    units: str = \"mm\",
    long_name: str = \"Accumulated surface runoff\",
    time: np.ndarray | None = None,
    crs_wkt: str | None = None,
    extra_global_attrs: Dict[str, Any] | None = None,
    extra_var_attrs: Dict[str, Any] | None = None,
) -> None:
    \"\"\"Write a 2D or 3D grid to a CF-compliant NetCDF file.

    Parameters
    ----------
    path:
        Output file path (will be overwritten if it exists).
    data:
        Array of shape (ny, nx) or (nt, ny, nx).
    x, y:
        1D coordinate arrays for x and y.
    var_name:
        Name of the data variable to create in the file.
    units:
        Physical units of the data (e.g. 'mm' for runoff).
    long_name:
        Descriptive name for the variable.
    time:
        Optional 1D time coordinate array (seconds since epoch, or other).
    crs_wkt:
        Optional WKT string describing the coordinate reference system.
    extra_global_attrs:
        Optional dictionary of additional global file attributes.
    extra_var_attrs:
        Optional dictionary of additional variable attributes.
    \"\"\"
    # Ensure data is a NumPy array.
    data = np.array(data)

    # Determine whether we have a time dimension (3D) or not (2D).
    has_time = data.ndim == 3

    # Extract spatial sizes from the last two dimensions of the data array.
    ny, nx = data.shape[-2], data.shape[-1]

    # Create the new NetCDF file for writing.
    ds = Dataset(path, \"w\")

    # Set global CF conventions attribute.
    ds.Conventions = CF_CONVENTIONS

    # If the caller provided extra global attributes, attach them.
    if extra_global_attrs:
        for k, v in extra_global_attrs.items():
            setattr(ds, k, v)

    # Create dimensions: y and x are always present.
    ds.createDimension(\"y\", ny)
    ds.createDimension(\"x\", nx)
    if has_time:
        # If data has a time dimension, create a corresponding time dimension.
        ds.createDimension(\"time\", data.shape[0])

    # Create coordinate variable for y.
    y_var = ds.createVariable(\"y\", \"f8\", (\"y\",))
    # CF-recommended attributes for projected y coordinate.
    y_var.standard_name = \"projection_y_coordinate\"
    y_var.units = \"m\"
    # Write the y coordinate values.
    y_var[:] = y

    # Create coordinate variable for x.
    x_var = ds.createVariable(\"x\", \"f8\", (\"x\",))
    # CF-recommended attributes for projected x coordinate.
    x_var.standard_name = \"projection_x_coordinate\"
    x_var.units = \"m\"
    # Write the x coordinate values.
    x_var[:] = x

    if has_time:
        # If we have a time dimension, create the time coordinate variable.
        t_var = ds.createVariable(\"time\", \"f8\", (\"time\",))
        t_var.standard_name = \"time\"
        # Use a generic \"seconds since epoch\" unit by default.
        t_var.units = \"seconds since 1970-01-01 00:00:00\"
        if time is not None:
            # If caller provided a specific time array, write it.
            t_var[:] = time
        else:
            # Otherwise just use a monotonic range.
            t_var[:] = np.arange(data.shape[0])

    # Optionally define a CRS / grid_mapping variable.
    grid_mapping_name = None
    if crs_wkt is not None:
        # Create a scalar variable called 'crs' to hold CRS metadata.
        grid_mapping_name = \"crs\"
        crs = ds.createVariable(grid_mapping_name, \"i4\")
        # Store the WKT representation and some basic ellipsoid info.
        crs.spatial_ref = crs_wkt
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563

    # Decide on the dimensions of the data variable.
    if has_time:
        dims = (\"time\", \"y\", \"x\")
    else:
        dims = (\"y\", \"x\")

    # Create the data variable with compression enabled.
    v = ds.createVariable(var_name, \"f4\", dims, zlib=True, complevel=4)
    # Write the data values to the file.
    v[:] = data
    # Set basic metadata for the variable.
    v.units = units
    v.long_name = long_name

    # If we defined a grid_mapping variable, attach its name here.
    if grid_mapping_name is not None:
        v.grid_mapping = grid_mapping_name

    # Attach any additional variable attributes provided by the caller.
    if extra_var_attrs:
        for k, v_attr in extra_var_attrs.items():
            setattr(v, k, v_attr)

    # Close the dataset to flush all data to disk.
    ds.close()
