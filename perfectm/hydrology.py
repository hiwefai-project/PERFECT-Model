\"\"\"Hydrological core of PERFECT-M.

This module implements:

- MPI-aware domain decomposition,
- optional GPU acceleration of the runoff calculation,
- accumulation of runoff over time using a simple CN-based model,
- computation of a binary flood-risk indicator,
- writing of NetCDF output files.

Most functions include line-by-line comments to make the logic as clear
as possible for students and collaborators.
\"\"\"

from __future__ import annotations

from typing import Tuple

import numpy as np
from mpi4py import MPI
import rasterio

from .config import cfg
from .netcdf_utils import write_cf_grid


def get_mpi_comm() -> Tuple[MPI.Comm, int, int]:
    \"\"\"Return the global MPI communicator and rank/size.

    Returns
    -------
    comm:
        The MPI communicator (usually MPI.COMM_WORLD).
    rank:
        Rank of the current process.
    size:
        Total number of processes in the communicator.
    \"\"\"
    # Use the default world communicator that includes all processes.
    comm = MPI.COMM_WORLD
    # Each process obtains its own rank (integer identifier).
    rank = comm.Get_rank()
    # The communicator also knows how many ranks it contains in total.
    size = comm.Get_size()
    return comm, rank, size


def decompose_rows(ny: int, size: int, rank: int) -> Tuple[int, int]:
    \"\"\"Compute the start and end row indices for a given MPI rank.

    The total number of rows ``ny`` is divided as evenly as possible among
    ``size`` ranks. This function tells each rank which subset of rows it
    is responsible for.

    Parameters
    ----------
    ny:
        Total number of rows in the global grid.
    size:
        Total number of MPI ranks.
    rank:
        Rank of the current process.

    Returns
    -------
    i_start:
        Index of the first row (inclusive) assigned to this rank.
    i_end:
        Index one past the last row (exclusive) assigned to this rank.
    \"\"\"
    # Compute the base number of rows per rank (integer division).
    base = ny // size
    # Compute how many rows are left over after the even split.
    remainder = ny % size

    # Ranks with index < remainder receive one extra row.
    if rank < remainder:
        # Ranks before 'remainder' get base + 1 rows.
        local_ny = base + 1
        # The starting index is rank * (base + 1).
        i_start = rank * (base + 1)
    else:
        # Ranks after 'remainder' get exactly 'base' rows.
        local_ny = base
        # Their starting index must skip all the larger chunks at the beginning.
        i_start = remainder * (base + 1) + (rank - remainder) * base

    # The end index is just start + number of rows.
    i_end = i_start + local_ny
    return i_start, i_end


def _try_import_cupy():
    \"\"\"Try to import CuPy for GPU acceleration.

    Returns
    -------
    module or None:
        The imported cupy module if available and cfg.runtime.use_gpu is True,
        otherwise None.
    \"\"\"
    # If the config does not enable GPU usage, do not even try to import CuPy.
    if not getattr(cfg.runtime, \"use_gpu\", False):
        return None

    try:
        # Try importing CuPy (GPU-accelerated NumPy-like library).
        import cupy as cp  # type: ignore[import]
    except Exception:
        # If anything goes wrong (module not installed, no CUDA, etc.),
        # fall back to CPU by returning None.
        return None

    # If the import succeeded, return the module object.
    return cp


def compute_runoff_local(
    cn_local: np.ndarray,
    radar_local: np.ndarray,
    alpha: float,
    lambda_ia: float,
    use_gpu: bool = False,
) -> np.ndarray:
    \"\"\"Compute runoff time series and accumulation for a local subdomain.

    Parameters
    ----------
    cn_local:
        2D array (ny_local, nx) of Curve Number values for this rank.
    radar_local:
        3D array (nt, ny_local, nx) of rainfall depth per time step.
    alpha:
        Scaling factor for potential maximum retention S.
    lambda_ia:
        Initial abstraction factor (Ia = lambda_ia * S).
    use_gpu:
        Whether to attempt GPU acceleration with CuPy.

    Returns
    -------
    runoff_acc:
        2D array (ny_local, nx) of accumulated runoff over all time steps.
    \"\"\"
    # Determine if we can actually use the GPU by importing CuPy.
    cp = _try_import_cupy() if use_gpu else None

    if cp is not None:
        # --- GPU implementation using CuPy ---

        # Transfer arrays to GPU memory by converting them to CuPy arrays.
        cn_gpu = cp.asarray(cn_local)
        radar_gpu = cp.asarray(radar_local)

        # Compute the potential maximum retention S on the GPU.
        S_gpu = alpha * (1000.0 / cp.clip(cn_gpu, 1.0, None) - 10.0)
        # Compute the initial abstraction Ia on the GPU.
        Ia_gpu = lambda_ia * S_gpu

        # Initialise the accumulated runoff array with zeros on the GPU.
        runoff_acc_gpu = cp.zeros_like(cn_gpu, dtype=cp.float32)

        # Loop over the time dimension on the GPU, operating on 2D slices.
        nt = radar_gpu.shape[0]
        for it in range(nt):
            # Extract rainfall depth for the current time step.
            P_t = radar_gpu[it, :, :]

            # Compute mask where rainfall exceeds initial abstraction Ia.
            mask = P_t > Ia_gpu

            # Compute runoff where P_t > Ia using the SCS-like formula.
            Q_t = cp.zeros_like(P_t, dtype=cp.float32)
            num = (P_t - Ia_gpu) ** 2
            den = P_t - Ia_gpu + S_gpu
            Q_t[mask] = num[mask] / cp.maximum(den[mask], 1e-6)

            # Accumulate runoff over time.
            runoff_acc_gpu += Q_t

        # Transfer the accumulated runoff back to host memory (NumPy array).
        runoff_acc = cp.asnumpy(runoff_acc_gpu)
    else:
        # --- CPU implementation using NumPy ---

        # Compute the potential maximum retention S on the CPU.
        S = alpha * (1000.0 / np.clip(cn_local, 1.0, None) - 10.0)
        # Compute the initial abstraction Ia.
        Ia = lambda_ia * S

        # Initialise the accumulated runoff to zeros.
        runoff_acc = np.zeros_like(cn_local, dtype=np.float32)

        # Number of time steps in the local radar data.
        nt = radar_local.shape[0]

        # Optionally, import tqdm for a progress bar if requested.
        use_tqdm = getattr(cfg.runtime, \"use_progress_bars\", False)
        if use_tqdm:
            try:
                from tqdm import trange  # type: ignore[import]
            except Exception:
                # If tqdm is not available, simply disable the progress bar.
                use_tqdm = False

        # Choose iterator: either a tqdm range or a normal range.
        iterator = trange(nt, desc=\"Runoff (local)\" if use_tqdm else None) if use_tqdm else range(nt)

        # Loop over time steps and compute incremental runoff.
        for it in iterator:
            # Extract rainfall depth for the current time step.
            P_t = radar_local[it, :, :]

            # Build a boolean mask where P_t exceeds Ia.
            mask = P_t > Ia

            # Allocate an array for the runoff at this time step.
            Q_t = np.zeros_like(P_t, dtype=np.float32)

            # Compute numerator and denominator of the SCS-like formula.
            num = (P_t - Ia) ** 2
            den = P_t - Ia + S

            # Use np.maximum to avoid division by very small numbers.
            Q_t[mask] = num[mask] / np.maximum(den[mask], 1e-6)

            # Accumulate runoff over time.
            runoff_acc += Q_t

    # Return the accumulated runoff array (on CPU).
    return runoff_acc


def runoff_mpi(
    dem: np.ndarray,
    cn: np.ndarray,
    radar: np.ndarray,
    dem_meta: dict,
) -> np.ndarray | None:
    \"\"\"High-level MPI wrapper to compute accumulated runoff.

    Parameters
    ----------
    dem:
        2D array of elevation values (only valid on rank 0, ignored elsewhere).
    cn:
        2D array of Curve Numbers (only valid on rank 0).
    radar:
        3D array of rainfall (time, y, x) (only valid on rank 0).
    dem_meta:
        Metadata for the DEM (used on rank 0 to derive coordinates and CRS).

    Returns
    -------
    runoff_global:
        2D array (ny, nx) of accumulated runoff on rank 0, or None on other ranks.
    \"\"\"
    # Obtain the MPI communicator and rank/size information.
    comm, rank, size = get_mpi_comm()

    # Broadcast the global shape information from rank 0 to all ranks.
    if rank == 0:
        ny, nx = cn.shape
        nt = radar.shape[0]
    else:
        ny = nx = nt = 0
    ny = comm.bcast(ny, root=0)
    nx = comm.bcast(nx, root=0)
    nt = comm.bcast(nt, root=0)

    # Each rank computes which subset of rows it is responsible for.
    i_start, i_end = decompose_rows(ny, size, rank)
    local_ny = i_end - i_start

    # Allocate local slices for CN and radar on each rank.
    cn_local = np.empty((local_ny, nx), dtype=np.float32)
    radar_local = np.empty((nt, local_ny, nx), dtype=np.float32)

    # Rank 0 scatters the CN and radar data to all ranks.
    if rank == 0:
        # Prepare CN chunks for each rank.
        cn_chunks = []
        radar_chunks = []
        for r in range(size):
            s, e = decompose_rows(ny, size, r)
            cn_chunks.append(cn[s:e, :].astype(np.float32))
            radar_chunks.append(radar[:, s:e, :].astype(np.float32))
    else:
        cn_chunks = None
        radar_chunks = None

    # Scatter CN slices so that each rank receives its chunk into cn_local.
    comm.Scatter(cn_chunks, cn_local, root=0)
    # Scatter radar slices so that each rank receives its chunk into radar_local.
    comm.Scatter(radar_chunks, radar_local, root=0)

    # Read hydrological parameters from the configuration.
    alpha = cfg.hydrology.alpha_retention
    lambda_ia = cfg.hydrology.lambda_initial_abstraction

    # Decide whether to use GPU based on config.
    use_gpu = getattr(cfg.runtime, \"use_gpu\", False)

    # Compute the local accumulated runoff for this rank.
    runoff_local = compute_runoff_local(
        cn_local=cn_local,
        radar_local=radar_local,
        alpha=alpha,
        lambda_ia=lambda_ia,
        use_gpu=use_gpu,
    )

    # Prepare a list to gather all local arrays back to rank 0.
    if rank == 0:
        recvbuf = []
        for r in range(size):
            s, e = decompose_rows(ny, size, r)
            recvbuf.append(np.empty((e - s, nx), dtype=np.float32))
    else:
        recvbuf = None

    # Gather the local runoff arrays into recvbuf on rank 0.
    comm.Gather(runoff_local, recvbuf, root=0)

    # On rank 0, assemble the global runoff array from the gathered chunks.
    if rank == 0:
        runoff_global = np.empty((ny, nx), dtype=np.float32)
        for r in range(size):
            s, e = decompose_rows(ny, size, r)
            runoff_global[s:e, :] = recvbuf[r]
        return runoff_global

    # On non-root ranks, return None to signal that nothing is assembled.
    return None


def compute_flood_risk(runoff: np.ndarray) -> np.ndarray:
    \"\"\"Compute a simple binary flood-risk indicator from accumulated runoff.

    Parameters
    ----------
    runoff:
        2D array (ny, nx) of accumulated runoff (mm).

    Returns
    -------
    flood_risk:
        2D uint8 array with 1 where runoff exceeds the threshold, 0 otherwise.
    \"\"\"
    # Read the threshold from the configuration file.
    thr = cfg.hydrology.flood_threshold_mm

    # Build a boolean mask where runoff is greater or equal to the threshold.
    mask = runoff >= thr

    # Allocate the output array with integer values 0/1.
    flood_risk = np.zeros_like(runoff, dtype=np.uint8)

    # Set 1 where the mask is true (flood risk present).
    flood_risk[mask] = 1

    # Return the flood-risk map.
    return flood_risk


def write_runoff_netcdf(runoff: np.ndarray, dem_meta: dict, path: str) -> None:
    \"\"\"Write the accumulated runoff to a NetCDF file.

    Parameters
    ----------
    runoff:
        2D array (ny, nx) of accumulated runoff.
    dem_meta:
        Metadata dictionary obtained when loading the DEM.
    path:
        Output file path for the NetCDF file.
    \"\"\"
    # The DEM metadata may be either a rasterio profile or a NetCDF meta structure.
    # Here we handle the rasterio profile case (GeoTIFF DEM) for coordinates and CRS.
    if \"transform\" in dem_meta and \"crs\" in dem_meta:
        # Extract the affine transform and CRS from the rasterio profile.
        transform = dem_meta[\"transform\"]
        crs = dem_meta[\"crs\"]
        # Get the spatial shape from the runoff array.
        ny, nx = runoff.shape
        # Build coordinate arrays for the cell centers.
        xs = np.arange(nx)
        ys = np.arange(ny)
        # Use rasterio.transform.xy to get the real-world coordinates of centers.
        x_coords, _ = rasterio.transform.xy(transform, [0] * nx, xs, offset=\"center\")
        _, y_coords = rasterio.transform.xy(transform, ys, [0] * ny, offset=\"center\")
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        # Convert CRS to WKT if available.
        crs_wkt = crs.to_wkt() if crs is not None else None
    else:
        # If we do not have transform/info, fall back to index-based coordinates.
        ny, nx = runoff.shape
        x_coords = np.arange(nx, dtype=float)
        y_coords = np.arange(ny, dtype=float)
        crs_wkt = None

    # Prepare some global attributes for provenance.
    global_attrs = {
        \"title\": cfg.project.name,
        \"source\": \"PERFECT-M hydrological model\",
        \"region\": cfg.project.region,
    }

    # Use the helper from netcdf_utils to write a CF-compliant NetCDF file.
    write_cf_grid(
        path=path,
        data=runoff,
        x=x_coords,
        y=y_coords,
        var_name=\"runoff\",
        units=\"mm\",
        long_name=\"Accumulated surface runoff\",
        crs_wkt=crs_wkt,
        extra_global_attrs=global_attrs,
    )


def write_flood_risk_netcdf(flood_risk: np.ndarray, dem_meta: dict, path: str) -> None:
    \"\"\"Write the binary flood-risk indicator to a NetCDF file.

    Parameters
    ----------
    flood_risk:
        2D uint8 array (ny, nx) of flood-risk flags (1 = risk, 0 = no risk).
    dem_meta:
        Metadata dictionary obtained when loading the DEM.
    path:
        Output file path for the NetCDF file.
    \"\"\"
    # The coordinate-building logic mirrors that in write_runoff_netcdf.
    if \"transform\" in dem_meta and \"crs\" in dem_meta:
        transform = dem_meta[\"transform\"]
        crs = dem_meta[\"crs\"]
        ny, nx = flood_risk.shape
        xs = np.arange(nx)
        ys = np.arange(ny)
        x_coords, _ = rasterio.transform.xy(transform, [0] * nx, xs, offset=\"center\")
        _, y_coords = rasterio.transform.xy(transform, ys, [0] * ny, offset=\"center\")
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        crs_wkt = crs.to_wkt() if crs is not None else None
    else:
        ny, nx = flood_risk.shape
        x_coords = np.arange(nx, dtype=float)
        y_coords = np.arange(ny, dtype=float)
        crs_wkt = None

    # Global attributes similar to runoff output.
    global_attrs = {
        \"title\": cfg.project.name,
        \"source\": \"PERFECT-M hydrological model\",
        \"region\": cfg.project.region,
    }

    # Use the same helper to write a CF-compliant NetCDF file.
    write_cf_grid(
        path=path,
        data=flood_risk.astype(np.int16),
        x=x_coords,
        y=y_coords,
        var_name=\"flood_risk\",
        units=\"1\",
        long_name=\"Binary flood-risk indicator\",
        crs_wkt=crs_wkt,
        extra_global_attrs=global_attrs,
    )
