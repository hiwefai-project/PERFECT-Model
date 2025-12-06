\"\"\"High-level workflow orchestration for PERFECT-M.

This module ties together:

- configuration loading,
- data loading (DEM, CN map, radar),
- MPI-based runoff computation,
- flood-risk computation,
- and output writing.

All heavy computations are implemented in :mod:`perfectm.hydrology`.
\"\"\"

from __future__ import annotations

from mpi4py import MPI

from ..config import cfg
from ..raster import load_dem, load_cn_map, load_radar_full
from ..hydrology import runoff_mpi, compute_flood_risk, write_runoff_netcdf, write_flood_risk_netcdf


def run_full_pipeline() -> None:
    \"\"\"Run the complete PERFECT-M workflow.

    This function is intended to be called from the main entry point
    and can be executed with either a single MPI rank or multiple ranks.
    \"\"\"
    # Obtain the MPI communicator and rank.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Rank 0 prints some basic information about the run.
    if rank == 0:
        print(\"[PERFECT-M] Starting run for project:\", cfg.project.name)
        print(\"[PERFECT-M] Region:\", cfg.project.region)

    # Step 1: load DEM and CN map on rank 0 only.
    if rank == 0:
        # Load the DEM for the study area.
        dem, dem_meta = load_dem()
        # Load the Curve Number map.
        cn, cn_meta = load_cn_map()
        # Load the full 3D radar dataset with time dimension.
        radar, radar_coords, radar_meta = load_radar_full()

        # Basic sanity checks on shapes.
        if cn.shape != dem.shape:
            raise ValueError(f\"CN map shape {cn.shape} does not match DEM shape {dem.shape}\")
        if radar.shape[1:] != dem.shape:
            raise ValueError(f\"Radar spatial shape {radar.shape[1:]} does not match DEM shape {dem.shape}\")
    else:
        # On non-root ranks, set placeholders (they will not be used directly).
        dem = None
        dem_meta = {}
        cn = None
        cn_meta = {}
        radar = None
        radar_coords = {}
        radar_meta = {}

    # Step 2: run the MPI-parallel runoff computation.
    runoff_global = runoff_mpi(dem=dem, cn=cn, radar=radar, dem_meta=dem_meta)

    # Step 3: only rank 0 has the assembled global runoff; it performs post-processing.
    if rank == 0:
        print(\"[PERFECT-M] Runoff computation completed. Writing outputs...\")

        # Compute flood-risk indicator from the accumulated runoff.
        flood_risk = compute_flood_risk(runoff_global)

        # Paths for output files are taken directly from the configuration.
        runoff_path = cfg.paths.runoff_output
        flood_risk_path = cfg.paths.flood_risk_output

        # Write runoff and flood-risk to NetCDF files.
        write_runoff_netcdf(runoff_global, dem_meta=dem_meta, path=runoff_path)
        write_flood_risk_netcdf(flood_risk, dem_meta=dem_meta, path=flood_risk_path)

        print(f\"[PERFECT-M] Runoff written to {runoff_path}\")
        print(f\"[PERFECT-M] Flood risk written to {flood_risk_path}\")
        print(\"[PERFECT-M] All done.\")

    # Ensure all ranks synchronize before finalizing.
    comm.Barrier()
