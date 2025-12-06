import glob
import os

# Percorsi dei file
DEM_FILEPATH = "data/campania_dem_combined.tiff"
RADAR_FILEPATH = "data/rdr0_d01_20250401Z0540_VMI_cropped.tiff" # dinosauro
# RADAR_FILEPATH = "rdr0_d01_20250314Z0710_VMI.tiff"
RADAR_FILES = sorted(glob.glob("path_to_radar_tiffs/*.tif"))
RADAR_CAMPANIA = "data/cropped.tif"
REPROJECTED_RADAR_FILEPATH = "data/reprojected_radar.tiff"
CN_MAP_FILEPATH = "data/aligned_cn.tif"
D8_FILEPATH = "D8_output.tiff"
ALIGNED_CN_FILEPATH = "aligned_cn.tif"