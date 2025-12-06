import cdsapi
import zipfile
import os
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GLC_ZIP_FILE = os.path.join(DATA_DIR, "glc_global.zip")
GLC_NETCDF_FILE = os.path.join(DATA_DIR, "C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.41.5.15.8.39.9.13.7.nc")
GLC_TIFF_FILE = os.path.join(DATA_DIR, "glc_global.tif")

def download_glc():
    c = cdsapi.Client()
    c.retrieve(
        "satellite-land-cover",
        {
            "variable": "all",
            "year": ["2022"],
            "version": ["v2_1_1"],
            "area": [41.5, 13.7, 39.9, 15.8]
        },
        GLC_ZIP_FILE
    )
    print("GLC scaricato con successo.")

def extract_netcdf(zip_file, destination_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        print("Contenuto del file ZIP:")
        print(zip_ref.namelist()) 
        
        zip_ref.extract('C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.41.5.15.8.39.9.13.7.nc', destination_dir)

    extracted_file = os.path.join(destination_dir, 'C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.41.5.15.8.39.9.13.7.nc')
    if os.path.exists(extracted_file):
        print(f"File {extracted_file} estratto correttamente!")
    else:
        print(f"Errore: il file {extracted_file} non è stato estratto.")

def convert_to_geotiff():
    ds = xr.open_dataset(GLC_NETCDF_FILE)
    variable = list(ds.data_vars.keys())[0]
    data = ds[variable].isel(time=0).values

    transform = from_bounds(ds.lon.min(), ds.lat.min(), ds.lon.max(), ds.lat.max(), data.shape[1], data.shape[0])

    with rasterio.open(
        GLC_TIFF_FILE,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    print("GLC convertito in GeoTIFF.")

def process_glc():
    download_glc()
    extract_netcdf(GLC_ZIP_FILE, DATA_DIR)
    convert_to_geotiff()