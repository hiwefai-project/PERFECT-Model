import os
import time
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import numpy as np
import subprocess

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

def refresh_token():
    global oauth  
    print("Rinnovo del token...")
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    print("Token rinnovato con successo.")

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["DEM"],
    output: { bands: 1 },
  }
}
function evaluatePixel(sample) {
  return [sample.DEM / 1000]
}
"""

campania_bbox = {
    "min_x": 13.7,
    "max_x": 15.8,
    "min_y": 39.9,
    "max_y": 41.5
}

resx = 0.0003
resy = 0.0003
tile_width = 0.5  
tile_height = 0.5 

tiles_directory = "italy_tiles"
os.makedirs(tiles_directory, exist_ok=True)  

num_tiles_x = int(np.ceil((campania_bbox["max_x"] - campania_bbox["min_x"]) / tile_width))
num_tiles_y = int(np.ceil((campania_bbox["max_y"] - campania_bbox["min_y"]) / tile_height))

def download_tile(tile_x, tile_y):
    try:
        token_info = oauth.token
        if not token_info or "expires_at" not in token_info or token_info["expires_at"] < time.time():
            refresh_token()
    except Exception as e:
        print(f"Errore durante il controllo del token: {e}")
        refresh_token()

    tile_min_x = campania_bbox["min_x"] + tile_x * tile_width
    tile_max_x = min(tile_min_x + tile_width, campania_bbox["max_x"])
    tile_min_y = campania_bbox["min_y"] + tile_y * tile_height
    tile_max_y = min(tile_min_y + tile_height, campania_bbox["max_y"])

    request = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "bbox": [tile_min_x, tile_min_y, tile_max_x, tile_max_y],
            },
            "data": [
                {
                    "type": "dem",
                    "dataFilter": {"demInstance": "COPERNICUS_30"},
                    "processing": {
                        "upsampling": "BILINEAR",
                        "downsampling": "BILINEAR",
                    },
                }
            ],
        },
        "output": {
            "resx": resx,
            "resy": resy,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": evalscript,
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = oauth.post(url, json=request)

    if response.status_code == 200:
        filename = os.path.join(tiles_directory, f"italy_dem_tile_{tile_x}_{tile_y}.tiff")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File DEM scaricato con successo: {filename}")
        return filename
    else:
        print(f"Errore durante il download del tile ({tile_x}, {tile_y}): {response.status_code}")
        print(response.text)
        return None

def process_dem():
    all_filenames = []
    for tile_x in range(num_tiles_x):
        for tile_y in range(num_tiles_y):
            filename = download_tile(tile_x, tile_y)
            if filename:
                all_filenames.append(filename)

    if all_filenames:
        combined_filename = os.path.join(os.getcwd(), "data/campania_dem_combined.tiff")
        gdal_command = f"gdal_merge.py -o {combined_filename} {' '.join(all_filenames)}"
        print(f"Esecuzione del comando GDAL: {gdal_command}")
        subprocess.run(gdal_command, shell=True)
        print(f"Tutti i tile sono stati combinati in: {combined_filename}")

        for filename in all_filenames:
            try:
                os.remove(filename)
                print(f"File temporaneo eliminato: {filename}")
            except OSError as e:
                print(f"Errore durante l'eliminazione di {filename}: {e}")