import os
import requests
from datetime import datetime, timedelta

def download_tiff_files(base_url, year, month, day, base_output_dir="data/radar"):
    # Creare la cartella di output basata sulla data
    output_dir = os.path.join(base_output_dir, f"{year}/{month:02}/{day:02}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterare per ogni intervallo di 10 minuti
    start_time = datetime(year, month, day, 0, 0)
    end_time = datetime(year, month, day, 23, 59)
    delta = timedelta(minutes=10)

    current_time = start_time
    while current_time <= end_time:
        # Formattare l'ora in formato ZHHMM
        time_str = current_time.strftime("Z%H%M")
        date_str = current_time.strftime("%Y%m%d")

        # Costruire l'URL con il nuovo formato
        file_name = f"rdr0_d01_{date_str}{time_str}_VMI.tiff"
        url = f"{base_url}/{year}/{month:02}/{day:02}/{file_name}"
        file_path = os.path.join(output_dir, file_name)

        # Scaricare il file
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {file_name}")
            else:
                print(f"File not found: {file_name}")
        except requests.RequestException as e:
            print(f"Error downloading {file_name}: {e}")

        # Passa al successivo intervallo di 10 minuti
        current_time += delta
        
def process_radar():
    base_url = "http://193.205.230.6/rdr0/"
    year = 2025
    month = 3
    day = 26

    download_tiff_files(base_url, year, month, day)