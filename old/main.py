from raster_utils import *
from hydrology import *
from termcolor import colored
import script.download 
import config  

def main():
    print(colored("Avvio del processo di elaborazione dei dati...", "cyan"))
    
    script.download.main()

    # Caricamento DEM
    print(colored("Caricamento del DEM in corso...", "green"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)
        
    # Crop del radar
    print(colored("Esecuzione del crop del radar...", "yellow"))
    crop_tiff_to_campania(config.RADAR_FILEPATH, config.RADAR_CAMPANIA)
    
    # Riallineamento della mappa CN al DEM
    print(colored("Riallineamento della mappa Curve Number al DEM...", "green"))
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    
    # Caricamento mappa Curve Number riallineata
    print(colored("Caricamento della mappa Curve Number riallineata...", "yellow"))
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)
    
    # Calcolo del deflusso superficiale (Runoff)
    print(colored("Calcolo del deflusso superficiale...", "blue"))
    runoff = calculate_accumulated_runoff("data/test_26mar", cn_map, MASK, config.DEM_FILEPATH, "runoff", output_format="netcdf")
    plot_territory_boundaries(dem_data, runoff)
    
    print(colored("Calcolo della direzione del flusso D8...", "blue"))
    d8_initialize(config.DEM_FILEPATH, config.D8_FILEPATH)
    # scalability_test(config.DEM_FILEPATH, config.D8_FILEPATH)
    # scalability_test(config.DEM_FILEPATH, config.D8_FILEPATH, MASK)
    
    flood_risk_map = compute_flood_risk(config.D8_FILEPATH, runoff)
    visualize_flood_risk_with_legend(flood_risk_map, dem_data)
    
    print(colored("Processo completato con successo!", "cyan"))

if __name__ == "__main__":
    main()