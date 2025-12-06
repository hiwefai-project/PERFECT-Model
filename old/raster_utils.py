import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from osgeo import gdal
import os
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.ndimage import sobel
from scipy.ndimage import binary_erosion

def load_and_plot_dem(file_path, title="Modello Digitale Elevazione (DEM)"):
    """
    Carica e visualizza un raster DEM.
    
    args:
        file_path (str): Percorso del file raster DEM.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster DEM.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        print(f"{title} Metadata:\n", profile)
        
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=14, fontweight="bold")
        plt.imshow(data, cmap='terrain')
        plt.colorbar(label='Altitudine (m)')
        plt.axis('off')  
        plt.show()
        
        return data, profile

def latlon_load_and_plot_dem(file_path, title="Modello Digitale Elevazione (DEM)"):
    """
    Carica e visualizza un raster DEM.
    
    args:
        file_path (str): Percorso del file raster DEM.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster DEM.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  
        profile = src.profile
        transform = src.transform  
        height, width = data.shape

        lon, lat = np.meshgrid(
            np.linspace(transform.c, transform.c + transform.a * width, width),
            np.linspace(transform.f, transform.f + transform.e * height, height)
        )

        data = np.where(data <= 0, np.nan, data)

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(lon, lat, data, cmap="terrain", shading="auto")
        plt.colorbar(label="Altitudine (m)")
        plt.xlabel("Longitudine")
        plt.ylabel("Latitudine")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.show()
        
        return data, profile
    
def load_and_plot_rainfall(file_path, title="Raster Pioggia"):
    """
    Carica e visualizza un raster delle precipitazioni.
    
    args:
        file_path (str): Percorso del file raster delle precipitazioni.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster delle precipitazioni.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        print(f"{title} Metadata:\n", profile)
        
        data = np.where(data <= 0, np.nan, data)
        
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=14, fontweight="bold")
        cmap = plt.cm.Blues
        cmap.set_bad(color='lightgray') 
        plt.imshow(data, cmap=cmap)
        plt.colorbar(label='Intensità di pioggia (mm/h)')
        plt.axis('off')  
        plt.show()
        
        return data, profile
    
def latlon_load_and_plot_rainfall(file_path, title="Raster Pioggia"):
    """
    Carica e visualizza un raster delle precipitazioni.
    
    args:
        file_path (str): Percorso del file raster delle precipitazioni.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster delle precipitazioni.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  
        profile = src.profile
        transform = src.transform  
        height, width = data.shape

        lon, lat = np.meshgrid(
            np.linspace(transform.c, transform.c + transform.a * width, width),
            np.linspace(transform.f, transform.f + transform.e * height, height)
        )

        data = np.where(data <= 0, np.nan, data)

        plt.figure(figsize=(8, 6))
        cmap = plt.cm.Blues
        plt.pcolormesh(lon, lat, data, cmap=cmap, shading="auto")
        plt.colorbar(label='Intensità di pioggia (mm/h)')
        plt.xlabel("Longitudine")
        plt.ylabel("Latitudine")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.show()
        
        return data, profile
    
def load_and_plot_land_cover(file_path, title="Mappa Land Cover"):
    """
    Carica e visualizza una mappa della land cover.
    
    args:
        file_path (str): Percorso del file raster della land cover.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster della land cover.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        print(f"{title} Metadata:\n", profile)
        
        water_value = 0  
        data = np.where(data == water_value, np.nan, data)
        
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=14, fontweight="bold")
        cmap = plt.cm.viridis
        cmap.set_bad(color='lightblue') 
        plt.imshow(data, cmap=cmap)
        plt.colorbar(label='Tipo di copertura terrestre')
        plt.axis('off')  
        plt.show()
        
        return data, profile

def latlon_load_and_plot_land_cover(file_path, title="Mappa Land Cover"):
    """
    Carica e visualizza una mappa della land cover.
    
    args:
        file_path (str): Percorso del file raster della land cover.
        title (str): Titolo della visualizzazione.
    
    returns:
        data (numpy.ndarray): Dati raster della land cover.
        profile (dict): Profilo del raster.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  
        profile = src.profile
        transform = src.transform  
        height, width = data.shape

        lon, lat = np.meshgrid(
            np.linspace(transform.c, transform.c + transform.a * width, width),
            np.linspace(transform.f, transform.f + transform.e * height, height)
        )

        data = np.where(data == 0, np.nan, data)

        plt.figure(figsize=(8, 6))
        cmap = plt.cm.viridis
        cmap.set_bad(color='lightblue') 
        plt.pcolormesh(lon, lat, data, cmap=cmap, shading="auto")
        plt.colorbar(label='Tipo di copertura terrestre')
        plt.xlabel("Longitudine")
        plt.ylabel("Latitudine")
        plt.title(title)
        plt.show()
        
        return data, profile

def plot_raster_with_latlon(file_path, title="Raster con coordinate geografiche"):
    """
    Carica un raster e lo visualizza utilizzando coordinate geografiche (latitudine/longitudine).
    
    args:
        file_path (str): Percorso del file raster.
        title (str): Titolo della visualizzazione.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  
        transform = src.transform  
        height, width = data.shape

        lon, lat = np.meshgrid(
            np.linspace(transform.c, transform.c + transform.a * width, width),
            np.linspace(transform.f, transform.f + transform.e * height, height)
        )

        data = np.where(data <= 0, np.nan, data)

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(lon, lat, data, cmap="terrain", shading="auto")
        plt.colorbar(label="Altitudine (m)")
        plt.xlabel("Longitudine")
        plt.ylabel("Latitudine")
        plt.title(title)
        plt.show()

def visualize_flood_risk(risk_map):
    """
    Visualizza la mappa del rischio alluvionale con una legenda chiara.

    :param risk_map: Array numpy contenente i dati di rischio alluvionale.
    """
    plt.figure(figsize=(10, 6))
    
    cmap = sns.color_palette("Reds", as_cmap=True)
    img = plt.imshow(risk_map, cmap=cmap, interpolation="nearest")

    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("Livello di Rischio di Alluvione", fontsize=12)
    
    plt.title("Mappa del Rischio Alluvionale", fontsize=14, fontweight="bold")
    plt.xlabel("Longitudine (pixel)", fontsize=12)
    plt.ylabel("Latitudine (pixel)", fontsize=12)
    
    plt.show()
    
def classify_risk_levels(risk_map):
    """
    Converte la mappa del rischio in classi discrete: basso, medio, alto, critico.
    
    :param risk_map: Array numpy contenente i dati di rischio alluvionale normalizzati tra 0 e 1.
    :return: Mappa classificata in 4 livelli di rischio.
    """
    classified_map = np.zeros_like(risk_map)

    classified_map[(risk_map >= 0.0) & (risk_map < 0.3)] = 1  # Basso (Verde)
    classified_map[(risk_map >= 0.3) & (risk_map < 0.6)] = 2  # Medio (Giallo)
    classified_map[(risk_map >= 0.6) & (risk_map < 0.8)] = 3  # Alto (Arancione)
    classified_map[(risk_map >= 0.8)] = 4  # Critico (Rosso)

    return classified_map

def visualize_flood_risk_with_legend(risk_map, background_map):
    """
    Visualizza la mappa del rischio alluvionale con una legenda che identifica le aree a rischio.
    Utilizza una mappa di sfondo in scala di grigi come riferimento geografico.

    :param risk_map: Array numpy contenente i dati di rischio alluvionale.
    :param background_map: Array numpy contenente i dati di sfondo (ad esempio, land cover).
    """
    # Classifica i livelli di rischio
    classified_map = classify_risk_levels(risk_map)

    # Colormap per il rischio alluvionale
    cmap = mcolors.ListedColormap(["green", "yellow", "orange", "red"])
    bounds = [1, 2, 3, 4, 5]  # Confini delle classi
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Visualizza la mappa di sfondo in scala di grigi
    plt.figure(figsize=(10, 6))
    plt.imshow(background_map, cmap="gray", alpha=0.5, vmin=np.nanmin(background_map), vmax=np.nanmax(background_map))

    # Sovrappone la mappa del rischio alluvionale
    img = plt.imshow(classified_map, cmap=cmap, norm=norm, alpha=0.7, interpolation="nearest")

    # Aggiunge la legenda
    cbar = plt.colorbar(img, ticks=[1.5, 2.5, 3.5, 4.5])
    cbar.set_ticklabels(["Basso", "Medio", "Alto", "Critico"])
    cbar.set_label("Livello di Rischio di Alluvione", fontsize=12)

    # Titoli e etichette
    plt.title("Mappa del Rischio Alluvionale", fontsize=14, fontweight="bold")
    plt.xlabel("Longitudine (pixel)", fontsize=12)
    plt.ylabel("Latitudine (pixel)", fontsize=12)

    # Nasconde gli assi
    plt.axis('off')
    plt.show()

def align_radar_to_dem(radar_tiff, dem_tiff, output_tiff):
    """
    Allinea un raster radar al DEM basandosi sulle coordinate centrali delle celle del DEM.

    args:
        radar_tiff (str): Percorso del file radar da allineare.
        dem_tiff (str): Percorso del file DEM di riferimento.
        output_tiff (str): Percorso del file raster allineato risultante.

    returns:
        None
    """
    # Carica il DEM
    with rasterio.open(dem_tiff) as dem_src:
        dem_data = dem_src.read(1)
        dem_profile = dem_src.profile
        dem_transform = dem_src.transform
        dem_width, dem_height = dem_src.width, dem_src.height
        dem_crs = dem_src.crs

        # Calcola le coordinate centrali delle celle del DEM
        x_coords = np.linspace(dem_transform[2] + dem_transform[0] / 2, 
                               dem_transform[2] + dem_transform[0] * dem_width - dem_transform[0] / 2, 
                               dem_width)
        y_coords = np.linspace(dem_transform[5] + dem_transform[4] / 2, 
                               dem_transform[5] + dem_transform[4] * dem_height - dem_transform[4] / 2, 
                               dem_height)
        xv, yv = np.meshgrid(x_coords, y_coords)

    # Carica il radar
    with rasterio.open(radar_tiff) as radar_src:
        radar_data = radar_src.read(1)
        radar_profile = radar_src.profile
        radar_transform = radar_src.transform
        radar_crs = radar_src.crs

        # Verifica se i CRS sono diversi e converte se necessario
        if radar_crs != dem_crs:
            print("⚠️ ATTENZIONE: Il CRS del radar è diverso da quello del DEM. Viene effettuata la conversione.")
            radar_src = rasterio.warp.reproject(
                source=radar_data,
                src_transform=radar_transform,
                src_crs=radar_crs,
                dst_crs=dem_crs,
                dst_shape=(dem_height, dem_width),
                resampling=rasterio.enums.Resampling.bilinear
            )
            radar_data = radar_src[0]

    # Interpola i valori del radar alle coordinate centrali del DEM
    radar_x = np.linspace(radar_src.bounds.left, radar_src.bounds.right, radar_src.width)
    radar_y = np.linspace(radar_src.bounds.top, radar_src.bounds.bottom, radar_src.height)
    radar_xv, radar_yv = np.meshgrid(radar_x, radar_y)

    # Normalizza le coordinate del DEM rispetto al radar
    radar_x_indices = (xv - radar_x[0]) / (radar_x[-1] - radar_x[0]) * (radar_src.width - 1)
    radar_y_indices = (yv - radar_y[0]) / (radar_y[-1] - radar_y[0]) * (radar_src.height - 1)

    # Interpolazione bilineare
    radar_aligned = map_coordinates(radar_data, [radar_y_indices.ravel(), radar_x_indices.ravel()], order=1, mode='nearest')
    radar_aligned = radar_aligned.reshape(dem_height, dem_width)

    # Salva il raster allineato
    aligned_profile = dem_profile.copy()
    aligned_profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(output_tiff, "w", **aligned_profile) as dst:
        dst.write(radar_aligned.astype(rasterio.float32), 1)

    print(f"✅ File radar allineato salvato in: {output_tiff}")
     
def crop_tiff_to_campania(input_tiff, output_tiff):
    """
    Esegue il cropping di un file TIFF sull'area della Campania.

    :param input_tiff: Percorso del file TIFF da cui eseguire il crop.
    :param output_tiff: Percorso del file TIFF risultante dopo il crop.
    """
    # Definisci il bounding box per la Campania
    campania_bbox = {
        "min_x": 13.7,  # Longitudine Ovest
        "max_x": 15.8,  # Longitudine Est
        "min_y": 39.9,  # Latitudine Sud
        "max_y": 41.5   # Latitudine Nord
    }

    # Verifica che il file di input esista
    if not os.path.exists(input_tiff):
        raise FileNotFoundError(f"Il file {input_tiff} non esiste.")

    try:
        # Usa gdal.Translate per eseguire il cropping
        gdal.Translate(
            output_tiff,
            input_tiff,
            projWin=[campania_bbox["min_x"], campania_bbox["max_y"], campania_bbox["max_x"], campania_bbox["min_y"]]
        )
        print(f"Cropping completato. File salvato in: {output_tiff}")
    except Exception as e:
        print(f"Errore durante il cropping: {e}")

def visualize_combined(dem_data, radar_original_data, radar_reprojected_data):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot DEM
    im0 = axes[0].imshow(dem_data, cmap='terrain')
    axes[0].set_title("DEM", fontsize=14)
    axes[0].axis('off')  # Nasconde gli assi
    fig.colorbar(im0, ax=axes[0], label='Altitudine (m)', fraction=0.046, pad=0.04)
    
    # Plot Radar originale
    im1 = axes[1].imshow(radar_original_data, cmap='Blues', vmin=np.min(radar_reprojected_data), vmax=np.max(radar_reprojected_data))
    axes[1].set_title("Radar Originale", fontsize=14)
    axes[1].axis('off')  # Nasconde gli assi
    fig.colorbar(im1, ax=axes[1], label='Intensità di pioggia (mm/h)', fraction=0.046, pad=0.04)
    
    # Plot Radar ricalibrato
    im2 = axes[2].imshow(radar_reprojected_data, cmap='Blues', vmin=np.min(radar_reprojected_data), vmax=np.max(radar_reprojected_data))
    axes[2].set_title("Radar Ricalibrato", fontsize=14)
    axes[2].axis('off')  # Nasconde gli assi
    fig.colorbar(im2, ax=axes[2], label='Intensità di pioggia (mm/h)', fraction=0.046, pad=0.04)
    
    plt.tight_layout()  
    plt.show()

def plot_runoff_on_land_cover(land_cover_data, runoff_data, title="Deflusso Superficiale su Land Cover"):
    """
    Visualizza il deflusso superficiale sovrapponendolo alla mappa della land cover.

    args:
        land_cover_data (numpy.ndarray): Dati raster della land cover.
        runoff_data (numpy.ndarray): Dati del deflusso superficiale.
        title (str): Titolo della visualizzazione.

    returns:
        None
    """
    # Gestione dei valori NaN nella land cover e nel runoff
    land_cover_data = np.where(land_cover_data <= 0, np.nan, land_cover_data)
    runoff_data = np.where(runoff_data < 0, 0, runoff_data)  # Assicura valori non negativi

    # Visualizza la land cover come sfondo
    plt.figure(figsize=(10, 8))
    plt.title(title)

    # Colormap per la land cover
    cmap_land_cover = plt.cm.viridis
    cmap_land_cover.set_bad(color='lightgray')  # Colora le aree NaN in grigio chiaro
    plt.imshow(land_cover_data, cmap="gray", alpha=0.5)

    # Sovrappone il deflusso superficiale
    cmap_runoff = plt.cm.Reds
    cmap_runoff.set_under(color='none')  # Non visualizza i valori bassi di runoff
    plt.imshow(runoff_data, cmap=cmap_runoff, alpha=0.6, vmin=0.1, label="Runoff")

    # Aggiunge una barra colore per il deflusso superficiale
    cbar = plt.colorbar(label="Deflusso Superficiale (mm)", fraction=0.046, pad=0.04)
    cbar.set_label("Deflusso Superficiale (mm)", rotation=270, labelpad=15)

    # Nasconde gli assi
    plt.axis('off')
    plt.show()

def plot_territory_boundaries(dem_data, runoff_data, title="Deflusso Superficiale su DEM"):
    """
    Visualizza il deflusso superficiale sovrapponendolo a una mappa DEM.
    Lo sfondo è il DEM in scala di grigi, mentre il deflusso è visualizzato in blu.

    args:
        dem_data (numpy.ndarray): Dati raster del DEM (Digital Elevation Model).
        runoff_data (numpy.ndarray): Dati del deflusso superficiale.
        title (str): Titolo della visualizzazione.

    returns:
        None
    """
    # Gestione dei valori NaN nel DEM e nel runoff
    dem_data = np.where(dem_data <= 0, np.nan, dem_data)  # Gestione di valori non validi nel DEM
    runoff_data = np.where(runoff_data < 0, 0, runoff_data)  # Assicura valori non negativi nel runoff

    # Crea la figura
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=14, fontweight="bold")

    # Visualizza il DEM come sfondo in scala di grigi
    plt.imshow(dem_data, cmap="gray", alpha=0.3, vmin=np.nanmin(dem_data), vmax=np.nanmax(dem_data))

    # Sovrappone il deflusso superficiale in blu
    cmap_runoff = plt.cm.Blues  # Usa una colormap in tonalità blu
    cmap_runoff.set_under(color='none')  # Non visualizza i valori bassi di runoff
    plt.imshow(runoff_data, cmap=cmap_runoff, alpha=0.7, vmin=0.1)

    # Aggiunge una barra colore per il deflusso superficiale
    cbar = plt.colorbar(label="Deflusso Superficiale (mm)", fraction=0.046, pad=0.04)
    cbar.set_label("Deflusso Superficiale (mm)", rotation=270, labelpad=15, fontsize=12)

    # Nasconde gli assi
    plt.axis('off')

    # Mostra la figura
    plt.show()
    
def sea_mask(input_tiff):
    with rasterio.open(input_tiff) as src:
        dem = src.read(1)  

    mask = (dem > 0).astype(np.uint8) # 1 per la terraferma, 0 per il mare
    
    return mask