import json
import requests
import geopandas as gpd
import folium
from pyproj import Transformer
from folium.plugins import MousePosition
import io
import os
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from shapely.geometry import Point

warnings.simplefilter('ignore', InsecureRequestWarning)

JSON_FILENAME = 'powiat_wfs_egib.json'

def load_config(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def get_wfs_large_area(teryt_code, lat, lon, data_sources, search_radius=500):
    if teryt_code not in data_sources:
        print(f"Błąd: Brak konfiguracji dla TERYT: {teryt_code}")
        return None, None

    config = data_sources[teryt_code]
    wfs_url = config['url']
    
    # Transformacja WGS84 -> PUWG1992 (EPSG:2180)
    # always_xy=True oznacza, że dostaniemy: X (Easting), Y (Northing)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x_2180, y_2180 = transformer.transform(lon, lat)
    
    print(f"Łączenie z: {config['name']}...")
    print(f" -> Szukam wokół punktu 2180: X={x_2180:.2f}, Y={y_2180:.2f}")

    def fetch_layer(layer_name):
        # Definiujemy dwie wersje BBOX: 
        # 1. Standardowa (minx, miny, maxx, maxy)
        bbox_standard = f"{x_2180 - search_radius},{y_2180 - search_radius},{x_2180 + search_radius},{y_2180 + search_radius}"
        # 2. Odwrócona dla WFS 2.0 (miny, minx, maxy, maxx)
        bbox_flipped = f"{y_2180 - search_radius},{x_2180 - search_radius},{y_2180 + search_radius},{x_2180 + search_radius}"

        # Lista prób do wykonania
        attempts = [
            ("Standard (X,Y)", bbox_standard),
            ("Odwrócony (Y,X)", bbox_flipped)
        ]

        for attempt_name, bbox_val in attempts:
            params = {
                'service': 'WFS',
                'version': config.get('version', '2.0.0'),
                'request': 'GetFeature',
                'typeNames': layer_name,
                'srsName': 'EPSG:2180',
                'bbox': bbox_val,
                'outputFormat': 'application/json' # GeoJSON
            }

            try:
                # print(f"   Próba {attempt_name}...") # Debug
                response = requests.get(wfs_url, params=params, timeout=20, verify=False)
                
                # Czasem serwer zwraca tekst błędu zamiast JSON
                if response.status_code != 200 or b"Exception" in response.content:
                    continue

                # Próba wczytania
                try:
                    gdf = gpd.read_file(io.BytesIO(response.content))
                except Exception:
                    # Jeśli read_file rzuci błędem (np. index out of bounds), to znaczy że JSON jest pusty/zły
                    continue

                if not gdf.empty:
                    print(f" -> Sukces! Pobrano {len(gdf)} obiektów (tryb {attempt_name})")
                    # Naprawa CRS jeśli brakuje
                    if gdf.crs is None:
                        gdf.set_crs(epsg=2180, inplace=True)
                    
                    # Obliczenie dystansu
                    center_point = Point(x_2180, y_2180)
                    gdf['dist_to_center'] = gdf.geometry.distance(center_point)
                    
                    return gdf.to_crs(epsg=4326)
            
            except Exception as e:
                pass
        
        print(f" -> Nie udało się pobrać warstwy {layer_name} (sprawdzono obie kombinacje osi).")
        return gpd.GeoDataFrame()

    parcels = fetch_layer(config['layer_parcels'])
    buildings = fetch_layer(config['layer_buildings'])
    
    return parcels, buildings

def create_filtered_map(parcels, buildings, lat, lon, filename_out):
    m = folium.Map(location=[lat, lon], zoom_start=18)

    folium.raster_layers.WmsTileLayer(
        url='https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution',
        layers='Raster',
        name='Ortofotomapa',
        fmt='image/png',
        transparent=True,
        attr='GUGiK'
    ).add_to(m)

    # 1. DZIAŁKI
    if parcels is not None and not parcels.empty:
        # Filtrujemy: te bardzo blisko (< 30m) i reszta tła
        target_parcels = parcels[parcels['dist_to_center'] < 30] 
        background_parcels = parcels[parcels['dist_to_center'] >= 30]

        # Szukanie kolumny z numerem
        cols = parcels.columns
        # Lista popularnych nazw kolumn z numerem działki
        candidates = ['numer', 'nr_dzialki', 'identyfikator', 'id', 'pelny_numer', 'label', 'name']
        id_col = next((c for c in candidates if c in cols), cols[0])

        # Tło
        folium.GeoJson(
            background_parcels,
            name='Działki (tło)',
            style_function=lambda x: {'color': 'silver', 'fillOpacity': 0, 'weight': 1},
            tooltip=folium.GeoJsonTooltip(fields=[id_col], aliases=['Nr:'])
        ).add_to(m)

        # Szukane
        if not target_parcels.empty:
            folium.GeoJson(
                target_parcels,
                name='SZUKANE',
                style_function=lambda x: {'color': '#FFD700', 'fillOpacity': 0.1, 'weight': 4},
                tooltip=folium.GeoJsonTooltip(fields=[id_col], aliases=['SZUKANA:'])
            ).add_to(m)

    # 2. BUDYNKI
    if buildings is not None and not buildings.empty:
        folium.GeoJson(
            buildings,
            name='Budynki',
            style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.5, 'weight': 1},
            popup=folium.GeoJsonPopup(fields=buildings.columns.tolist()[:4])
        ).add_to(m)

    folium.LayerControl().add_to(m)
    folium.Marker([lat, lon], popup="Cel", icon=folium.Icon(color="blue", icon="star")).add_to(m)
    
    m.save(filename_out)
    print(f"\nMapa zapisana: {filename_out}")

if __name__ == "__main__":
    print("--- EGiB WFS v2.1 (Auto-Flip Fix) ---")
    data_sources = load_config(JSON_FILENAME)
    
    if data_sources:
        user_teryt = input("Kod TERYT (np. 1016): ").strip()
        if user_teryt in data_sources:
            try:
                # Domyślne: Tomaszów Maz.
                def_lat, def_lon = "51.531209", "20.026557"
                in_lat = input(f"Lat [Enter={def_lat}]: ").strip() or def_lat
                in_lon = input(f"Lon [Enter={def_lon}]: ").strip() or def_lon
                
                gdf_p, gdf_b = get_wfs_large_area(user_teryt, float(in_lat), float(in_lon), data_sources, search_radius=200)

                if (gdf_p is not None and not gdf_p.empty) or (gdf_b is not None and not gdf_b.empty):
                    create_filtered_map(gdf_p, gdf_b, float(in_lat), float(in_lon), f"mapa_{user_teryt}.html")
                else:
                    print("Brak danych (sprawdź czy punkt na pewno leży w granicach powiatu).")
            except ValueError:
                print("Błąd danych.")
        else:
            print("Nieznany kod TERYT.")