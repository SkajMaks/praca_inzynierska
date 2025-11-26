import requests
import io
import urllib3
import re
import os
import json
import tempfile
from urllib.parse import urlencode
from PIL import Image
from owslib.wms import WebMapService
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box, Point
import pyproj
from shapely.ops import transform
import streamlit as st
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Wyłączamy ostrzeżenia o braku weryfikacji SSL (częste w urzędach)
urllib3.disable_warnings(InsecureRequestWarning)

# --- KONFIGURACJA URL ---

ULDK_URL = "https://uldk.gugik.gov.pl/"
WMS_ORTO_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution"
WFS_BU_URL = "https://mapy.geoportal.gov.pl/wss/service/wfsBU/guest"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}

# ------------------------------------------------------------------------------
# --- FUNKCJE ULDK / SŁOWNIKOWE ------------------------------------------------
# ------------------------------------------------------------------------------

def fetch_uldk_data(request_type, object_id=None):
    """Pobiera dane słownikowe z ULDK (woj/pow/gm/obr)."""
    params = {"request": request_type, "result": "teryt,nazwa"}
    if object_id:
        params["id"] = object_id
    try:
        response = requests.get(
            ULDK_URL, params=params, headers=HEADERS, timeout=10, verify=False
        )
        response.encoding = response.apparent_encoding
        text = response.text
        if "error" in text.lower() or response.status_code != 200:
            return []
        data = []
        for line in text.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                data.append({"id": parts[0], "name": f"{parts[1]} [{parts[0]}]"})
        return data
    except Exception:
        return []

def get_voivodeships():
    return [{"id": "14", "name": "mazowieckie"}, {"id": "10", "name": "łódzkie"}]

def get_counties(id): return fetch_uldk_data("GetCountyById", id)
def get_communes(id): return fetch_uldk_data("GetCommuneById", id)
def get_precincts(id): return fetch_uldk_data("GetRegionById", id)

def check_precinct_format(pid):
    try:
        return requests.get(
            ULDK_URL,
            params={"request": "GetParcelsInRegion", "region": pid, "cnt": 5},
            headers=HEADERS, verify=False
        ).text
    except Exception:
        return ""

# ------------------------------------------------------------------------------
# --- GEOLOKALIZACJA -----------------------------------------------------------
# ------------------------------------------------------------------------------

def get_coordinates_from_address(address):
    """Zwraca (lat, lon) dla podanego adresu przy użyciu Nominatim (EPSG:4326)."""
    url = "https://nominatim.openstreetmap.org/search"
    try:
        r = requests.get(
            url,
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "pl"},
            headers=HEADERS, timeout=5
        )
        d = r.json()
        if d:
            return float(d[0]["lat"]), float(d[0]["lon"])
    except Exception:
        pass
    return None, None

def get_plot_id_from_xy(lat, lon):
    """Pobiera identyfikator działki z ULDK na podstawie współrzędnych."""
    try:
        r = requests.get(
            ULDK_URL,
            params={"request": "GetParcelByXY", "xy": f"{lon},{lat},4326", "result": "id"},
            headers=HEADERS, verify=False, timeout=10
        )
        if r.status_code == 200 and "error" not in r.text.lower():
            parts = r.text.strip().split()
            for p in reversed(parts):
                if "." in p: return p
            if parts and parts[-1] != "0": return parts[-1]
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# --- GEOMETRIA DZIAŁKI --------------------------------------------------------
# ------------------------------------------------------------------------------

def get_plot_geometry_from_full_id(full_id):
    """Pobiera geometrię działki (WKT) z ULDK na podstawie identyfikatora."""
    clean_id = re.sub(r"[^\w\._/-]", "", str(full_id))
    params = {"request": "GetParcelById", "id": clean_id, "result": "geom_wkt,srid"}
    try:
        response = requests.get(
            ULDK_URL, params=params, headers=HEADERS, timeout=15, verify=False
        )
        if response.status_code != 200:
            return None, None, f"HTTP Error: {response.status_code}"

        text = response.text
        lines = text.strip().split("\n")
        data_line = next((line for line in lines if "POLYGON" in line.upper()), None)
        
        if not data_line and lines and ";" in lines[-1]:
            data_line = lines[-1]
        if not data_line:
            return None, None, f"Błąd API GUGiK:\n{text}"

        match = re.search(r"(MULTI)?POLYGON", data_line, re.IGNORECASE)
        if not match:
            return None, None, f"Nie rozpoznano formatu WKT:\n{data_line}"

        wkt_str = data_line[match.start() : data_line.rfind(")") + 1]
        srid = 2180
        digits = re.findall(r"\d{4}", data_line.replace(wkt_str, ""))
        if digits: srid = int(digits[-1])

        return wkt.loads(wkt_str), srid, "OK"
    except Exception as e:
        return None, None, str(e)

# ------------------------------------------------------------------------------
# --- FUNKCJE POMOCNICZE -------------------------------------------------------
# ------------------------------------------------------------------------------

def get_square_bbox(geom, buffer_meters=20):
    """Zwraca kwadratowy BBOX wokół geometrii powiększony o bufor."""
    minx, miny, maxx, maxy = geom.bounds
    minx -= buffer_meters; miny -= buffer_meters
    maxx += buffer_meters; maxy += buffer_meters
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    side = max(maxx - minx, maxy - miny) / 2
    return (cx - side, cy - side, cx + side, cy + side)

def get_orthophoto(bbox, size=(512, 512)):
    """Pobiera ortofotomapę z WMS."""
    try:
        wms = WebMapService(WMS_ORTO_URL, timeout=30, headers=HEADERS)
        img = wms.getmap(
            layers=["Raster"], srs="EPSG:2180", bbox=bbox, size=size,
            format="image/png", transparent=True
        )
        return Image.open(io.BytesIO(img.read()))
    except Exception:
        return Image.new("RGB", size, (128, 128, 128))

def transform_geom_to_4326(geom):
    """Transformuje geometrię z EPSG:2180 do EPSG:4326."""
    project = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:2180"), pyproj.CRS("EPSG:4326"), always_xy=True
    ).transform
    return transform(project, geom)

def bbox_2180_to_4258(bbox_2180):
    """Przelicza BBOX EPSG:2180 -> EPSG:4258 (lat/lon)."""
    minx, miny, maxx, maxy = bbox_2180
    transformer = pyproj.Transformer.from_crs("EPSG:2180", "EPSG:4258", always_xy=True)
    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    lonlats = [transformer.transform(x, y) for x, y in zip(xs, ys)]
    lons, lats = [p[0] for p in lonlats], [p[1] for p in lonlats]
    return (min(lons), min(lats), max(lons), max(lats))

# ------------------------------------------------------------------------------
# --- WFS GUGiK (BDOT10k) ------------------------------------------------------
# ------------------------------------------------------------------------------

def get_bdot10k_buildings(bbox_2180):
    """Pobiera budynki INSPIRE z GUGiK."""
    debug_info = []
    bbox_4258 = bbox_2180_to_4258(bbox_2180)
    bbox_str = ",".join(map(str, [round(x, 6) for x in bbox_4258]))
    
    url = f"{WFS_BU_URL}?service=WFS&version=2.0.0&request=GetFeature&srsName=EPSG:4258&bbox={bbox_str}&outputFormat=application/json&typeNames=bu-core2d:Building"
    debug_info.append(f"Query: {url}")
    
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=60)
        if r.status_code != 200:
            debug_info.append(r.text[:800])
            return gpd.GeoDataFrame(), "\n".join(debug_info)
        
        # Zapis do pliku tmp (bezpieczniejsze dla geopandas)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
            
        gdf = gpd.read_file(tmp_path)
        if gdf.empty: return gpd.GeoDataFrame(), "\n".join(debug_info)
        
        if gdf.crs is None: gdf.set_crs(epsg=4258, inplace=True)
        return gdf.to_crs(epsg=2180), "\n".join(debug_info)
    except Exception as e:
        debug_info.append(f"Error: {e}")
        return gpd.GeoDataFrame(), "\n".join(debug_info)

# ------------------------------------------------------------------------------
# --- POWIATOWE WFS (EGiB) - DIAGNOSTYKA + AUTO-FLIP + FORMAT FALLBACK ---------
# ------------------------------------------------------------------------------

def load_wfs_config(filename='powiat_wfs_egib.json'):
    if not os.path.exists(filename): return None
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

def get_neighboring_wfs_layer(teryt_code, center_x_2180, center_y_2180, data_sources, layer_key='layer_parcels', search_radius=300):
    """
    Pobiera warstwę z WFS (EGiB) z pełną diagnostyką i obsługą GML przez pliki tymczasowe.
    """
    logs = []
    logs.append(f"--- START: Pobieranie {layer_key} dla TERYT {teryt_code} ---")
    
    if teryt_code not in data_sources:
        logs.append("❌ BŁĄD: Brak kodu TERYT w pliku JSON.")
        return gpd.GeoDataFrame(), logs

    config = data_sources[teryt_code]
    wfs_url = config['url']
    logs.append(f"📍 Urząd: {config.get('name', 'Nieznany')}")
    
    if layer_key not in config:
        logs.append(f"❌ BŁĄD: Brak definicji warstwy '{layer_key}'.")
        return gpd.GeoDataFrame(), logs
        
    layer_name = config[layer_key]
    version = config.get('version', '1.0.0') # Domyślnie 1.0.0 dla bezpieczeństwa
    logs.append(f"🗺️ Warstwa: {layer_name} | Wersja: {version}")
    
    # BBOX Definitions
    # Standard: X=Wschód, Y=Północ (GIS)
    bbox_std = f"{center_x_2180 - search_radius},{center_y_2180 - search_radius},{center_x_2180 + search_radius},{center_y_2180 + search_radius}"
    # Flipped: X=Północ, Y=Wschód (Geodezja) - Kraków często tego wymaga!
    bbox_flip = f"{center_y_2180 - search_radius},{center_x_2180 - search_radius},{center_y_2180 + search_radius},{center_x_2180 + search_radius}"

    # Strategie
    strategies = [
        # Próba 1: Odwrócone osie (najczęstsze dla MapServera PL) + GML
        {"name": "Flipped (GML)",      "bbox": bbox_flip, "format": None},
        # Próba 2: Standardowe osie + GML
        {"name": "Standard (GML)",     "bbox": bbox_std,  "format": None},
        # Próba 3: GeoJSON (dla nowoczesnych serwerów)
        {"name": "GeoJSON",            "bbox": bbox_std,  "format": "application/json"},
    ]

    for i, strat in enumerate(strategies):
        logs.append(f"\n🔹 PRÓBA {i+1}: {strat['name']}")
        
        params = {
            'service': 'WFS',
            'version': version,
            'request': 'GetFeature',
            'typeNames': layer_name,
            'typeName': layer_name,
            'srsName': 'EPSG:2180',
            'bbox': strat["bbox"]
        }
        if strat["format"]:
            params['outputFormat'] = strat["format"]

        try:
            # Logowanie linku
            req_prep = requests.Request('GET', wfs_url, params=params).prepare()
            logs.append(f"   🔗 LINK: {req_prep.url}")

            response = requests.Session().send(req_prep, timeout=25, verify=False)
            logs.append(f"   HTTP Status: {response.status_code}")
            
            if response.status_code != 200:
                logs.append(f"   ⚠️ Błąd HTTP. Treść: {response.text[:200]}")
                continue
            
            if "exception" in response.text[:300].lower():
                logs.append(f"   ⚠️ ExceptionReport (XML).")
                continue

            # --- KLUCZOWA ZMIANA: ZAPIS DO PLIKU TYMCZASOWEGO ---
            # Geopandas nie radzi sobie z GML z pamięci (BytesIO). Musi mieć plik na dysku.
            suffix = ".json" if strat["format"] == "application/json" else ".gml"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                gdf = gpd.read_file(tmp_path)
            except Exception as e_read:
                logs.append(f"   ⚠️ Błąd odczytu pliku geopandas: {e_read}")
                gdf = gpd.GeoDataFrame()
            finally:
                # Sprzątanie pliku
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

            if not gdf.empty:
                logs.append(f"✅ SUKCES! Pobrano {len(gdf)} obiektów.")
                if gdf.crs is None:
                    gdf.set_crs(epsg=2180, inplace=True)
                return gdf.to_crs(epsg=4326), logs
            else:
                logs.append("   ⚠️ Pusty wynik (0 obiektów w pliku).")
        
        except Exception as e:
            logs.append(f"   ❌ Wyjątek połączenia: {str(e)}")
            pass
            
    logs.append("\n❌ WSZYSTKIE PRÓBY NIEUDANE.")
    return gpd.GeoDataFrame(), logs