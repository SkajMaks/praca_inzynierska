import streamlit as st

# === 1. KONFIGURACJA STRONY ===
st.set_page_config(page_title="Detekcja Budynków AI", layout="wide")

import sys
import os
import gc
import math
import matplotlib.pyplot as plt
import shapely.geometry
import numpy as np

# === 2. IMPORTY LEKKIE ===
try:
    import geopandas as gpd
    import folium
    from streamlit_folium import st_folium
    from shapely.geometry import box
    import rasterio.features
    from PIL import Image
    import scipy.ndimage as nd
except ImportError as e:
    st.error(f"Brakuje bibliotek: {e}")
    st.stop()

try:
    import utils
except ImportError as e:
    st.error(f"Błąd importu 'utils.py': {e}")
    st.stop()

# === INICJALIZACJA SESJI ===
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'full_plot_id' not in st.session_state: st.session_state.full_plot_id = None
if 'prob_map' not in st.session_state: st.session_state.prob_map = None
if 'img_pil' not in st.session_state: st.session_state.img_pil = None
if 'bbox' not in st.session_state: st.session_state.bbox = None
if 'geom_2180' not in st.session_state: st.session_state.geom_2180 = None

# Wczytanie konfiguracji powiatów
@st.cache_resource
def load_wfs_sources():
    return utils.load_wfs_config('powiat_wfs_egib.json')

WFS_SOURCES = load_wfs_sources()

# === 3. MODEL I INFERENCJA ===
MODEL_PATH = "unet_inria.pth" 
PATCH_SIZE = 256

@st.cache_resource
def load_model_pipeline():
    try:
        import torch
        if not os.path.exists(MODEL_PATH):
            return None, None, f"Brak pliku modelu: {MODEL_PATH}"
            
        try:
            from models_unet import UNet
        except ImportError:
            return None, None, "Brak pliku 'models_unet.py'"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(n_channels=3, n_classes=2)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, "OK"
    except Exception as e:
        return None, None, str(e)

def predict_with_overlap(model, device, image_pil, patch_size=256, overlap=0.5):
    import torch
    import torch.nn.functional as F
    if model is None: return np.zeros((256, 256)), "Brak modelu"
    try:
        img = np.array(image_pil)
        if len(img.shape) == 3 and img.shape[2] == 4: img = img[:, :, :3]
        rgb = np.moveaxis(img, -1, 0)
        C, H, W = rgb.shape
        
        if H <= patch_size and W <= patch_size:
            stride = patch_size
        else:
            stride = int(patch_size * (1 - overlap))
            
        pad_h = (math.ceil(H / stride) * stride) + patch_size
        pad_w = (math.ceil(W / stride) * stride) + patch_size
        
        padded = np.zeros((C, pad_h, pad_w), dtype=rgb.dtype)
        padded[:, :H, :W] = rgb
        
        sum_map = np.zeros((pad_h, pad_w), dtype=np.float32)
        count_map = np.zeros((pad_h, pad_w), dtype=np.float32)
        
        total_steps = ((pad_h - patch_size) // stride + 1) * ((pad_w - patch_size) // stride + 1)
        progress_bar = st.progress(0, text=f"Analiza AI ({W}x{H} px)...")
        step = 0
        
        with torch.no_grad():
            for y in range(0, pad_h - patch_size + 1, stride):
                for x in range(0, pad_w - patch_size + 1, stride):
                    patch = padded[:, y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                    logits = model(patch_tensor)
                    probs = F.softmax(logits, dim=1)[0, 1]
                    sum_map[y:y+patch_size, x:x+patch_size] += probs.cpu().numpy()
                    count_map[y:y+patch_size, x:x+patch_size] += 1.0
                    step += 1
                    if step % 5 == 0 or total_steps < 10:
                        progress_bar.progress(min(step / max(total_steps, 1), 1.0))
        
        progress_bar.empty()
        count_map[count_map == 0] = 1
        return (sum_map / count_map)[:H, :W], None
    except Exception as e: return None, str(e)

# === 4. UI BOCZNE ===

st.title("🏘️ Detekcja Budynków AI")
st.sidebar.title("📍 Wyszukiwarka")
search_method = st.sidebar.radio("Metoda:", ["Adres", "Numer działki"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Ustawienia")
res_option = st.sidebar.select_slider(
    "Rozdzielczość zdjęcia",
    options=["Natywna (256px)", "Szybka (512px)", "Średnia (1024px)", "Dokładna (1536px)", "Ultra (2048px)"],
    value="Dokładna (1536px)" 
)
IMG_SIZE = int(res_option.split('(')[1].split('px')[0])

def start_search_address():
    if not st.session_state.address_input:
        st.toast("Wpisz adres!", icon="⚠️")
        return
    lat, lon = utils.get_coordinates_from_address(st.session_state.address_input)
    if lat and lon:
        found_id = utils.get_plot_id_from_xy(lat, lon)
        if found_id:
            st.session_state.full_plot_id = found_id.strip()
            run_full_analysis()
        else: st.toast("Punkt poza działką.", icon="❌")
    else: st.toast("Nie znaleziono adresu.", icon="❌")

def start_search_id():
    if st.session_state.plot_num_input:
        st.session_state.full_plot_id = st.session_state.plot_num_input.strip()
        run_full_analysis()

def run_full_analysis():
    st.session_state.analyzed = False
    st.session_state.prob_map = None
    
    geom_2180, srid, debug_msg = utils.get_plot_geometry_from_full_id(st.session_state.full_plot_id)
    
    if geom_2180 is None:
        st.toast("Błąd geometrii", icon="❌")
        st.session_state.error_msg = debug_msg
    else:
        st.session_state.geom_2180 = geom_2180
        st.session_state.bbox = utils.get_square_bbox(geom_2180, buffer_meters=40)
        
        st.session_state.img_pil = utils.get_orthophoto(st.session_state.bbox, size=(IMG_SIZE, IMG_SIZE))
        
        model, device, status = load_model_pipeline()
        if model:
            prob_map, error = predict_with_overlap(model, device, st.session_state.img_pil, patch_size=PATCH_SIZE, overlap=0.5)
            if prob_map is not None:
                st.session_state.prob_map = prob_map
                st.session_state.analyzed = True
            else: st.toast(f"Błąd AI: {error}", icon="❌")

if search_method == "Adres":
    st.sidebar.text_input("Adres", key="address_input", on_change=start_search_address)
    st.sidebar.button("🔍 Szukaj", on_click=start_search_address, type="primary")
else:
    st.sidebar.text_input("ID Działki", key="plot_num_input", on_change=start_search_id)
    st.sidebar.button("Analizuj ID", on_click=start_search_id, type="primary")

# === 5. WYNIKI ===

if st.session_state.analyzed and st.session_state.prob_map is not None:
    st.info(f"Analizowana działka: **{st.session_state.full_plot_id}**")
    
    area_sqm = st.session_state.geom_2180.area
    area_ar = area_sqm / 100.0
    st.caption(f"📐 Powierzchnia geometryczna działki: **{area_sqm:.2f} m²** ({area_ar:.2f} ar)")
    
    col1, col2 = st.columns(2)
    
    # Obraz
    col1.image(st.session_state.img_pil, caption=f"Ortofotomapa ({IMG_SIZE}x{IMG_SIZE} px)", use_container_width=True)
    
    # Overlay
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(st.session_state.img_pil, alpha=0.8) 
    ax.imshow(st.session_state.prob_map, cmap='inferno', vmin=0, vmax=1, alpha=0.5)
    ax.axis('off')
    col2.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    threshold = col2.slider("Czułość (Próg)", 0.1, 0.9, 0.5, 0.05)
    
    # Binaryzacja
    binary_mask = (st.session_state.prob_map > threshold).astype(np.uint8)
    binary_mask = nd.binary_opening(binary_mask, structure=np.ones((3,3))).astype(np.uint8)
    binary_mask = nd.binary_fill_holes(binary_mask).astype(np.uint8)

    # === MAPA INTERAKTYWNA ===
    st.subheader("🗺️ Mapa interaktywna")
    centroid = utils.transform_geom_to_4326(st.session_state.geom_2180.centroid)
    
    # Mapa bez białego tła
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=19, tiles=None)

    # Ortofotomapa Tło
    folium.raster_layers.WmsTileLayer(
        url='https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution',
        layers='Raster', name='Ortofotomapa', fmt='image/png', transparent=False, attr='GUGiK'
    ).add_to(m)

    # --- SĄSIEDZI Z POWIATU ---
    plot_id_str = str(st.session_state.full_plot_id)
    teryt_powiat = plot_id_str[:4] if len(plot_id_str) >= 4 else None
    
    debug_logs_parcels = []
    debug_logs_buildings = []

    if WFS_SOURCES and teryt_powiat in WFS_SOURCES:
        # A. Działki Tło
        with st.spinner("Pobieranie działek sąsiednich..."):
            gdf_parcels, debug_logs_parcels = utils.get_neighboring_wfs_layer(
                teryt_powiat, st.session_state.geom_2180.centroid.x, st.session_state.geom_2180.centroid.y,
                WFS_SOURCES, layer_key='layer_parcels', search_radius=300
            )
            if not gdf_parcels.empty:
                cols = gdf_parcels.columns
                id_col = next((c for c in ['numer', 'nr_dzialki', 'identyfikator', 'id', 'oznaczenie'] if c in cols), cols[0])
                folium.GeoJson(
                    gdf_parcels, name="Działki (EGiB)",
                    style_function=lambda x: {'color': 'yellow', 'fillOpacity': 0, 'weight': 1},
                    tooltip=folium.GeoJsonTooltip(fields=[id_col], aliases=['Nr działki:'])
                ).add_to(m)

        # B. Budynki Kontekst
        with st.spinner("Pobieranie budynków ewidencji..."):
            gdf_buildings, debug_logs_buildings = utils.get_neighboring_wfs_layer(
                teryt_powiat, st.session_state.geom_2180.centroid.x, st.session_state.geom_2180.centroid.y,
                WFS_SOURCES, layer_key='layer_buildings', search_radius=300
            )
            if not gdf_buildings.empty:
                popup_fields = gdf_buildings.columns.tolist()[:4]
                folium.GeoJson(
                    gdf_buildings, name="Budynki (EGiB)",
                    style_function=lambda x: {'color': '#ff4d4d', 'fillColor': '#ff4d4d', 'fillOpacity': 0.5, 'weight': 1},
                    popup=folium.GeoJsonPopup(fields=popup_fields)
                ).add_to(m)

    # Analizowana Działka
    plot_gdf = gpd.GeoDataFrame({'geometry': [st.session_state.geom_2180], 'ID': [st.session_state.full_plot_id]}, crs="EPSG:2180").to_crs(epsg=4326)
    folium.GeoJson(
        plot_gdf, name="Analizowana Działka",
        style_function=lambda x: {'color':'#FFD700', 'fillOpacity':0, 'weight':4},
        tooltip="Twoja działka"
    ).add_to(m)
    
    # Wyniki AI
    if binary_mask.sum() > 0:
        try:
            W_px, H_px = st.session_state.img_pil.size
            bbox = st.session_state.bbox
            pw = (bbox[2]-bbox[0])/W_px; ph = (bbox[3]-bbox[1])/H_px
            tf = rasterio.transform.from_origin(bbox[0], bbox[3], pw, ph)
            shapes = rasterio.features.shapes(binary_mask.astype(np.int16), transform=tf)
            polys = [shapely.geometry.shape(g) for g, v in shapes if v == 1]
            polys = [p for p in polys if p.area > 2]
            
            if polys:
                gdf_ai = gpd.GeoDataFrame({'geometry': polys}, crs="EPSG:2180").to_crs(epsg=4326)
                folium.GeoJson(
                    gdf_ai, name="Wykryte przez AI",
                    style_function=lambda x: {'color':'#00FF00', 'weight':2, 'fillColor':'#00FF00', 'fillOpacity':0.4},
                    tooltip="AI Wykrycie"
                ).add_to(m)
        except Exception: pass

    folium.LayerControl().add_to(m)
    st_folium(m, width=1000, height=600)
    
    # Diagnostyka
    st.markdown("---")
    with st.expander("🛠️ Diagnostyka WFS (Powiaty)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Logi Działek")
            st.text_area("Szczegóły:", value="\n".join(debug_logs_parcels) if debug_logs_parcels else "Brak logów", height=300)
        with c2:
            st.subheader("Logi Budynków")
            st.text_area("Szczegóły:", value="\n".join(debug_logs_buildings) if debug_logs_buildings else "Brak logów", height=300)

elif 'error_msg' in st.session_state and not st.session_state.analyzed:
    st.error("Błąd analizy.")
    with st.expander("Info"): st.code(st.session_state.get('error_msg', ''))
else:
    st.info("👈 Panel boczny: Wpisz adres i szukaj.")