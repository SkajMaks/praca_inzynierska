import pandas as pd
import json

URL = "https://geoinformatyka.com.pl/raporty/analiza_uslug_wfs.html"

tables = pd.read_html(URL)
df = tables[0]  # pierwsza tabela – trzeba będzie sprawdzić nazwy kolumn

# przykład: filtr tylko EGIB
df_egib = df[df["Nazwa zbioru"].str.contains("Ewidencja Gruntów i Budynków", na=False)]

config = {}
for _, row in df_egib.iterrows():
    teryt = str(row["TERYT"])
    url_caps = row["URL"]
    base_url = url_caps.split("?")[0]

    # tu musisz na podstawie kolumn z warstwami wyciągnąć pełne nazwy typu ewns:budynki
    # (z raportu widać, że są w kolumnie "Warstwy")

    # przykład „na sztywno”, do dalszego dopracowania:
    warstwy = str(row["Warstwy"])
    if "budynki" not in warstwy or "dzialki" not in warstwy:
        continue  # pomijamy powiaty bez obu warstw

    # heurystycznie: weź pierwszy token zawierający 'budynki' i 'dzialki'
    layer_buildings = [w for w in warstwy.split() if "budynki" in w][0]
    layer_parcels = [w for w in warstwy.split() if "dzialki" in w][0]

    config[teryt] = {
        "name": row["Organ zgłaszający"],
        "url": base_url,
        "version": str(row["Wersja WFS"]),
        "layer_buildings": layer_buildings,
        "layer_parcels": layer_parcels,
        "crs": "EPSG:2180",  # dla EGiB to praktycznie standard
    }

with open("powiat_wfs_egib.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
