#!/usr/bin/env python3
"""
Create an interactive Folium map showing the three study sites in Karnataka.

Output: figures/study_area_interactive.html
"""

import folium
import geopandas as gpd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ── Study site info ────────────────────────────────────────────────────────
SITES = {
    'Mangaluru': {
        'lat': 12.9141, 'lon': 74.8560,
        'climate': 'Tropical monsoon (Am)',
        'elevation': '0 – 50 m',
        'rainfall': '3,000 – 4,000 mm/yr',
        'color': 'red',
        'icon': 'tint',
    },
    'Mysore': {
        'lat': 12.3537, 'lon': 76.4054,
        'climate': 'Semi-arid to sub-humid (BSh/Cwa)',
        'elevation': '700 – 900 m',
        'rainfall': '~800 mm/yr',
        'color': 'blue',
        'icon': 'leaf',
    },
    'Shivamogga': {
        'lat': 13.9719, 'lon': 75.4704,
        'climate': 'Western Ghats transition (Am/Cwa)',
        'elevation': '500 – 800 m',
        'rainfall': '~2,000 mm/yr',
        'color': 'green',
        'icon': 'tree-conifer',
    },
}

# ── Load shapefiles ───────────────────────────────────────────────────────
mysore_shp = gpd.read_file(ROOT / 'study area' / 'mysore_shp' / 'Shapefile_Polygon.shp')
mysore_shp = mysore_shp.to_crs(epsg=4326)

shivamogga_shp = gpd.read_file(ROOT / 'study area' / 'shivamogga_shp' /
                                'shivmogga_rectangle_study_area.shp')
shivamogga_shp = shivamogga_shp.to_crs(epsg=4326)

# ── Create map centred on Karnataka ──────────────────────────────────────
center_lat = sum(s['lat'] for s in SITES.values()) / 3
center_lon = sum(s['lon'] for s in SITES.values()) / 3

m = folium.Map(location=[center_lat, center_lon], zoom_start=8,
               tiles='OpenStreetMap', control_scale=True)

# ── Helper: popup HTML ───────────────────────────────────────────────────
def make_popup(name, info):
    html = f"""
    <div style="font-family: Arial, sans-serif; width: 260px;">
        <h3 style="margin:0 0 8px; color: {info['color']};">{name}</h3>
        <table style="border-collapse:collapse; font-size:13px;">
            <tr><td style="padding:3px 8px;"><b>Coordinates</b></td>
                <td style="padding:3px 8px;">{info['lat']:.4f}°N, {info['lon']:.4f}°E</td></tr>
            <tr><td style="padding:3px 8px;"><b>Climate</b></td>
                <td style="padding:3px 8px;">{info['climate']}</td></tr>
            <tr><td style="padding:3px 8px;"><b>Elevation</b></td>
                <td style="padding:3px 8px;">{info['elevation']}</td></tr>
            <tr><td style="padding:3px 8px;"><b>Rainfall</b></td>
                <td style="padding:3px 8px;">{info['rainfall']}</td></tr>
        </table>
    </div>
    """
    return folium.Popup(html, max_width=300)

# ── Add markers for each site ────────────────────────────────────────────
for name, info in SITES.items():
    folium.Marker(
        location=[info['lat'], info['lon']],
        popup=make_popup(name, info),
        tooltip=name,
        icon=folium.Icon(color=info['color'], icon='info-sign'),
    ).add_to(m)

# ── Add Mangaluru circle (since no shapefile) ────────────────────────────
folium.Circle(
    location=[SITES['Mangaluru']['lat'], SITES['Mangaluru']['lon']],
    radius=5000,  # 5 km radius
    color='red', fill=True, fill_color='red', fill_opacity=0.2,
    tooltip='Mangaluru study area (approx.)',
).add_to(m)

# ── Add Mysore polygon ──────────────────────────────────────────────────
mysore_geojson = json.loads(mysore_shp.to_json())
folium.GeoJson(
    mysore_geojson,
    name='Mysore Study Area',
    style_function=lambda x: {
        'fillColor': '#2980B9',
        'color': '#1A5276',
        'weight': 2,
        'fillOpacity': 0.3,
    },
    tooltip='Mysore study area (shapefile polygon)',
).add_to(m)

# ── Add Shivamogga polygon ──────────────────────────────────────────────
shivamogga_geojson = json.loads(shivamogga_shp.to_json())
folium.GeoJson(
    shivamogga_geojson,
    name='Shivamogga Study Area',
    style_function=lambda x: {
        'fillColor': '#27AE60',
        'color': '#1E8449',
        'weight': 2,
        'fillOpacity': 0.3,
    },
    tooltip='Shivamogga study area (shapefile polygon)',
).add_to(m)

# ── Add layer control ───────────────────────────────────────────────────
folium.LayerControl().add_to(m)

# ── Fit bounds to show all 3 sites ───────────────────────────────────────
lats = [s['lat'] for s in SITES.values()]
lons = [s['lon'] for s in SITES.values()]

# Include shapefile extents
for gdf in [mysore_shp, shivamogga_shp]:
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    lons.extend([bounds[0], bounds[2]])
    lats.extend([bounds[1], bounds[3]])

sw = [min(lats) - 0.2, min(lons) - 0.2]
ne = [max(lats) + 0.2, max(lons) + 0.2]
m.fit_bounds([sw, ne])

# ── Add title ────────────────────────────────────────────────────────────
title_html = """
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
     z-index: 9999; background: white; padding: 10px 24px; border-radius: 8px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
  <h3 style="margin:0; color:#2C3E50;">Study Area Locations in Karnataka, India</h3>
  <p style="margin:4px 0 0; font-size:12px; color:#666;">
    Soil Moisture Forecasting Project — Mangaluru · Mysore · Shivamogga
  </p>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ── Save ─────────────────────────────────────────────────────────────────
out_path = FIG_DIR / 'study_area_interactive.html'
m.save(str(out_path))
print(f"✓ Saved interactive map: {out_path}")
