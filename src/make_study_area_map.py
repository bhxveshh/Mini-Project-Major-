#!/usr/bin/env python3
"""
Create a publication-quality study area map showing the three research locations
in Karnataka, India: Mangaluru, Mysore, and Shivamogga.

Output: figures/fig_study_area_map.png
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box, Polygon, MultiPolygon
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ── Study site coordinates ─────────────────────────────────────────────────
SITES = {
    'Mangaluru': {'lat': 12.9141, 'lon': 74.8560,
                  'climate': 'Tropical monsoon', 'color': '#E74C3C'},
    'Mysore':    {'lat': 12.3537, 'lon': 76.4054,
                  'climate': 'Semi-arid plateau', 'color': '#2980B9'},
    'Shivamogga':{'lat': 13.9719, 'lon': 75.4704,
                  'climate': 'Western Ghats transition', 'color': '#27AE60'},
}

# ── Load shapefiles ───────────────────────────────────────────────────────
mysore_shp = gpd.read_file(ROOT / 'study area' / 'mysore_shp' / 'Shapefile_Polygon.shp')
mysore_shp = mysore_shp.to_crs(epsg=4326)

shivamogga_shp = gpd.read_file(ROOT / 'study area' / 'shivamogga_shp' /
                                'shivmogga_rectangle_study_area.shp')
shivamogga_shp = shivamogga_shp.to_crs(epsg=4326)

# ── Simplified Karnataka boundary (approximate) ───────────────────────────
# Using natural earth for India admin boundaries. If not available, draw a
# simplified Karnataka outline from known vertices.
try:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres') if hasattr(gpd.datasets, 'get_path')
                          else gpd.datasets.get_path('naturalearth_lowres'))
    india = world[world['name'] == 'India']
    has_india = True
except Exception:
    has_india = False

# Approximate Karnataka boundary (simplified polygon for context)
karnataka_coords = [
    (74.05, 11.60), (74.80, 11.70), (75.90, 11.80), (76.40, 11.60),
    (77.10, 11.70), (77.60, 12.00), (77.80, 12.50), (78.00, 13.00),
    (78.30, 13.50), (78.40, 14.00), (78.10, 14.50), (77.80, 15.00),
    (77.50, 15.50), (77.00, 15.90), (76.50, 16.30), (76.00, 16.50),
    (75.50, 16.60), (75.00, 16.80), (74.50, 17.00), (74.10, 16.80),
    (73.80, 16.50), (73.70, 16.00), (74.00, 15.50), (74.10, 15.00),
    (74.30, 14.50), (74.50, 14.00), (74.60, 13.50), (74.20, 13.00),
    (74.00, 12.50), (74.05, 11.60),
]
karnataka_poly = Polygon(karnataka_coords)
karnataka_gdf = gpd.GeoDataFrame(geometry=[karnataka_poly], crs='EPSG:4326')

# ── Arabian Sea polygon for context ───────────────────────────────────────
sea_poly = box(73.0, 11.0, 74.05, 17.5)
sea_gdf = gpd.GeoDataFrame(geometry=[sea_poly], crs='EPSG:4326')

# ── Create figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')

# Plot sea
sea_gdf.plot(ax=ax, color='#D6EAF8', edgecolor='none', alpha=0.5)

# Plot Karnataka boundary
karnataka_gdf.plot(ax=ax, facecolor='#FDEBD0', edgecolor='#7F8C8D',
                   linewidth=1.5, alpha=0.7)

# Plot India outline for context
if has_india:
    india.boundary.plot(ax=ax, edgecolor='#5D6D7E', linewidth=0.8, alpha=0.5)

# Plot study area polygons
mysore_shp.plot(ax=ax, facecolor=SITES['Mysore']['color'], edgecolor='black',
                linewidth=1.5, alpha=0.5, label='Mysore study area')

shivamogga_shp.plot(ax=ax, facecolor=SITES['Shivamogga']['color'], edgecolor='black',
                    linewidth=1.5, alpha=0.5, label='Shivamogga study area')

# Plot Mangaluru as a marker
ax.plot(SITES['Mangaluru']['lon'], SITES['Mangaluru']['lat'],
        'o', color=SITES['Mangaluru']['color'], markersize=14,
        markeredgecolor='black', markeredgewidth=1.5, zorder=5,
        label='Mangaluru (point)')

# ── Labels with boxes ────────────────────────────────────────────────────
bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.9)

for name, info in SITES.items():
    # Offset labels so they don't overlap markers
    if name == 'Mangaluru':
        off_x, off_y = -0.6, 0.25
    elif name == 'Mysore':
        off_x, off_y = 0.35, -0.25
    else:
        off_x, off_y = 0.35, 0.20

    label_text = f"{name}\n({info['lat']:.2f}°N, {info['lon']:.2f}°E)\n{info['climate']}"
    ax.annotate(label_text,
                xy=(info['lon'], info['lat']),
                xytext=(info['lon'] + off_x, info['lat'] + off_y),
                fontsize=9, fontweight='bold', ha='center', va='center',
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                zorder=6)

# Also mark centroids of the shapefile polygons
for name, gdf, color in [('Mysore', mysore_shp, SITES['Mysore']['color']),
                          ('Shivamogga', shivamogga_shp, SITES['Shivamogga']['color'])]:
    centroid = gdf.geometry.centroid.iloc[0]
    ax.plot(centroid.x, centroid.y, 's', color=color, markersize=8,
            markeredgecolor='black', markeredgewidth=1, zorder=5)

# ── Map extent ────────────────────────────────────────────────────────────
ax.set_xlim(73.5, 79.0)
ax.set_ylim(11.3, 17.5)

# ── Grid ──────────────────────────────────────────────────────────────────
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
ax.tick_params(labelsize=10)

# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title('Study Area Locations in Karnataka, India',
             fontsize=16, fontweight='bold', pad=15)

# ── North arrow ──────────────────────────────────────────────────────────
arrow_x = 78.3
arrow_y_base = 16.5
arrow_dy = 0.5
ax.annotate('N', xy=(arrow_x, arrow_y_base + arrow_dy + 0.05),
            fontsize=14, fontweight='bold', ha='center', va='bottom')
ax.annotate('', xy=(arrow_x, arrow_y_base + arrow_dy),
            xytext=(arrow_x, arrow_y_base),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# ── Scale bar ─────────────────────────────────────────────────────────────
# Approximate scale: 1 degree latitude ≈ 111 km
# Draw a manual scale bar at the bottom
scale_x = 74.0
scale_y = 11.6
bar_len_deg = 1.0   # ~111 km
ax.plot([scale_x, scale_x + bar_len_deg], [scale_y, scale_y],
        'k-', linewidth=3, zorder=7)
ax.plot([scale_x, scale_x], [scale_y - 0.05, scale_y + 0.05],
        'k-', linewidth=2, zorder=7)
ax.plot([scale_x + bar_len_deg, scale_x + bar_len_deg],
        [scale_y - 0.05, scale_y + 0.05], 'k-', linewidth=2, zorder=7)
ax.text(scale_x + bar_len_deg / 2, scale_y - 0.15, '≈ 111 km',
        ha='center', va='top', fontsize=10, fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=SITES['Mangaluru']['color'], edgecolor='black',
                   label='Mangaluru (point marker)'),
    mpatches.Patch(facecolor=SITES['Mysore']['color'], edgecolor='black',
                   alpha=0.5, label='Mysore study area (polygon)'),
    mpatches.Patch(facecolor=SITES['Shivamogga']['color'], edgecolor='black',
                   alpha=0.5, label='Shivamogga study area (polygon)'),
    mpatches.Patch(facecolor='#FDEBD0', edgecolor='#7F8C8D',
                   label='Karnataka (approx.)'),
    mpatches.Patch(facecolor='#D6EAF8', edgecolor='none',
                   label='Arabian Sea'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor='gray', title='Map Legend',
          title_fontsize=10)

# ── Annotations ───────────────────────────────────────────────────────────
ax.text(73.7, 14.5, 'Arabian\nSea', fontsize=11, fontstyle='italic',
        color='#2471A3', ha='center', alpha=0.7, rotation=90)
ax.text(76.3, 16.8, 'KARNATAKA', fontsize=13, fontstyle='italic',
        color='#7F6C3B', ha='center', alpha=0.6, fontweight='bold')

# ── Save ──────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = FIG_DIR / 'fig_study_area_map.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✓ Saved static map: {out_path}")
