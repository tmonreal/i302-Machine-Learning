import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Datos de referencia
lat_origin = -34.445755
lon_origin = -58.530565
north_origin = 109.1
east_origin = -98

# Escalas aproximadas (grados por metro)
scale_lat = 1 / 111000
scale_lon = 1 / 91000

# Función para convertir coordenadas locales a geográficas
def local_to_geo(north, east):
    lat = lat_origin + (north - north_origin) * scale_lat
    lon = lon_origin + (east - east_origin) * scale_lon
    return lat, lon

# Estimaciones de estudiantes
estimaciones_locales = {
    'Zimmerman': (178.0, -234.1),
    'Lebrero': (60.0, -120.0),
    'Viñas Canale': (167.0, -439.0),
    'Basso': (67.4, -54.7),
    'Tissera': (-21.4, -166.3),
    'Arbelaiz': (125.5, -126.2),
    'Colombini': (124.6, -469.7)
}

# Convertir a coordenadas geográficas
data_geo = []
for nombre, (norte, este) in estimaciones_locales.items():
    lat, lon = local_to_geo(norte, este)
    data_geo.append({'name': nombre, 'geometry': Point(lon, lat)})

# Agregar el punto exacto
data_geo.append({'name': 'Exacto', 'geometry': Point(lon_origin, lat_origin)})

# Crear GeoDataFrame
gdf = gpd.GeoDataFrame(data_geo, crs='EPSG:4326')

# Reproyectar a Web Mercator para agregar mapas base
gdf = gdf.to_crs(epsg=3857)

# Graficar sobre mapa satelital
fig, ax = plt.subplots(figsize=(10, 10))
colors = [
    "#186378", "#377eb8", "#4daf4a", "#984ea3",
    "#c96400", "#e7298a", "#a65628", "#f5782a"
]

# Dibujar puntos y etiquetas con fondo blanco y mejor visibilidad
for (x, y, label), color in zip(zip(gdf.geometry.x, gdf.geometry.y, gdf['name']), colors):
    if label == "Exacto":
        ax.scatter(x, y, color='red', marker='*', s=200, edgecolor='black', zorder=6)
        ax.text(x - 10, y + 10, "Posición\n Exacta",
                fontsize=15, fontweight='bold',
                color='red',
                bbox=dict(facecolor='white', edgecolor='red', pad=3),
                zorder=10)
    else:
        ax.scatter(x, y, color=color, s=120, edgecolor='black', linewidth=0.5, zorder=5)
        ax.text(x - 20, y + 7, label,
                fontsize=14, fontweight='bold',
                color=color,
                bbox=dict(facecolor='white', edgecolor=color, pad=2),
                zorder=10)


# Agregar mapa base
#ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
#ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)

ax.set_axis_off()
plt.tight_layout()

# Guardar imagen
plt.savefig("estimaciones_satelital_1.png", dpi=300)