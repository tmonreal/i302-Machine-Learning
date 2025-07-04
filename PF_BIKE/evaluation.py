import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re

# Ruta a la carpeta con los archivos
folder_path = '/home/tmonreal/Desktop/i302-Machine-Learning/PF_BIKE/Students_Predictions'
file_pattern = os.path.join(folder_path, '*_delta*min.csv')
file_list = glob.glob(file_pattern)

# Estaciones de interés
selected_stations = [29, 54, 130, 96, 161]
station_names = {
    29: "Parque Centenario",
    54: "Acuña de Figueroa",
    130: "Retiro II",
    96: "Carlos Gardel",
    161: "Humahuaca"
}

# Función para extraer el número de estación
def extract_station_id(value):
    try:
        match = re.search(r'(\d+)', str(value))
        return int(match.group(1)) if match else None
    except:
        return None

# Leer archivos y construir DataFrame
all_data = []

for file_path in file_list:
    try:
        df = pd.read_csv(file_path)
        group_name = os.path.basename(file_path).replace('.csv', '')
        station_col = next((col for col in df.columns if "station" in col.lower() or "estacion" in col.lower()), None)
        r2_col = next((col for col in df.columns if col.lower() == 'r2'), None)

        if station_col and r2_col:
            df['parsed_station'] = df[station_col].apply(extract_station_id)
            df_filtered = df[df['parsed_station'].isin(selected_stations)]

            for _, row in df_filtered.iterrows():
                station_id = int(row['parsed_station'])
                r2_value = row[r2_col]
                if pd.notna(r2_value):
                    all_data.append({
                        "id_estacion": station_id,
                        "grupo": group_name,
                        "r2": r2_value
                    })
    except Exception as e:
        print(f"⚠️ Error en {file_path}: {e}")

# Convertir a DataFrame
df_all = pd.DataFrame(all_data)

# Separar en delta30 y delta60
df_30 = df_all[df_all['grupo'].str.contains('delta30min')]
df_60 = df_all[df_all['grupo'].str.contains('delta60min')]

# Asignar colores únicos por grupo (combinando ambos)
todos_los_grupos = sorted(set(df_30['grupo'].unique()).union(df_60['grupo'].unique()))
cmap = plt.get_cmap('tab20')
colors = dict(zip(todos_los_grupos, cmap.colors[:len(todos_los_grupos)]))

# Función para graficar
def graficar(df_subset, delta_label, output_file):
    ncols = 3
    nrows = 2  # siempre 6 subplots para poder usar uno como leyenda
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8))
    axs = axs.flatten()

    for idx, station_id in enumerate(selected_stations):
        ax = axs[idx]
        df_station = df_subset[df_subset["id_estacion"] == station_id].sort_values("r2", ascending=False)

        grupos_station = df_station["grupo"].tolist()
        r2_values = df_station["r2"].tolist()
        bar_colors = [colors[grupo] for grupo in grupos_station]

        ax.bar(range(len(grupos_station)), r2_values, color=bar_colors)
        nombre = station_names.get(station_id, f"Estación {station_id}")
        ax.set_title(nombre)
        ax.set_xticks([])
        ax.set_ylim(-2, max(r2_values + [1]))

    # Leyenda en el 6to subplot
    ax_legend = axs[-1]
    ax_legend.axis("off")  # desactivar ejes

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=g, markerfacecolor=c, markersize=10)
        for g, c in colors.items() if g in df_subset['grupo'].unique()
    ]
    ax_legend.legend(handles=legend_handles, title="Grupos", loc='center', ncol=2, fontsize=10)

    # Si hay más de 6 subplots, ocultar los sobrantes
    for j in range(len(selected_stations), len(axs) - 1):
        fig.delaxes(axs[j])

    plt.suptitle(f"Comparación de R² por estación - {delta_label}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Gráfico guardado en: {output_file}")
# Ejecutar ambos gráficos
graficar(df_30, "delta30min", "/home/tmonreal/Desktop/i302-Machine-Learning/PF_BIKE/r2_delta30min.png")
graficar(df_60, "delta60min", "/home/tmonreal/Desktop/i302-Machine-Learning/PF_BIKE/r2_delta60min.png")

"""
29: Parque Centenario
54: Acuña de Figueroa
130: Retiro II
96: Carlos Gardel
161: Humahuaca
"""
