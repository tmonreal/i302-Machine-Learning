import pandas as pd
import streamlit as st

# Configuración general
st.set_page_config(layout="wide")

# ========== Sección 1 ==========
st.title("🚗 Predicción de Precios de SUVs - Test Set")

# Cargar resultados de SUVs
results_df = pd.read_csv('/home/tmonreal/Desktop/i302-Machine-Learning/PF_SUV/test_results_suv.csv')
results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.2f}")
results_df['R²'] = results_df['R²'].apply(lambda x: f"{x:.2f}")

# Función para agregar emojis
def add_place_emojis(row):
    place = row.name + 1
    if place == 1:
        return "🥇 " + row['Alumnos']
    elif place == 2:
        return "🥈 " + row['Alumnos']
    elif place == 3:
        return "🥉 " + row['Alumnos']
    else:
        return row['Alumnos']

# Agregar emojis a la columna "Alumnos"
results_df['Alumnos'] = results_df.apply(add_place_emojis, axis=1)
results_df.index += 1

# Mostrar tabla
st.table(results_df.style.set_table_styles([{
    'selector': 'td',
    'props': [('font-size', '18px')]
}]))

# ========== Sección 2 ==========
st.markdown("---")
st.title("🚲 Modelado Bicicletas Públicas GCBA - Test Set")

# Mostrar imágenes comparativas por estación
st.markdown("### 📊 Comparación de R² por estación")
st.image("/home/tmonreal/Desktop/i302-Machine-Learning/PF_BIKE/r2_delta30min.png", caption="Delta 30 minutos", use_container_width=True)
st.image("/home/tmonreal/Desktop/i302-Machine-Learning/PF_BIKE/r2_delta60min.png", caption="Delta 60 minutos", use_container_width=True)
