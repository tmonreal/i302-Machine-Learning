import pandas as pd
import streamlit as st
#import plotly.express as px

# Load the results
results_df = pd.read_csv('Proyecto_Final/Evaluation_Results/test_results_fair.csv')
results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.2f}")  # Format RMSE to 2 decimal places
results_df['R²'] = results_df['R²'].apply(lambda x: f"{x:.2f}")  # Format R2 to 2 decimal places

# Streamlit App
st.set_page_config(layout="wide") 
st.title("Predicción de Precios de SUVs - Test Set")

# Add emojis and adjust index
def add_place_emojis(row):
    place = row.name + 1
    if place == 1:
        return "🥇 " + row['Alumnos']
    elif place == 2:
        return "🥈 " + row['Alumnos']
    elif place == 3:
        return "🥉 " + row['Alumnos']
    else:
        return row['Alumnos']  # Return 'Alumnos' without appending place number

# Apply the function to add emojis and adjust index
results_df['Alumnos'] = results_df.apply(add_place_emojis, axis=1)

# Display the leaderboard with index starting from 1
results_df.index += 1  # Adjust index to start from 1
#st.table(results_df)
st.table(results_df.style.set_table_styles([{
    'selector': 'td',
    'props': [('font-size', '18px')]
}]))
