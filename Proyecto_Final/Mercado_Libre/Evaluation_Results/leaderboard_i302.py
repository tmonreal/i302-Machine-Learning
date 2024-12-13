import pandas as pd
import streamlit as st
#import plotly.express as px

# Load the results
results_df = pd.read_csv('/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Mercado_Libre/Evaluation_Results/test_results.csv')
results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.2f}")  # Format RMSE to 2 decimal places
results_df['R²'] = results_df['R²'].apply(lambda x: f"{x:.2f}")  # Format R2 to 2 decimal places

# Streamlit App
st.set_page_config(layout="wide") 
st.title("Predicción de Precios de Alquiler en el AMBA - Test Set")

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
    'props': [('font-size', '17px')]
}]))

results2_df = pd.read_csv('/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Evaluation_Results/test_results.csv')
#Accuracy,Precision,Recall,F1 Score,AUC-ROC
results2_df['Accuracy'] = results2_df['Accuracy'].apply(lambda x: f"{x:.2f}")  # Format RMSE to 2 decimal places
results2_df['Precision'] = results2_df['Precision'].apply(lambda x: f"{x:.2f}")  # Format R2 to 2 decimal places
results2_df['Recall'] = results2_df['Recall'].apply(lambda x: f"{x:.2f}")  # Format RMSE to 2 decimal places
results2_df['F1 Score'] = results2_df['F1 Score'].apply(lambda x: f"{x:.2f}")  # Format RMSE to 2 decimal places
results2_df['AUC-ROC'] = results2_df['AUC-ROC'].apply(lambda x: f"{x:.2f}")  # Format R2 to 2 decimal places

st.title("Predicción de Riesgos de Default en Créditos Personales - Test Set")
st.table(results2_df.style.set_table_styles([{
    'selector': 'td',
    'props': [('font-size', '17px')]
}]))