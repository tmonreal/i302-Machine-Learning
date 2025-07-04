import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
predictions_folder = '/home/tmonreal/Desktop/i302-Machine-Learning/PF_SUV/Students_Predictions'
output_path = '/home/tmonreal/Desktop/i302-Machine-Learning/PF_SUV/test_results_suv.csv'

# Load the original dataset
dataset_path = '/home/tmonreal/Desktop/i302-Machine-Learning/PF_SUV/SUVS_2025-test.csv' #mezcla de pesos y dolares
df_original = pd.read_csv(dataset_path)

# Initialize results list
results_list = []

# Iterate through prediction files
for filename in os.listdir(predictions_folder):
    if filename.endswith('.csv') and "USD" in filename:
        predictions_path = os.path.join(predictions_folder, filename)
        predictions = pd.read_csv(predictions_path)
        
        # Extract conversion rate, model name, and student names from filename
        # filename_parts: e.g. ['Arbelaiz', 'Ostrovsky', 'XGBoost', 'USD1280']
        filename_parts = filename.split('.')[0].split('_')

        # Detectar automáticamente el índice de la parte que contiene "USD"
        usd_index = next(i for i, part in enumerate(filename_parts) if part.startswith("USD"))

        # Extraer valores con índices relativos
        conversion_rate_usd = int(filename_parts[usd_index][3:])
        model_name = filename_parts[usd_index - 1]
        student_names = ' '.join(filename_parts[:usd_index - 1])
        
        # Convert prices from ARS to USD based on the student's conversion rate
        df_student_ground_truth = df_original.copy()
        df_student_ground_truth['Real_Price_USD'] = df_student_ground_truth.apply(
            lambda row: round(row['Precio'] / conversion_rate_usd, 1) if row['Moneda'] == '$' else row['Precio'], axis=1
        )
        
        # Create a ground truth DataFrame for the student
        ids = range(0, len(df_student_ground_truth)) # Assuming IDs start from 0 and increment by 1
        student_ground_truth_df = pd.DataFrame({'id': ids, 'Real_Price_USD': df_student_ground_truth['Real_Price_USD']})
        
        # Merge predictions with the student's ground truth
        merged = pd.merge(predictions, student_ground_truth_df, on='id')
        
        # Check for NaN values
        if merged[['Real_Price_USD', 'Predicted_Price_USD']].isnull().any().any():
            print(f"File {filename} contains NaN values in the columns 'Real_Price_USD' or 'Predicted_Price_USD'.")
            merged = merged.dropna(subset=['Real_Price_USD', 'Predicted_Price_USD'])
        
        # Calculate evaluation metrics
        rmse = round(np.sqrt(mean_squared_error(merged['Real_Price_USD'], merged['Predicted_Price_USD'])),2)
        r2 = round(r2_score(merged['Real_Price_USD'], merged['Predicted_Price_USD']),3)
        
        # Append results to the list
        results_list.append({'Modelo': model_name, 'Alumnos': student_names, 'RMSE': rmse, 'R²': r2})

# Save results to CSV
results_df = pd.DataFrame(results_list).sort_values(by='RMSE', ascending=True)
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")