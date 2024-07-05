import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
predictions_folder = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/Student´s Predictions'
output_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/Evaluation Results/test_results_fair.csv'

# Load the original dataset
dataset_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/pf_suvs_test_i302_1s2024.csv'
df_original = pd.read_csv(dataset_path)

# Initialize results list
results_list = []

# Iterate through prediction files
for filename in os.listdir(predictions_folder):
    if filename.endswith('.csv') and "USD" in filename:
        predictions_path = os.path.join(predictions_folder, filename)
        predictions = pd.read_csv(predictions_path)
        
        # Extract conversion rate, model name, and student names from filename
        filename_parts = filename.split('.')[0].split('_')
        conversion_rate_usd = int(filename_parts[2][3:])
        model_name = filename_parts[1]
        student_names = ' '.join(filename_parts[3:])
        
        # Convert prices from ARS to USD based on the student's conversion rate
        df_student_ground_truth = df_original.copy()
        df_student_ground_truth['Real_Price_USD'] = df_student_ground_truth.apply(
            lambda row: round(row['Precio'] / conversion_rate_usd, 1) if row['Moneda'] == '$' else row['Precio'], axis=1
        )
        
        # Create a ground truth DataFrame for the student
        ids = range(1, len(df_student_ground_truth) + 1)  # Assuming IDs start from 1 and increment by 1
        student_ground_truth_df = pd.DataFrame({'id': ids, 'Real_Price_USD': df_student_ground_truth['Real_Price_USD']})
        
        # Merge predictions with the student's ground truth
        merged = pd.merge(predictions, student_ground_truth_df, on='id')
        
        # Check for NaN values
        if merged[['Real_Price_USD', 'Predicted_Price_USD']].isnull().any().any():
            print(f"File {filename} contains NaN values in the columns 'Real_Price_USD' or 'Predicted_Price_USD'.")
            merged = merged.dropna(subset=['Real_Price_USD', 'Predicted_Price_USD'])
        
        # Calculate evaluation metrics
        rmse = round(np.sqrt(mean_squared_error(merged['Real_Price_USD'], merged['Predicted_Price_USD'])),2)
        r2 = round(r2_score(merged['Real_Price_USD'], merged['Predicted_Price_USD']),2)
        
        # Append results to the list
        results_list.append({'Modelo': model_name, 'Alumnos': student_names, 'RMSE': rmse, 'R2': r2})

# Save results to CSV
results_df = pd.DataFrame(results_list).sort_values(by='RMSE', ascending=True)
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")