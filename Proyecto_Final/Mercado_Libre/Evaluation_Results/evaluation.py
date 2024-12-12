import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
predictions_folder = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Mercado_Libre/Student_Predictions'
ground_truth_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Mercado_Libre/Evaluation_Results/ground_truth_test.csv'
output_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Mercado_Libre/Evaluation_Results/test_results.csv'

ground_truth = pd.read_csv(ground_truth_path)
results_list = []

for filename in os.listdir(predictions_folder):
    if filename.endswith('.csv'):
        predictions_path = os.path.join(predictions_folder, filename)
        predictions = pd.read_csv(predictions_path)
        
        merged = pd.merge(predictions, ground_truth, on='id')
        
        if merged[['precio_pesos_constantes', 'predicted_price']].isnull().any().any():
            print(f"File {filename} contains NaN values in the columns 'precio_pesos_constantes' or 'predicted_price'.")
            merged = merged.dropna(subset=['precio_pesos_constantes', 'predicted_price'])
        
        rmse = np.sqrt(mean_squared_error(merged['precio_pesos_constantes'], merged['predicted_price']))
        r2 = r2_score(merged['precio_pesos_constantes'], merged['predicted_price'])
        
        # Extract model name and student names from filename
        filename_parts = filename.split('.')[0].split('_')
        model_name = filename_parts[-2]
        student_names = ' '.join(filename_parts[:-2])
        
        # Append to results list
        results_list.append({'Modelo': model_name, 'Alumnos': student_names, 'RMSE': rmse, 'R2': r2})

results_df = pd.DataFrame(results_list).sort_values(by='RMSE', ascending=True)
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")