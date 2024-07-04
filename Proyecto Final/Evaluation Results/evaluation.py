import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
predictions_folder = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/Student´s Predictions'
ground_truth_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/Evaluation Results/ground_truth_test.csv'
output_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/Evaluation Results/test_results.csv'

ground_truth = pd.read_csv(ground_truth_path)
results_list = []

for filename in os.listdir(predictions_folder):
    if filename.endswith('.csv'):
        predictions_path = os.path.join(predictions_folder, filename)
        predictions = pd.read_csv(predictions_path)
        
        merged = pd.merge(predictions, ground_truth, on='id')
        
        if merged[['Real_Price_USD', 'Predicted_Price_USD']].isnull().any().any():
            print(f"File {filename} contains NaN values in the columns 'Real_Price_USD' or 'Predicted_Price_USD'.")
            merged = merged.dropna(subset=['Real_Price_USD', 'Predicted_Price_USD'])
        
        rmse = np.sqrt(mean_squared_error(merged['Real_Price_USD'], merged['Predicted_Price_USD']))
        r2 = r2_score(merged['Real_Price_USD'], merged['Predicted_Price_USD'])
        
        # Extract model name and student names from filename
        filename_parts = filename.split('.')[0].split('_')
        model_name = filename_parts[1]
        student_names = ' '.join(filename_parts[2:])
        
        # Append to results list
        results_list.append({'Modelo': model_name, 'Alumnos': student_names, 'RMSE': rmse, 'R2': r2})

results_df = pd.DataFrame(results_list).sort_values(by='RMSE', ascending=True)
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")