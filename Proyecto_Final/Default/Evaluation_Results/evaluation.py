import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define paths
predictions_folder = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Student_Prediction'
ground_truth_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Evaluation_Results/ground_truth_test.csv'
output_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Evaluation_Results/test_results.csv'

ground_truth = pd.read_csv(ground_truth_path)
results_list = []

for filename in os.listdir(predictions_folder):
    if filename.endswith('.csv'):
        predictions_path = os.path.join(predictions_folder, filename)
        predictions = pd.read_csv(predictions_path)
        
        # Aseguramos que ambas columnas 'id' sean del mismo tipo de datos
        ground_truth['id'] = ground_truth['id'].astype(str)  # Cambiar a tipo string
        predictions['id'] = predictions['id'].astype(str)  # Cambiar a tipo string
        
        # Merging the predictions with the ground truth
        merged = pd.merge(predictions, ground_truth, on='id')
        
        if merged[['DEFAULT_SIGUIENTE_MES', 'predicted_default']].isnull().any().any():
            print(f"File {filename} contains NaN values in the columns 'DEFAULT_SIGUIENTE_MES' or 'predicted_default'.")
            merged = merged.dropna(subset=['DEFAULT_SIGUIENTE_MES', 'predicted_default'])
        
        # Classification metrics
        y_true = merged['DEFAULT_SIGUIENTE_MES']
        y_pred = merged['predicted_default']
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)
        
        # Extract model name and student names from filename
        filename_parts = filename.split('.')[0].split('_')
        model_name = filename_parts[-2]
        student_names = ' '.join(filename_parts[:-2])
        
        # Append the results to the list
        results_list.append({
            'Modelo': model_name,
            'Alumnos': student_names,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-ROC': auc_roc
        })

# Creating the results dataframe and sorting by F1 score or another metric
results_df = pd.DataFrame(results_list).sort_values(by='F1 Score', ascending=False)

# Saving the results to a CSV
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")