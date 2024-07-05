import pandas as pd

# Load the original dataset
dataset_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/pf_suvs_test_i302_1s2024.csv' 
df = pd.read_csv(dataset_path)

# Remove 'Precio' and 'Moneda' columns
df = df.drop(columns=['Precio', 'Moneda'])

# Add an ID column
df.insert(0, 'id', range(1, len(df) + 1))  # Insert ID column at the beginning

# Save the transformed dataset to a new CSV file
output_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/pf_suvs_test_ids_i302_1s2024.csv' # Specify where you want to save the transformed test set
df.to_csv(output_path, index=False)

print(f"Transformed test set saved to {output_path}")
