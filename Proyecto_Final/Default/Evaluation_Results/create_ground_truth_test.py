import pandas as pd

# Define the input and output file paths
input_file = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Data/OFFICIAL/TEST_SET_TO_UPLOAD/finanzas_test+target.xlsx'
output_file = '/home/linar/Desktop/ML/Clases/i302/Proyecto_Final/Default/Evaluation_Results/ground_truth_test.csv'

# Load the data into a dataframe
df = pd.read_excel(input_file)

# Select only the 'id' and 'precio_pesos_constantes' columns
df_selected = df[['id', 'DEFAULT_SIGUIENTE_MES']]

# Save the selected columns to a new CSV file
df_selected.to_csv(output_file, index=False)

print(f"File saved successfully at {output_file}")