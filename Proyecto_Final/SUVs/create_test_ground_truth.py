import pandas as pd

# Load the original dataset
dataset_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/pf_suvs_test_i302_1s2024.csv' 
df_original = pd.read_csv(dataset_path)

# Define conversion rate from ARS to USD (as of June 26, 2024)
# dolar mayorista segun https://www.cronista.com/MercadosOnline/moneda.html?id=ARSIB
conversion_rate_ars_to_usd = 914  # 914 ARS = 1 USD

# Convert prices from ARS to USD based on 'Moneda' column
df_original['Real_Price_USD'] = df_original.apply(lambda row: round(row['Precio'] / conversion_rate_ars_to_usd, 1) if row['Moneda'] == '$' else row['Precio'], axis=1)

# Extract IDs and Real Prices (in USD)
ids = range(1, len(df_original) + 1)  # Assuming IDs start from 1 and increment by 1
real_prices_usd = df_original['Real_Price_USD']  # Real prices converted to USD

# Create a DataFrame for ground truth test set
ground_truth_df = pd.DataFrame({'id': ids, 'Real_Price_USD': real_prices_usd})

# Save the ground truth test set to a new CSV file
ground_truth_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/ground_truth_test.csv'  # Specify where to save the ground truth test set
ground_truth_df.to_csv(ground_truth_path, index=False)

print(f"Ground truth test set saved to {ground_truth_path}")