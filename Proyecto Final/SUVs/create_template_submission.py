import pandas as pd

# Definir la cantidad de IDs que quieres (en este caso, 8551 IDs)
num_ids = 8551

# Crear un DataFrame con IDs y precios predichos inicializados (por ejemplo, todos en cero)
template_df = pd.DataFrame({
    'id': range(1, num_ids + 1),
    'Predicted_Price_USD': [0.0] * num_ids  # Todos los precios predichos inicializados en cero
})

# Guardar el template CSV para predicciones de alumnos
template_path = '/home/linar/Desktop/ML/Clases/i302/Proyecto Final/SUVs/predictions_template.csv'  # Especifica dónde guardar el template CSV
template_df.to_csv(template_path, index=False)

print(f"Template CSV para predicciones de alumnos guardado en {template_path}")
