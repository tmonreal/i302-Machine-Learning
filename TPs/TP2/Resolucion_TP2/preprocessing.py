import numpy as np
import pandas as pd

def one_hot_encoder(df, column):
    """
    Codifica una columna categórica de un DataFrame en formato one-hot.

    Args:
    df (pandas.DataFrame): DataFrame que contiene la columna categórica a codificar.
    column (str): Nombre de la columna a codificar.

    Returns:
    pandas.DataFrame: DataFrame con la columna codificada en one-hot.
    """
    # Extraer la columna categórica como un array de numpy
    data = df[column].values
    
    # Identificar las categorías únicas
    categories = np.unique(data)

    # Crear una matriz de ceros de tamaño (n_samples, n_categories)
    one_hot_matrix = np.zeros((data.shape[0], len(categories)))

    # Llenar la matriz con 1s en la posición correspondiente a la categoría
    for i, category in enumerate(categories):
        one_hot_matrix[data == category, i] = 1

    # Crear un DataFrame con la matriz one-hot
    one_hot_df = pd.DataFrame(one_hot_matrix, columns=[f"{column}_{cat}" for cat in categories])
    
    # Concatenar el DataFrame original con el DataFrame one-hot
    df_encoded = pd.concat([df.drop(column, axis=1), one_hot_df], axis=1)
    
    return df_encoded