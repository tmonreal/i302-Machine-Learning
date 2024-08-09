import numpy as np
import pandas as pd

def train_val_split(df, train_size=0.7, random_state=None):
    """
    Divide el DataFrame en dos subconjuntos: entrenamiento y validación.
    
    Parámetros:
    - df: DataFrame de pandas que contiene los datos.
    - train_size: Proporción del conjunto de datos a utilizar para el entrenamiento (por defecto 0.7).
    - random_state: Semilla para la generación de números aleatorios (opcional).
    
    Devuelve:
    - train_df: DataFrame de pandas para entrenamiento.
    - val_df: DataFrame de pandas para validación.
    """
    
    # Asegurarse de que el tamaño del conjunto de entrenamiento sea válido
    if not (0 < train_size < 1):
        raise ValueError("train_size debe estar en el rango (0, 1).")
    
    # Fijar la semilla para reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)
    
    # Mezclar el DataFrame para asegurar una división aleatoria
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calcular el índice de división
    split_index = int(train_size * len(shuffled_df))
    
    # Dividir el DataFrame
    train_df = shuffled_df[:split_index]
    val_df = shuffled_df[split_index:]
    
    return train_df, val_df


def cross_val():
    return