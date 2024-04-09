import numpy as np
import pandas as pd
from itertools import product

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
    
    def transform(self, X):
        if self.mean_ is None or self.var_ is None:
            raise ValueError("Scaler not fitted. Call fit() before transform().")
        
        return (X - self.mean_) / np.sqrt(self.var_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
def one_hot_encode(df):
    """
    Perform one-hot encoding on categorical columns of a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing categorical columns to encode.

    Returns:
        pandas.DataFrame: DataFrame with categorical columns replaced by one-hot encoded columns.
    """
    # Identify categorical columns
    categorical_columns = df.columns.tolist()
    
    # Perform one-hot encoding on each categorical column
    for col in categorical_columns:
        one_hot_encoded = pd.get_dummies(df[col], prefix=col).astype(int)
        df = pd.concat([df, one_hot_encoded], axis=1)
        df.drop(col, axis=1, inplace=True)
    
    return df
  
def train_val_split(X, y, val_size=0.2, random_state=None):
  """
  Shuffle and split data into training and testing sets.

  Parameters:
      X: DataFrame of features.
      y: Series of target variable.
      val_size: Proportion of data for the val set (0 to 1).
      random_state: Seed for shuffling (optional).

  Returns:
      X_train, X_val, y_train, y_val: Split data.
  """

  if random_state:
    np.random.seed(random_state)
  shuffled_df = X.sample(frac=1, random_state=random_state)  

  split_index = int(len(shuffled_df) * (1 - val_size))
  X_train, X_val = shuffled_df.iloc[:split_index], shuffled_df.iloc[split_index:]
  y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

  return X_train, X_val, y_train, y_val

def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=None):
    """
    Shuffle and split data into training, validation, and testing sets.

    Parameters:
        X: DataFrame of features.
        y: Series of target variable.
        val_size: Proportion of data for the validation set (0 to 1).
        test_size: Proportion of data for the test set (0 to 1).
        random_state: Seed for shuffling (optional).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split data.
    """

    if random_state:
        np.random.seed(random_state)
    shuffled_df = X.sample(frac=1, random_state=random_state)

    val_test_index = int(len(shuffled_df) * (1 - val_size - test_size))
    test_index = int(len(shuffled_df) * (1 - test_size))

    X_train = shuffled_df.iloc[:val_test_index]
    X_val = shuffled_df.iloc[val_test_index:test_index]
    X_test = shuffled_df.iloc[test_index:]

    y_train = y.iloc[:val_test_index]
    y_val = y.iloc[val_test_index:test_index]
    y_test = y.iloc[test_index:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_test(x, y, train_ratio, val_ratio, test_ratio):
    # Calculate the number of samples 
    train_samples = int(len(x) * train_ratio)
    val_samples = int(len(x) * val_ratio)
    test_samples = int(len(x) * test_ratio)
    
    # Shuffle the indices
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    # Split the indices into training, validation, and test indices
    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:train_samples + val_samples]
    test_indices = indices[train_samples + val_samples:train_samples + val_samples + test_samples]
    return train_indices, val_indices, test_indices