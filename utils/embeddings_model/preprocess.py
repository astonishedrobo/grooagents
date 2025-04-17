import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def load_and_preprocess_data(data_path, continuous_cols, categorical_cols=None, scaler_type="standard"):
    """
    Loads data from a CSV file, preprocesses it, and returns it as a pandas DataFrame.

    Args:
        data_path (str): Path to the CSV data file.
        continuous_cols (list): List of names of continuous feature columns.
        categorical_cols (list, optional): List of names of categorical feature columns.
                                          Defaults to None.
        scaler_type (str, optional): Type of scaler to use ('standard' or 'minmax').
                                     Defaults to "standard".

    Returns:
        pd.DataFrame: Preprocessed pandas DataFrame.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print()
    if continuous_cols:
    #   print(f"Continuous Columns: {continuous_cols}")
      # Check for any non-numeric values
      for col in continuous_cols:
          if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: column '{col}' is not numeric.")
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce to nan so they will be removed.
          print(f"Checking for NaN values in column '{col}': {df[col].isnull().sum()}")

      df = df.dropna(subset=continuous_cols)

      # Scale continuous features
      if scaler_type == "standard":
          scaler = StandardScaler()
      elif scaler_type == "minmax":
          scaler = MinMaxScaler()
      else:
          raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

      print(f"Scaling continuous columns: {continuous_cols}")
      df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
      print("First 5 rows after scaling:")
      print(df[continuous_cols].head())


    if categorical_cols:
        print(f"Categorical Columns: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols)

    return df