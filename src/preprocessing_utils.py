import pandas as pd
import numpy as np
from Levenshtein import distance as levenshtein_distance
import re

def min_levenshtein_distance_one_hot(df:pd.DataFrame, column_name:str, known_values:list[str]) -> pd.DataFrame:
    """
    Applies the minimum Levenshtein distance one-hot encoding to a specified column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        column_name (str): The name of the column to be processed.
        known_values (list[str]): List of known values for one-hot encoding.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    for value in known_values:
        df[value] = 0

    for i, input_value in enumerate(df[column_name]):
        dists = [levenshtein_distance(str(input_value).lower(), value.lower()) for value in known_values]
        idx_min = np.argmin(dists)
        matched_value = known_values[idx_min]
        df.at[i, matched_value] = 1

    df.drop(columns=[column_name], inplace=True)
    return df

def extract_l_motor(motor_str):
    """
    Extracts the engine size in liters from a string.

    Parameters:
        motor_str (str): The string containing the engine size.

    Returns:
        float: The extracted engine size in liters, or NaN if not found.
    """
    if pd.isna(motor_str):
        return np.nan
    match = re.search(r'(\d+[.,]?\d*)', motor_str)
    if match:
        value = float(match.group(1).replace(',', '.'))
        if 0 < value < 7:
            return value
    return np.nan

def clean_km_value(value):
    """
    Cleans a single value of kilometers.

    Parameters:
        value (str or float): The value to be cleaned.

    Returns:
        float: The cleaned value in kilometers, or NaN if not valid.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace('.', '').replace(',', '.')
    try:
        km = float(value)
        if 0 < km < 1e6:  # Assuming a reasonable range for kilometers
            return km
    except ValueError:
        pass
    return np.nan

def calcular_promedios_por_modelo(df:pd.DataFrame):
    df_temp = df.copy()
    df_temp["Con cámara de retroceso"] = df_temp["Con cámara de retroceso"].map({"Sí": 1, "No": 0})
    promedios = df_temp.groupby("Modelo")["Con cámara de retroceso"].mean()
    promedios = promedios.fillna(-1)
    return promedios.to_dict()

def rellenar_nans_por_modelo(df, promedios, feature='Con cámara de retroceso', default=-1):
    df[feature] = df[feature].map({"Sí": 1, "No": 0})
    mask_nans = df[feature].isna()
    modelos_nan = df.loc[mask_nans, "Modelo"]
    valores_rellenar = modelos_nan.map(promedios).fillna(default)
    df.loc[mask_nans, feature] = valores_rellenar