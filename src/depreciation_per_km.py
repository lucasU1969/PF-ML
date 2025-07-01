import pandas as pd


def estimate_depreciation_per_km(samples:pd.DataFrame, km:float, model) -> pd.Series: 
    """
    Estimates the depreciation per kilometer for a given set of samples using a trained model.

    Parameters:
        samples (pd.DataFrame): DataFrame containing the samples for which to estimate depreciation.
        km (float): The number of kilometers to use for the depreciation calculation.
        model: A trained regression model that can predict prices based on the samples.

    Returns:
        pd.Series: A Series containing the estimated depreciation per kilometer for each sample.
    """
    price_0 = model.predict(samples)
    samples_plus_km = samples.copy()
    samples_plus_km['Kilómetros'] += km
    price_km = model.predict(samples_plus_km)
    depreciation = (price_km - price_0) / km
    depreciation = pd.Series(depreciation, index=samples.index)
    return depreciation