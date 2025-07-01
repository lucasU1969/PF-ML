import pandas as pd
from src.preprocessing_utils import min_levenshtein_distance_one_hot, extract_l_motor, clean_km_value, calcular_promedios_por_modelo, rellenar_nans_por_modelo
from src.knowns import KNOWN_BRANDS, KNOWN_MODELS, precio_dolar, KNOWN_CURRENCIES, KNOWN_SELLERS, KNOWN_COLORS, KNOWN_FUELS, KNOWN_TRANSMISSIONS
import numpy as np


"""
train, val, test = data.split()

prec = Preprocessor(train)

train = prec.preprocess(train)
val = prec.preprocess(val)
test = prec.preprocess(test)

model.train(train)
e_tpo_preds = model.predict(val)

preds = ln(e_tpo_preds)
"""

class Preprocessor:

    def __init__(self, raw_train_df: pd.DataFrame):
        """
        """
        self.train_df = raw_train_df.drop(columns=['Unnamed: 0']).copy()

        self.train_df['LitrosMotor'] = self.train_df['Motor'].apply(extract_l_motor)
        self.train_df['LitrosMotor_extraido_de_Titulo'] = 0
        self.train_df['LitrosMotor_extraido_de_Version'] = 0
        filter_nulls = self.train_df['LitrosMotor'].isna()
        liters_from_title = self.train_df.loc[filter_nulls, 'Título'].apply(extract_l_motor)
        indices_title = liters_from_title[liters_from_title.notna()].index
        self.train_df.loc[indices_title, 'LitrosMotor'] = liters_from_title.loc[indices_title]
        self.train_df.loc[indices_title, 'LitrosMotor_extraido_de_Titulo'] = 1

        self.train_df['Kilómetros'] = self.train_df['Kilómetros'].apply(clean_km_value)

    def unnamed(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        if 'Unnamed: 0' in test_df.columns:
            test_df.drop(columns=['Unnamed: 0'], inplace=True)
        return test_df


    def marca(self, test_df:pd.DataFrame, known_brands: list[str]=KNOWN_BRANDS) -> pd.DataFrame: 
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Marca', known_brands)
        return test_df
    
    def modelo(self, test_df:pd.DataFrame,  known_models: list[str]=KNOWN_MODELS) -> pd.DataFrame:
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Modelo', known_models)
        return test_df
    
    def año(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        median_year = self.train_df['Año'].median()
        test_df['Año'] = test_df['Año'].apply(
            lambda x: median_year if pd.isna(x) or x > 2025 else x
        )
        return test_df
    
    def version(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        """
        """
        if 'Versión' in test_df.columns: 
            test_df.drop(columns=['Versión'], inplace=True)
        return test_df
    
    def color(self, test_df:pd.DataFrame, known_colors: list[str]=KNOWN_COLORS) -> pd.DataFrame:
        """
        """
        test_df['Color'] = test_df['Color'].apply(lambda x: x if isinstance(x, str) else 'Otro')
        test_df = min_levenshtein_distance_one_hot(test_df, 'Color', known_colors)
        return test_df
    
    def tipo_de_combustible(self, test_df:pd.DataFrame, known_fuels: list[str]=KNOWN_FUELS) -> pd.DataFrame:
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Tipo de combustible', known_fuels)
        return test_df
    
    def puertas(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        test_df['Puertas'] = test_df['Puertas'].apply(
            lambda x: float(x) if 2 <= x <= 10 else np.nan # Use np.nan instead of pd.NA
        )
        test_df['Puertas'] = test_df['Puertas'].astype(float) 
        median_puertas = self.train_df['Puertas'].median()
        test_df['Puertas'] = test_df['Puertas'].fillna(median_puertas).infer_objects(copy=False)
        return test_df

    def transmision(self, test_df:pd.DataFrame, known_transmissions: list[str]=KNOWN_TRANSMISSIONS) -> pd.DataFrame:
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Transmisión', known_transmissions)
        return test_df
    
    def motor(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        """
        """
        test_df['LitrosMotor'] = test_df['Motor'].apply(extract_l_motor)

        test_df['LitrosMotor_extraido_de_Titulo'] = 0
        test_df['LitrosMotor_extraido_de_Version'] = 0

        filter_nulls = test_df['LitrosMotor'].isna()
        liters_from_title = test_df.loc[filter_nulls, 'Título'].apply(extract_l_motor)
        indices_title = liters_from_title[liters_from_title.notna()].index
        test_df.loc[indices_title, 'LitrosMotor'] = liters_from_title.loc[indices_title]
        test_df.loc[indices_title, 'LitrosMotor_extraido_de_Titulo'] = 1  

        filter_nulls = test_df['LitrosMotor'].isna()
        liters_from_version = test_df.loc[filter_nulls, 'Versión'].apply(extract_l_motor)
        indices_version = liters_from_version[liters_from_version.notna()].index
        test_df.loc[indices_version, 'LitrosMotor'] = liters_from_version
        test_df.loc[indices_version, 'LitrosMotor_extraido_de_Version'] = 1

        test_df['LitrosMotor'] = test_df['LitrosMotor'].fillna(self.train_df['LitrosMotor'].mean())
        test_df['LitrosMotor'] = test_df['LitrosMotor'].astype(float)

        test_df.drop(columns=['Motor'], inplace=True, errors='ignore')

        return test_df
    
    def tipo_de_carroceria(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        """
        """
        test_df.drop(columns=['Tipo de carrocería'], inplace=True)
        return test_df
    
    def con_camara_de_retroceso(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        camara_retroceso_mean_per_model = calcular_promedios_por_modelo(self.train_df)
        rellenar_nans_por_modelo(test_df, camara_retroceso_mean_per_model, feature='Con cámara de retroceso')
        return test_df

    def kilometros(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        test_df['Kilómetros'] = test_df['Kilómetros'].apply(clean_km_value)
        median_km = self.train_df['Kilómetros'].median()
        test_df['Kilómetros'] = test_df['Kilómetros'].fillna(median_km)
        test_df['Kilómetros'] = test_df['Kilómetros'].astype(float)
        return test_df
    
    def titulo(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        if 'Título' in test_df.columns:
            test_df.drop(columns=['Título'], inplace=True)
        return test_df
    
    def precio(self, test_df:pd.DataFrame) -> pd.DataFrame:
        mask = test_df['Moneda'] == '$'
        test_df.loc[mask, 'Precio'] = test_df.loc[mask, 'Precio'] / precio_dolar
        return test_df
    
    def moneda(self, test_df:pd.DataFrame, known_currencies:list[str]=KNOWN_CURRENCIES) -> pd.DataFrame:
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Moneda', known_currencies)
        return test_df
    
    def descripcion(self, test_df:pd.DataFrame, drop:bool=True) -> pd.DataFrame:
        """
        """
        test_df['Caracteres_descripcion'] = test_df['Descripción'].apply(lambda x: len(str(x)))
        if drop:
            test_df.drop(columns=['Descripción'], inplace=True)
        return test_df
    
    def tipo_de_vendedor(self, test_df:pd.DataFrame, known_sellers:list[str]=KNOWN_SELLERS) -> pd.DataFrame:
        """
        """
        test_df = min_levenshtein_distance_one_hot(test_df, 'Tipo de vendedor', known_sellers)
        return test_df
    
    def preprocess(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        """
        methods:list[function] = [
            self.descripcion,
            self.precio,
            self.color,
            self.kilometros,
            self.marca,
            self.puertas,
            self.con_camara_de_retroceso,
            self.modelo,
            self.tipo_de_combustible,
            self.transmision,
            self.tipo_de_vendedor,
            self.moneda,
            self.motor,
            self.tipo_de_carroceria,
            self.año,
            self.version,
            self.titulo,
        ]

        for method in methods:
            test_df = method(test_df)

        return test_df