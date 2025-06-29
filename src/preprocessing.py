import pandas as pd
from preprocessing_utils import min_levenshtein_distance_one_hot, extract_l_motor, clean_km_value, calcular_promedios_por_modelo, rellenar_nans_por_modelo
from knowns import KNOWN_BRANDS, KNOWN_MODELS, precio_dolar, KNOWN_CURRENCIES, KNOWN_SELLERS, KNOWN_COLORS, KNOWN_FUELS, KNOWN_TRANSMISSIONS



class Preprocessor:

    def __init__(self, raw_train_df: pd.DataFrame):
        """
        Initializes the Preprocessor with the raw training DataFrame.

        Parameters:
            raw_train_df (pd.DataFrame): The raw training DataFrame to be preprocessed.
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


    def marca(self, test_df:pd.DataFrame, known_brands: list[str]=KNOWN_BRANDS) -> pd.DataFrame: 
        """
        Cleans the 'Marca' column of the training DataFrame.
        Creates a new one-hot column for each known brand.

        Parameters:
            known_brands (list[str]): List of known brands to create one-hot columns.
        """
        min_levenshtein_distance_one_hot(test_df, 'Marca', known_brands)
        return test_df
    
    def modelo(self, test_df:pd.DataFrame,  known_models: list[str]=KNOWN_MODELS) -> pd.DataFrame:
        """
        Cleans the 'Modelo' column of the training DataFrame.
        Creates a new one-hot column for each known brand.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Modelo' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Modelo', known_models)
        return test_df
    
    def año(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the 'Año' column of the training DataFrame.
        Replaces NaN values and years greater than 2025 with the median year of the column in the training set.
        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Año' column.
        """
        median_year = self.train_df['Año'].median()
        test_df['Año'] = test_df['Año'].apply(
            lambda x: median_year if pd.isna(x) or x > 2025 else x
        )
        return test_df
    
    def version(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        return test_df
    
    def color(self, test_df:pd.DataFrame, known_colors: list[str]=KNOWN_COLORS) -> pd.DataFrame:
        """
        Cleans the 'Color' column of the training DataFrame.
        Creates a new one-hot column for each known color.

        Parameters:
            known_colors (list[str]): List of known colors to create one-hot columns.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Color' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Color', known_colors)
        return test_df
    
    def tipo_de_combustible(self, test_df:pd.DataFrame, known_fuels: list[str]=KNOWN_FUELS) -> pd.DataFrame:
        """
        Cleans the 'Tipo de combustible' column of the training DataFrame.
        Creates a new one-hot column for each known fuel type.

        Parameters:
            known_fuels (list[str]): List of known fuel types to create one-hot columns.
        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Tipo de combustible' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Tipo de combustible', known_fuels)
        return test_df
    
    def puertas(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the 'Puertas' column of the training DataFrame.
        Replaces values outside the interval [2, 10] with NaN and fills NaN values with the mean number of doors per model.
        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Puertas' column.
        """
        test_df['Puertas'] = test_df['Puertas'].apply(
            lambda x: x if 2 <= x <= 10 else pd.NA
        )
        mean_doors_per_model = self.train_df.groupby('Modelo')['Puertas'].mean()
        test_df['Puertas'] = test_df.apply(
            lambda row: mean_doors_per_model[row['Modelo']] if pd.isna(row['Puertas']) else row['Puertas'], axis=1
        )
        return test_df

    def transmision(self, test_df:pd.DataFrame, known_transmissions: list[str]=KNOWN_TRANSMISSIONS) -> pd.DataFrame:
        """
        Cleans the 'Transmisión' column of the training DataFrame.
        Creates a new one-hot column for each known transmission type.

        Parameters:
            known_transmissions (list[str]): List of known transmission types to create one-hot columns.
        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Transmisión' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Transmisión', known_transmissions)
        return test_df
    
    def motor(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        """
        Cleans the 'Motor' column of the training DataFrame.
        Extracts the engine size in liters from the 'Motor' column and creates a new column 'LitrosMotor'.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Motor' column and new 'LitrosMotor' column.
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

        test_df['LitrosMotor'].fillna(self.train_df['LitrosMotor'].mean(), inplace=True)
        test_df['LitrosMotor'] = test_df['LitrosMotor'].astype(float)

        return test_df
    
    def tipo_de_carroceria(self, test_df:pd.DataFrame) -> pd.DataFrame: 
        """
        Drop the 'Tipo de carrocería' column from the training DataFrame.
        Returns:
            pd.DataFrame: The DataFrame with 'Tipo de carrocería' column dropped.
        """
        test_df.drop(columns=['Tipo de carrocería'], inplace=True)
        return test_df
    
    def con_camara_de_retroceso(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the 'Con cámara de retroceso' column of the training DataFrame.
        Replaces NaN values with the mean value of the column grouped by 'Modelo'.
        If the mean is NaN, it fills with -1.
        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Con cámara de retroceso' column.
        """
        camara_retroceso_mean_per_model = calcular_promedios_por_modelo(self.train_df)
        rellenar_nans_por_modelo(test_df, camara_retroceso_mean_per_model, feature='Con cámara de retroceso')
        return test_df

    def kilometros(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the 'Kilómetros' column of the training DataFrame.
        Replaces NaN values with the median kilometers of the column in the training set.
        Converts the column to float type after cleaning.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Kilómetros' column.
        """
        test_df['Kilómetros'] = test_df['Kilómetros'].apply(clean_km_value)
        median_km = self.train_df['Kilómetros'].median()
        test_df['Kilómetros'].fillna(median_km, inplace=True)
        test_df['Kilómetros'] = test_df['Kilómetros'].astype(float)
        return test_df
    
    def titulo(self, test_df:pd.DataFrame) -> pd.DataFrame:
        return test_df
    
    def precio(self, test_df:pd.DataFrame) -> pd.DataFrame:
        mask = test_df['Moneda'] == '$'
        test_df.loc[mask, 'Precio'] = test_df.loc[mask, 'Precio'] / precio_dolar
        return test_df
    
    def moneda(self, test_df:pd.DataFrame, known_currencies:list[str]=KNOWN_CURRENCIES) -> pd.DataFrame:
        """
        Cleans the 'Moneda' column of the training DataFrame.
        Creates a new one-hot column for each known currency.

        Parameters:
            known_currencies (list[str]): List of known currencies to create one-hot columns.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Moneda' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Moneda', known_currencies)
        return test_df
    
    def descripcion(self, test_df:pd.DataFrame, drop:bool=True) -> pd.DataFrame:
        """
        Cleans the 'Descripción' column of the training DataFrame.
        Optionally drops the column.

        Parameters:
            drop (bool): Whether to drop the 'Descripción' column. Defaults to True.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Descripción' column.
        """
        test_df['Caracteres_descripcion'] = test_df['Descripción'].apply(lambda x: len(str(x)))
        if drop:
            test_df.drop(columns=['Descripción'], inplace=True)
        return test_df
    
    def tipo_de_vendedor(self, test_df:pd.DataFrame, known_sellers:list[str]=KNOWN_SELLERS) -> pd.DataFrame:
        """
        Cleans the 'Tipo de vendedor' column of the training DataFrame.
        Creates a new one-hot column for each known seller type.

        Parameters:
            known_sellers (list[str]): List of known seller types to create one-hot columns.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'Tipo de vendedor' column.
        """
        min_levenshtein_distance_one_hot(test_df, 'Tipo de vendedor', known_sellers)
        return test_df
    
    def preprocess(self, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        Applies all preprocessing steps to the test DataFrame.

        Parameters:
            test_df (pd.DataFrame): The DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        methods:list[function] = [
            self.marca,
            self.modelo,
            self.año,
            self.version,
            self.color,
            self.tipo_de_combustible,
            self.puertas,
            self.transmision,
            self.motor,
            self.tipo_de_carroceria,
            self.con_camara_de_retroceso,
            self.kilometros,
            self.titulo,
            self.precio,
            self.moneda,
            self.descripcion,
            self.tipo_de_vendedor
        ]
        
        for method in methods:
            test_df = method(test_df)

        return test_df