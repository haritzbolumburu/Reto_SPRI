import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_regression, f_classif, SelectKBest, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,  confusion_matrix, r2_score, mean_squared_error


RANDOM_STATE = 42


def leer_excel(ruta:str, nombre_archivo:str) -> pd.DataFrame:
    """Función que lee un archivo excel y devuelve un dataframe de pandas.
    Args:
        path (str): carpeta donde se encuentra el archivo excel a leer
        nombre_archivo (str): nombre del archivo excel a leer, incluyendo la extensión
    """
    df = pd.read_excel(os.path.join(ruta, nombre_archivo))
    return df

def miss_min(df:pd.DataFrame, columna:str) -> None:
    """Función que devuelve el número de missings de una columna de un dataframe.
    Args:
        df (DataFrame): dataframe que incluya la columna
        columna (str): nombre de la columna
    """
    print(columna)
    print(f"shape: {df[df[columna].isna()].shape}")
    try:
      print(f"min: {df[df[columna].notna()][columna].min()}")
    except: print("variable no numerica")
    return None


# OUTLIERS

def Q1(x) -> float:
    """Función que devuelve el primer cuartil de una serie de pandas.
    Args:
        x (Series): serie de pandas
    Returns:
        float: primer cuartil
    """
    return x.quantile(0.25)

def Q3(x) -> float:
    """Función que devuelve el tercer cuartil de una serie de pandas.
    Args:
        x (Series): serie de pandas
    Returns:
        float: tercer cuartil
    """
    return x.quantile(0.75)

def IQR(x) -> float:
    """Función que devuelve el rango intercuartílico de una serie de pandas.
    Args:
        x (Series): serie de pandas
    Returns:
        float: rango intercuartílico
    """
    return x.quantile(0.75)-x.quantile(0.25)


def saber_empresa_outliersMAX(df:pd.DataFrame, columnas:list) -> None:
    """Función que devuelve las 6 primeras instancias de un dataframe ordenado de mayor a menor según las diferentes columnas 
    introducidas junto con la variable 'Nombre_sabi' y 'año'. 
    Args:
        df (DataFrame): dataframe que incluya todas las variables
        columnas (list): lista de columnas
    """
    for var in columnas:
        print(df.sort_values(by = var, ascending =  False)[['Nombre_sabi',var,'year']][:6])
    return None

def saber_empresa_outliersMIN(df:pd.DataFrame, columnas:list) -> None:
    """Función que devuelve las 6 primeras instancias de un dataframe ordenado de menor a mayor según las diferentes columnas 
    introducidas junto con la variable 'Nombre_sabi' y 'año'. 
    Args:
        df (DataFrame): dataframe que incluya todas las variables
        columnas (list): lista de columnas
    """
    for var in columnas:
        print(df.sort_values(by = var, ascending =  True)[['Nombre_sabi',var,'year']][:6])
    return None



# IMPUTACIÓN MEDIANTE MODELOS

def mejor_imputador_clasif(variable, dftot, vars_miss_obj) -> object:
    """Función que devuelve el mejor modelo para imputar una variable categórica mediante un modelo de clasificación.
    Args:
        variable (str): variable a imputar
        dftot (DataFrame): dataframe que incluya la variable a imputar y las variables predictoras
        vars_miss_obj (list): lista de variables categóricas con missings
    Returns:
        modelo: mejor modelo para imputar la variable
    """
    # Prepare the data
    df_pr = dftot.drop(columns='valuation_2022')

    # Como es solo para evaluar y no para imputar, se quitan las instancias con missings para ver qué tal imputa las que sí sabemos
    df_pr = df_pr.dropna()
    
    X = df_pr.drop(columns=vars_miss_obj, axis=1)
    X = X.select_dtypes(include=['float64', 'int64'])
    y = df_pr[variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the models
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    # Fit the models on the training data
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # Evaluate the models on the testing data
    rf_score = rf_model.score(X_test, y_test)
    gb_score = gb_model.score(X_test, y_test)

    # Print the average scores and their standard deviations
    print(f"RandomForestClassifier para la variable {variable} tiene un accuracy de {rf_score:.4f}")
    print(f"GradientBoostingClassifier para la variable {variable} tiene un accuracy de {gb_score:.4f}")

    # Select the best model
    if rf_score >= gb_score:
        mejor_modelo = rf_model
    else:
        mejor_modelo = gb_model

    return mejor_modelo


# CLASIFICACIÓN

def modelo_base_clasif(features, dftot) -> float:
    """Función que devuelve el accuracy de un modelo de clasificación con las variables introducidas.
    Args:
        features (list): lista de variables predictoras
        dftot (DataFrame): dataframe que incluya las variables predictoras y la variable objetivo
    Returns:
        float: accuracy del modelo
    """
    X = dftot[features]
    try:
        X = X.drop(columns=['valuation_2022'])
    except:
        pass
    y = dftot['Porcentaje_adquisicion_cat']
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Este método tiene un accuracy de {accuracy.round(4)}')
    return accuracy

# Mejor modelo en base a accuracy
def mejor_modelo_clasif(features, new_feat, new_features, corr_features, dftot) -> list:
    """Función que devuelve el mejor modelo de clasificación en base a su accuracy.
    Args:
        features (list): lista de variables predictoras mediante el método 1 de selección de variables
        new_feat (list): lista de variables predictoras mediante el método 2 de selección de variables
        new_features (list): lista de variables predictoras mediante el método 3 de selección de variables
        corr_features (list): lista de variables predictoras mediante el método 4 de selección de variables
        dftot (DataFrame): dataframe que incluye todas las variables predictoras y la variable objetivo
    Returns:
        list: lista de variables predictoras del mejor modelo
    """
    modelos = [features, new_feat, new_features, corr_features, dftot.columns]
    accuracy = []
    for modelo in modelos:
        accuracy.append(modelo_base_clasif(modelo, dftot))
    mejor_modelo = modelos[np.argmax(accuracy)]
    print(f'El mejor método es el {np.argmax(accuracy)+1}, con un accuracy de {np.max(accuracy).round(4)}')
    return mejor_modelo


# REGRESIÓN

def modelo_base_regr(features, dftot) -> float:
    """Función que devuelve el RMSE y el R2 de un modelo de regresión con las variables introducidas.
    Args:
        features (list): lista de variables predictoras
        dftot (DataFrame): dataframe que incluya las variables predictoras y la variable objetivo
    Returns:
        float: R2 del modelo
    """
    X = dftot[features]
    y = dftot['valuation_2022']
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X.loc[:, col] = le.fit_transform(X.loc[:, col])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f'Este método tiene un RMSE de {rmse.round(4)}, un R2 de {r2.round(4)}')
    return r2

# Mejor modelo en base a R2
def mejor_modelo_regr(features, new_feat, new_features, corr_features, dftot) -> object:
    """Función que devuelve el mejor modelo de regresión en base a su R2.
    Args:
        features (list): lista de variables predictoras mediante el método 1 de selección de variables
        new_feat (list): lista de variables predictoras mediante el método 2 de selección de variables
        new_features (list): lista de variables predictoras mediante el método 3 de selección de variables
        corr_features (list): lista de variables predictoras mediante el método 4 de selección de variables
        dftot (DataFrame): dataframe que incluye todas las variables predictoras y la variable objetivo
    Returns:
        list: lista de variables predictoras del mejor modelo
    """
    modelos = [features, new_feat, new_features, corr_features, dftot.columns]
    r2 = []
    for modelo in modelos:
        r2.append(modelo_base_regr(modelo, dftot))
    mejor_modelo = modelos[np.argmax(r2)]
    print(f'El mejor método es el {np.argmax(r2)+1}, con un R2 de {np.max(r2).round(4)}')
    return mejor_modelo



# MEJOR MÉTODO PARA NORMALIZAR DE LOS DATOS

# Mejor opción para regresión
def random_forest_regr(df) -> float:
    """Calcula el RMSE de un modelo de Random Forest Regressor básico. Sirve para comparar conjuntos de datos.
    Args:
        df (pandas.dataframe): dataframe con las variables predictoras y la variable objetivo, la cual debe llamarse 'valuation 2022'.
    Returns:
        mape: error absoluto porcentual medio del modelo entrenado de Random Forest Regressor.
    """
    df = df.dropna(subset=['valuation_2022']) # filtramos las filas con valores nulos en la variable objetivo
    try:
        df = df.drop(columns=['first_funding_date', 'last_funding_date'])
    except:
        pass
    X = df.drop('valuation_2022', axis=1)
    y = df['valuation_2022']

    # Label encoding
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return r2

# Mejor opción para clasificación
def random_forest_clasif(df) -> float:
    """Calcula el accuracy de un modelo de Random Forest Classifier básico. Sirve para comparar conjuntos de datos.
    Args:
        df (pandas.dataframe): dataframe con las variables predictoras y la variable objetivo, la cual debe llamarse 'Porcentaje_adquisicion_cat'.
    Returns:
        accuracy: accuracy del modelo entrenado de Random Forest Classifier.
    """
    try:
        df = df.drop(columns=['first_funding_date', 'last_funding_date', 'valuation_2022'])
    except:
        pass
    X = df.drop('Porcentaje_adquisicion_cat', axis=1)
    y = df['Porcentaje_adquisicion_cat'].astype('int')
    
    # Label encoding
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions).round(4)
    return accuracy



# FUSIÓN DE LOS DATOS DE 2020 Y 2021

def is_valid_pair(row1, row2) -> bool:
    """Función que comprueba si un par de filas cumple las condiciones.
    Args:
        row1 (pandas Series): La primera fila a comparar.
        row2 (pandas Series): La segunda fila a comparar.
    Returns:
        bool: True si el par de filas cumple las condiciones, False en caso contrario.
    """
    return row1['year'] == 2021 and row2['year'] == 2020

def get_average(row1, row2) -> dict:
    """Función que calcula la diferencia porcentual entre dos filas.
    Args:
        row1 (pandas Series): La primera fila a comparar.
        row2 (pandas Series): La segunda fila a comparar.
    Returns:
        dict: Un diccionario con las diferencias calculadas. Las variables categóricas mantienen el valor de la primera fila.
    """
    average = {}
    for col in row1.index:
        if isinstance(row1[col], str) or col == 'valuation_2022' or col == 'Porcentaje_adquisicion_cat' or col == 'dias_operando':
            # Si la columna es de tipo string o es una de las variables target, mantener el valor de la fila de 2021
            average[col] = row1[col]
        else:
            # Si la columna es numérica, calcular la media ponderada, salvo que el valor de la fila de 2020 sea 0
            if row2[col] == 0:
                average[col] = 0
            else:
                # row1 es la fila de 2021 y row2 es la fila de 2020: row1*0.75 + row2*0.25
                average[col] = row1[col] * 0.75 + row2[col] * 0.25
    return average

def calculate_averages(df) -> pd.DataFrame:
    """Función que itera sobre un DataFrame y calcula la media pondera entre 2020 y 2021.
    Iteraremos sobre los datos tratando las filas a pares. Crearemos una función que comprueba que la combinación de las dos instancias próximas es 2021 primero y 2020 después. Si eso es así, otra función calculará la media ponderada para cada variable, y esa fila se añadirá al nuevo dataframe.
    Si la comprobación de la combinación de 2021 primero y 2020 después falla, se imprimirá un error por pantalla y se ignorará ese par de filas.
    De esta manera, se reduce el número de instancias a la mitad: 60 instancias para 60 empresas, teniendo una fila por empresa.
    Args:
        df (pandas DataFrame): El DataFrame que contiene los datos.
    Returns:
        pandas DataFrame: Un nuevo DataFrame con la media ponderada calculada.
    """
    averages = []
    for i in range(len(df) - 1):
        if is_valid_pair(df.iloc[i], df.iloc[i + 1]):
            averages.append(get_average(df.iloc[i], df.iloc[i + 1]))
    return pd.DataFrame(averages)

def get_difference(row1, row2) -> dict:
    """Función que calcula la diferencia porcentual entre dos filas.
    Args:
        row1 (pandas Series): La primera fila a comparar.
        row2 (pandas Series): La segunda fila a comparar.
    Returns:
        dict: Un diccionario con las diferencias calculadas. Las variables categóricas mantienen el valor de la primera fila.
    """
    difference = {}
    for col in row1.index:
        if isinstance(row1[col], str) or col == 'valuation_2022' or col == 'Porcentaje_adquisicion_cat' or col == 'dias_operando':
            # Si la columna es de tipo string o es una de las variables target, mantener el valor de la fila de 2021
            difference[col] = row1[col]
        else:
            # Si la columna es numérica, calcular la diferencia porcentual, salvo que el valor de la fila de 2020 sea 0
            if row2[col] == 0:
                difference[col] = 0
            else:
                difference[col] = (row1[col] - row2[col]) / row2[col] * 100
    return difference

def calculate_differences(df) -> pd.DataFrame:
    """Función que itera sobre un DataFrame y calcula las diferencias.
    Iteraremos sobre los datos tratando las filas a pares. Crearemos una función que comprueba que la combinación de las dos instancias próximas es 2021 primero y 2020 después. Si eso es así, otra función calculará (2021-2020)/2020*100 para cada variable, y esa fila que representa la diferencia entre los años se añadirá al nuevo dataframe.
    Si la comprobación de la combinación de 2021 primero y 2020 después falla, se imprimirá un error por pantalla y se ignorará ese par de filas.
    De esta manera, se reduce el número de instancias a la mitad: 60 instancias para 60 empresas, teniendo una fila por empresa.
    Args:
        df (pandas DataFrame): El DataFrame que contiene los datos.
    Returns:
        pandas DataFrame: Un nuevo DataFrame con las diferencias calculadas.
    """
    differences = []
    for i in range(len(df) - 1):
        if is_valid_pair(df.iloc[i], df.iloc[i + 1]):
            differences.append(get_difference(df.iloc[i], df.iloc[i + 1]))
    return pd.DataFrame(differences)



# PREPROCESAMIENTO INICIAL

def prepro_df1(df1):
    
    ## DF SABI 1
    df1['Fecha constitucion']=pd.to_datetime(df1['Fecha constitucion'], format='%Y/%m/%d')

    return df1


def prepro_df2(df2, df2_2):
    ## DF SABI 2 (variables financieras)
    ### Union y eliminacion de instancias
    #Se juntan el df2 y el df2_2, que contienen ambos datos financieros, pero distintas variables
    df2 = df2.merge(df2_2, on=['Codigo_NIF', 'year'], how='outer') # por las claves primarias: "Codigo_NIF" y "year"

    # Se quitan las instancias de 2020 para las empresas que se han creado a finales de 2020 o en 2021
    df2 = df2[df2.isna().mean(axis=1)<0.5]

    # Hay algunos NaNs camuflados como 'n.s.', los convertiremos a NaNs
    df2 = df2.replace('n.s.', np.nan)


    # COLINEALIDADES ENTRE LAS VARIABLES FINANCIERAS
    # El balance cuadra a la perfección


    # Missings en las variables principales
    vars_balance = ['Inmovilizado mil EUR', 'Activo circulante mil EUR', 'Total activo mil EUR', \
                        'Fondos propios mil EUR', 'Pasivo fijo mil EUR', 'Pasivo líquido mil EUR', \
                        'Total pasivo y capital propio mil EUR']


    ### Imputación de variables financieras según reglas financieras

    # IMPUTACIONES (I): Activo

    # Hay algunos missings, los rellenamos según las reglas contables:
    df2['Inmovilizado mil EUR'] = df2['Inmovilizado mil EUR'].fillna(df2['Total activo mil EUR'] - df2['Activo circulante mil EUR'])
    df2['Activo circulante mil EUR'] = df2['Activo circulante mil EUR'].fillna(df2['Total activo mil EUR'] - df2['Inmovilizado mil EUR'])

    # Missings imputados según las reglas contables


    # IMPUTACIONES (II): Pasivo

    # Hay algunos missings, los rellenamos según las reglas contables:
    df2['Pasivo fijo mil EUR'] = df2['Pasivo fijo mil EUR'].fillna(df2['Total pasivo y capital propio mil EUR'] - df2['Fondos propios mil EUR'] - df2['Pasivo líquido mil EUR'])
    df2['Pasivo líquido mil EUR'] = df2['Pasivo líquido mil EUR'].fillna(df2['Total pasivo y capital propio mil EUR'] - df2['Fondos propios mil EUR'] - df2['Pasivo fijo mil EUR'])

    # Quedan 2 missings en cada columna que tenía missings, pero no se puede imputar nada más
    # Se trata de empresas que no tienen pasivo fijo ni pasivo líquido, por lo que los NaNs se imputan como 0s

    df2['Pasivo fijo mil EUR'] = df2['Pasivo fijo mil EUR'].fillna(0)
    df2['Pasivo líquido mil EUR'] = df2['Pasivo líquido mil EUR'].fillna(0)

    # Missings imputados según las reglas contables


    # IMPUTACIONES (III): Decimales en el Activo

    # Las instancias que no coinciden no coindicen por un problema de decimales, por lo que se considera que son equivalentes
    df2['Total activo mil EUR'] = df2['Inmovilizado mil EUR'] + df2['Activo circulante mil EUR']
    # 0 missings


    # IMPUTACIONES (IV): Decimales en el Patrimonio Neto

    # de nuevo es un problema de decimales, por lo que se considera que son iguales
    df2['Fondos propios mil EUR'] = df2['Capital suscrito mil EUR'] + df2['Otros fondos propios mil EUR']
    # 0 missings



    ### Corrección de variables

    # 1. Fondo de maniobra = Activo circulante - Pasivo líquido
    # No cuadra en la mayoría de los casos, por lo que reescribiremos esa variable según las reglas contables
    df2['Fondo de maniobra mil EUR'] = df2['Activo circulante mil EUR'] - df2['Pasivo líquido mil EUR']
    # 0 missings



    ### Selección de variables

    # CUENTA DE PÉRDIDAS Y GANANCIAS

    # A: Resultado de explotación = Ingresos de explotación - Gastos de explotación
    # B: Resultado financiero = Ingresos financieros - Gastos financieros
    # C: Resultado ordinario antes de impuestos = Resultado de explotación (A) + Resultado financiero (B)
    # D: Resultado del ejercicio = Resultado ordinario antes de impuestos (C) - Impuesto de sociedades

    # Variables clave:
    vars_perd_ganancias = ['Resultado Explotación mil EUR', 'Resultado financiero mil EUR',
                            'Resultado del Ejercicio mil EUR', 'Costes de los trabajadores / Ingresos de explotación (%) %',
                            'Tesorería mil EUR', 'Cash flow mil EUR', 'Período de cobro (días) días',
                            'Valor agregado mil EUR']
    # El Resultado ordinario antes de impuestos no se incluye porque sería redundante

    # Ratios ya calculados
    vars_ratios = df2.columns[df2.columns.str.contains('%')] # ratios ya calculados
    # pocos missings, se imputarán con KNN

    # Otras variables seleccionadas necesarias para calcular nuevas variables o ejecucion de alguna tarea del reto
    vars_otros = ['EBITDA mil EUR', 'EBIT mil EUR', 'Fondo de maniobra mil EUR', 'Impuestos sobre sociedades mil EUR', 'Dotaciones para amortiz. de inmovil. mil EUR',
                    'Deudas financieras mil EUR', 'Importe neto Cifra de Ventas mil EUR', 'Existencias mil EUR', 'Acreedores comerciales mil EUR']

    # Variables identificativas de la empresa 
    vars_identificativas = ['Codigo_NIF', 'year', 'Número empleados']

    vars = []
    vars.extend(vars_identificativas)
    vars.extend(vars_balance)
    vars.extend(vars_perd_ganancias)
    vars.extend(vars_ratios)
    vars.extend(vars_otros)
    df2_final = df2[vars]

    return df2_final


def prepro_df3(df3):

    ## DF 3 (Dealroom)

    # first funding date
    df3['first_funding_date'] = df3['first_funding_date'].fillna("jan/1900")
    for index, row in df3.iterrows():
        try:
            df3['first_funding_date'][index]=pd.to_datetime(row['first_funding_date'], format='%b/%Y')
        except: df3['first_funding_date'][index]=pd.to_datetime(row['first_funding_date'], format='%Y')

    # last funding date
    df3['last_funding_date'] = df3['last_funding_date'].fillna("jan/1900")
    for index, row in df3.iterrows():
        try:
            df3['last_funding_date'][index]=pd.to_datetime(row['last_funding_date'], format='%b/%Y')
        except: df3['last_funding_date'][index]=pd.to_datetime(row['last_funding_date'], format='%Y')

    # Las instancias cuya fecha de financiación sea 1900/1/1 no tienen rondas de financiación y las rellenamos con el valor "sin ronda".
    df3.loc[df3['first_funding_date'] == dt.datetime(1900,1,1), 'last_round'] = df3[df3['first_funding_date'] == dt.datetime(1900,1,1)]['last_round'].fillna("sin ronda")

    # Todas las empresas que tienen total_funding=0 son las que tampoco tienen ronda de financiación.
    df3.loc[df3['total_funding'] == 0, 'last_funding'] = df3[df3['total_funding'] == 0]['last_funding'].fillna(0)

    return df3


def agrupar_dataframes(df1, df2_final, df3):

    # AGRUPAR DATAFRAMES

    ### Agrupación y selección de la variable del número de empleados
    dftot = df1.merge(df3, on="Codigo_NIF", how="outer")
    dftot = dftot.merge(df2_final, on="Codigo_NIF", how="outer")

    # Se pone el Codigo_NIF como índice
    

    # Hay dos columnas con el numero de empleados así que nos quedamos con el que menos missings tenga, en este caso la denominada n_empleados (la del SABI)
    dftot = dftot.drop(columns="n_empleados_dealroom")
    

    ## ANÁLISIS DE LA VARIABLE DEL NÚMERO DE EMPLEADOS (2/2)

    # Tengo una variable "n_empleados"
    # Y una variable "growth_stage"
    # Quiero imputar la columna "n_empleados" según la media de la columna "n_empleados" de las empresas que estén en el mismo "growth_stage"

    dftot['Número empleados'] = dftot.groupby('growth_stage')['Número empleados'].transform(lambda x: x.fillna(x.mean()))
    dftot.set_index('Codigo_NIF', inplace=True)

    # En Linkedin indican que tienen entre 11 y 50 empleados, de modo que se imputa con 30
    dftot.loc[dftot['Número empleados'].isna(), 'Número empleados'] = 30
    dftot = dftot.loc[:,~dftot.columns.duplicated()]

    return dftot



def crear_variables(dftot):

    # CREACIÓN DE VARIABLES

    # RATIOS FINANCIEROS ADICIONALES

    # Ratios con la deuda
    dftot['ratio_deuda_ebitda'] = dftot['Deudas financieras mil EUR'] / dftot['EBITDA mil EUR']
    dftot['ratio_deuda_activos'] = dftot['Deudas financieras mil EUR'] / dftot['Total activo mil EUR']
    dftot['ratio_deuda_patrimonio'] = dftot['Deudas financieras mil EUR'] / dftot['Fondos propios mil EUR']

    # Ratios con el EBITDA
    dftot['ratio_ebitda_activos'] = dftot['EBITDA mil EUR'] / dftot['Total activo mil EUR']
    dftot['ratio_ebitda_patrimonio'] = dftot['EBITDA mil EUR'] / dftot['Fondos propios mil EUR']

    # Ratios con las ventas
    dftot['ratio_ventas_ebitda'] = dftot['Importe neto Cifra de Ventas mil EUR'] / dftot['EBITDA mil EUR']
    dftot['ratio_ventas_activos'] = dftot['Importe neto Cifra de Ventas mil EUR'] / dftot['Total activo mil EUR']
    dftot['ratio_ventas_patrimonio'] = dftot['Importe neto Cifra de Ventas mil EUR'] / dftot['Fondos propios mil EUR']

    # Prueba ácida: medida más afinada de la capacidad que tiene una empresa para afrontar sus deudas a corto con elementos de activo, puesto que resta de estos elementos los que forman parte del inventario.
    dftot['ratio_prueba_acida'] = (dftot['Activo circulante mil EUR'] - dftot['Existencias mil EUR']) / dftot['Pasivo líquido mil EUR']

    # Periodo promedio de cobro = (Cuentas por cobrar * días del año) / Ventas anuales en cuenta corriente
    dftot['ratio_periodo_prom_cobro'] = (dftot['Acreedores comerciales mil EUR'] * 365) / dftot['Importe neto Cifra de Ventas mil EUR']


    # CREACIÓN DE VARIABLES TEMPORALES

    # Número de días operando
    dftot['dias_operando'] = dt.datetime.now() - dftot['Fecha constitucion']

    # Diferencia entre la fecha actual y la fecha de la última financiación
    dftot['dias_desde_ult_round'] = dt.datetime.now() - dftot['last_funding_date']

    # Se convierten las variables temporales a días y a numéricas para poder emplearlas para el modelado
    dftot['dias_operando'] = dftot['dias_operando'].dt.days.astype('int')
    dftot['dias_desde_ult_round'] = dftot['dias_desde_ult_round'].dt.days.astype('int')

    return dftot


def imputacion_numericas(dftot):

    # IMPUTACIONES

    # Variables financieras: agrupadas en 6 categorías de variables

    df_numericos = dftot.select_dtypes(include=['float64', 'int64'])
    df_numericos.columns
    df_numericos=df_numericos.drop(columns=['Codigo primario CNAE 2009', 'valuation_2022', 'year'])

    vars_resultados=df_numericos[['Resultado Explotación mil EUR', 'Resultado financiero mil EUR', 'Resultado del Ejercicio mil EUR', 'Impuestos sobre sociedades mil EUR', 'Deudas financieras mil EUR', 'Importe neto Cifra de Ventas mil EUR', 'Margen de beneficio (%) %']].columns
    vars_rentabilidad=df_numericos[['Rentabilidad económica (%) %', 'Rentabilidad financiera (%) %', 'Rentabilidad sobre capital empleado (%) %', 'Rentabilidad sobre el activo total (%) %', 'Rentabilidad sobre recursos propios (%) %', 'Margen de beneficio (%) %']].columns
    vars_recursos=df_numericos[['Inmovilizado mil EUR', 'Dotaciones para amortiz. de inmovil. mil EUR', 'Activo circulante mil EUR', 'Total activo mil EUR', 'Fondos propios mil EUR', 'Pasivo fijo mil EUR', 'Pasivo líquido mil EUR', 'Total pasivo y capital propio mil EUR', 'Valor agregado mil EUR', 'Costes de los trabajadores / Ingresos de explotación (%) %', 'Existencias mil EUR', 'Acreedores comerciales mil EUR', 'Fondo de maniobra mil EUR']].columns
    vars_liquidos=df_numericos[['Tesorería mil EUR', 'Cash flow mil EUR', 'Período de cobro (días) días', 'Liquidez general %', 'Ratio de liquidez %', 'Ratio de solvencia %', 'Coeficiente de solvencia (%) %', 'Ratio de cobertura de intereses %']].columns
    vars_rotacion_activos=df_numericos[['Rotación de activos netos %', 'Rotación de las existencias %',  'ratio_ventas_activos', 'ratio_ventas_ebitda', 'ratio_ventas_patrimonio', 'ratio_periodo_prom_cobro', 'ratio_prueba_acida', 'ratio_deuda_activos']].columns
    vars_apalancamiento=df_numericos[['ratio_deuda_ebitda', 'ratio_deuda_patrimonio', 'ratio_ebitda_activos', 'ratio_ebitda_patrimonio', 'Apalancamiento (%) %', 'Endeudamiento (%) %', 'Ratios de autonomía financiera a medio y largo plazo %']].columns

    imputer = KNNImputer(n_neighbors=10, weights='distance')
    # 10 vecinos y ponderación por distancia para evitar una desmedida influencia de posibles outliers

    dftot[vars_resultados] = imputer.fit_transform(dftot[vars_resultados])
    dftot[vars_rentabilidad] = imputer.fit_transform(dftot[vars_rentabilidad])
    dftot[vars_recursos] = imputer.fit_transform(dftot[vars_recursos])
    dftot[vars_liquidos] = imputer.fit_transform(dftot[vars_liquidos])
    dftot[vars_rotacion_activos] = imputer.fit_transform(dftot[vars_rotacion_activos])
    dftot[vars_apalancamiento] = imputer.fit_transform(dftot[vars_apalancamiento])

    # Variables donde aún quedan missings tras la imputación
    vars_miss = dftot.columns[dftot.isna().any()]

    # La target valuation_2022 de momento no se imputa, ya que es la variable que se quiere predecir
    vars_miss = vars_miss.drop('valuation_2022')

    # Variables continuas a imputar
    vars_miss_cont = dftot[vars_miss].select_dtypes(include=['float64', 'int64']).columns

    # Imputación de las variables continuas mediante KNN (mismo método que para las variables financieras)
    dftot[vars_miss_cont] = imputer.fit_transform(dftot[vars_miss_cont])

    return dftot, vars_miss


def imputacion_categoricas(dftot, vars_miss):

    # Imputar variables categóricas con mejor imputador para cada una
    print(' ', '-'*80, '\n', 'IMPUTACIÓN DE LAS VARIABLES CATEGÓRICAS MEDIANTE MODELOS', '\n', '-'*80)

    # Variables categóricas a imputar
    vars_miss_obj = dftot[vars_miss].select_dtypes(include=['object']).columns

    for var in vars_miss_obj:
        mejor_modelo = mejor_imputador_clasif(var, dftot, vars_miss_obj)

        num_vars = dftot.select_dtypes(include=['float64', 'int64']).columns

        # Dividir el conjunto de datos en train y test
        X_train = dftot[num_vars][~dftot[var].isna()]
        y_train = dftot[var][~dftot[var].isna()]

        X_test = dftot[num_vars][dftot[var].isna()]

        # Imputar valores faltantes con SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Entrenar el modelo
        mejor_modelo.fit(X_train, y_train)

        # Predecir valores faltantes en el conjunto de datos original
        y_pred = mejor_modelo.predict(X_test)

        dftot.loc[dftot[var].isna(), var] = y_pred

        print(f"Variable {var} imputada con el modelo {mejor_modelo}")
        print('\n')
    
    return dftot


def calcular_fcf(dftot):

    # CÁLCULO DEL FREE CASH FLOW

    # variacion del activo fijo (capex):
    # capex = activo no corriente 2021 - (activo no corriente 2020 - dotaciones para amortizacion del inmovilizado 2020)
    dftot['capex'] = dftot[dftot['year']==2021]['Inmovilizado mil EUR']-(dftot[dftot['year']==2020]['Inmovilizado mil EUR'] - dftot[dftot['year']==2020]['Dotaciones para amortiz. de inmovil. mil EUR'])

    # variación del activo circulante:
    dftot['activo_circ_variacion'] = dftot[dftot['year']==2021]['Activo circulante mil EUR'] - dftot[dftot['year']==2020]['Activo circulante mil EUR']

    # free cash flow = ebit - impuestos + dotacion_amort - capex - activo circulante
    # se corrigen capex y activo circulante porque están sumados pero aún no se han cobrado así que no son cash (clientes y proveedores)
    dftot['free_cash_flow'] = dftot['EBIT mil EUR'] - dftot['Impuestos sobre sociedades mil EUR'] + dftot['Dotaciones para amortiz. de inmovil. mil EUR'] \
        - dftot['capex'] - dftot['activo_circ_variacion']

    return dftot


def feature_selection(dftot):

    # FEATURE SELECTION

    # features : según el modelo de Random Forest
    # new_feat : según f_classif o f_regression
    # new_features : según Recursive Feature Elimination
    # corr_features : según correlación con la variable objetivo
    dftot = dftot.drop(columns=['capex', 'activo_circ_variacion', 'free_cash_flow'])
    df_aux = dftot.copy()

    #### Para regresión

    dftot = df_aux.copy()
    dftot = dftot.drop(columns=['Codigo primario CNAE 2009', 'Nombre_sabi', 'Localidad', 'Fecha constitucion', 'Codigo consolidacion', 'Forma juridica', 'Forma juridica detallada', 'Estado', 'Estado detallado', 'website', 'name_dealroom', 'profile_url', 'tagline'])

    # Se filtran los NaN de la target
    dftot = dftot[~dftot['valuation_2022'].isna()]

    X = dftot.drop(columns=['valuation_2022'])
    y = dftot['valuation_2022']


    # Label encoding sobre todas las categóricas
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])

    # Split train-test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    # Modelo a partir del cual se sacarán las features más importantes
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    forest_importances = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=True)
    umbral = 0.01
    forest_importances = forest_importances.to_frame().reset_index()
    forest_importances.columns = ['feature', 'importance']
    features = list(forest_importances[forest_importances['importance'] > umbral]['feature'])

    # Según f_regression
    selector = SelectKBest(f_regression, k=20)
    selector = selector.fit(x_train, y_train)
    mask = selector.get_support() # True si es importante, False si no lo es
    new_feat = []
    for bool, feature in zip(mask, x_train.columns):
        if bool:
            new_feat.append(feature)

    # Recursive feature elimination
    rf_RFE = RandomForestRegressor(random_state=RANDOM_STATE)
    selector = RFE(rf_RFE, n_features_to_select=20)
    selector = selector.fit(x_train, y_train)
    rfe_mask = selector.get_support()
    new_features = []
    for bool, feature in zip(rfe_mask, x_train.columns):
        if bool:
            new_features.append(feature)

    # Variables más correlacionadas con la variable objetivo
    corr = dftot.corr()
    corr = corr.sort_values(by=['valuation_2022'], ascending=False)
    corr_features = corr['valuation_2022'].head(20).index[1:]
    corr.iloc[1:6, :]['valuation_2022']


    # Modelo base: predecir la variable objetivo con cada combinación de variables con un Random Forest
    print(' ', '-'*80, '\n', 'SELECCIÓN DE VARIABLES PARA REGRESIÓN MEDIANTE DISTINTOS ALGORITMOS', '\n', '-'*80)
    X = dftot[mejor_modelo_regr(features, new_feat, new_features, corr_features, dftot)] # selecciona el mejor modelo

    # Guardar X en path limpios
    X.to_csv('Datos/Limpios/df_regr_prefusion.csv')



    #### Para clasificación

    dftot = df_aux.copy()
    dftot = dftot.drop(columns=['Codigo primario CNAE 2009', 'Nombre_sabi', 'Localidad', 'Fecha constitucion', 'Codigo consolidacion', 'Forma juridica', 'Forma juridica detallada', 'Estado', 'Estado detallado', 'website', 'name_dealroom', 'profile_url', 'tagline'])

    X = dftot.drop(columns=['Porcentaje_adquisicion_cat']).drop(columns=['valuation_2022'])
    y = dftot['Porcentaje_adquisicion_cat']

    # Label encoding sobre todas las categóricas
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])

    # Split train-test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    # Modelo a partir del cual se sacarán las features más importantes
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    forest_importances = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=True)
    umbral = 0.015
    forest_importances = forest_importances.to_frame().reset_index()
    forest_importances.columns = ['feature', 'importance']
    features = list(forest_importances[forest_importances['importance'] > umbral]['feature'])

    # Según f_classif
    selector = SelectKBest(f_classif, k=20)
    selector = selector.fit(x_train, y_train)
    mask = selector.get_support() # True si es importante, False si no lo es
    new_feat = []
    for bool, feature in zip(mask, x_train.columns):
        if bool:
            new_feat.append(feature)

    # Recursive feature elimination
    rf_RFE = RandomForestClassifier(random_state=RANDOM_STATE)
    selector = RFE(rf_RFE, n_features_to_select=20)
    selector = selector.fit(x_train, y_train)
    rfe_mask = selector.get_support()
    new_features = []
    for bool, feature in zip(rfe_mask, x_train.columns):
        if bool:
            new_features.append(feature)

    # Variables más correlacionadas con la variable objetivo
    corr = dftot.corr()
    corr = corr.sort_values(by=['Porcentaje_adquisicion_cat'], ascending=False)
    corr_features = corr['Porcentaje_adquisicion_cat'].head(20).index[1:]
    corr.iloc[1:6, :]['Porcentaje_adquisicion_cat']

    # Modelo base: predecir la variable objetivo con cada combinación de variables con un Random Forest
    print(' ', '-'*80, '\n', 'SELECCIÓN DE VARIABLES PARA CLASIFICACIÓN MEDIANTE DISTINTOS ALGORITMOS', '\n', '-'*80)
    X = dftot[mejor_modelo_clasif(features, new_feat, new_features, corr_features, dftot)] # selecciona el mejor modelo

    # Guardar X en path limpios
    X.to_csv('Datos/Limpios/df_clasif_prefusion.csv')

    try:
        X = X.drop(columns=['valuation_2022'])
    except:
        pass
    y = dftot['Porcentaje_adquisicion_cat']
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)

    print(rf.__class__.__name__, ':', round(accuracy_score(y_test, predictions),4), '\n', confusion_matrix(y_test, predictions))


    ### Actualización de datos
    df = dftot.copy() # ninguna selección de variables se acerca al error o al accuracy del modelo con todas las variables

    return df


def normalizar(dfa_1, dfa_2, df):

    # NORMALIZACIÓN DE DATOS

    #### Opción A: datos sin normalizar

    #### Opción B: (x - mean)/std
    # (x - mean(x)) / std(x) por cada columna numérica, y se mantiene el valor de 2021 para las variables categóricas
    dfb = df.copy()
    scaler = StandardScaler()
    for col in dfb.select_dtypes(include='number').columns:
        if col == 'Porcentaje_adquisicion_cat' or col == 'valuation_2022':
            pass
        dfb[col] = scaler.fit_transform(dfb[col].values.reshape(-1, 1))

    #### Opción C: (x - min)/(max - min)
    # (x - min(x)) / (max(x) - min(x)) por cada columna numérica, y se mantiene el valor de 2021 para las variables categóricas
    dfc = df.copy()
    scaler = MinMaxScaler()
    for col in dfc.select_dtypes(include='number').columns:
        if col == 'Porcentaje_adquisicion_cat' or col == 'valuation_2022':
            pass
        dfc[col] = scaler.fit_transform(dfc[col].values.reshape(-1, 1))


    #### Mejor opción para regresión
    print(' ', '-'*80, '\n', 'MEJOR MÉTODO DE NORMALIZACIÓN PARA REGRESIÓN', '\n', '-'*80)

    # Opción A: sin escalar
    print(f'Sin escalar: {random_forest_regr(dfa_1).round(4)} de R2')
    # Opción B: con StandardScaler
    print(f'Con StandardScaler: {random_forest_regr(dfb).round(4)} de R2')
    # Opción C: con MinMaxScaler
    print(f'Con MinMaxScaler: {random_forest_regr(dfc).round(4)} de R2')

    print(f'El mejor modelo es el de la opción {np.argmax([random_forest_regr(dfa_1), random_forest_regr(dfb), random_forest_regr(dfc)]) + 1} con un R2 de {(np.max([random_forest_regr(dfa_1), random_forest_regr(dfb), random_forest_regr(dfc)])).round(4)}')


    #### Mejor opción para clasificación
    print(' ', '-'*80, '\n', 'MEJOR MÉTODO DE NORMALIZACIÓN PARA CLASIFICACIÓN', '\n', '-'*80)

    # Opción A: sin escalar
    print(f'Sin escalar: {random_forest_clasif(dfa_1)} de accuracy')
    # Opción B: con StandardScaler
    print(f'Con StandardScaler: {random_forest_clasif(dfb)} de accuracy')
    # Opción C: con MinMaxScaler
    print(f'Con MinMaxScaler: {random_forest_clasif(dfc)} de accuracy')

    print(f'El mejor modelo es el de la opción {np.argmax([random_forest_clasif(dfa_1), random_forest_clasif(dfb), random_forest_clasif(dfc)]) + 1} con un accuracy de {(np.max([random_forest_clasif(dfa_1), random_forest_clasif(dfb), random_forest_clasif(dfc)])).round(4)}')


    # Para ambos casos el mejor método es no escalar los datos
    df = dfa_2.copy()

    return df


def fusionar_datos_media_ponderada(df, df2):
    
    # FUSIÓN DE LOS DATOS DE 2020 Y 2021 MEDIANTE UNA MEDIA PONDERADA

    df_aux = df.copy()

    # Se añade la columna del año
    df2_reset = df2.reset_index()
    df_year = df2_reset['year']
    df_aux.reset_index(inplace=True)
    df = pd.concat([df_aux, df_year], axis=1)
    df.columns = df_aux.columns.tolist() + ['year']
    df = df.loc[:,~df.columns.duplicated()] # si hay columnas duplicadas, se elimina una de ellas

    # Ignore funding dates
    df = df.drop(columns=['first_funding_date', 'last_funding_date'])
    df['Porcentaje_adquisicion_cat'] = df['Porcentaje_adquisicion_cat'].astype('object')

    # Se calculan la media ponderada entre 2020 y 2021
    df_diff = calculate_averages(df) # la función clave

    # La columna 'year' ya no es necesaria, ya que todas las filas indican la diferencia entre dos años
    df_diff = df_diff.drop(columns=['year'])
    df_diff.set_index('Codigo_NIF', inplace=True)

    return df_diff


def fusionar_datos_diferencia(df, df2):
    
    # FUSIÓN DE LOS DATOS DE 2020 Y 2021 CON DIFERENCIAS PORCENTUALES

    df_aux = df.copy()

    # Se añade la columna del año
    df2_reset = df2.reset_index()
    df_year = df2_reset['year']
    df_aux.reset_index(inplace=True)
    df = pd.concat([df_aux, df_year], axis=1)
    df.columns = df_aux.columns.tolist() + ['year']
    df = df.loc[:,~df.columns.duplicated()] # si hay columnas duplicadas, se elimina una de ellas

    # Ignore funding dates
    df = df.drop(columns=['first_funding_date', 'last_funding_date'])
    df['Porcentaje_adquisicion_cat'] = df['Porcentaje_adquisicion_cat'].astype('object')

    # Se calculan las diferencias porcentuales entre 2020 y 2021
    df_diff = calculate_differences(df) # la función clave

    # La columna 'year' ya no es necesaria, ya que todas las filas indican la diferencia entre dos años
    df_diff = df_diff.drop(columns=['year'])
    df_diff.set_index('Codigo_NIF', inplace=True)

    return df_diff