# Generales y preprocesamiento
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

# Preparación para modelos
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from imblearn.combine import SMOTEENN

# Modelos
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier, BaggingRegressor, BaggingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, TweedieRegressor, LogisticRegression, SGDClassifier , Lasso, ElasticNet, Ridge
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import  DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor, XGBClassifier
from tpot import TPOTRegressor, TPOTClassifier

# Validación y guardado de modelos
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, accuracy_score, confusion_matrix, mean_absolute_error, classification_report, ConfusionMatrixDisplay
import pickle
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import warnings


RANDOM_STATE = 42


def carga_datos_limpios(df: str, DATOS_LIMPIOS: str) -> pd.DataFrame:
    """Carga de datos limpios de la carpeta datos_limpios

    Args:
        df (str): nombre del dataframe a cargar
        DATOS_LIMPIOS (str): ruta de la carpeta datos limpios

    Returns:
        df (pd.DataFrame): dataframe cargado
    """
    return pd.read_csv(os.path.join(DATOS_LIMPIOS, df + '.csv'), index_col=0)


def quitar_variables_ident(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las variables que identifican a la empresa

    Args:
        df (pd.DataFrame): dataframe a limpiar

    Returns:
        df (pd.DataFrame): dataframe limpio
    """
    try:
        df = df.drop(columns=['Codigo primario CNAE 2009', 'year', 'Nombre_sabi', 'Localidad', 'Fecha constitucion', \
                                'Codigo consolidacion', 'Forma juridica', 'Forma juridica detallada', 'Estado', \
                                'Estado detallado', 'website', 'name_dealroom', 'profile_url', 'tagline'])
    except:
        pass
    return df


def label_encoder_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica label encoder a las variables categóricas

    Args:
        df (pd.DataFrame): dataframe a transformar

    Returns:
        df (pd.DataFrame): dataframe transformado
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df



##### REGRESION

def vars_correlacion_reg(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Devuelve las variables con una correlacion mayor a un umbral

    Args:
        df (pd.DataFrame): dataframe a analizar
        umbral (float): umbral de correlacion

    Returns:
        df_final (pd.DataFrame): dataframe con las variables seleccionadas
    """
    correlaciones=df.corr()
    correlaciones=correlaciones.loc[:,"valuation_2022"].sort_values(ascending=False)
    corr_abs = correlaciones.abs()
    high_corr = corr_abs[corr_abs > umbral].reset_index()
    high_corr.columns = ['Variable', 'Correlation']
    variable_list = pd.Series(high_corr['Variable'].unique().tolist()).unique().tolist()
    print(f'Hay {len(variable_list)} variables con una correlacion mayor a {umbral}')
    df_final=df[variable_list]

    return df_final



def vars_importantes_rf_reg(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Devuelve las variables mas importantes segun un random forest

    Args:
        df (pd.DataFrame): dataframe a analizar
        umbral (float): umbral de importancia

    Returns:
        df_final (pd.DataFrame): dataframe con las variables seleccionadas
    """
    independientes = df.drop(['valuation_2022'], axis=1)
    dependiente = df['valuation_2022']

    rf=RandomForestRegressor(random_state=RANDOM_STATE)
    rf.fit(independientes, dependiente)

    print(dict(zip(independientes.columns, rf.feature_importances_.round(2))))

    # plot de las variables en orden descendente
    forest_importances=pd.Series(rf.feature_importances_, index=independientes.columns).sort_values(ascending=True)
    plt.figure(figsize=(10,10))
    plt.barh(forest_importances.index, forest_importances)
    plt.xlabel("Random Forest Feature Importance")

    # crear dataframe con importancia de caracteristicas
    forest_importances=forest_importances.to_frame().reset_index()
    forest_importances.columns=['feature', 'importance']

    importantes= list(forest_importances[forest_importances['importance'] > umbral]['feature'])
    print(f"Hay {len(importantes)} variables importantes con el umbral {umbral}")
    importantes.append('valuation_2022')
    df_final=df[importantes]

    return df_final



def normalizacion_reg(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los datos de un dataframe con StandardScaler y MinMaxScaler y los une con la variable dependiente

    Args:
        df (pd.DataFrame): dataframe a normalizar

    Returns:
        df1 (pd.DataFrame): dataframe normalizado con StandardScaler
        df2 (pd.DataFrame): dataframe normalizado con MinMaxScaler
    """
    df=df.reset_index()
    target=df[['valuation_2022', 'Codigo_NIF']]
    df=df.drop(['valuation_2022','Codigo_NIF'], axis=1)
    scaler = StandardScaler()
    normalizer = MinMaxScaler()

    df1 = scaler.fit_transform(df)
    df2=normalizer.fit_transform(df)
 
    df1 = pd.DataFrame(df1, columns=df.columns)
    df1['valuation_2022']=target['valuation_2022']
    df1['Codigo_NIF']=target['Codigo_NIF']
    df1.set_index('Codigo_NIF', inplace=True)

    df2 = pd.DataFrame(df2, columns=df.columns)
    df2['valuation_2022']=target['valuation_2022']
    df2['Codigo_NIF']=target['Codigo_NIF']
    df2.set_index('Codigo_NIF', inplace=True)

    return df1, df2



def modelo_simple_reg(modelo: object, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas):
    """Entrena un modelo de regresion y devuelve el MSE y R2

    Args:
        modelo (object): modelo a entrenar
        X_train (pd.DataFrame): dataframe de entrenamiento (independientes)
        y_train (pd.DataFrame): dataframe de entrenamiento (dependiente)
        X_test (pd.DataFrame): dataframe de test (independientes)
        y_test (pd.DataFrame): dataframe de test (dependiente)

    Returns:
        modelo (object): modelo entrenado
        r2 (float): R2 del modelo
        rmse (float): RMSE del modelo
    """
    print(str(modelo).split('(')[0])
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #y_pred = np.exp(y_pred)
    y_pred = inv_boxcox(y_pred, lambdas)
    #y_pred = y_pred / 100
    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))
    r2=r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    print('RMSE: ', rmse)
    mape=mean_absolute_percentage_error(y_test, y_pred)
    print('MAPE:', mape)
    print('\n')
    return modelo, r2, rmse


def probar_dataset_reg(df: pd.DataFrame) -> pd.DataFrame:
    """Entrena un modelo de regresion y devuelve el MSE y R2

    Args:
        df (pd.DataFrame): dataframe a entrenar

    Returns:
        None
    """
    X = df.drop(columns=['valuation_2022'])
    y = df['valuation_2022']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    #y_train=y_train*100
    y_train, lambdas = boxcox(y_train)
    #y_train = np.log(y_train)
    print("            ")

    # Random Forest
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    modelo_rf,mape_rf,rmse = modelo_simple_reg(rf, X_train, y_train, X_test, y_test, lambdas)

    # Linear Regression
    lr = LinearRegression()
    modelo_lr,mape_lr,rmse = modelo_simple_reg(lr, X_train, y_train, X_test, y_test, lambdas)

    # Support Vector Regressor
    svr = SVR()
    modelo_svr,mape_svr,rmse = modelo_simple_reg(svr, X_train, y_train, X_test, y_test, lambdas)

    # KNN regressor
    knn=KNeighborsRegressor()
    modelo_knn,mape_knn,rmse = modelo_simple_reg(knn, X_train, y_train, X_test, y_test, lambdas)

    # Decission Tree Regressor
    dtr=DecisionTreeRegressor(random_state=RANDOM_STATE)
    modelo_dtr,mape_dtr,rmse = modelo_simple_reg(dtr, X_train, y_train, X_test, y_test, lambdas)

    # Tweedie Regressor
    twe=TweedieRegressor()
    modelo_twe,mape_twe,rmse = modelo_simple_reg(twe, X_train, y_train, X_test, y_test, lambdas)

    # Mejor modelo de todos según RMSE
    mejor_modelo = np.argmin([mape_rf, mape_lr, mape_svr, mape_knn, mape_dtr, mape_twe])
    print('Mejor modelo:', [rf, lr, svr, knn, dtr, twe][mejor_modelo],', que tiene un R2 de:', [mape_rf, mape_lr, mape_svr, mape_knn, mape_dtr, mape_twe][mejor_modelo])
    print("              ")

    return None


def prediccion_regr(model: object, xtest: pd.DataFrame, ytest: pd.DataFrame, lambdas: float):
    """Entrena un modelo de regresion y devuelve el MSE y R2

    Args:
        model (object): modelo con el que se va a predecir
        xtest (pd.DataFrame): variables independientes de test
        ytest (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox

    Returns:
        r2 (float): R2 del modelo
        rmse (float): RMSE del modelo
    """
    prediccion = model.predict(xtest)

    prediccion = inv_boxcox(prediccion, lambdas)

    print('R2:', round(r2_score(ytest, prediccion),2), '\n', 'MAPE:',round(mean_absolute_percentage_error(ytest, prediccion),4),'\n',
       'MAE :',round(mean_absolute_error(ytest, prediccion),4),'\n', 'RMSE :',round(np.sqrt(mean_squared_error(ytest, prediccion)),4))
    r2=round(r2_score(ytest, prediccion),2)
    rmse=round(np.sqrt(mean_squared_error(ytest, prediccion)),4)
    
    return r2, rmse



def entrenamiento_regr(xtr:pd.DataFrame, ytr:pd.DataFrame, models_list:list, model_hyperparameters:dict) -> None:
    """Entrena un modelo de regresion y devuelve el MSE y R2

    Args:
        xtr (pd.DataFrame): datos de entrenamiento (variables independientes)
        ytr (pd.DataFrame): datos de entrenamiento (target)
        models_list (list): lista de modelos a probar
        model_hyperparameters (dict): diccionario con los hiperparametros de cada modelo a probar

    Returns:
        None
    """
    model_keys=list(model_hyperparameters.keys())

    result = []
    i=0

    for model in models_list:
       key=model_keys[i]
       i+=1
       params = model_hyperparameters[key]

       classifier=GridSearchCV(model,params,cv=5,scoring='r2',refit=True)

       classifier.fit(xtr,ytr)
       result.append({
        'model_used':model,
        'highest_score':classifier.best_score_,
        'best hyperparameters':classifier.best_params_,
       })
       print(result[i-1])
    
    return None



def transform_regr(df_regr: pd.DataFrame, df_regr_prefusion: pd.DataFrame) -> pd.DataFrame:
    """Transforma los datos de la datos de regresion

    Args:
        df_regr (pd.DataFrame): datos de regresion
        df_regr_prefusion (pd.DataFrame): datos de regresion con dos filas por empresa

    Returns:
        df_regr (pd.DataFrame): datos de regresion transformada
        df_regr_prefusion (pd.DataFrame): datos de regresion con dos filas por empresa transformada
    """
    df_regr.set_index("Codigo_NIF", inplace=True)
    df_regr=df_regr.drop(['index', 'ownerships'], axis=1)
    df_regr_prefusion = df_regr_prefusion.drop(columns=['first_funding_date', 'last_funding_date', 'Free capital mil EUR'])
    RANDOM_STATE = 42
    # Limpieza de variables
    df_regr=quitar_variables_ident(df_regr)
    df_regr_prefusion=quitar_variables_ident(df_regr_prefusion)
    # LabelEncoder
    df_regr = label_encoder_categoricas(df_regr)
    df_regr_prefusion=label_encoder_categoricas(df_regr_prefusion)
    #Transformacion
    df_regr=np.log(df_regr+1)
    df_regr_prefusion=np.log(df_regr_prefusion+1)
    df_regr=df_regr.dropna(axis=1)
    df_regr_prefusion=df_regr_prefusion.dropna(axis=1)

    return df_regr, df_regr_prefusion


def seleccion_variables(df_regr: pd.DataFrame, df_regr_prefusion: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las variables mas importantes de los datos de regresion

    Args:
        df_regr (pd.DataFrame): datos de regresion
        df_regr_prefusion (pd.DataFrame): datos de regresion con dos filas por empresa

    Returns:
        df_regr (pd.DataFrame): datos de regresion con las variables seleccionadas
    """
    # correlacion
    df_regr_cor=vars_correlacion_reg(df_regr, 0.15)
    df_regr_pref_cor=vars_correlacion_reg(df_regr_prefusion, 0.2)
    # feature imiportance
    df_regr_imp=vars_importantes_rf_reg(df_regr,0.01 )
    df_regr_pref_imp=vars_importantes_rf_reg(df_regr_prefusion,0.02 )
    # ratios
    df_regr_ratio = df_regr.filter(regex='^(ratio|valuation)')
    print(f'Contiene {df_regr_ratio.shape[1]} columnas')
    df_regr_pref_ratio = df_regr_prefusion.filter(regex='^(ratio|valuation)')
    print(f'Contiene {df_regr_pref_ratio.shape[1]} columnas')

    df_regr_cor_esc, df_regr_cor_nor = normalizacion_reg(df_regr_cor)
    df_regr_pref_cor_esc, df_regr_pref_cor_nor = normalizacion_reg(df_regr_pref_cor)
    df_regr_imp_esc, df_regr_imp_nor  = normalizacion_reg(df_regr_imp)
    df_regr_pref_imp_esc, df_regr_pref_imp_nor = normalizacion_reg(df_regr_pref_imp)
    df_regr_ratio_esc, df_regr_ratio_nor =normalizacion_reg(df_regr_ratio)
    df_regr_pref_ratio_esc, df_regr_pref_ratio_nor = normalizacion_reg(df_regr_pref_ratio)

    df_lista=[df_regr_cor_esc, df_regr_cor_nor,df_regr_pref_cor_esc, df_regr_pref_cor_nor,df_regr_imp_esc, df_regr_imp_nor,df_regr_pref_imp_esc, df_regr_pref_imp_nor,
            df_regr_ratio_esc, df_regr_ratio_nor,df_regr_pref_ratio_esc, df_regr_pref_ratio_nor]
    df_lista_nombre=['df_regr_cor_esc', 'df_regr_cor_nor','df_regr_pref_cor_esc', 'df_regr_pref_cor_nor','df_regr_imp_esc', 'df_regr_imp_nor','df_regr_pref_imp_esc', 'df_regr_pref_imp_nor',
            'df_regr_ratio_esc', 'df_regr_ratio_nor','df_regr_pref_ratio_esc', 'df_regr_pref_ratio_nor']

    df_lista_original=[df_regr_cor,df_regr_pref_cor, df_regr_imp, df_regr_pref_imp, df_regr_ratio, df_regr_pref_ratio ]
    df_original_nombres=['df_regr_cor','df_regr_pref_cor', 'df_regr_imp', 'df_regr_pref_imp', 'df_regr_ratio', 'df_regr_pref_ratio']

    i=0
    for df in df_lista_original:
        print(df_original_nombres[i])
        probar_dataset_reg(df)
        i+=1

    i=0
    for df in df_lista:
        print(df_lista_nombre[i])
        probar_dataset_reg(df)
        i+=1

    return df_regr_imp

def guardar_csv(df: pd.DataFrame, nombre: str) -> None:
    """Guarda un dataframe en un csv

    Args:
        df (pd.DataFrame): dataframe a guardar
        nombre (str): nombre del csv donde se guardará el dataframe

    Returns:
        None
    """
    df.to_csv(nombre, index=False)
    return None

def separar_train_test_regr(df_regr_imp: pd.DataFrame):
    """Separa los datos de regresion en train y test

    Args:
        df_regr_imp (pd.DataFrame): datos de regresion con las variables seleccionadas

    Returns:
        X_train (pd.DataFrame): datos de train
        X_test (pd.DataFrame): datos de test
        y_train (pd.DataFrame): target de train
        y_test (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox
    """
    independinetes=df_regr_imp.drop('valuation_2022', axis=1)
    target=df_regr_imp['valuation_2022']

    X_train, X_test, y_train, y_test = train_test_split(independinetes, target, test_size=0.2, random_state=RANDOM_STATE)

    # transformamos y_train
    prin=y_train

    y_train, lambdas = boxcox(y_train)

    return X_train, X_test, y_train, y_test, lambdas

    
def automatico_regr(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas: float):
    """Entrena un modelo de regresion con TPOT

    Args:
        X_train (pd.DataFrame): datos de train
        y_train (pd.DataFrame): target de train
        X_test (pd.DataFrame): datos de test
        y_test (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox

    Returns:
        r2 (float): r2 del modelo
        rmse (float): rmse del modelo
    """
    tpot = TPOTRegressor(generations=10, population_size=50, verbosity=2,random_state=RANDOM_STATE, scoring="r2")
    tpot.fit(X_train, y_train)
    r2,rmse=prediccion_regr(tpot, X_test, y_test, lambdas)
    return r2, rmse


def modelos_simples_regr(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas: float):
    """Entrena un modelo de regresion con los modelos simples

    Args:
        X_train (pd.DataFrame): datos de train
        y_train (pd.DataFrame): target de train
        X_test (pd.DataFrame): datos de test
        y_test (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox

    Returns:
        model (object): modelo de regresion
        r2 (float): r2 del modelo
        rmse (float): rmse del modelo
    """
    model_list=[KNeighborsRegressor(), DecisionTreeRegressor(random_state=0), SVR(), TweedieRegressor(), LinearRegression()]

    model_hyperparameters={"kn":{
        "n_neighbors": [3,5,7],
        "leaf_size": [15,30],
        'weights':['uniform', 'distance']
    },
    "tree_reg":{
        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        "max_depth":[None, 3, 5,10],
        'min_samples_split':[2,5,10]
    },
    "svr":{
        'kernel':['linear', 'rbf'],
        "C":[1,3,5],
        "gamma":["scale", "auto"]
    },
    "tweedie":{
        "link":['auto', 'identity', 'log']
    },
    'linear':{}
    }

    warnings.filterwarnings('ignore')
    entrenamiento_regr(xtr=X_train, ytr=y_train,models_list=model_list, model_hyperparameters=model_hyperparameters)
    tw=TweedieRegressor(link='auto')
    modelo,r2, rmse=modelo_simple_reg(tw, X_train, y_train, X_test, y_test, lambdas)

    return modelo, r2, rmse


def stacking_regr(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas: float) -> dict:
    """Entrena un modelo de regresion con stacking

    Args:
        X_train (pd.DataFrame): datos de train
        y_train (pd.DataFrame): target de train
        X_test (pd.DataFrame): datos de test
        y_test (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox

    Returns:
        modelos (dict): diccionario con los modelos de regresion
    """    
    # level 0
    level0 = list()
    level0.append(('ls', Lasso()))
    level0.append(('dt', DecisionTreeRegressor(random_state=1234)))
    level0.append(('svr', SVR()))
    level0.append(('ridge', Ridge()))

    # level 1
    level1 = DecisionTreeRegressor(random_state=1234)

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)

    r2, rmse = prediccion_regr(model, X_test, y_test, lambdas)
    modelos=dict()
    hiper={"R2": r2, "RMSE":rmse}
    modelos['stacking_sin_opt']=hiper

    space = {
    'alpha' : hp.uniform("alpha", 0.5,1),
    'max_depth': hp.choice('max_depth', range(1, 100)),
    'selection': hp.choice('selection',['cyclic', 'random']),
    'min_samples_split': hp.choice('min_samples_split', range(2, 15)),
    'C': hp.uniform('C', 0.01, 10),
    'gamma': hp.uniform('gamma', 0.01, 10),
    'alpha2' : hp.uniform("alpha2", 0.5,1),
    'max_depth2': hp.choice('max_depth2', range(1, 100)),
    'min_samples_split2': hp.choice('min_samples_split2', range(2, 15))
    }


    def objective(space):
        """Funcion objetivo para optimizar los hiperparametros

        Args:
            space (dict): diccionario con los hiperparametros

        Returns:
            dict: diccionario con el score y el status
        """
        # level 0
        level0 = list()
        level0.append(('ls', Lasso(alpha=space['alpha'], selection=space['selection'])))
        level0.append(('dt', DecisionTreeRegressor(random_state=1234, max_depth=space['max_depth'], min_samples_split=space['min_samples_split'])))
        level0.append(('svr', SVR(C=space['C'], gamma=space['gamma'])))
        level0.append(('ridge', Ridge(alpha=space['alpha2'])))

        # level 1
        level1 = DecisionTreeRegressor(random_state=1234,max_depth=space['max_depth2'], min_samples_split=space['min_samples_split2'] )

        # define the stacking ensemble
        model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trial = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trial)
    print("best {}".format(best))

    best = space_eval(space, best)

    # level 0
    level0 = list()
    level0.append(('ls', Lasso(alpha=best['alpha'], selection=best['selection'])))
    level0.append(('dt', DecisionTreeRegressor(random_state=1234, max_depth=best['max_depth'], min_samples_split=best['min_samples_split'] )))
    level0.append(('svr', SVR(C=best['C'], gamma=best['gamma'])))
    level0.append(('ridge', Ridge(alpha=best['alpha2'])))

    # level 1
    level1 = DecisionTreeRegressor(random_state=1234, max_depth=best['max_depth2'], min_samples_split=best['min_samples_split2'])

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)

    r2, rmse = prediccion_regr(model, X_test, y_test, lambdas)

    hiper = {"R2": r2, "RMSE": rmse}
    modelos['stacking_opt'] = hiper

    return modelos
    


def bagging_regr(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas: float) -> dict:
    """Entrena un modelo de regresion con bagging

    Args:
        X_train (pd.DataFrame): datos de train
        y_train (pd.DataFrame): target de train
        X_test (pd.DataFrame): datos de test
        y_test (pd.DataFrame): target de test
        lambdas (float): lambda de la transformacion boxcox
    
    Returns:
        modelos (dict): diccionario con los modelos de regresion
    """
    # BAGGING REGRESSOR
    
    linear = LinearRegression()
    dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
    svr = SVR()
    lasso = Lasso(random_state=RANDOM_STATE)
    ridge = Ridge(random_state=RANDOM_STATE)
    elastic = ElasticNet(random_state=RANDOM_STATE)

    models = [linear, dt, svr, lasso, ridge, elastic]

    for model in models:

        bagging = BaggingRegressor(base_estimator=model, n_estimators=100, random_state=RANDOM_STATE)
        bagging.fit(X_train, y_train)
        prediccion = bagging.predict(X_test)
        prediccion = np.exp(prediccion)
        prediccion = inv_boxcox(prediccion, lambdas)
        prediccion = prediccion / 100

        print(model.__class__.__name__, ':','\n',   'R2:', round(r2_score(y_test, prediccion),2), '\n', 'MAPE:',round(mean_absolute_percentage_error(y_test, prediccion),4),'\n',
        'MAE :',round(mean_absolute_error(y_test, prediccion),4),'\n', 'RMSE :',round(np.sqrt(mean_squared_error(y_test, prediccion)),4))
    
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    r2, rmse = prediccion_regr(rf, X_test, y_test, lambdas)
    modelos=dict()
    hiper={"R2":r2, "RMSE":rmse}
    modelos['Baggingregressor_sin_opt']=hiper

    space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(2, 10)),
    'bootstrap': hp.choice('bootstrap', [True, False])
    }

    def objective(space):
        """Funcion objetivo para optimizar los hiperparametros

        Args:
            space (dict): diccionario con los hiperparametros

        Returns:
            dict: diccionario con el score y el status
        """
        rf = RandomForestRegressor(n_estimators=space['n_estimators'], max_depth=space['max_depth'], min_samples_split=space['min_samples_split'], min_samples_leaf=space['min_samples_leaf'], bootstrap=space['bootstrap'], random_state=RANDOM_STATE)
        score = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)

    print("best {}".format(best))

    best = space_eval(space, best)
    rf = RandomForestRegressor(random_state=RANDOM_STATE, bootstrap=best['bootstrap'],  min_samples_leaf=best['min_samples_leaf'], min_samples_split=best['min_samples_split'], n_estimators=best['n_estimators'])
    rf.fit(X_train, y_train)
    r2, rmse = prediccion_regr(rf, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['Baggingregressor_opt']=hiper

    # EXTREMELY RANDOMIZED TREE

    et = ExtraTreesRegressor(random_state=RANDOM_STATE, )
    et.fit(X_train, y_train)
    r2, rmse=prediccion_regr(et, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['ExtraTreesRegressor_sin_opt']=hiper

    space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(2, 10)),
    'bootstrap': hp.choice('bootstrap', [True, False])
    }

    def objective(space):
        """Funcion objetivo para optimizar los hiperparametros

        Args:
            space (dict): diccionario con los hiperparametros

        Returns:
            dict: diccionario con el score y el status
        """
        et = ExtraTreesRegressor(n_estimators=space['n_estimators'], max_depth=space['max_depth'], min_samples_split=space['min_samples_split'], min_samples_leaf=space['min_samples_leaf'], bootstrap=space['bootstrap'], random_state=RANDOM_STATE)
        score = cross_val_score(et, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)

    print("best {}".format(best))

    best=space_eval(space, best)
    et = ExtraTreesRegressor(random_state=RANDOM_STATE,max_depth=best['max_depth'], min_samples_leaf=best['min_samples_leaf'], min_samples_split=best['min_samples_split'], n_estimators=best['n_estimators'] )
    et.fit(X_train, y_train)
    r2, rmse=prediccion_regr(et, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['ExtraTreesRegressor_opt']=hiper

    return modelos
    

def boosting_regr(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, lambdas: list) -> dict:
    """Funcion para optimizar los hiperparametros de los modelos de boosting

    Args:
        X_train (pd.DataFrame): dataframe con las variables predictoras de entrenamiento
        y_train (pd.DataFrame): dataframe con la variable objetivo de entrenamiento
        X_test (pd.DataFrame): dataframe con las variables predictoras de test
        y_test (pd.DataFrame): dataframe con la variable objetivo de test
        lambdas (list): lista con los valores de lambda

    Returns:
        dict: diccionario con los modelos y sus hiperparametros
    """
    # ADABOOST

    ada = AdaBoostRegressor(random_state=RANDOM_STATE)
    ada.fit(X_train, y_train)
    r2, rmse=prediccion_regr(ada, X_test, y_test, lambdas)
    modelos=dict()
    hiper={"R2":r2, "RMSE":rmse}
    modelos['Adaboost_sin_opt']=hiper


    space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1),
    'loss': hp.choice('loss', ['linear', 'square', 'exponential'])
    }

    def objective(space):
        ada = AdaBoostRegressor(n_estimators=space['n_estimators'], learning_rate=space['learning_rate'], loss=space['loss'], random_state=RANDOM_STATE)
        score = cross_val_score(ada, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)

    print("best {}".format(best))

    best = space_eval(space, best)

    ada = AdaBoostRegressor(n_estimators=best['n_estimators'], learning_rate=best['learning_rate'], loss=best['loss'], random_state=RANDOM_STATE)
    ada.fit(X_train, y_train)
    r2, rmse=prediccion_regr(ada, X_test, y_test, lambdas)

    hiper={"R2":r2, "RMSE":rmse}
    modelos['Adaboost_opt']=hiper

    # Gradient Descent Boosting

    params = {
        "n_estimators": 1000,
    }

    gb = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    gb.fit(X_train, y_train)
    r2,rmse=prediccion_regr(gb, X_test, y_test, lambdas)

    hiper={"R2":r2, "RMSE":rmse}
    modelos['GradientBoostingRegressor_sin_opt']=hiper

    
    space = { 
    'loss':hp.choice('loss', ['squared_error', 'absolute_error']),
    'n_estimators': hp.quniform('n_esimator', 10, 100,10),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'min_samples_split':hp.quniform('min_samples_split', 2,30,4),
    'min_samples_leaf':hp.quniform('min_samples_leaf', 2,21,3)
    }
    def objective(space):
        model = GradientBoostingRegressor( loss=space['loss'],n_estimators= int(round(space['n_estimators'])), learning_rate=space['learning_rate'],
                                        min_samples_split=round(space['min_samples_split']) , min_samples_leaf=int(round(space['min_samples_leaf'])))
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    
    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 10, 
    trials= trials)
    print("best: {}".format(best))

    best_rf = space_eval(space, best)

    et = GradientBoostingRegressor(loss=best_rf['loss'],n_estimators= round(best_rf['n_estimators']), learning_rate=best_rf['learning_rate'],
                                        min_samples_split=round(best_rf['min_samples_split']) , min_samples_leaf=round(best_rf['min_samples_leaf']))
    et.fit(X_train, y_train)
    r2,rmse=prediccion_regr(et, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['GradientBoostingRegressor_opt']=hiper

    # Hist Gradient Boosting

    gb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    r2,rmse=prediccion_regr(gb, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['HistGradientBoostingRegressor_sin_opt']=hiper

    space = { 
    'loss':hp.choice('loss', ['squared_error', 'absolute_error']),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'max_leaf_nodes':hp.quniform('max_leaf_nodes', 5,40,5),
    'min_samples_leaf':hp.quniform('min_samples_leaf', 2,21,3)
    }
    def objective(space):
        model = HistGradientBoostingRegressor( loss=space['loss'], learning_rate=space['learning_rate'],
                                        max_leaf_nodes=round(space['max_leaf_nodes']) , min_samples_leaf=int(round(space['min_samples_leaf'])))
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 10, 
    trials= trials)
    print("best: {}".format(best))

    best_rf = space_eval(space, best)

    gb = HistGradientBoostingRegressor(loss=best_rf['loss'], learning_rate=best_rf['learning_rate'],
                                        max_leaf_nodes=round(best_rf['max_leaf_nodes']) , min_samples_leaf=round(best_rf['min_samples_leaf']))
    gb.fit(X_train, y_train)
    r2,rmse=prediccion_regr(gb, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['HistGradientBoostingRegressor_opt']=hiper

    return modelos


def advanced_boosting_regr(X_train,y_train, X_test,y_test, lambdas):
    
    # XGBOOST

    xgb = XGBRegressor(random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    r2, rmse=prediccion_regr(xgb, X_test, y_test, lambdas)
    modelos=dict()
    hiper={"R2":r2, "RMSE":rmse}
    modelos['XGBOOST_sin_opt']=hiper

    space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1),
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
    'gamma': hp.uniform('gamma', 0.01, 1),
    'subsample': hp.uniform('subsample', 0.01, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.01, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 1)
    }

    def objective(space):
        xgb = XGBRegressor(n_estimators=space['n_estimators'], learning_rate=space['learning_rate'], max_depth=space['max_depth'], min_child_weight=space['min_child_weight'], gamma=space['gamma'], subsample=space['subsample'], colsample_bytree=space['colsample_bytree'], reg_alpha=space['reg_alpha'], reg_lambda=space['reg_lambda'], random_state=RANDOM_STATE)
        score = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)

    print("best {}".format(best))


    best = space_eval(space, best)

    xgb = XGBRegressor(n_estimators=best['n_estimators'], learning_rate=best['learning_rate'], max_depth=best['max_depth'], min_child_weight=best['min_child_weight'], gamma=best['gamma'], subsample=best['subsample'], colsample_bytree=best['colsample_bytree'], reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'], random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    r2, rmse=prediccion_regr(xgb, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['XGBOOST_opt']=hiper    


    # CATBOOST


    cat = CatBoostRegressor(random_state=RANDOM_STATE, silent=True)
    cat.fit(X_train, y_train)
    r2, rmse=prediccion_regr(cat, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['CATBOOST_sin_opt']=hiper   

    space = {
    'n_estimators': hp.choice('n_estimators', range(50, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1),
    'depth': hp.choice('depth', range(1, 10)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.01, 1),
    'bagging_temperature': hp.uniform('bagging_temperature', 0.001, 1),
    'random_strength': hp.uniform('random_strength', 0.001, 1)
    }

    def objective(space):
        cat = CatBoostRegressor(n_estimators=space['n_estimators'], learning_rate=space['learning_rate'], depth=space['depth'], l2_leaf_reg=space['l2_leaf_reg'], bagging_temperature=space['bagging_temperature'], random_strength=space['random_strength'], random_state=RANDOM_STATE, silent=True)
        score = cross_val_score(cat, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)

    print("best {}".format(best))

    best = space_eval(space, best)

    cat = CatBoostRegressor(n_estimators=best['n_estimators'], 
                            learning_rate=best['learning_rate'], 
                            depth=best['depth'], 
                            l2_leaf_reg=best['l2_leaf_reg'], 
                            bagging_temperature=best['bagging_temperature'], 
                            random_strength=best['random_strength'], 
                            random_state=RANDOM_STATE,
                            silent=True)
    cat.fit(X_train, y_train)
    r2, rmse= prediccion_regr(cat, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['CATBOOST_opt']=hiper 

    # LIGHT GBM

    lg = LGBMRegressor(random_state=RANDOM_STATE, silent=True, objective='regression')
    lg.fit(X_train, y_train)
    r2, rmse=prediccion_regr(lg, X_test, y_test, lambdas)
    hiper={"R2":r2, "RMSE":rmse}
    modelos['LIGHTGBM_sin_opt']=hiper 

    space = { #criterion 
    'num_iterations': hp.uniform('num_iterations', 1, 50),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'max_depth' :hp.quniform('max_depth', 3,12,3),
    'subsample':hp.quniform('subsample', 0.3,1,0.1),
    'feature_fraction':hp.quniform('feature_fraction', 0.3,1,0.1)
    }
    def objective(space):
        model = LGBMRegressor( num_iterations= round(space['num_iterations']), learning_rate=space['learning_rate'],max_depth=round(space['max_depth']),
                            subsample=space['subsample'],feature_fraction=space['feature_fraction'], random_state=0, silent=True )
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 20, 
    trials= trials)
    print("best: {}".format(best))

    best = space_eval(space, best)

    lg = LGBMRegressor(feature_fraction=best['feature_fraction'],
                            learning_rate=best['learning_rate'],
                            max_depth=round(best['max_depth']),
                            num_iterations=round(best['num_iterations']),
                            subsample=best['subsample'],
                            random_state=RANDOM_STATE,
                            silent=True)
    lg.fit(X_train, y_train)
    r2, rmse=prediccion_regr(lg, X_test, y_test, lambdas)

    hiper={"R2":r2, "RMSE":rmse}
    modelos['LIGHTGBM_opt']=hiper 

    return modelos








    
























#### CLASIFICACION


def vars_correlacion_clasif(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Calcula las variables con una correlacion mayor a un umbral

    Args:
        df (pd.DataFrame): dataframe con las variables
        umbral (float): umbral de correlacion para seleccionar las variables

    Returns:
        df_final (pd.DataFrame): dataframe con las variables con una correlacion mayor a un umbral
    """
    correlaciones=df.corr()
    correlaciones=correlaciones.loc[:,"Porcentaje_adquisicion_cat"].sort_values(ascending=False)
    corr_abs = correlaciones.abs()
    high_corr = corr_abs[corr_abs > umbral].reset_index()
    high_corr.columns = ['Variable', 'Correlation']
    variable_list = pd.Series(high_corr['Variable'].unique().tolist()).unique().tolist()
    print(f'Hay {len(variable_list)} variables con una correlacion mayor a {umbral}')
    df_final=df[variable_list]

    return df_final


def vars_importantes_rf_clasif(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Calcula las variables con una importancia mayor a un umbral
    
    Args:
        df (pd.DataFrame): dataframe con las variables
        umbral (float): umbral de importancia para seleccionar las variables

    Returns:
        df_final (pd.DataFrame): dataframe con las variables con una importancia mayor a un umbral
    """

    independientes = df.drop(['Porcentaje_adquisicion_cat'], axis=1)
    dependiente = df['Porcentaje_adquisicion_cat']

    rf=RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(independientes, dependiente)

    print(dict(zip(independientes.columns, rf.feature_importances_.round(2))))

    # plot de las variables en orden descendente
    forest_importances=pd.Series(rf.feature_importances_, index=independientes.columns).sort_values(ascending=True)
    plt.figure(figsize=(10,10))
    plt.barh(forest_importances.index, forest_importances)
    plt.xlabel("Random Forest Feature Importance")

    # crear dataframe con importancia de caracteristicas
    forest_importances=forest_importances.to_frame().reset_index()
    forest_importances.columns=['feature', 'importance']

    importantes= list(forest_importances[forest_importances['importance'] > umbral]['feature'])
    print(f"Hay {len(importantes)} variables importantes con el umbral {umbral}")
    importantes.append('Porcentaje_adquisicion_cat')
    df_final=df[importantes]

    return df_final



def normalizacion_clasif(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las variables de un dataframe

    Args:
        df (pd.DataFrame): dataframe con las variables

    Returns:
        df1 (pd.DataFrame): dataframe df1 con las variables normalizadas
        df2 (pd.DataFrame): dataframe df2 con las variables normalizadas
    """
    df=df.reset_index()
    target=df[['Porcentaje_adquisicion_cat', 'Codigo_NIF']]
    df=df.drop(['Porcentaje_adquisicion_cat','Codigo_NIF'], axis=1)
    scaler = StandardScaler()
    normalizer = MinMaxScaler()
    
    df1 = scaler.fit_transform(df)
    df2=normalizer.fit_transform(df)
 
    df1 = pd.DataFrame(df1, columns=df.columns)
    df1['Porcentaje_adquisicion_cat']=target['Porcentaje_adquisicion_cat']
    df1['Codigo_NIF']=target['Codigo_NIF']
    df1.set_index('Codigo_NIF', inplace=True)

    df2 = pd.DataFrame(df2, columns=df.columns)
    df2['Porcentaje_adquisicion_cat']=target['Porcentaje_adquisicion_cat']
    df2['Codigo_NIF']=target['Codigo_NIF']
    df2.set_index('Codigo_NIF', inplace=True)

    return df1, df2


def modelo_simple_clasif(modelo, X_train, y_train, X_test, y_test):
    print(str(modelo).split('(')[0])
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(modelo.__class__.__name__, ':', round(accuracy_score(y_test, y_pred),4), '\n', confusion_matrix(y_test, y_pred),'\n', '-'*30)
    print('\n')
    return modelo, round(accuracy_score(y_test, y_pred),4)


def probar_dataset_clasif(df, balanceo):
    X = df.drop(columns=['Porcentaje_adquisicion_cat'])
    y = df['Porcentaje_adquisicion_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_train, y_train = balanceo.fit_resample(X_train, y_train)

    print("            ")

    # Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    modelo_rf,ac_rf = modelo_simple_clasif(rf, X_train, y_train, X_test, y_test)

    # Linear Regression
    sgcd = SGDClassifier(random_state=RANDOM_STATE)
    modelo_sgcd,ac_sgcd = modelo_simple_clasif(sgcd, X_train, y_train, X_test, y_test)

    # Support Vector Clasifier
    svc = SVC(random_state=RANDOM_STATE)
    modelo_svc,ac_svc = modelo_simple_clasif(svc, X_train, y_train, X_test, y_test)

    # KNN classifier

    knn=KNeighborsClassifier()
    modelo_knn,ac_knn = modelo_simple_clasif(knn, X_train, y_train, X_test, y_test)

    #Decission Tree Classifier
    dtr=DecisionTreeClassifier(random_state=RANDOM_STATE)
    modelo_dtr,ac_dtr = modelo_simple_clasif(dtr, X_train, y_train, X_test, y_test)

    #Gaussian NB
    gnb=GaussianNB()
    modelo_gnb,ac_gnb = modelo_simple_clasif(gnb, X_train, y_train, X_test, y_test)



    # Mejor modelo de todos según accuracy
    mejor_modelo = np.argmax([ac_rf, ac_sgcd, ac_svc, ac_knn, ac_dtr, ac_gnb])
    print('Mejor modelo:', [rf, sgcd, svc, knn, dtr, gnb][mejor_modelo],', que tiene un accuracy de:', [ac_rf, ac_sgcd, ac_svc, ac_knn, ac_dtr, ac_gnb][mejor_modelo])
    print("              ")
    

    return None



def entrenamiento_clasif(xtr:pd.DataFrame, ytr:pd.DataFrame, models_list:list, model_hyperparameters:dict)-> pd.DataFrame:
    model_keys=list(model_hyperparameters.keys())

    result = []
    i=0

    for model in models_list:
       key=model_keys[i]
       i+=1
       params = model_hyperparameters[key]

       classifier=GridSearchCV(model,params,cv=5,scoring='accuracy',refit=True)

       classifier.fit(xtr,ytr)
       result.append({
        'model_used':model,
        'highest_score':classifier.best_score_,
        'best hyperparameters':classifier.best_params_,
       })
       print(result[i-1])



def prediccion_clasif(modelo, xtest, ytest):
    predicciones=modelo.predict(xtest)
    print("Classification report for classifier %s:\n%s\n"
      % (modelo, classification_report(ytest, predicciones, digits=4)))
    print("Confusion matrix:\n%s" % confusion_matrix(ytest, predicciones))


    cm = confusion_matrix(ytest, predicciones, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])

    disp.plot()
    plt.show()

    acc=accuracy_score(ytest, predicciones)

    return acc


def transformacion_clasif(df_clasif, df_clasif_prefusion):

    df_clasif['Porcentaje_adquisicion_cat']=df_clasif['Porcentaje_adquisicion_cat'].astype('category')
    df_clasif['Porcentaje_adquisicion_cat']=df_clasif['Porcentaje_adquisicion_cat'].cat.rename_categories([0,1,2])
    df_clasif['Porcentaje_adquisicion_cat'].value_counts()

    df_clasif_prefusion['Porcentaje_adquisicion_cat']=df_clasif_prefusion['Porcentaje_adquisicion_cat'].astype('category')
    df_clasif_prefusion['Porcentaje_adquisicion_cat']=df_clasif_prefusion['Porcentaje_adquisicion_cat'].cat.rename_categories([0,1,2])
    df_clasif_prefusion['Porcentaje_adquisicion_cat'].value_counts()

    df_clasif.set_index("Codigo_NIF", inplace=True)
    df_clasif_prefusion = df_clasif_prefusion.drop(columns=['first_funding_date', 'last_funding_date','valuation_2022', 'Free capital mil EUR'])
    df_clasif = df_clasif.drop(['index','ownerships'], axis=1)

    df_clasif=quitar_variables_ident(df_clasif)
    df_clasif_prefusion=quitar_variables_ident(df_clasif_prefusion)

    df_clasif = label_encoder_categoricas(df_clasif)
    df_clasif_prefusion=label_encoder_categoricas(df_clasif_prefusion)

    return df_clasif, df_clasif_prefusion



def seleccion_variables_clasif(df_clasif, df_clasif_prefusion):
    df_clasif_cor=vars_correlacion_clasif(df_clasif, 0.04)
    df_clasif_pref_cor=vars_correlacion_clasif(df_clasif_prefusion, 0.08)

    df_clasif_imp=vars_importantes_rf_clasif(df_clasif,0.017)
    df_clasif_pref_imp=vars_importantes_rf_clasif(df_clasif_prefusion,0.017 )

    df_clasif_ratio = df_clasif.filter(regex='^(ratio|Porcentaje)')
    print(f'Contiene {df_clasif_ratio.shape[1]} columnas')
    df_clasif_pref_ratio = df_clasif_prefusion.filter(regex='^(ratio|Porcentaje)')
    print(f'Contiene {df_clasif_pref_ratio.shape[1]} columnas')

    df_originales=[df_clasif_cor, df_clasif_pref_cor, df_clasif_imp, df_clasif_pref_imp, df_clasif_ratio, df_clasif_pref_ratio ]
    df_originales_nombre=['df_clasif_cor', 'df_clasif_pref_cor', 'df_clasif_imp', 'df_clasif_pref_imp', 'df_clasif_ratio', 'df_clasif_pref_ratio']

    
    df_clasif_cor_esc, df_clasif_cor_nor = normalizacion_clasif(df_clasif_cor)
    df_clasif_pref_cor_esc, df_clasif_pref_cor_nor = normalizacion_clasif(df_clasif_pref_cor)
    df_clasif_imp_esc, df_clasif_imp_nor = normalizacion_clasif(df_clasif_imp)
    df_clasif_pref_imp_esc, df_clasif_pref_imp_nor = normalizacion_clasif(df_clasif_pref_imp)
    df_clasif_ratio_esc, df_clasif_ratio_nor = normalizacion_clasif(df_clasif_ratio)
    df_clasif_pref_ratio_esc, df_clasif_pref_ratio_nor = normalizacion_clasif(df_clasif_pref_ratio)

    df_lista=[df_clasif_cor_esc, df_clasif_cor_nor,df_clasif_pref_cor_esc, df_clasif_pref_cor_nor,df_clasif_imp_esc, df_clasif_imp_nor,df_clasif_pref_imp_esc, df_clasif_pref_imp_nor,
            df_clasif_ratio_esc, df_clasif_ratio_nor,df_clasif_pref_ratio_esc, df_clasif_pref_ratio_nor]

    df_lista_nombre=['df_clasif_cor_esc', 'df_clasif_cor_nor','df_clasif_pref_cor_esc', 'df_clasif_pref_cor_nor','df_clasif_imp_esc', 'df_clasif_imp_nor','df_clasif_pref_imp_esc', 'df_clasif_pref_imp_nor',
            'df_clasif_ratio_esc', 'df_clasif_ratio_nor','df_clasif_pref_ratio_esc', 'df_clasif_pref_ratio_nor']

    warnings.filterwarnings('ignore')

    i=0
    for df in df_originales:
        print(df_originales_nombre[i])
        probar_dataset_clasif(df, SMOTEENN(random_state=0))
        i+=1

    i=0
    for df in df_lista:
        print(df_lista_nombre[i])
        probar_dataset_clasif(df, SMOTEENN(random_state=0))
        i+=1

    print(f'El mejor dataset para clasificación de los probados es: {"df_clasif_cor"}')

    return df_clasif_cor

def separar_train_tets_clasif(df_clasif_cor):
        X = df_clasif_cor.drop(columns=['Porcentaje_adquisicion_cat'])
        y = df_clasif_cor['Porcentaje_adquisicion_cat']

        X, y = SMOTEENN(random_state=0).fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test


def modelo_automatico_clasif(X_train,y_train ,X_test,y_test):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
    model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=RANDOM_STATE, n_jobs=7)
    # perform the search
    model.fit(X_train, y_train)

    acc=prediccion_clasif(model, X_test, y_test)

    return acc

def modelos_simples(X_train,y_train ,X_test,y_test):

    model_list=[KNeighborsClassifier(), DecisionTreeClassifier(random_state=0), SVC(),  SGDClassifier()]

    model_hyperparameters={"kn":{
        "n_neighbors": [3,5,7],
        "leaf_size": [15,30],
        'weights':['uniform', 'distance']
    },
    "tree_clas":{
        "max_depth":[None, 3, 5,10],
        'min_samples_split':[0,2,5,10]
    },
    "svc":{
        'kernel':['linear', 'rbf'],
        "C":[1,3,5],
        "gamma":["scale", "auto"]
    },


    'sgdc':{
        "loss":['squared_error', 'log']
    }
    }

    entrenamiento_clasif(xtr=X_train, ytr=y_train, models_list=model_list, model_hyperparameters=model_hyperparameters)

    sv=SVC(C= 3, gamma= 'scale', kernel= 'rbf')
    modelo,acc=modelo_simple_clasif(tw, X_train, y_train, X_test, y_test)

    return modelo, acc

def stacking_clasif(X_train,y_train, X_test,y_test):

    # level 0
    level0 = list()
    level0.append(('lr', LogisticRegression(random_state=RANDOM_STATE)))
    level0.append(('svm', SVC(random_state=RANDOM_STATE)))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('dt', DecisionTreeClassifier(random_state=RANDOM_STATE)))
    level0.append(('sgd', SGDClassifier(random_state=RANDOM_STATE)))

    # level 1 - meta model
    level1 = LogisticRegression(random_state=RANDOM_STATE)

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)
    acc=prediccion_clasif(model, X_test, y_test)
    modelo=dict()
    modelo['stacking_sin_opt']=acc

    space = {
        "lr__C": hp.uniform("lr__C", 0.01, 1.0),
        "knn__n_neighbors" : hp.randint("knn__n_neighbors", 1, 11),
        "cart__max_depth" : hp.randint("cart__max_depth", 1, 4),
        "svm__C":  hp.uniform("svm__C", 0.01, 1.0),
        "final_estimator__solver": hp.choice("final_estimator__solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
        "final_estimator__C": hp.uniform("final_estimator__C", 0.01, 1.0),
        "final_estimator__class_weight": hp.choice("final_estimator__class_weight", ["balanced", None])}


    def objective(space):

        # level 0
        level0 = list()
        level0.append(('lr', LogisticRegression(random_state=RANDOM_STATE, C = space['lr__C'])))
        level0.append(('svm', SVC(random_state=RANDOM_STATE, C = space['svm__C'])))
        level0.append(('knn', KNeighborsClassifier(n_neighbors = space['knn__n_neighbors'])))
        level0.append(('dt', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth = space['cart__max_depth'])))

        # level 1 - meta model
        level1 = LogisticRegression(random_state=RANDOM_STATE, solver = space['final_estimator__solver'], C = space['final_estimator__C'], class_weight = space['final_estimator__class_weight'])

        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
        score = cross_val_score(model, X_train, y_train, cv=5).mean()
        return{'loss':-score, 'status':STATUS_OK}
    
    trials = Trials()
    best = fmin(fn= objective,
        space= space,
        algo= tpe.suggest,
        max_evals = 25,
        trials= trials)
    print("best: {}".format(best))

    best_stacking = space_eval(space, best)
    # level 0
    level0 = list()
    level0.append(('lr', LogisticRegression(random_state=RANDOM_STATE, C = best_stacking['lr__C'])))
    level0.append(('svm', SVC(random_state=RANDOM_STATE, C = best_stacking['svm__C'])))
    level0.append(('knn', KNeighborsClassifier(n_neighbors = best_stacking['knn__n_neighbors'])))
    level0.append(('dt', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth = best_stacking['cart__max_depth'])))

    # level 1 - meta model
    level1 = LogisticRegression(random_state=RANDOM_STATE, solver = best_stacking['final_estimator__solver'], C = best_stacking['final_estimator__C'], class_weight = best_stacking['final_estimator__class_weight'])

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)
    acc=prediccion_clasif(model, X_test, y_test)
    modelo['stacking_opt']=acc

    return modelo

def bagging_clasif(X_train,y_train, X_test,y_test):

    logistic_regression = LogisticRegression(random_state=RANDOM_STATE)
    svc = SVC(random_state=RANDOM_STATE)
    knn = KNeighborsClassifier()
    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    sgd=SGDClassifier(random_state=RANDOM_STATE)
    gaussian_nb = GaussianNB()


    models = [logistic_regression, knn, decision_tree, sgd, gaussian_nb, svc]
    for model in models:
    
        bc = BaggingClassifier(random_state=RANDOM_STATE, base_estimator=model)
        bc.fit(X_train, y_train)
        predictions = bc.predict(X_test)

        print(model.__class__.__name__, ':', round(accuracy_score(y_test, predictions),4), '\n', confusion_matrix(y_test, predictions),'\n', '-'*30)

    # random forest

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    acc=prediccion_clasif(rf, X_test, y_test)
    modelo=dict()
    modelo['random_forest_sin_opt']=acc

    space = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'max_depth': hp.quniform('max_depth', 10, 1200, 10),
    'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
    'n_estimators' : hp.randint('n_estimators', 0, 2000)
    }

    def objective(space):
    
        model = RandomForestClassifier(criterion = space['criterion'], 
            #max_depth = space['max_depth'],
            max_features = space['max_features'],
            min_samples_leaf = space['min_samples_leaf'],
            min_samples_split = space['min_samples_split'],
            n_estimators = space['n_estimators'])
            
        accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials = Trials()
    best = fmin(fn= objective,
        space= space,
        algo= tpe.suggest,
        max_evals = 25,
        trials= trials)
    print("best: {}".format(best))

    best_rf = space_eval(space, best)


    rf = RandomForestClassifier(criterion=best_rf['criterion'], 
            max_depth=int(best_rf['max_depth']), 
            max_features=best_rf['max_features'], 
            min_samples_leaf=best_rf['min_samples_leaf'], 
            min_samples_split=best_rf['min_samples_split'], 
            n_estimators=best_rf['n_estimators'])

    rf.fit(X_train, y_train)

    acc=prediccion_clasif(rf, X_test, y_test)

    modelo['random_forest_opt']=acc

    # extra randomised trees

    et = ExtraTreesClassifier(random_state=RANDOM_STATE, )
    et.fit(X_train, y_train)
    acc= prediccion_clasif(et, X_test, y_test)
    modelo['ExtraTreesClassifier_sin_opt']=modelo

    space = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'max_depth': hp.quniform('max_depth', 10, 1200, 10),
    'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
    'n_estimators' : hp.randint('n_estimators', 0, 2000)
    }

    def objective(space):

        model = ExtraTreesClassifier(criterion = space['criterion'], 
            #max_depth = space['max_depth'],
            max_features = space['max_features'],
            min_samples_leaf = space['min_samples_leaf'],
            min_samples_split = space['min_samples_split'],
            n_estimators = space['n_estimators'])
            
        accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials = Trials()
    best = fmin(fn= objective,
        space= space,
        algo= tpe.suggest,
        max_evals = 25,
        trials= trials)
    print("best: {}".format(best))

    best_rf = space_eval(space, best)


    rf = ExtraTreesClassifier(criterion=best_rf['criterion'], 
            max_depth=int(best_rf['max_depth']), 
            max_features=best_rf['max_features'], 
            min_samples_leaf=best_rf['min_samples_leaf'], 
            min_samples_split=best_rf['min_samples_split'], 
            n_estimators=best_rf['n_estimators'])

    rf.fit(X_train, y_train)

    acc=prediccion_clasif(rf, X_test, y_test)
    modelo['ExtraTreesClassifier_opt']=acc


    return modelo


def boosting_clasif(X_train,y_train, X_test,y_test):

    # adaboost
    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    ada.fit(X_train, y_train)
    acc=prediccion_clasif(ada, X_test, y_test)
    modelo=dict()
    modelo['adaboost_sin_opt']=acc


    space = {'n_estimators': hp.quniform('n_estimators', 10, 1200, 10),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1),
    }

    def objective(space):

        model = AdaBoostClassifier(n_estimators = int(space['n_estimators']),
            learning_rate = space['learning_rate'])
            
        accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

        return {'loss': -accuracy, 'status': STATUS_OK }

    trials = Trials()
    best = fmin(fn= objective,
        space= space,
        algo= tpe.suggest,
        max_evals = 25,
        trials= trials)

    print("best: {}".format(best))

    best=space_eval(space, best)

    ada = AdaBoostClassifier(n_estimators=int(best['n_estimators']),
            learning_rate=best['learning_rate'])

    ada.fit(X_train, y_train)
    acc=prediccion_clasif(ada, X_test, y_test)
    modelo['adaboost_opt']=acc

    # Gradient descent boosting
    params = {
        "n_estimators": 1000,
    }

    gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gbc.fit(X_train, y_train)
    acc=prediccion_clasif(gbc, X_test, y_test)
    modelo['GradientBoostingClassifier_sin_opt']=acc

    
    space = { 
    'n_estimators': hp.quniform('n_esimator', 10, 100,10),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'min_samples_split':hp.quniform('min_samples_split', 2,30,4),
    'min_samples_leaf':hp.quniform('min_samples_leaf', 2,21,3)
    }
    def objective(space):
        model = GradientBoostingClassifier(n_estimators= int(round(space['n_estimators'])), learning_rate=space['learning_rate'],
                                        min_samples_split=round(space['min_samples_split']) , min_samples_leaf=int(round(space['min_samples_leaf'])))
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    

    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 10, 
    trials= trials)
    print("best: {}".format(best))

    best_rf = space_eval(space, best)

    et = GradientBoostingClassifier(n_estimators= round(best_rf['n_estimators']), learning_rate=best_rf['learning_rate'],
                                        min_samples_split=round(best_rf['min_samples_split']) , min_samples_leaf=round(best_rf['min_samples_leaf']))
    et.fit(X_train, y_train)
    acc=prediccion_clasif(et, X_test, y_test)
    modelo['GradientBoostingClassifier_sin_opt']=acc

    params = {
        "n_estimators": 1000,
    }

    # hist gradinet boosting

    gbc = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    gbc.fit(X_train, y_train)
    acc=prediccion_clasif(gbc, X_test, y_test)
    modelo['HistGradientBoostingClassifier_sin_opt']=acc

    
    space = { 
    'n_estimators': hp.quniform('n_esimator', 10, 100,10),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'max_depth': hp.quniform('max_depth', 10, 1200, 10),
    'min_samples_leaf':hp.quniform('min_samples_leaf', 1,21,3)
    }
    def objective(space):
        model = GradientBoostingClassifier( n_estimators= int(round(space['n_estimators'])), learning_rate=space['learning_rate'],
                                        max_depth=round(space['max_depth']) , min_samples_leaf=int(round(space['min_samples_leaf'])))
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 10, 
    trials= trials)
    print("best: {}".format(best))

    best_gb = space_eval(space, best)

    gb = HistGradientBoostingClassifier(n_estimators= round(best_gb['n_estimators']), learning_rate=best_gb['learning_rate'],
                                            min_samples_leaf=round(best_gb['min_samples_leaf']))
    gb.fit(X_train, y_train)
    acc=prediccion_clasif(gb, X_test, y_test)
    modelo['HistGradientBoostingClassifier_opt']=acc

    return modelo 


def advanced_boosting_clasif(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Modelo de clasificación avanzado

    Args:
        X_train (pd.DataFrame): datos de entrenamiento
        y_train (pd.DataFrame): etiquetas de entrenamiento
        X_test (pd.DataFrame): datos de test
        y_test (pd.DataFrame): etiquetas de test

    Returns:
        modelo (dict): diccionario con los modelos y sus respectivas métricas
    """
    # XGBoost
    xgb = XGBClassifier(random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    acc=prediccion_clasif(xgb, X_test, y_test)
    modelo=dict()
    modelo['xgboost_sin_opt']=acc

    # Optimized XGBoost
    space = {'n_estimators': hp.quniform('n_estimators', 10, 1200, 10),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1),
    'max_depth': hp.quniform('max_depth', 10, 1200, 10),
    'min_child_weight': hp.uniform('min_child_weight', 0, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'subsample': hp.uniform('subsample', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
    }
        #X = df_clasif_imp.drop(columns=['Porcentaje_adquisicion_cat'])
        #y = df_clasif_imp['Porcentaje_adquisicion_cat']

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #X_train, y_train = space['balanceo'].fit_resample(X_train, y_train)
    def objective(space: dict) -> dict:
        """Optimización de hiperparámetros

        Args:
            space (dict): diccionario con los hiperparámetros

        Returns:
            dict: diccionario con el valor de la métrica y el estado de la optimización
        """
        model = XGBClassifier(n_estimators = int(space['n_estimators']),
            learning_rate = space['learning_rate'],
            max_depth = int(space['max_depth']),
            min_child_weight = space['min_child_weight'],
            gamma = space['gamma'],
            subsample = space['subsample'],
            colsample_bytree = space['colsample_bytree'],
            reg_alpha = space['reg_alpha'],
            reg_lambda = space['reg_lambda'])
            
        accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials = Trials()
    best = fmin(fn= objective,
        space= space,
        algo= tpe.suggest,
        max_evals = 25,
        trials= trials)

    print("best: {}".format(best))

    best=space_eval(space, best)

    xgb = XGBClassifier(n_estimators=int(best['n_estimators']),
            learning_rate=best['learning_rate'],
            max_depth=int(best['max_depth']),
            min_child_weight=best['min_child_weight'],
            gamma=best['gamma'],
            subsample=best['subsample'],
            colsample_bytree=best['colsample_bytree'],
            reg_alpha=best['reg_alpha'],
            reg_lambda=best['reg_lambda'])

    xgb.fit(X_train, y_train)
    acc=prediccion_clasif(xgb, X_test, y_test)
    modelo['xgboost_opt']=acc

    # CatBoost
    cat = CatBoostClassifier(random_state=RANDOM_STATE,  silent=True)
    cat.fit(X_train, y_train)
    acc=prediccion_clasif(cat, X_test, y_test)
    modelo['catboost']=acc
    
    # Light gbm
    lgbm = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    lgbm.fit(X_train, y_train)
    acc=prediccion_clasif(lgbm, X_test, y_test)
    modelo['lightgbm_sinopt']=acc


    space = { 
    'num_iterations': hp.uniform('num_iterations', 1, 50),
    'learning_rate': hp.quniform('learning_rate', 0.1,1,0.1),
    'max_depth' :hp.quniform('max_depth', 3,12,3),
    'subsample':hp.quniform('subsample', 0.3,1,0.1),
    'feature_fraction':hp.quniform('feature_fraction', 0.3,1,0.1)
    }
    def objective(space: dict) -> dict:
        """Optimización de hiperparámetros

        Args:
            space (dict): diccionario con los hiperparámetros

        Returns:
            dict: diccionario con el valor de la métrica y el estado de la optimización
        """
        model = lgb.LGBMClassifier( num_iterations= round(space['num_iterations']), learning_rate=space['learning_rate'],max_depth=round(space['max_depth']),
                            subsample=space['subsample'],feature_fraction=space['feature_fraction'], random_state=0, silent=True )
        accuracy = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_absolute_percentage_error').mean()

        return {'loss': abs(accuracy), 'status': STATUS_OK } 
    
    trials = Trials()
    best = fmin(fn= objective,
    space= space, 
    algo= tpe.suggest, 
    max_evals = 20, 
    trials= trials)
    print("best: {}".format(best))


    best = space_eval(space, best)

    lg = lgb.LGBMClassifier(feature_fraction=best['feature_fraction'],
                            learning_rate=best['learning_rate'],
                            max_depth=round(best['max_depth']),
                            num_iterations=round(best['num_iterations']),
                            subsample=best['subsample'],
                            random_state=RANDOM_STATE,
                            silent=True)
    lg.fit(X_train, y_train)
    acc=prediccion_clasif(lg, X_test, y_test)
    modelo['lightgbm_opt']=acc

    return modelo