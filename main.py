import os
import pandas as pd
import preprocesamiento as prepro
import modelado as model


# CONTROL EJECUCIÓN
PREPROCESAR = True

# PATHS
DATOS_ORIGINALES = os.path.join("Datos", "Originales")
DATOS_LIMPIOS = os.path.join("Datos", "Limpios")
MODELOS = os.path.join('Modelos')
GRAFICOS = os.path.join('Graficos')

RANDOM_STATE = 42

###############################################################

if not os.path.exists(DATOS_LIMPIOS):
    os.makedirs(DATOS_LIMPIOS)

if not os.path.exists(MODELOS):
    os.makedirs(MODELOS)

if not os.path.exists(GRAFICOS):
    os.makedirs(GRAFICOS)

####### PREPROCESAMIENTO #######

if PREPROCESAR:
    print("PREPROCESAMIENTO")

    # Carga de datos
    df1 = prepro.leer_excel(DATOS_ORIGINALES, 'df_sabi_modif_1.xlsx')
    df2 = prepro.leer_excel(DATOS_ORIGINALES, 'df_sabi_modif_2_new.xlsx')
    df2_2 = prepro.leer_excel(DATOS_ORIGINALES, 'df_sabi_parte3.xlsx')
    df3 = prepro.leer_excel(DATOS_ORIGINALES, 'df_dealroom_modif.xlsx')

    # Preprocesamiento inicial
    df1 = prepro.prepro_df1(df1)
    df2_final = prepro.prepro_df2(df2, df2_2) # incluye el análisis de colinealidad, imputación y selección de variables contables
    df3 = prepro.prepro_df3(df3)

    # Agrupación de dataframes
    dftot = prepro.agrupar_dataframes(df1, df2_final, df3)
    dftot.to_csv(os.path.join(DATOS_LIMPIOS, "para_outliers.csv"), index=False)

    # Creación de variables
    dftot = prepro.crear_variables(dftot)

    # Imputaciones
    dftot,vars_miss = prepro.imputacion_numericas(dftot)
    dftot = prepro.imputacion_categoricas(dftot, vars_miss)
    
    # Cálculo del Free Cash Flow
    dftot = prepro.calcular_fcf(dftot)
    dftot.to_csv(os.path.join(DATOS_LIMPIOS, "FCF.csv"), index=True)

    # Feature Selection
    df = prepro.feature_selection(dftot)
    
    # Normalización de datos
    dfa_1 = pd.read_csv(os.path.join(DATOS_LIMPIOS,'df_regr_prefusion.csv'))
    dfa_2 = pd.read_csv(os.path.join(DATOS_LIMPIOS,'df_clasif_prefusion.csv'))
    df = prepro.normalizar(dfa_1, dfa_2, df)

    # Fusionar datos de 2020 y 2021
    df_regr = prepro.fusionar_datos_media_ponderada(df, df2)
    df_clasif = prepro.fusionar_datos_diferencia(df, df2)

    # ESCRITURA DATOS LIMPIOS PARA MODELO

    # Clasificación
    df_clasif = df_clasif.drop(columns=['valuation_2022'])
    df_clasif.reset_index(inplace=True)
    df_clasif.to_csv(os.path.join(DATOS_LIMPIOS, "df_clasif.csv"))

    # Regresión
    df_regr = df_regr.drop(columns=['Porcentaje_adquisicion_cat'])
    df_regr = df_regr.dropna(subset=['valuation_2022'])
    df_regr.reset_index(inplace=True)
    df_regr.to_csv(os.path.join(DATOS_LIMPIOS, "df_regr.csv"))

    print("FIN DEL PREPROCESAMIENTO, DATOS LIMPIOS GUARDADOS")
    print('-'*80, '\n'*3, '-'*80)


