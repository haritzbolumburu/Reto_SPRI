Reto7: Equipo Amarillo
==========
Reto desarrollado con Spri para la valoración de startups.

Proceso seguido
--------

El proyecto comienza importando los datos y realizando un análisis exploratorio inicial para entender las características y la distribución de los mismos. Luego, se limpian e imputan empleando KNN para las variables numéricas y Random Forest y Gradient Boosting para las variables categóricas.

A continuación, se emplean modelos de clasificación para predecir si una empresa será adquirida total o parcialmente, así como de regresión para predecir la valoración para el año 2022.

Los datos más relevantes de los análisis descriptivos así como los mejores modelos de predicción se desplegarán en una aplicación multifuncional desarrollada en Flask, que proveerá al usuario de información descriptiva y prescriptiva de las startups en general o de la empresa que él/ella elija.

Requerimientos de ejecución
------------

Para ejecutar este proyecto, es necesario importar el fichero "environment.yml" para crear un environment que ya contiene todos los paquetes necesarios para la ejecución del proyecto.

Se presupone que dentro del proyecto, existe la carpeta "Datos", con la carpeta "Originales" en ella, la cual incluye los ficheros con los datos originales. La carpeta "Limpios" se creará con la ejecución del proyecto, y recibirá sus ficheros de manera automática, así como las carpetas "Graficos" y "Modelos".

La estructura inicial debe ser la siguiente:
- Datos
  - Originales
    - df_dealroom_modif.xlsx
    - df_sabi_modif_1.xlsx
    - df_sabi_modif_2_new.xlsx
    - df_sabi_parte3.xlsx

Una vez que se tengan todos los paquetes instalados y todos los datos originales introducidos en la carpeta, se debe ejecutar en primer lugar el archivo principal del proyecto, llamado "main.py", para que se realice todo el preprocesamiento y análisis. Posteriormente, se ejecutarán los notebooks relacionados con los modelos de clasificación y regresión.

El código está bien documentado y se ha organizado en diferentes módulos y funciones para facilitar su comprensión, mantenimiento y despliegue.
