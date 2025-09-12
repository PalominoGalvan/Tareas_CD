# -*- coding: utf-8 -*-
"""
 Tarea 1 - Intro Ciencia de Datos
    Jessica Rubí Lara Rosales  
    Luis Erick Palomino Galván 
    Eric Moreles
"""

# Directorio de trabajo (cambialo al tuyo)
from os import chdir
chdir("C:/Users/Rubi/Documents/Intro_Ciencia_Datos/tarea1/final")
   
# Cargar pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuracion inicial
pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
pd.set_option("display.max_rows", None)     # Mostrar todas las filas
pd.set_option("display.width", None)        # Ajuste automático al ancho de la terminal
pd.set_option("display.max_colwidth", None) # No truncar el contenido de celdas

# Cargamos archivo
file_path = "./data.csv" 


# Leemos el archivo CSV con pandas. 
df = pd.read_csv(file_path)
#print(df.head())

# Quitamos los primeros tres renglones porque no hay datos ahí
df = pd.read_csv(file_path,skiprows=3)
print(df.head(11))

# Maneras de trabajar los datos
# ========= Forma 1 ============

# Quitamos las primeras 10 columnas corresponden a características que en este momento no vamos a usar
df_n = df.drop(range(9))
# indexamos por el año
df_n = df_n.set_index('Site Code')
print(df_n.head(25))


# ========= Forma 2 ============

# Transponemos el DataFrame: usando como pivotal 'Site Code' 
df_t = df.set_index("Site Code").T

# si quieres codificar los sitios por numericos
# Reiniciamos el índice para que las etiquetas de muestra (BRO, CAV, CAZ, etc)
# no queden como índice, sino como una columna llamada "Sample".
#df_t = df_t.reset_index().rename(columns={"index": "Sample"})

print(df_t.head(25))

# Quitamos las primeras 10 columnas corresponden a características que en este momento no vamos a usar
df_t = df_t.drop(columns=['Site  name', 'Country', 'Latitude', 'Longitude', 'Species', 
                          'First year CE', 'Last year CE', 'elevation a.s.l.', 'Year CE'])



# Imprimimos los nombres de las columnas que nos quedan para confirmar la limpieza.
print(df_t.head())
print(df_t.columns)

# ================ Exploración inicial de datos ===============

# Valores faltantes NaN por sitio
print('Numero de datos faltantes por sitio')
print(df_t.isna().sum(axis = 1))

# porcentaje
# 2005 - 1600 + 1 = 406años -- 100%
print('Porcentaje de datos faltantes por sitio')
print(df_t.isna().sum(axis = 1) * 100/406)


# Valores faltantes NaN por año


nan_counts =df_n.isna().sum(axis = 1)
print('Numero de datos faltantes por año')
print(nan_counts )

#imprimimos solo los que les falta 9 o más
print(nan_counts[nan_counts >= 10]  )
print('Porcentaje de datos faltantes por año')
# porcentaje 25 -- 100%
print(nan_counts * 4)