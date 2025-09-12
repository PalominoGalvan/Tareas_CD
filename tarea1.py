"""
 Tarea 1 - Intro Ciencia de Datos
    Jessica Rubí Lara Rosales  
    Luis Erick Palomino Galván 
    Eric Moreles
"""

# Directorio de trabajo (cambialo al tuyo)
from os import chdir
chdir("C:/Users/Rubi/Documents/Intro_Ciencia_Datos/tarea1")
   
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

# FORMA 1 
# Quitamos las primeras 10 columnas corresponden a características que en este momento no vamos a usar
df_n = df.drop(range(9))
print(df_n.head(25))
