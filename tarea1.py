# -*- coding: utf-8 -*-
"""
 Tarea 1 - Intro Ciencia de Datos
    Jessica Rubí Lara Rosales  
    Luis Erick Palomino Galván 
    Eric Moreles
"""

# Directorio de trabajo (cambialo al tuyo)
from os import chdir
chdir("C:/Users/Rubi/Documents/Intro_Ciencia_Datos/tarea1/entregable")
   
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


#Separamos la metadata de la base de datos en una dataframe separada para mejor manejo
df_info = df.iloc[:9]

#print(df_info)

#Transponemos
df_info = df_info.transpose()

#Y fjiamos las columnas como el primer renglon, al mismo tiempo que desechamos ese renglon,
#esto tiene el efecto de que nuestras columnas ahora son todas las diferentes variables de informacion,
#i.e. "Site Code", "Site Name", etc.
df_info.columns = df_info.iloc[0]
df_info = df_info.drop(df_info.index[0])

#Limpiamos un poco los datos, pues por ejemplo los codigos de los sitios vienen con varios espacios en blanco
#y los valores numericos viene como strings
df_info.columns = ['Site name' if x=='Site  name' else x for x in df_info.columns]
df_info.index = df_info.index.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
df_info["First year CE"] = pd.to_numeric(df_info["First year CE"], errors="coerce").astype("Int64")
df_info["Last year CE"]  = pd.to_numeric(df_info["Last year CE"], errors="coerce").astype("Int64")
df_info["Latitude"] = pd.to_numeric(df_info["Latitude"], errors="coerce")
df_info["Longitude"] = pd.to_numeric(df_info["Longitude"], errors="coerce")
df_info["elevation a.s.l."] = pd.to_numeric(df_info["elevation a.s.l."], errors="coerce")

# Maneras de trabajar los datos
# ========= Forma 1 ============

# Quitamos las primeras 10 columnas corresponden a características que en este momento no vamos a usar
df_n = df.iloc[10:]
# indexamos por el año
df_n = df_n.set_index('Site Code')
df_n.index = df_n.index.astype(int)
df_n.columns = df_n.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
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
df_t = df_t.iloc[10:]

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

print("Porcentaje de datos faltantes en los rangos dados por First year y Last year de cada sitio")

#Como se explica en el print de arriba, este for calcula y guarda el porcentaje de datos faltantes
#relativo a el periodo de tiempo denotado por el rango dado
missing_percent = {} #Donde guardamos el porcentaje faltante
count = {} #Guardamos el numero total de datos

for sitio in df_n.columns: #Para los sitios
    sitio_info = df_info.loc[sitio] #Extraemos la informacion
    first_year = sitio_info["First year CE"]#Obtenemos cota menor y superior
    last_year = sitio_info["Last year CE"]

    sitio_datos = df_n.loc[(df_n.index >= first_year) & (df_n.index <= last_year), sitio] #De nuestra dataframe con los datos, extraemos los que estan en el rango

    missing_percent[sitio] = sitio_datos.isna().sum()/len(sitio_datos) * 100 #Calculamos el porcentaje faltante
    count[sitio] = last_year - first_year #Contamos cuantos datos son simplemente restando menor a mayor

missing_percent = pd.Series(missing_percent)
count = pd.Series(count)

for sitio in missing_percent.index:
    print(f"{sitio}: {missing_percent[sitio]:.2f}% faltantes, con {count[sitio]} datos")



# Boxplots 
# Convertimos todas las columnas a numéricas
plt.figure(figsize=(15,6))
plt.xticks(rotation=90)
plt.ylabel('13CVDB')
plt.title('Boxplot por sitio')
df_n = df_n.apply(pd.to_numeric, errors = 'coerce')

df_n.boxplot()


# Heatmap de datos faltantes
plt.figure(figsize=(12, 6))
sns.heatmap(df_n)
plt.xticks(rotation=90)
plt.xlabel('Sitios')
plt.ylabel('Año')
plt.title('Mapa de calor de valores faltantes (NaN)')
plt.tight_layout()
plt.show()


