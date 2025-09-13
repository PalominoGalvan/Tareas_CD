# -*- coding: utf-8 -*-
"""
 Tarea 1 - Intro Ciencia de Datos
    Jessica Rubí Lara Rosales  
    Luis Erick Palomino Galván 
    Eric Moreles
"""

# Directorio de trabajo (cambialo al tuyo)
from os import chdir
chdir("/home/erick-palomino/Introducción a Ciencia de Datos")

#cargar paquetes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from matplotlib.widgets import Slider
from matplotlib.ticker import MultipleLocator

# Configuracion inicial
pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
pd.set_option("display.max_rows", None)     # Mostrar todas las filas
pd.set_option("display.width", None)        # Ajuste automático al ancho de la terminal
pd.set_option("display.max_colwidth", None) # No truncar el contenido de celdas

# Cargamos archivo
file_path = "data.csv" 


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



# ===============================================================
# ===================== Exploración gráfica =====================
# ===============================================================


# ---------------------------------------------------------
# Regresion lineal para dectección de posibles outliers
# ---------------------------------------------------------

for columna in df_n.columns:
    
    #transformamos los indices de la fila en una nueva columna
    df_sub = df_n.reset_index().rename(columns={"index": "Sample"})
    #eliminamos todas las filas donde haya NaN
    df_sub = df_sub[['Site Code', columna]].dropna()
    
    
    # Seleccionamos los años (Site Code) como X y los datos de carbono 13 (columna) como Y
    # Antes de usar x e y
    df_sub[columna] = pd.to_numeric(df_sub[columna].astype(str).str.replace(",", "."), errors='coerce')
    df_sub = df_sub.dropna(subset=[columna, 'Site Code'])
    x = df_sub['Site Code'].to_numpy(dtype=float)
    y = df_sub[columna].to_numpy(dtype=float)


    # Añadir intercepto, el vector de unos de la regresion lineal
    X1 = np.column_stack([np.ones(x.shape[0]), x])
    
    # Calcular beta con mínimos cuadrados
    beta_hat, *_ = np.linalg.lstsq(X1, y, rcond=None)
    
    # Calcular matriz hat H
    H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    leverages = np.diag(H)
    
    # Regla práctica de corte
    n, p = X1.shape
    threshold = 2*p/n
    
    # Visualización
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, c="blue", alpha=0.6)
    plt.plot(x, X1 @ beta_hat, c="red")
    plt.xlabel("Años")
    plt.ylabel("δ13C")
    plt.title(columna)
    
    # Resaltar puntos con leverage alto
    outliers = leverages > threshold
    plt.scatter(x[outliers], y[outliers], facecolors="none", edgecolors="r", s=100, label="Posible outlier")

    #Ajustados
    y_fitted = X1 @ beta_hat

    #Residuales
    residuals = y - y_fitted

    #Desviación estandar de los residuales
    std_residuals = np.std(residuals)

    #Residuales estandarizados
    standardized_residuals = residuals / std_residuals

    #Barreras para detectar posibles outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(y_fitted, standardized_residuals, c="purple", alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='-', linewidth=2)
    plt.axhline(y=2, color='green', linestyle=':', linewidth=2, label='Sospechoso de ser outlier')
    plt.axhline(y=-2, color='green', linestyle=':', linewidth=2)
    plt.axhline(y=3, color='blue', linestyle=':', linewidth=2, label='Altamente probable de ser outlier')
    plt.axhline(y=-3, color='blue', linestyle=':', linewidth=2)
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Residuos Estandarizados")
    plt.title(f"Residuos Estandarizados vs. Valores Ajustados {columna}")
    plt.legend()


### CASO ESPECIAL POR LECTURA

# Reemplazar coma decimal -> punto decimal SOLO en los datos
df.iloc[1:, 1:] = df.iloc[1:, 1:].replace(",", ".", regex=True)

# ===================== Limpieza =====================

# Los metadatos son las primeras ~9 filas: 'Site name', 'Country', 'Latitude', etc.
metadata = df.iloc[:9].set_index("Site Code")
print("\nMetadata de los sitios:")
print(metadata)

# Los datos de series temporales empiezan a partir de 'Year CE'
df_data = df.iloc[9:].set_index("Site Code")
df_data.index.name = "Year"

# Limpiar nombres de columnas (quitar espacios y puntos)
df_data.columns = df_data.columns.str.strip().str.replace(".", "", regex=False)

# Convertimos valores a float
df_data = df_data.apply(pd.to_numeric, errors="coerce")

print("\nColumnas disponibles:")
print(df_data.columns.tolist())

# ---------------------------------------------------------
# Grafica iterativa
# ---------------------------------------------------------

# Convertimos índice a numérico
df_data.index = pd.to_numeric(df_data.index, errors="coerce")

# Eliminamos filas con índice NaN
df_data = df_data[~df_data.index.isna()]

# Convertimos a entero
df_data.index = df_data.index.astype(int)



# Seleccionemos algunos sitios para graficar
sitios_interes = ["BRO", "CAV", "CAZ", "COL", "DRA", "FON", "GUT" ,"ILO",
                  "INA" ,"AHI","LAI" ,"LIL","LOC" ,"NIE1","NIE2","PAN" ,
                  "PED" ,"POE" ,"REN" ,"SER","SUW" ,"VIG" ,"VIN" ,"WIN","WOB" 
]

# Filtramos solo esos sitios
df_sel = df_data[sitios_interes]

# Año inicial
year0 = 1600

fig, ax = plt.subplots(figsize=(14,8))
plt.subplots_adjust(bottom=0.25)

bars = ax.bar(df_sel.columns, df_sel.loc[year0])
ax.set_ylim(df_sel.min().min(), df_sel.max().max())
ax.set_ylabel("δ13C (‰)")
ax.set_title(f"Año: {year0}")

# Eje para slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Año", 1600, 2005, valinit=year0, valstep=1)

# Función que actualiza barras
def update(val):
    year = int(slider.val)
    for bar, h in zip(bars, df_sel.loc[year]):
        bar.set_height(h)
    ax.set_title(f"Año: {year}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()

# ---------------------------------------------------------
# Histogramas, densidades y boxplots para imputaciones
# ---------------------------------------------------------

fig, axes = plt.subplots(3, len(sitios_interes), figsize=(20, 10))
# Ejes organizados por filas: [0] Histogramas, [1] Densidades, [2] Boxplots

for i, col in enumerate(sitios_interes):
    if col not in df_data.columns:
        print(f"Advertencia: El sitio '{col}' no está en los datos.")
        continue

    col_data = df_data[col].dropna()

    # Histograma
    sns.histplot(col_data, ax=axes[0, i], kde=False, color="skyblue")
    axes[0, i].set_title(f"Histograma: {col}")
    
    # Densidad (KDE)
    sns.kdeplot(col_data, ax=axes[1, i], fill=True, color="orange")
    axes[1, i].set_title(f"Densidad: {col}")

    # Boxplot
    sns.boxplot(y=col_data, ax=axes[2, i], color="lightgreen")
    axes[2, i].set_title(f"Boxplot: {col}")

plt.tight_layout()
plt.show()

# ================== General ==================

plt.figure(figsize=(12, 6))
for sitio in sitios_interes:
    plt.plot(df_data.index, df_data[sitio], label=sitio, linewidth=1)
plt.title("Evolución ð13C a lo largo del tiempo para un subconjunto de site code")
plt.xlabel("Año")
plt.ylabel("Z-score δ¹³C")
plt.legend(ncol=3, fontsize=8)  # Ajusta columnas y tamaño de leyenda si hay muchos sitios
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.tight_layout()
plt.show()

# ================== Gráfica Z-score ==================
df_zscore = (df_data - df_data.mean()) / df_data.std()
plt.figure(figsize=(10, 6))
for sitio in sitios_interes:
    plt.plot(df_zscore.index, df_zscore[sitio], label=sitio, linewidth=1)
plt.title("Todos los sitios - Normalización Z-score")
plt.xlabel("Año")
plt.ylabel("Z-score δ¹³C")
plt.legend(ncol=3, fontsize=8)  # Ajusta columnas y tamaño de leyenda si hay muchos sitios
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.tight_layout()
plt.show()

# ================== Gráfica Min-Max ==================
df_minmax = (df_data - df_data.min()) / (df_data.max() - df_data.min())
plt.figure(figsize=(12, 6))
for sitio in sitios_interes:
    plt.plot(df_minmax.index, df_minmax[sitio], label=sitio, linewidth=1)
plt.title("Todos los sitios - Normalización Min-Max")
plt.xlabel("Año")
plt.ylabel("Min-Max δ¹³C")
plt.legend(ncol=4, fontsize=8)
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Heatmap de datos faltantes
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.heatmap(df_n)
plt.xticks(rotation=90)
plt.xlabel('Sitios')
plt.ylabel('Año')
plt.title('Mapa de calor de valores faltantes (NaN)')
plt.tight_layout()
plt.show()
