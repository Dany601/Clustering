# Análisis de Datos de Profundidad y Preparación para Clustering 
    Este bloque de código realiza lo siguiente:

1. **Importación de librerías**:
   - `pandas` para manejar y analizar datos.
   - `matplotlib.pyplot` para visualizaciones gráficas.
   - `sklearn.preprocessing.StandardScaler` y `sklearn.cluster.KMeans` para preprocesamiento y algoritmos de aprendizaje no supervisado (clustering).

2. **Carga de datos**:
   - Se obtiene un archivo CSV desde una URL pública, cargándolo en un DataFrame de pandas.
   - El índice del DataFrame se establece en la columna `DEPTH_MD`.

3. **Limpieza de datos**:
   - Se eliminan filas con valores faltantes (`NaN`) usando `dropna()`.

4. **Resumen estadístico**:
   - `describe()` genera estadísticas descriptivas de las columnas numéricas (como media, desviación estándar, mínimo, y máximos).
           
         import pandas as pd
         import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            url = 'https://raw.githubusercontent.com/andymcdgeo/Andys_YouTube_Notebooks/main/Data/force2020_data_unsupervised_learning.csv'
            df =  pd.read_csv(url, index_col = 'DEPTH_MD')
            df
            df.dropna(inplace = True)
            df.describe()

