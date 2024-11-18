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
# Estandarización de Datos
Este bloque realiza las siguientes operaciones:

1. **Estadísticas descriptivas**:
   - `df.describe()` genera un resumen estadístico (como media, desviación estándar, mínimos y máximos) para las columnas numéricas del DataFrame.

2. **Escalado de datos**:
   - Se inicializa un escalador estándar con `StandardScaler()`, que normaliza los datos (restando la media y dividiendo por la desviación estándar).
   - `scaler.fit_transform(df)` escala todas las columnas del DataFrame `df` y devuelve una versión escalada de los datos en formato de arreglo `NumPy`.

3. **Escalado selectivo y reasignación**:
   - Se seleccionan las columnas `['RHOB', 'NPHI', 'GR', 'PEF', 'DTC']` para escalarlas.
   - Los valores escalados se asignan a nuevas columnas `['RHOB_T', 'NPHI_T', 'GR_T', 'PEF_T', 'DTC_T']`, que representan las versiones transformadas de las columnas originales.

4. **Resultado**:
   - El DataFrame `df` incluye tanto las columnas originales como las nuevas columnas escaladas.

            df.describe()
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            df[['RHOB_T', 'NPHI_T', 'GR_T', 'PEF_T', 'DTC_T']] = scaler.fit_transform(df[['RHOB', 'NPHI', 'GR', 'PEF', 'DTC']])
            df

# Función para Optimizar el Número de Clusters y Grafica de inercia
Este bloque de código realiza las siguientes tareas relacionadas con el uso de *k-means* para determinar el número óptimo de clusters:

1. **Definición de la función `optimize_kmeans`**:
   - **Parámetros**:
     - `data`: conjunto de datos a clusterizar.
     - `max_k`: número máximo de clusters para probar.
   - **Cálculos dentro de la función**:
     - Se inicializan listas vacías `means` (para guardar el número de clusters) e `inertias` (para almacenar las inercias asociadas a cada número de clusters).
     - Se ejecuta un bucle para probar diferentes números de clusters, desde 1 hasta `max_k - 1`.
     - Para cada valor de clusters (`k`), se ajusta el modelo de *k-means* (`KMeans`) al conjunto de datos y se calcula la inercia (suma de distancias cuadradas de los puntos a sus centroides asignados).
     - Los valores de `k` e inercia se almacenan en sus respectivas listas.

2. **Visualización del método del codo**:
   - Se genera un gráfico de línea (método del codo) que muestra cómo cambia la inercia en función del número de clusters.
   - **Ejes del gráfico**:
     - Eje X: número de clusters (`means`).
     - Eje Y: inercia correspondiente (`inertias`).
   - El gráfico ayuda a identificar el número óptimo de clusters, donde la inercia deja de disminuir significativamente (el "codo").

3. **Llamado a la función**:
   - `optimize_kmeans(df[['RHOB_T', 'NPHI_T']], 10)`:
     - Aplica el análisis al subconjunto de datos compuesto por las columnas escaladas `RHOB_T` y `NPHI_T`.
     - Evalúa el modelo para un número de clusters desde 1 hasta 9 (`max_k - 1`).

           def optimize_kmeans(data, max_k):
            means = []
            inertias = []
        
            for k in range(1, max_k):
                kmeans = KMeans(n_clusters = k)
                kmeans.fit(data)
                means.append(k)
                inertias.append(kmeans.inertia_)
        
            fig = plt.subplots(figsize = (10, 5))
            plt.plot(means, inertias, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.grid(True)
            plt.xticks(means)
            plt.show()
        
            optimize_kmeans(df[['RHOB_T', 'NPHI_T']], 10)
# Aplicar K-Means con 3 Clusters
Este bloque de código realiza las siguientes operaciones para aplicar el algoritmo de *k-means* con 3 clusters:

1. **Creación del modelo `KMeans`**:
   - Se inicializa un modelo de *k-means* con `n_clusters=3`, especificando que se dividirán los datos en 3 grupos o clusters.

2. **Ajuste del modelo**:
   - `kmeans.fit(df[['NPHI_T', 'RHOB_T']])` entrena el modelo utilizando las columnas escaladas `NPHI_T` y `RHOB_T` del DataFrame como características.

3. **Asignación de etiquetas**:
   - `kmeans.labels_` contiene las etiquetas de cluster asignadas a cada fila del DataFrame.
   - Estas etiquetas se añaden al DataFrame `df` en una nueva columna llamada `kmenas_3`, indicando a qué cluster pertenece cada fila según el modelo entrenado.

4. **Resultado**:
   - El DataFrame `df` ahora incluye una nueva columna (`kmenas_3`) que clasifica cada punto de datos en uno de los 3 clusters creados por el modelo *k-means*.

            kmeans = KMeans(n_clusters = 3)
            kmeans.fit(df[['NPHI_T', 'RHOB_T']])
            df['kmenas_3'] = kmeans.labels_
            df
# Gráfico de dispersión 
Este bloque de código realiza las siguientes acciones:

1. **Visualización con `scatter`**:
   - `plt.scatter(x=df['NPHI'], y=df['RHOB'], c = df['kmenas_3'])`:
     - Crea un gráfico de dispersión (scatter plot) utilizando las columnas `NPHI` y `RHOB` como las coordenadas en los ejes X e Y, respectivamente.
     - El parámetro `c = df['kmenas_3']` colorea los puntos según las etiquetas de los clusters (almacenadas en la columna `kmenas_3`), con un color diferente para cada grupo.
   - `plt.xlim(-0.1, 1)` y `plt.ylim(3, 1.5)` ajustan los límites de los ejes X e Y para enfocar la visualización en un rango específico de los datos.
   - `plt.show()` muestra el gráfico.

2. **Ciclo para aplicar *k-means* con diferentes números de clusters**:
   - Se ejecuta un bucle `for k in range(1, 6)` para probar *k-means* con 1 a 5 clusters.
     - En cada iteración:
       - Se crea un modelo de *k-means* con `n_clusters=k` (es decir, el número de clusters cambia en cada iteración).
       - Se ajusta el modelo a las columnas escaladas `['NPHI_T', 'RHOB_T']` del DataFrame.
       - Se agrega una nueva columna en el DataFrame (`df[f'kmenas_{k}']`), que contiene las etiquetas de los clusters para el número de clusters `k`.

3. **Resultado**:
   - Después de ejecutar el bucle, el DataFrame `df` contiene 5 nuevas columnas (`kmenas_1`, `kmenas_2`, `kmenas_3`, `kmenas_4`, `kmenas_5`), cada una correspondiente a las etiquetas de cluster generadas para diferentes números de clusters (de 1 a 5).

            plt.scatter(x=df['NPHI'], y=df['RHOB'], c = df['kmenas_3'])
            plt.xlim(-0.1,1)
            plt.ylim(3,1.5)
            plt.show()
            for k in range(1, 6):
                kmeans = KMeans(n_clusters = k)
                kmeans.fit(df[['NPHI_T', 'RHOB_T']])
                df[f'kmenas_{k}'] = kmeans.labels_
            df
# Visualización de los Gráficos múltiples de Clusters (de 1 a 5)

Este bloque de código realiza lo siguiente para visualizar los resultados de los clusters en múltiples subgráficos:

1. **Creación de subgráficos**:
   - `fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))` crea una figura con 1 fila y 5 columnas de subgráficos. El tamaño de la figura se ajusta a 20x5 pulgadas para una visualización amplia.
   - `fig.axes` accede a los ejes (subgráficos) creados dentro de la figura.

2. **Bucle para graficar los clusters**:
   - Se usa un bucle `for i, ax in enumerate(fig.axes, start=1)` que recorre cada uno de los subgráficos (`ax`) y su índice (`i`), comenzando desde 1.
   - Dentro del bucle:
     - `ax.scatter(x=df['NPHI'], y=df['RHOB'], c=df[f'kmenas_{i}'])` crea un gráfico de dispersión para cada subgráfico (`ax`), utilizando las columnas `NPHI` y `RHOB` como los valores de los ejes X e Y, y coloreando los puntos según las etiquetas de cluster correspondientes a cada número de clusters (`kmenas_1`, `kmenas_2`, etc.).
     - `ax.set_ylim(3, 1.5)` y `ax.set_xlim(0, 1)` ajustan los límites de los ejes Y y X para cada gráfico.
     - `ax.set_title(f'Clusters: {i}')` agrega un título a cada subgráfico indicando el número de clusters representados.

              fig, axs = plt.subplots(nrows=1, ncols=5,  figsize = (20, 5))
                for i, ax in enumerate(fig.axes, start=1):
                    ax.scatter(x=df['NPHI'], y=df['RHOB'], c = df[f'kmenas_{i}'])
                    ax.set_ylim(3,1.5)
                    ax.set_xlim(0,1)
                    ax.set_title(f'Clusters: {i}')
3. **Resultado**:
   - El resultado es una figura con 5 subgráficos, cada uno mostrando el gráfico de dispersión de los datos con el número de clusters correspondiente (de 1 a 5).

     
