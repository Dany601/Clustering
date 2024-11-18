# Proyecto de Clustering - Análisis de Datos con K-Means y Visualización

Este proyecto aplica técnicas de clustering, específicamente el algoritmo de K-Means, para agrupar datos basados en sus características y visualizar los resultados en Python. Se utiliza la biblioteca `pandas` para manejar los datos, `scikit-learn` para el modelo de K-Means, y `matplotlib` para las visualizaciones gráficas.

## Objetivo
Aplicar técnicas de clustering con K-Means para agrupar un conjunto de datos, evaluar el número óptimo de clusters mediante el método de la inercia y visualizar los resultados obtenidos.

## Contenido del Proyecto
### 1. Carga y Preprocesamiento de Datos
Se trabaja con un conjunto de datos de un archivo CSV, que contiene información de varios atributos. Los pasos incluyen:

- **Lectura de datos**: Cargar los datos desde una URL utilizando `pandas`.
- **Limpieza de datos**: Eliminación de valores faltantes (NaN) utilizando el método `dropna()`.
- **Estandarización**: Escalado de los datos utilizando `StandardScaler` de `scikit-learn` para normalizar las características relevantes, como `NPHI`, `RHOB`, entre otras.

### 2. Aplicación de K-Means
El objetivo principal es aplicar el algoritmo de K-Means para agrupar los datos en clusters. Se realizan los siguientes pasos:

- **Determinación del número de clusters**: Utilizando el método de inercia, se evalúa cuál es el número óptimo de clusters al visualizar cómo cambia la inercia con el número de clusters.
- **Entrenamiento del modelo K-Means**: Se ejecuta K-Means con un número predefinido de clusters (en este caso 3 clusters) y se asignan las etiquetas de cluster a cada fila de datos.

### 3. Visualización de Resultados
- **Visualización de clusters**: Se crea un gráfico de dispersión con `matplotlib` donde los puntos se colorean según el cluster al que pertenecen. 
- **Gráficos comparativos**: Utilizando múltiples subgráficos, se muestran los resultados de aplicar K-Means con diferentes números de clusters (de 1 a 5). Esto permite observar cómo cambia la agrupación de los datos al variar el número de clusters.

### 4. Evaluación del Modelo
La evaluación se realiza observando cómo cambia la asignación de puntos con diferentes valores de K (número de clusters). Se analiza:

- **La inercia**: Un indicador del ajuste del modelo, donde valores más bajos de inercia indican un mejor ajuste.
- **Distribución de los clusters**: Cómo los puntos se agrupan en función del número de clusters seleccionado.

### 5. Implementación en Python
El proyecto se desarrolló utilizando Python, con las siguientes bibliotecas principales:

- **Pandas**: Para la manipulación y limpieza de los datos.
- **Scikit-learn**: Para aplicar el algoritmo de K-Means y realizar la estandarización.
- **Matplotlib**: Para la visualización gráfica de los resultados.

#### Pasos del Algoritmo
1. **Preprocesamiento de los datos**: Lectura del archivo CSV, eliminación de valores nulos y normalización de las columnas de interés.
2. **Optimización de K-Means**: Evaluación del número óptimo de clusters usando el método de inercia.
3. **Aplicación de K-Means**: Creación y ajuste del modelo K-Means con el número de clusters elegido.
4. **Visualización**: Gráficos de dispersión mostrando los clusters y la evolución de los resultados al variar el número de clusters.

### 6. Resultados y Visualización
1. **Visualización de los clusters**: 
   - Se generaron gráficos de dispersión para cada valor de K (1 a 5), donde se muestra cómo cambian los clusters al variar el número de agrupaciones.
   - Se resaltaron las celdas visitadas y el camino final seguido por el algoritmo para visualizar cómo se agrupan los puntos en función de las características `NPHI` y `RHOB`.

2. **Inercia y número de clusters**:
   - Se analizó la inercia en función del número de clusters, ayudando a determinar el número más adecuado de clusters para los datos.

### 7. Evaluación del Modelo
El desempeño del modelo se evaluó en función de:

- **Número de clusters**: El mejor número de clusters se determinó observando cómo cambiaba la inercia y la asignación de puntos.
- **Visualización de la asignación de clusters**: Se compararon diferentes configuraciones del modelo observando cómo las etiquetas de cluster se asignaban a los datos con distintos valores de K.

## Enlaces de Referencia
- **Repositorio de GitHub Explicacion Codigo**: [Enlace a GitHub](https://github.com/Dany601/Clustering/blob/main/Scripts.md)
- **Colab Notebook**: [Enlace a Colab](https://colab.research.google.com/drive/1ilaoA1D6glKjx-7OPLrnCxXJDJeQmPXC?usp=sharing#scrollTo=3pZcbAvfVvB-)

