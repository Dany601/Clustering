# Carga de datos e importacion de librerias
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
df.describe()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df[['RHOB_T', 'NPHI_T', 'GR_T', 'PEF_T', 'DTC_T']] = scaler.fit_transform(df[['RHOB', 'NPHI', 'GR', 'PEF', 'DTC']])
df

#Función para Optimizar el Número de Clusters y Grafica de inercia
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
kmeans = KMeans(n_clusters = 3)
kmeans.fit(df[['NPHI_T', 'RHOB_T']])
df['kmenas_3'] = kmeans.labels_
df

# Gráfico de dispersión
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
fig, axs = plt.subplots(nrows=1, ncols=5,  figsize = (20, 5))
for i, ax in enumerate(fig.axes, start=1):
    ax.scatter(x=df['NPHI'], y=df['RHOB'], c = df[f'kmenas_{i}'])
    ax.set_ylim(3,1.5)
    ax.set_xlim(0,1)
    ax.set_title(f'Clusters: {i}')
