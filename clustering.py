#############################
## 3 PARTE - CLUSTERING #####
#############################

import pandas as pd
import pickle
from pathlib import Path

from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans

# Directorio base
BASE_DIR = Path(__file__).resolve().parent

# cargamos datos de entrenamiento y etiquetas
X_train = (
    pd.read_csv(BASE_DIR / 'practica_X_train.csv', sep=';')
      .drop(columns=[0], errors='ignore')
      .astype(float)
)
Y_train = (
    pd.read_csv(BASE_DIR / 'practica_Y_train.csv', sep=';')
      .drop(columns=[0], errors='ignore')
)
y_train = Y_train['Air_Quality']

# cargamos el pipeline de preprocesado
with open(BASE_DIR / 'pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Transformar X_train
X_train_pre = pipeline.transform(X_train)

# realizamos clustering con KMeans
n_clusters = len(y_train.unique())  # uno por cada clase
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X_train_pre)
cluster_train = kmeans.predict(X_train_pre)

# matriz de contingencia (clusters vs clases)
conf_mat = pd.crosstab(cluster_train, y_train,
                       rownames=['cluster'], colnames=['class'])
print("Contingency Matrix (cluster vs class):")
print(conf_mat)

# mapear cada cluster a la clase mayoritaria
cluster_to_class = {}
for idx, cluster in enumerate(conf_mat.index):
    # seleccionar la clase con mayor frecuencia en ese cluster
    col_idx = conf_mat.values[idx].argmax()
    cluster_to_class[cluster] = conf_mat.columns[col_idx]
print("Cluster -> Clase mapping:")
print(cluster_to_class)

# asignamos los clusters y etiquetas al conjunto de test
X_test = (
    pd.read_csv(BASE_DIR / 'practica_X_test.csv', sep=';')
      .drop(columns=[0], errors='ignore')
      .astype(float)
)
X_test_pre = pipeline.transform(X_test)
cluster_test = kmeans.predict(X_test_pre)
assigned_labels = [cluster_to_class[c] for c in cluster_test]

# 8) Guardar resultados de clustering + etiquetas
output_df = pd.DataFrame({
    'Score': range(1, len(assigned_labels) + 1),
    'Air_Quality': assigned_labels
})
output_df.to_csv(BASE_DIR / 'practica3_Y_test.csv', sep=';', index=False)
print("Resultados de clustering guardados en practica3_Y_test.csv")
