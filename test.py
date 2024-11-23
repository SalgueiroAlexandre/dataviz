import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import procrustes
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D  # Import pour les graphiques 3D

import matplotlib
matplotlib.use('MacOSX')  

# Lire les fichiers avec des espaces comme séparateurs
Test1 = pd.read_csv('exampledata/3-holdout/pasture/matlab/test_pasture.0', sep=' ', header=None)
Train1 = pd.read_csv('exampledata/3-holdout/pasture/matlab/train_pasture.0', sep=' ', header=None)

# Lire le fichier avec des espaces comme séparateurs
Test1 = pd.read_csv('exampledata/3-holdout/pyrim10/matlab/test_pyrim10.0', sep=' ', header=None)
Train1 = pd.read_csv('exampledata/3-holdout/pyrim10/matlab/train_pyrim10.0', sep=' ', header=None)

# Données brutes
X = pd.concat([Train1, Test1])
X.columns = [f'x{i}' for i in range(1, len(X.columns))] + ['y']

# Convertir la colonne 'y' en type entier
X['y'] = X['y'].astype(int)

X_features = X.iloc[:, :-1]
y = X['y']

# MDS classique avec distances euclidiennes
dist = pairwise_distances(X_features, metric='euclidean')
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
transformed_mds = mds.fit_transform(dist)



# PCA
pca = PCA(n_components=2)
transformed_pca = pca.fit_transform(X_features)

# Inverser l'axe 2 de la PCA
transformed_pca[:, 1] = -transformed_pca[:, 1]

# Appliquer l'analyse de Procrustes
mtx1, mtx2, disparity = procrustes(transformed_pca, transformed_mds)

# Normaliser les sorties entre 0 et 1
scaler = MinMaxScaler()
transformed_mds_scaled = scaler.fit_transform(transformed_mds)
transformed_pca_scaled = scaler.fit_transform(transformed_pca)

# Obtenir les classes uniques et les couleurs associées
classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

# Créer une figure avec deux sous-graphiques côte à côte
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Graphique MDS normalisé
for cls, color in zip(classes, colors):
    idx = y == cls
    axes[0].scatter(transformed_mds_scaled[idx, 0], transformed_mds_scaled[idx, 1], label=f'Classe {cls}', color=color)
axes[0].legend(title='Classes')
axes[0].set_title('MDS normalisé')
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')
axes[0].grid(True)

# Graphique PCA normalisé
for cls, color in zip(classes, colors):
    idx = y == cls
    axes[1].scatter(transformed_pca_scaled[idx, 0], transformed_pca_scaled[idx, 1], label=f'Classe {cls}', color=color)
axes[1].legend(title='Classes')
axes[1].set_title('PCA normalisé')
axes[1].set_xlabel('Composante 1')
axes[1].set_ylabel('Composante 2')
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Afficher la courbe de stress (pour choisir le nombre de dimensions)
stress_list = []
dimensions = range(1, 11)
for n in dimensions:
    mds = MDS(n_components=n, dissimilarity='precomputed', random_state=42, normalized_stress=True, metric=False)
    mds.fit(dist)
    # ajouter le stress normalisé par rapport à la somme des carrés des distances
    stress_list.append(mds.stress_ )

plt.figure()
plt.plot(dimensions, stress_list, marker='o')
plt.title("Courbe de stress")
plt.xlabel("Nombre de dimensions")
plt.ylabel("Stress")
plt.grid(True)
plt.show()
    




# On observe le même résultat  !!!
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS

# Obtenir la liste des métriques disponibles
available_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys())

# Afficher les métriques disponibles
metrics = ['l2', 'canberra', 'l1', 'matching', 'hamming', 'chebyshev', 'braycurtis', 'sqeuclidean', 'correlation',
           'cityblock', 'minkowski', 'euclidean', 'cosine', 'nan_euclidean', 'manhattan']

# for metric in metrics: 
#     mds = MDS(n_components = 2, dissimilarity='precomputed')
#     dist = pairwise_distances(X_features, metric=metric)
#     transformed = mds.fit_transform(X=dist)
#     for cls, color in zip(classes, colors):
#         idx = y == cls
#         plt.scatter(transformed[idx, 0], transformed[idx, 1], label=f'Classe {cls}', color=color)

#     plt.legend()
#     plt.grid(True)
#     plt.xlabel("MDS1")
#     plt.xlabel("MDS2")
#     plt.title(metric)
#     plt.show()

        


# Non metric MDSn distances are not preserved but the order of the points is preserved

for metric in metrics: 
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=False)
    dist = pairwise_distances(X_features, metric=metric)
    transformed = mds.fit_transform(X=dist)
    for cls, color in zip(classes, colors):
        idx = y == cls
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=f'Classe {cls}', color=color)
    plt.legend(title='Classes')
    plt.title(f'MDS non métrique avec métrique {metric}')
    plt.show()

# MDS en 3D
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
transformed_mds = mds.fit_transform(dist)

# PCA en 3D
pca = PCA(n_components=3)
transformed_pca = pca.fit_transform(X_features)

# Visualisation en 3D
classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

fig = plt.figure(figsize=(14, 6))

# MDS 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for cls, color in zip(classes, colors):
    idx = y == cls
    ax1.scatter(transformed_mds[idx, 0], transformed_mds[idx, 1], transformed_mds[idx, 2], label=f'Classe {cls}', color=color)
ax1.set_title('MDS 3D')
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')
ax1.set_zlabel('Dimension 3')
ax1.legend(title='Classes')
ax1.grid(True)

# PCA 3D
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for cls, color in zip(classes, colors):
    idx = y == cls
    ax2.scatter(transformed_pca[idx, 0], transformed_pca[idx, 1], transformed_pca[idx, 2], label=f'Classe {cls}', color=color)
ax2.set_title('PCA 3D')
ax2.set_xlabel('Composante 1')
ax2.set_ylabel('Composante 2')
ax2.set_zlabel('Composante 3')
ax2.legend(title='Classes')
ax2.grid(True)

plt.tight_layout()
plt.show()

for metric in metrics:
    dist = pairwise_distances(X_features, metric=metric)
    mds = MDS(n_components=3, dissimilarity='precomputed', metric=False)
    transformed = mds.fit_transform(dist)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cls, color in zip(classes, colors):
        idx = y == cls
        ax.scatter(transformed[idx, 0], transformed[idx, 1], transformed[idx, 2], label=f'Classe {cls}', color=color)
    ax.set_title(f'MDS non métrique 3D avec métrique {metric}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.legend(title='Classes')
    ax.grid(True)
    plt.show()