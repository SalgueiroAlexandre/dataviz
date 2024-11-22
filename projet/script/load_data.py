import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import numpy as np

# Lire le fichier avec des espaces comme séparateurs
Test1 = pd.read_csv('../../exampledata/3-holdout/pyrim10/matlab/test_pyrim10.0', sep=' ', header=None)
Train1 = pd.read_csv('../../exampledata/3-holdout/pyrim10/matlab/train_pyrim10.0', sep=' ', header=None)

Test1 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/test_pasture.0', sep=' ', header=None)
Train1 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/train_pasture.0', sep=' ', header=None)
#Données brutes
X = pd.concat([Train1, Test1])
X.columns = [f'x{i}' for i in range(1, len(X.columns))] + ['y']

# Convertir la colonne 'y' en type catégoriel si nécessaire
X['y'] = X['y'].astype(int)

metrics = ['l2', 'canberra', 'l1', 'matching', 'hamming', 'chebyshev', 'braycurtis', 'sqeuclidean', 'correlation',
           'cityblock', 'minkowski', 'euclidean', 'cosine', 'nan_euclidean', 'manhattan']

# Boucle pour les métriques en 2D
for metric in metrics:
    dist = pairwise_distances(X, metric=metric)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, metric=False)
    transformed = mds.fit_transform(dist)

    # Créer une figure
    plt.figure()

    # Obtenir les classes uniques
    classes = np.unique(X['y'])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

    # Tracer chaque classe avec une couleur spécifique
    for cls, color in zip(classes, colors):
        idx = X['y'] == cls
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=f'Classe {cls}', color=color)

    plt.legend(title='Classes')
    plt.title(f'MDS avec métrique {metric}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

# Boucle pour les métriques en 3D
from mpl_toolkits.mplot3d import Axes3D

for metric in metrics:
    dist = pairwise_distances(X, metric=metric)
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    transformed = mds.fit_transform(dist)

    # Créer une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer chaque classe avec une couleur spécifique
    for cls, color in zip(classes, colors):
        idx = X['y'] == cls
        ax.scatter(transformed[idx, 0], transformed[idx, 1], transformed[idx, 2], label=f'Classe {cls}', color=color)

    ax.legend(title='Classes')
    plt.title(f'MDS en 3D avec métrique {metric}')
    plt.show()

# Analyse en composantes principales (ACP) en 3D
pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(X.drop(columns='y'))

# plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], pca_transformed[:, 2], c=X['y'], cmap='rainbow')
plt.title('ACP en 3D')
plt.show()