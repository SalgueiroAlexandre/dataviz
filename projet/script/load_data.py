import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances
# Lire le fichier avec des espaces comme séparateurs
Test1 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/test_pasture.0', sep=' ', header=None)
Train1 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/train_pasture.0', sep=' ', header=None)
#Train2 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/train_pasture.1', sep=' ', header=None)
#Train3 = pd.read_csv('../../exampledata/3-holdout/pasture/matlab/train_pasture.2', sep=' ', header=None)


# données brute
X = pd.concat([Train1, Test1])
X.columns = [f'x{i}' for i in range(1, len(X.columns))] + ['y']
X = X.drop(columns=['x17'])
# donnée sans les données booléennes
X_sans_bool = X.iloc[:,4:-1]
# normalisation des données
X_sans_bool = (X_sans_bool - X_sans_bool.mean()) / X_sans_bool.std()
# données encodées
X_dummies = pd.get_dummies(X,columns=['x1','x2','x3','x4'])
X_dummies = X_dummies.drop(columns=['y'])


metrics = ['l2', 'canberra', 'l1', 'matching', 'hamming', 'chebyshev', 'braycurtis', 'sqeuclidean', 'correlation', 'cityblock', 'minkowski', 'euclidean', 'cosine', 'nan_euclidean', 'manhattan']
for metric in metrics:
    dist = pairwise_distances(X_dummies, metric=metric)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    transformed = mds.fit_transform(dist)
    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=X['y'])
    plt.title(metric)
    plt.show()

for metric in metrics:
    dist = pairwise_distances(X_dummies, metric=metric)
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    transformed = mds.fit_transform(dist)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=X['y'])
    plt.colorbar(sc)
    plt.title('MDS with 3 Components '+metric)
    plt.show()