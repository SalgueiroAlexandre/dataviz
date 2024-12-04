import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('preferences_genres_musique.csv')
df.set_index('Personne', inplace=True)
print(df)

# Définir les genres
genres = ['Classique', 'Jazz', 'Rock', 'Pop', 'Hip-Hop', 'Électronique']
n_genres = len(genres)

# Initialiser une matrice de dissimilarité
dissimilarity = np.zeros((n_genres, n_genres))

# Calculer la dissimilarité pour chaque paire
for i in range(n_genres):
    for j in range(n_genres):
        if i != j:
            # Différence des rangs pour chaque individu
            diff = df[genres[i]] - df[genres[j]]
            # Calculer la dissimilarité (somme des carrés des différences)
            dissimilarity[i, j] = np.sum(diff ** 2)

# Appliquer MDS metric=False
mds = MDS(dissimilarity='precomputed', random_state=42, metric=False)
pos = mds.fit_transform(dissimilarity)

# Visualiser les résultats
plt.scatter(pos[:, 0], pos[:, 1])
for i, genre in enumerate(genres):
    plt.annotate(genre, (pos[i, 0], pos[i, 1]))
plt.title('Visualisation des dissimilarités entre genres musicaux sans métrique')
plt.show()

# Appliquer MDS metric=True
mds = MDS(dissimilarity='precomputed', random_state=42, metric=True)
pos = mds.fit_transform(dissimilarity)

# Visualiser les résultats
plt.scatter(pos[:, 0], pos[:, 1])
for i, genre in enumerate(genres):
    plt.annotate(genre, (pos[i, 0], pos[i, 1]))
plt.title('Visualisation des dissimilarités entre genres musicaux avec métrique')
plt.show()

# ACP
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pos_pca = pca.fit_transform(dissimilarity)

# Visualiser les résultats
plt.scatter(pos_pca[:, 0], pos_pca[:, 1])
for i, genre in enumerate(genres):
    plt.annotate(genre, (pos_pca[i, 0], pos_pca[i, 1]))
plt.title('Visualisation des dissimilarités entre genres musicaux avec ACP')
plt.show()

# visualisation de l'explication de la variance par les composantes principales
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée')
plt.title('Explication de la variance par les composantes principales')
plt.show()
