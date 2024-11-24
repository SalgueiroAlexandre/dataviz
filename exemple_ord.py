import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Classements des préférences
data = {
    'Individu': ['A', 'B', 'C', 'D', 'E'],
    'Classique': [1, 1, 1, 4, 1],
    'Jazz': [2, 2, 2, 3, 3],
    'Rock': [3, 3, 3, 2, 2],
    'Pop': [4, 4, 4, 1, 4]
}

df = pd.DataFrame(data)
df.set_index('Individu', inplace=True)
print(df)

genres = ['Classique', 'Jazz', 'Rock', 'Pop']
n_genres = len(genres)

# Initialiser une matrice de dissimilarité
dissimilarity = np.zeros((n_genres, n_genres))

# Calculer la dissimilarité pour chaque paire
for i in range(n_genres):
    for j in range(n_genres):
        if i != j:
            # Différence des rangs pour chaque individu
            diff = df[genres[i]] - df[genres[j]]
            # Somme des carrés des différences
            dissimilarity[i, j] = np.sum(diff**2)
        else:
            dissimilarity[i, j] = 0  # La dissimilarité avec soi-même est zéro

# Convertir en DataFrame pour une meilleure lisibilité
dissimilarity_df = pd.DataFrame(dissimilarity, index=genres, columns=genres)
print(dissimilarity_df)

from sklearn.manifold import MDS

# Créer une instance du MDS non métrique
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True, random_state=42)

# Appliquer le MDS aux données de dissimilarité
mds_results = mds.fit_transform(dissimilarity)

# Créer un DataFrame des résultats pour faciliter la manipulation
mds_df = pd.DataFrame(mds_results, index=genres, columns=['Dim1', 'Dim2'])
print(mds_df)

import matplotlib.pyplot as plt

# Taille de la figure
plt.figure(figsize=(8, 6))

# Tracer les points
plt.scatter(mds_df['Dim1'], mds_df['Dim2'])

# Ajouter les étiquettes pour chaque point
for genre in genres:
    plt.text(mds_df.loc[genre, 'Dim1'] + 0.05, mds_df.loc[genre, 'Dim2'] + 0.05, genre, fontsize=12)

# Ajouter les axes et le titre
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('MDS non métrique des genres musicaux')

# Afficher le graphique
plt.grid(True)
plt.show()
