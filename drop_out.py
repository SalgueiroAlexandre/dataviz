import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import numpy as np

# Lire les fichiers avec des espaces comme séparateurs
data = pd.read_csv('dropout_student\data.csv', sep=';')

# replace target values with 0 and 1
data['Target'] = data['Target'].replace('Graduate', 1)
data['Target'] = data['Target'].replace('Dropout', 0)
data['Target'] = data['Target'].replace('Enrolled', 1)


# Transformer une colonne de valeurs entières en une colonne catégorielle
bins = [0, 33, 66, 100, 133, 166, 200]  # Définir les intervalles pour les catégories
labels = [1, 2, 3, 4, 5, 6]  # Définir les labels pour les catégories
data['Previous qualification (grade)'] = pd.cut(data['Previous qualification (grade)'], bins=bins, labels=labels, include_lowest=True)

bins = [0, 25, 50, 75, 100]  # Définir les intervalles pour les catégories
labels = [ 1, 2, 3, 4]  # Définir les labels pour les catégories
data['Unemployment rate'] = pd.cut(data['Unemployment rate'], bins=bins, labels=labels, include_lowest=True)

bins = [0, 25, 50, 75, 100]  # Définir les intervalles pour les catégories
labels = [1, 2, 3, 4]  # Définir les labels pour les catégories
data['Inflation rate'] = pd.cut(data['Inflation rate'], bins=bins, labels=labels, include_lowest=True)

# Supprimer plusieurs colonnes
colonnes_a_supprimer = ['Age at enrollment', 'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)',
       'GDP']  # Remplacez par les noms des colonnes que vous souhaitez supprimer
data = data.drop(columns=colonnes_a_supprimer)
data.dropna(inplace=True)
data = data.head(200)

# MDS classique avec distances euclidiennes
# dist = pairwise_distances(data, metric='euclidean')
spearman_corr = 1 - data.T.corr(method='spearman')
spearman_dist = spearman_corr.values
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, metric=False)
transformed_mds = mds.fit_transform(spearman_dist)

# afficher les données
plt.scatter(transformed_mds[:, 0], transformed_mds[:, 1], c=data['Target'], cmap='viridis')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.title('MDS')
plt.show()

