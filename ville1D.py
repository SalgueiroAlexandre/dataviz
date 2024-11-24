import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

cities = pd.read_csv('C:\\Users\\alexe\\Downloads\\simplemaps_worldcities_basicv1.77\\worldcities.csv', sep = ',')
print(cities.head())

cities = cities.dropna()
cities = cities[['city', 'lat', 'lng', 'population']]
# récuperer les 5 villes les plus peupler
cities = cities.sort_values('population', ascending=False).head(7)
# Calculer les distances entre les villes
dist = pairwise_distances(cities[['lat', 'lng']])
# MDS metric=False
mds_false = MDS(n_components=1, dissimilarity='precomputed', random_state=42, metric=False)
transformed_mds_false = mds_false.fit_transform(dist)

# MDS metric=True
mds_true = MDS(n_components=1, dissimilarity='precomputed', random_state=42, metric=True)
transformed_mds_true = mds_true.fit_transform(dist)

# Créer une figure avec deux sous-graphiques côte à côte
fig, axs = plt.subplots(1, 2, figsize=(20, 5))

# Premier graphique
axs[0].scatter(transformed_mds_true, [0]*len(transformed_mds_true), s=cities['population']/1000, c='red', alpha=0.5)
for i, txt in enumerate(cities['city']):
    axs[0].annotate(txt, (transformed_mds_true[i], 0), fontsize=12)
axs[0].set_title('Villes les plus peuplées du monde (metric=True)')

# Deuxième graphique
axs[1].scatter(transformed_mds_false, [0]*len(transformed_mds_false), s=cities['population']/1000, c='red', alpha=0.5)
for i, txt in enumerate(cities['city']):
    axs[1].annotate(txt, (transformed_mds_false[i], 0), fontsize=12)
axs[1].set_title('Villes les plus peuplées du monde (metric=False)')

plt.show()
