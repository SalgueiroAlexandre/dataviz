import csv
import random

# Liste des genres musicaux
genres = ['Classique', 'Jazz', 'Rock', 'Pop', 'Hip-Hop', 'Électronique']

# Nombre de personnes
n_personnes = 100

# Ouvrir le fichier CSV en écriture
with open('preferences_genres_musique.csv', mode='w', newline='', encoding='utf-8') as fichier_csv:
    writer = csv.writer(fichier_csv)

    # Écrire l'en-tête
    header = ['Personne'] + genres
    writer.writerow(header)

    for i in range(1, n_personnes + 1):
        # Pondérations pour chaque genre
        # On donne une pondération plus élevée au Classique et au Jazz
        poids_genres = {
            'Classique': 1.7,
            'Jazz': 1.5,
            'Rock': 1.0,
            'Pop': 1.0,
            'Hip-Hop': 0.8,
            'Électronique': 0.6
        }

        # Générer les rangs pour chaque genre pour la personne i
        genres_pondérés = []
        for genre, poids in poids_genres.items():
            # Ajouter le genre plusieurs fois selon son poids pour augmenter sa probabilité d'être choisi tôt
            genres_pondérés.extend([genre] * int(poids * 10))

        # Mélanger la liste pondérée pour plus de variabilité
        random.shuffle(genres_pondérés)

        # Créer un classement unique en sélectionnant les genres sans répétition
        genres_classement = []
        genres_utilisés = set()
        for genre in genres_pondérés:
            if genre not in genres_utilisés:
                genres_classement.append(genre)
                genres_utilisés.add(genre)
            if len(genres_classement) == len(genres):
                break

        # Attribuer les rangs (1 pour le genre préféré)
        rangs = {genre: rang + 1 for rang, genre in enumerate(genres_classement)}

        # Créer la ligne pour la personne i
        ligne = [f'Personne_{i}'] + [rangs[genre] for genre in genres]
        writer.writerow(ligne)