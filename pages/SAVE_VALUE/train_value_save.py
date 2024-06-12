"""
MODULE: train_value_save

DESCRIPTION: Le module contient des fonction permettant de sauvegarder les données de à 
             utiliser pour le test et le resultat de l'entrainement

PROCEDUIRE:

sauvegarder_resultats(filename, x_test, y_test, history)  -> None   :    pour la sauvegarde des données

FONCTIONS:
charger_resultats(filename : str)     -> (x_test, y_test, history)  :    pour le charger des données sauvegardées
"""

import numpy as np
import json


def sauvegarder_resultats(filename, x_test, y_test, history):
    """
        Procédure de sauvegarde des données à utiliser pour le test et le résultat de l'entrainement
        @param: filename: Le nom du chemin ou les données seront sauvegardées
                x_test  : données d'entrer à utiliser pour la validation
                y_test  : données de sortie (données à prédire au cours de la validation)
    """
    # Convertir les données de test et les étiquettes de test en listes pour JSON
    x_test_list = x_test.tolist()
    y_test_list = y_test.tolist()
    
    # Convertir l'historique d'entraînement en un dictionnaire sérialisable en JSON
    history_dict = history.history

    # Créer un dictionnaire avec tous les résultats
    results = {
        'x_test': x_test_list,
        'y_test': y_test_list,
        'history': history_dict
    }

    # Écrire le dictionnaire dans un fichier JSON
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)




def charger_resultats(filename):

    """
        Cette focntion permet de charger les données d'un entrainement suvegarder dans un fichier
        @param : finame   -> chemin du fichier où se trouve les données
        @return : x_test  -> données d'entrée à utiliser pour la valisation
                  y_test  -> données de sortie à utiliser pour la validation
                  history -> resulats de l'entrainement pré_entrainer
    """

    # Lire le fichier JSON
    with open(filename, 'r') as file:
        results = json.load(file)
    
    # Reconstruire les données de test et les étiquettes de test à partir des listes
    x_test = np.array(results['x_test'])
    y_test = np.array(results['y_test'])
    
    # L'historique est déjà un dictionnaire compatible
    history = results['history']

    return x_test, y_test, history