import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import json

def entrantement_porte_et( epoch =2000, batch_size_ = 10,  verbose= 1):
    """
    La fonction permet d'entrainer un modèle séquentiel (perceptron multicouche) 
    représentant la porte logique ET. La fonction sauvegarde le modèle entraîné 
    dans le fichier 'and_gate_model.h5'.
    
    @param: tailles_des_donnees (La taille des données à utiliser pour l'entraînement et la validation)
            test_size_percente (Pourcentage des données qui seront réservé pour la validation)
    @return: Le score de la prédiction
    """

    # Générer des vecteurs A et B avec des 0 et des 1 aléatoires
    entre = np.array([[0,0], [0,1], [1, 0], [1, 1]])
    sortie = np.array([0, 0, 0, 1])

    
    # Définir le modèle séquentiel
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(entre, sortie, epochs=epoch, batch_size=batch_size_, verbose=verbose)

    # Évaluation du modèle
    loss, accuracy = model.evaluate(entre, sortie, verbose=0)


    # Sauvegarde du modèle entraîné
    model.save('pages/PORTE_ET/and_gate_model_new.keras')
    
    valeurs = {"batch_size": batch_size_, "epoque":epoch, "verbose":verbose, "accuracy": accuracy, "loss": loss}
    chemin_du_fichier = "pages/PORTE_ET/parametres.json"

    sauvegarder_parametres_json(chemin_du_fichier, valeurs)

    return accuracy, loss



def sauvegarder_parametres_json(nom_fichier, parametres_resultats):
    # Définir les paramètres
    # Écrire les paramètres dans un fichier JSON
    with open(nom_fichier, 'w') as f:
        json.dump(parametres_resultats, f)



