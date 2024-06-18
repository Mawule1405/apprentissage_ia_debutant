import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import json
#import SAVE_VALUE.train_value_save as tvs

def rns(nombre_de_neuronne=110, batch_size=20, validation_split = 0.01,  verbose=1, epochs=10):
    # Chargement de l'ensemble de données
    (x_entrainement, y_entrainement), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Prétraitement des images
    x_entrainement = x_entrainement.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
    y_entrainement = to_categorical(y_entrainement)
    y_test = to_categorical(y_test)

    # Définition de l'architecture du modèle MLP
    modele = Sequential()
    modele.add(Dense(784, activation='relu', input_shape=(28*28,)))
    modele.add(Dense(nombre_de_neuronne, activation='relu'))
    modele.add(Dense(10, activation='softmax'))

    # Compilation du modèle
    modele.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    history = modele.fit(x_entrainement, y_entrainement, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
   
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']

    # Si vous utilisez une validation_split dans fit, vous pouvez également accéder aux métriques de validation
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    # Sauvegarde de l'historique d'entraînement et des paramètres utilisés
    
    params = {
        "nombre_de_neuronne": nombre_de_neuronne,
        "batch_size": batch_size,
        "verbose": verbose,
        "epochs": epochs,
        "train_loss" : train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }
    modele.save('pages/RNS/model_de_rns.keras')
    # Accéder aux métriques d'entraînement
    with open('pages/RNS/resultats_rns.json', 'w') as json_file:
        json.dump({"params": params, "history": history.history}, json_file)

    return True



def sauvegarder_resultats(filename,history):
    """
        Procédure de sauvegarde des données à utiliser pour le test et le résultat de l'entrainement
        @param: filename: Le nom du chemin ou les données seront sauvegardées
                x_test  : données d'entrer à utiliser pour la validation
                y_test  : données de sortie (données à prédire au cours de la validation)
    """
    # Convertir les données de test et les étiquettes de test en listes pour JSON

    # Convertir l'historique d'entraînement en un dictionnaire sérialisable en JSON
    history_dict = history.history

    # Créer un dictionnaire avec tous les résultats
    results = {
        'history': history_dict
    }

    # Écrire le dictionnaire dans un fichier JSON
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

rns(epochs = 20)