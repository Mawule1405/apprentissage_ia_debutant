"""

"""
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import json

#import pages.SAVE_VALUE.train_value_save as tvs

def rnc(nombre_de_filtres=32, taille_filtre=(3, 3), taille_pooling=(2, 2), batch_size=128, verbose=1, validation_split=0.2, epochs=10):
    # Chargement de l'ensemble de données
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Prétraitement des images pour CNN
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Définition de l'architecture du modèle CNN
    model = Sequential()
    model.add(Conv2D(nombre_de_filtres, kernel_size=taille_filtre, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=taille_pooling))
    model.add(Dropout(0.25))  # Dropout pour éviter le surapprentissage
    
    # Ajout d'une autre couche convolutive pour améliorer l'apprentissage des caractéristiques
    model.add(Conv2D(nombre_de_filtres * 2, kernel_size=taille_filtre, activation='relu'))
    model.add(MaxPooling2D(pool_size=taille_pooling))
    model.add(Dropout(0.25))
    
    # Flatten les données pour les passer à des couches Dense
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))  # Couche de sortie pour la classification en 10 classes

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    # Accéder aux métriques d'entraînement
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']

    # Si vous utilisez une validation_split dans fit, vous pouvez également accéder aux métriques de validation
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    # Sauvegarde de l'historique d'entraînement et des paramètres utilisés
    
    params = {
        "nombre_de_filtre": nombre_de_filtres,
        "taille_filtre": taille_filtre,
        "taille_pooling": taille_pooling,
        "batch_size": batch_size,
        "verbose": verbose,
        "epochs": epochs,
        "train_loss" : train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }
    model.save('pages/RNC/model_de_rnc.keras')
    with open('pages/RNC/resultats_rnc.json', 'w') as json_file:
        json.dump({"params": params, "history": history.history}, json_file)

    return True







