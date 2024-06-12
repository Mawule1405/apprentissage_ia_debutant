import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

def rnc(nombre_de_filtres=32, taille_filtre=(3, 3), taille_pooling=(2, 2), min_delta=0.00001, batch_size=128, patience=10, verbose=1, validation_split=0.2, epochs=50):
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

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience, verbose=verbose)
    model_checkpoint = ModelCheckpoint('pages/RNC/model_rnc.keras', monitor='val_accuracy', verbose=verbose, save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # Entraînement du modèle
    history = model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)

    sauvegarder_resultats('pages/RNC/resultats_rnc.json', x_test, y_test, history)

    
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Retourner les métriques 
    return accuracy, val_accuracy



# Exécuter la fonction CNN
rnc()

def diagramme():

    x_test, y_test, history = charger_resultats('resultats_rnc.json')
    # Évaluation du modèle
    best_model = keras.models.load_model('pages/RNC/model_rnc.keras')
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print('Précision sur le test :', test_accuracy)

    # Afficher la matrice de confusion
    y_pred = best_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    # Afficher l'évolution des métriques d'entraînement et de validation
    plt.plot(history.history['accuracy'], label='Précision Entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision Validation')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Afficher la courbe d'erreur pendant l'entraînement et la validation
    plt.plot(history.history['loss'], label='Erreur (Entraînement)')
    plt.plot(history.history['val_loss'], label='Erreur (Validation)')
    plt.xlabel('Époque')
    plt.ylabel('Erreur (Perte)')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()



