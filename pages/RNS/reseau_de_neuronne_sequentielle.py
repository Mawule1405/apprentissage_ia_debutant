import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def rns(nombre_de_neuronne = 110, min_delta =0.00001,batch_size=3000, patientce= 10, verbose= 1, validation_split=0.2, epoch= 50):
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

    # Callbacks
    arret_precoce = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patientce, verbose=verbose)
    meilleur_modele = ModelCheckpoint('pages/RNS/model_de_rns.keras', monitor='val_accuracy', verbose=verbose, save_best_only=True)
    callbacks = [arret_precoce, meilleur_modele]

    # Entraînement du modèle
    historique = modele.fit(x_entrainement, y_entrainement, validation_split=validation_split, epochs=epoch, callbacks=callbacks)

    # Évaluation du modèle
    modele_sauvegarde = keras.models.load_model('pages/RNS/model_de_rns.keras')
    perte_test, precision_test = modele_sauvegarde.evaluate(x_test, y_test)
    print('Précision sur le test :', precision_test)

    # Afficher la matrice de confusion
    y_predictions = modele_sauvegarde.predict(x_test)
    y_predictions_classes = np.argmax(y_predictions, axis=1)
    y_reel = np.argmax(y_test, axis=1)

    matrice_confusion = confusion_matrix(y_reel, y_predictions_classes)
    affichage_matrice_confusion = ConfusionMatrixDisplay(confusion_matrix=matrice_confusion, display_labels=np.arange(10))
    affichage_matrice_confusion.plot(cmap=plt.cm.Blues)
    plt.show()

    # Afficher l'évolution des métriques d'entraînement et de validation
    plt.plot(historique.history['accuracy'], label='précision du test')
    plt.plot(historique.history['val_accuracy'], label='précision de validation')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.ylim([0.8, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Afficher la courbe d'erreur pendant l'entraînement et la validation
    plt.plot(historique.history['loss'], label='erreur (entraînement)')
    plt.plot(historique.history['val_loss'], label='erreur (validation)')
    plt.xlabel('Époque')
    plt.ylabel('Erreur (perte)')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()

rns()