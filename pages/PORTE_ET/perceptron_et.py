import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

def entrantement_porte_et(taille_des_donnees=200, test_size_percente=0.25, epoch =40, batch_size_ = 10, nombre_neuronne= 4, verbose= 0):
    """
    La fonction permet d'entrainer un modèle séquentiel (perceptron multicouche) 
    représentant la porte logique ET. La fonction sauvegarde le modèle entraîné 
    dans le fichier 'and_gate_model.h5'.
    
    @param: tailles_des_donnees (La taille des données à utiliser pour l'entraînement et la validation)
            test_size_percente (Pourcentage des données qui seront réservé pour la validation)
    @return: Le score de la prédiction
    """
    vector_size = taille_des_donnees

    # Générer des vecteurs A et B avec des 0 et des 1 aléatoires
    A = np.random.randint(0, 2, size=vector_size)
    B = np.random.randint(0, 2, size=vector_size)

    # Calculer les résultats de l'opération AND
    Output = A & B

    # Créer un DataFrame avec les vecteurs A, B et le résultat Output
    df = pd.DataFrame({
        'A': A,
        'B': B,
        'Output': Output
    })

    # Séparation des features (entrées) et de la target (sortie)
    X = df[['A', 'B']].values  # .values pour obtenir des numpy arrays
    y = df['Output'].values

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percente, random_state=42)

    # Définir le modèle séquentiel
    model = Sequential()
    model.add(Dense(nombre_neuronne, input_dim=2, activation='relu'))  # Couche d'entrée et une couche cachée
    model.add(Dense(1, activation='sigmoid'))  # Couche de sortie

    # Compiler le modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size_, verbose=verbose)

    # Évaluation du modèle
    _, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
    

    # Sauvegarde du modèle entraîné
    model.save('pages/PORTE_ET/and_gate_model.keras')

    return accuracy


