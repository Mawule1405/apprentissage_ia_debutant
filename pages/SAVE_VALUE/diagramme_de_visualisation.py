"""
MODULE: Diagramme de visualisation

DESCRIPTION:  Ce module contient des fonctions et procédures permettant de créer les diagrammes de visualisation du résultats de l'entrainement

diagramme_de_perte (x_test, x_test, history, model) -> None : permet de construire le diagramme des pertes
                                                              en comparant les valeurs d'entrainement au valeurs d tes

diagramme_de_validation(x_test, y_test, history, model) -> None : permet de construire le digramme de validtion
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ImageTk, ImageOps
import customtkinter as ctk
from tensorflow.keras.models import load_model



def diagramme( history, best_model):

    # Évaluation du modèle
    (x_entrainement, y_entrainement), (x_test, y_test) = keras.datasets.mnist.load_data()
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    

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


def matrice_de_confusion(best_model):
    # Charger les données
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliser les données d'entrée
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Aplatir les images de test
    x_test_flattened = x_test.reshape(-1, 28*28)
    # Effectuer la prédiction
    y_pred = best_model.predict(x_test_flattened)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Comme les labels ne sont pas en one-hot encoding, on peut directement les utiliser
    y_true = y_test

    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

    # Afficher la matrice de confusion
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()


def images_mal_predictes(model, parent_frame):
    # Charger les données
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliser les données d'entrée
    x_test_normalized = x_test / 255.0

    # Aplatir les images de test pour correspondre à la forme d'entrée du modèle
    x_test_flattened = x_test_normalized.reshape(-1, 28*28)

    # Effectuer les prédictions
    y_pred = model.predict(x_test_flattened)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Identifier les images mal prédites
    misclassified_indices = np.where(y_pred_classes != y_test)[0]

   
    # Ajouter un label pour afficher le nombre d'images mal prédites
    num_misclassified = len(misclassified_indices)
    label_num_images = ctk.CTkLabel(master=parent_frame, text=f"Nombre d'images males prédites : {num_misclassified}", font=("Arial", 14, "bold"))
    label_num_images.pack(pady=10)

    # Créer un CTkScrollFrame
    scroll_frame = ctk.CTkScrollableFrame(master=parent_frame, width=1000, height=800)
    scroll_frame.pack(fill="both", expand=True)

    
   

    # Afficher les images mal prédites
    columns = 9  # Nombre d'images par ligne
    image_size = 100  # Taille des images redimensionnées

    for i, idx in enumerate(misclassified_indices):
        img = x_test[idx]
        true_label = y_test[idx]
        predicted_label = y_pred_classes[idx]

        # Convertir l'image en format compatible avec Tkinter
        img = Image.fromarray((img * 255).astype(np.uint8))  # Convertir en image 8 bits
        img = img.resize((image_size, image_size))  # Redimensionner pour une meilleure visibilité
        img = ImageOps.colorize(img.convert('L'), black="white", white="red")  # Coloriser l'image

        # Convertir en PhotoImage pour Tkinter
        img = ImageTk.PhotoImage(img)

        # Calculer la position dans la grille
        row = i // columns
        column = i % columns

        # Créer un cadre pour chaque image
        img_frame = ctk.CTkFrame(master=scroll_frame)
        img_frame.grid(row=row + 1, column=column, padx=7, pady=5)  # Commencer à la ligne 1 pour laisser de la place au label

        # Afficher l'image
        img_label = ctk.CTkLabel(master=img_frame,text="", image=img)
        img_label.image = img  # Pour empêcher l'image d'être garbage collected
        img_label.pack()

        # Afficher le label vrai et prédit
        text_label = ctk.CTkLabel(master=img_frame, text=f"Vrai: {true_label}, Prédit: {predicted_label}")
        text_label.pack()





def plot_loss_curve(history):
    """
    Trace la courbe de perte (loss) sur les données d'entraînement.

    Parameters:
    - history : objet historique retourné par model.fit()

    Returns:
    - None (affiche le plot)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Perte (entraînement)')
    plt.plot(history['val_loss'], label='Perte (validation)')
    plt.title('Courbe de perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()


def plot_validation_curve(history, metric='accuracy'):
    """
    Trace la courbe de la métrique de validation spécifiée (par exemple, précision/accuracy).

    Parameters:
    - history : objet historique retourné par model.fit()
    - metric : nom de la métrique à tracer (par défaut 'accuracy')

    Returns:
    - None (affiche le plot)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history[metric], label=f'{metric.capitalize()} (entraînement)')
    plt.plot(history[f'val_{metric}'], label=f'{metric.capitalize()} (validation)')
    plt.title(f'Courbe de {metric}')
    plt.xlabel('Époque')
    plt.ylabel(f'{metric.capitalize()}')
    plt.legend()
    plt.show()



def matrice_de_confusion_rnc():
    # Charger les données
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliser les données d'entrée
    x_test_normalized = x_test / 255.0

    # Redimensionner les données d'entrée pour le modèle CNN
    x_test_reshaped = x_test_normalized.reshape(-1, 28, 28, 1)

    # Charger le meilleur modèle
    best_model = load_model('pages/RNC/model_de_rnc.keras')

    # Effectuer les prédictions
    y_pred = best_model.predict(x_test_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Identifier les classes réelles
    y_true = y_test

    # Calculer et afficher la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

# Utilisation de la fonction matrice_de_confusion


def images_mal_predictes_rnc(parent_frame):
    # Charger les données MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliser les données d'entrée et les mettre en forme pour le CNN
    x_test_normalized = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Charger le modèle CNN pré-entraîné
    model = load_model('pages/RNC/model_de_rnc.keras')

    # Effectuer les prédictions
    y_pred = model.predict(x_test_normalized)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Identifier les images mal prédites
    misclassified_indices = np.where(y_pred_classes != y_test)[0]

    # Ajouter un label pour afficher le nombre d'images mal prédites
    num_misclassified = len(misclassified_indices)
    label_num_images = ctk.CTkLabel(master=parent_frame, text=f"Nombre d'images mal prédites : {num_misclassified}", font=("Arial", 20, "bold"))
    label_num_images.pack(pady=10)

    # Créer un CTkScrollFrame pour afficher les images
    scroll_frame = ctk.CTkScrollableFrame(master=parent_frame, width=1000, height=800)
    scroll_frame.pack(fill="both", expand=True)

    # Afficher les images mal prédites
    columns = 10  # Nombre d'images par ligne
    image_size = 100  # Taille des images redimensionnées

    for i, idx in enumerate(misclassified_indices):
        img = x_test[idx]
        true_label = y_test[idx]
        predicted_label = y_pred_classes[idx]

        # Convertir l'image en format compatible avec Tkinter et coloriser en rouge
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((image_size, image_size))
        img = ImageOps.colorize(img.convert('L'), black="white", white="red")

        # Convertir en PhotoImage pour Tkinter
        img = ImageTk.PhotoImage(img)

        # Calculer la position dans la grille
        row = i // columns
        column = i % columns

        # Créer un cadre pour chaque image
        img_frame = ctk.CTkFrame(master=scroll_frame)
        img_frame.grid(row=row + 1, column=column, padx=7, pady=5)  # Commencer à la ligne 1 pour laisser de la place au label

        # Afficher l'image
        img_label = ctk.CTkLabel(master=img_frame, text="", image=img)
        img_label.image = img  # Pour empêcher l'image d'être garbage collected
        img_label.pack()

        # Afficher le label vrai et prédit
        text_label = ctk.CTkLabel(master=img_frame, text=f"Vrai: {true_label}, Prédit: {predicted_label}")
        text_label.pack()
