import customtkinter as ctk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import ttk, messagebox
import tkinter as tk
import pandas as pd
from tkinter import messagebox
import numpy as np
import json

import pages.PORTE_XOR.perceptron_xor as xor
import pages.SAVE_VALUE.training_progress_bar as pstp

def switch (affiche, page):

    for widget in affiche.winfo_children():
        widget.destroy()
    
    page()


# Fonction pour prédire le résultat avec le modèle AI
def predict_xor( entry_a, entry_b, result_label):

    # Charger le modèle pré-entraîné
    # Chargement du modèle
    model = load_model('pages/PORTE_XOR/xor_gate_model_new.keras')  # et .keras

    try:
        # Récupérer les valeurs des entrées
        a = int(entry_a.get())
        b = int(entry_b.get())

        
        # Validation des entrées : doivent être 0 ou 1
        if a not in [0, 1] or b not in [0, 1]:
           messagebox.showwarning("Prédition", "Valeur incorrecte! veillez vérifier vos données")
        else:
        
            # Préparer les données pour la prédiction en DataFrame avec les bons noms de colonnes
            input_data = pd.DataFrame([[a, b]], columns=['A', 'B'])
            
            # Utiliser le modèle pour prédire la sortie
            prediction = model.predict(input_data)
            
            # Afficher le résultat dans la fenêtre
          
            # Arrondir la sortie pour obtenir une valeur binaire
            predicted_value = int(np.round(prediction[0][0]))

            result_label.configure(text=f"{predicted_value}")

    except ValueError as ve:
        # Afficher un message d'erreur si les entrées ne sont pas valides
        messagebox.showerror("Erreur", str(ve))



def build_percetron_xor():

    """
    Perceptron simple XOR
    """

    fen = ctk.CTkToplevel()
    fen.geometry("700x500")
    fen.resizable(width= False, height= False)
    fen.configure(fg_color = "#2f5d42")
    fen.title("Perceptron XOR")
    fen.attributes('-topmost', True)

    #Définition des fenetres
    boutons_frame = ctk.CTkFrame(fen, width=300, border_width=2, corner_radius=0)
    boutons_frame.pack(fill="both",padx=2, pady=1, side="left")

    affichage_frame= ctk.CTkFrame(fen,width=800, corner_radius=0, fg_color="#fff")
    affichage_frame.pack(fill="y",side="left", padx=1, pady=1)

    utilisation_perceptron(affichage_frame)
    #Placement des boutons
    ttk.Button(boutons_frame, text="Entrainement", command= lambda: switch(affichage_frame, page= entrainementPerceptron)).pack(fill="x", padx=2, side="top")
    ttk.Button(boutons_frame, text="Prédition",  command= lambda: switch(affichage_frame, page = utilisationPerceptron )).pack(fill="x", padx=2, side="top")
    ttk.Button(boutons_frame, text="Quitter",  command= lambda: fen.destroy()).pack(fill="x", padx=2, side="bottom")


    #Les fonctions
    def entrainementPerceptron():
        entrainement_perceptron(affichage_frame)

    def utilisationPerceptron():
        utilisation_perceptron(affichage_frame)


    fen.mainloop()



def utilisation_perceptron(affichage_frame):

    """
    Interface d' utilisation du perceptron pour prédire un résultat
    @param:  affiche de la zone d' affichage 
    """
    
    
    #Entete de la fenetre
    ctk.CTkLabel(affichage_frame, text="UTILISATION DU PERCEPTRON \nMULTICOUCHES PORTE XOR", font=("Garamone", 20)).pack(side="top", pady=10, padx=50)
    
    #Zone de saisi des valeurs
    zone_valeur = ctk.CTkFrame(affichage_frame, border_color="#000", border_width=2, corner_radius=0, bg_color="#000")
    zone_valeur.pack(fill = "x", side="top", padx= 10, pady= 10)
    
    
    #zone A
    frame_A = ctk.CTkFrame(zone_valeur, height=300, width=300, corner_radius=1, border_width=2)
    frame_A.pack(fill="x", side="left")
    
    ctk.CTkLabel(frame_A, text="Valeur A").pack(pady=10)
    valeur_A = ctk.CTkEntry(frame_A, placeholder_text="0", width=280, height=35)
    valeur_A.pack(pady=10, padx=15)
    
    #zone de B
    frame_B = ctk.CTkFrame(zone_valeur, height=300, width=300, corner_radius=1, border_width=2)
    frame_B.pack(fill="both", side= "right")
    
    ctk.CTkLabel(frame_B, text="Valeur B").pack(pady=10)
    valeur_B = ctk.CTkEntry(frame_B, placeholder_text="0", width=280, height=35)
    valeur_B.pack(pady=10, padx=15)
    
    #bouton de validation
    ctk.CTkButton(affichage_frame, text="Exécuter", font=("Garamone", 15), command= lambda: predict_xor(valeur_A, valeur_B, result)).pack(pady= 10)
    
    #resultat
    result_frame= ctk.CTkFrame(affichage_frame, border_width=1)
    result_frame.pack(fill="both", padx="10")
    result = ctk.CTkLabel(result_frame, text="0", font=("Garamone", 250))
    result.pack(fill="both")



def entrainement_perceptron(affichage_frame):

    """
    Interface d' entrainement du modèle
    @param   : zone d'affichage des widgets
    """
    parametres = charger_parametres_json("pages/PORTE_XOR/parametres.json")

    ctk.CTkLabel(affichage_frame, text="ENTRAINEMENT DU PERCEPTRON\n MULTICOUCHES XOR", font=("Garamone", 20)).pack(padx=20, pady= 20)
    
    setting = ttk.Labelframe(affichage_frame, text="Paramètre d'entrainement" , width=800, height=300, relief="solid", border=3)
    setting.pack(fill= "x", padx= 10)

    

    #Paramètre: la taille de données utilisé pendant une époque
    ctk.CTkLabel(setting, text="Batch size").place(x=30, y=20)
    batch_size = ctk.CTkEntry(setting, placeholder_text=10, width= 250)
    batch_size.place(x=30, y=50)
    batch_size.insert(0, parametres["batch_size"])

    #Paramètre: Nombre d'époque
    ctk.CTkLabel(setting, text="Epoque").place(x=330, y=20)
    epoque = ctk.CTkEntry(setting, placeholder_text=40, width= 250)
    epoque.place(x=330, y=50)
    epoque.insert(0, parametres["epoque"])


    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="Verbose").place(x=30, y=90)
    verbose = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    verbose.place(x=30, y=120)
    verbose.insert(0, parametres["verbose"])

    progress = pstp.TrainingProgress(affichage_frame)
    ctk.CTkButton(affichage_frame, text="Exécuter", command= lambda: entrainement(les_entrys, result, loss, progress)).pack(pady=5)


    accuracy =f"Accuracy: {parametres['accuracy']*100:.2f}%"
    result  = ctk.CTkLabel(affichage_frame, text= accuracy, font=("Garamone", 20), fg_color="transparent")
    result.pack(fill="x",padx=5, pady=20,side="left")

    loss_v = f"Loss: {parametres['loss']*100:.2f}%"
    loss  = ctk.CTkLabel(affichage_frame, text=loss_v, font=("Garamone", 20), fg_color="transparent")
    loss.pack(fill="x",padx=5, pady=20, side="right")

    les_entrys = {
        
        "batch_size": batch_size,
        "epoque": epoque,
        "verbose": verbose,
    }


def entrainement(les_entry: dict, afficheLabel, lossLabel, progress):

    """
    Fonction pour entrainer le model en choisissant des paramètres
    @param:      les_entrys (Le dictionnaire des entrys)
                 afficheLabel  Le label pour afficher le résultat
    """


   
    batch_size = les_entry["batch_size"].get()
    epoque = les_entry["epoque"].get()
    verbose = les_entry["verbose"].get()



    try:
        batch_size = int(batch_size)
        les_entry["batch_size"].configure(border_color= "black")
    except:
        batch_size=0
        les_entry["batch_size"].configure(border_color= "red")

    
    try:
        epoque = int(epoque)
        les_entry["epoque"].configure(border_color= "black")
    except:
        epoque=0
        les_entry["epoque"].configure(border_color= "red")
    
    try:
        verbose = int(verbose)
        les_entry["verbose"].configure(border_color= "black")
    except:
        verbose = 0
        les_entry["verbose"].configure(border_color= "red")

    

    
    reponse = all([ batch_size,  epoque])

    if reponse:
        resultat = xor.entrantement_porte_xor( batch_size_=batch_size,  verbose= verbose, epoch= epoque, progress_bar= progress)
        afficheLabel.configure(text= f"Accuracy : {resultat[0] * 100:.2f}%")
        lossLabel.configure(text= f"Loss : {resultat[1] * 100:.2f}%")

        
    else:
        messagebox.showwarning("Paramètre d'entrainement", "Valeur incorrecte, veuillez vérifier!")


def charger_parametres_json(nom_fichier):
    # Charger les paramètres depuis le fichier JSON
    with open(nom_fichier, 'r') as f:
        parametres = json.load(f)
    
    return parametres







