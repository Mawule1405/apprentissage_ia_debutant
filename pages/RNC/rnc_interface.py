import customtkinter as ctk
from tkinter import ttk
from tensorflow.keras.models import load_model
import json


import pages.SAVE_VALUE.diagramme_de_visualisation as psv

def affichage_resultat(frame):

    """
    
    """
    params = None
    history = None

    model = load_model("pages/RNC/model_de_rnc.keras")
    with open('pages/RNC/resultats_rnc.json', 'r') as json_file:
        data = json.load(json_file)
        params = data['params']
        history = data['history']
    
    #Zone des boutons
    btn_frame = ctk.CTkFrame(frame, width=300)
    btn_frame.pack(fill="y", padx=2, pady=2, side="left")

    #Zone des résultats
    resultat_frame = ctk.CTkFrame(frame)
    resultat_frame.pack(fill="both", padx=2, pady=2, side="left")


    #les boutons
    entrainement = ttk.Button(btn_frame, text="Entraînement")
    entrainement.pack(fill="x", padx=2, pady=2)

    courbe_de_perte = ttk.Button(btn_frame, text="Courbe de perte", command= lambda: psv.plot_loss_curve(history))
    courbe_de_perte.pack(fill="x", padx=2, pady=2)

    courbe_de_valdation = ttk.Button(btn_frame, text="Courbe de validation", command= lambda: psv.plot_validation_curve(history))
    courbe_de_valdation.pack(fill="x", padx=2, pady=2)

    matrice_de_confusion = ttk.Button(btn_frame, text="Matrice de confusion", command= lambda: psv.matrice_de_confusion_rnc() )
    matrice_de_confusion.pack(fill="x", padx=2, pady=2)

    images_mal_prédicte = ttk.Button(btn_frame, text="20 images mals prédictes", command= lambda: psv.images_mal_predictes_rnc( resultat_frame))
    images_mal_prédicte.pack(fill="x", padx=2, pady=2)




def entrainement_perceptron(affichage_frame, parametres):

    """
    Interface d' entrainement du modèle
    @param   : zone d'affichage des widgets
    """


    ctk.CTkLabel(affichage_frame, text="ENTRAINEMENT D'UN CNN", font=("Garamone", 20)).pack(padx=20, pady= 20)
    
    setting = ttk.Labelframe(affichage_frame, text="Paramètre d'entrainement" , width=800, height=300, relief="solid", border=3)
    setting.pack(fill= "x", padx= 10)

    

    #Paramètre: la taille de données utilisé pendant une époque
    ctk.CTkLabel(setting, text="Taille de données par epoque").place(x=30, y=20)
    batch_size = ctk.CTkEntry(setting, placeholder_text=10, width= 250)
    batch_size.place(x=30, y=50)
    batch_size.insert(0, parametres["batch_size"])

    #Paramètre: Nombre d'époque
    ctk.CTkLabel(setting, text="Nombre d'époque").place(x=330, y=20)
    epoque = ctk.CTkEntry(setting, placeholder_text=40, width= 250)
    epoque.place(x=330, y=50)
    epoque.insert(0, parametres["epoque"])


    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="La verbose").place(x=30, y=90)
    verbose = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    verbose.place(x=30, y=120)
    verbose.insert(0, parametres["verbose"])

    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="La verbose").place(x=330, y=90)
    nb_couche = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    nb_couche.place(x=330, y=120)
    nb_couche.insert(0, parametres["nombre_de_couche"])

    ctk.CTkButton(affichage_frame, text="Exécuter", command= lambda: "entrainement(les_entrys, result, loss)").pack(pady=5)


    accuracy =f"Accuracy: {parametres['train_accuracy']*100:.2f}%"
    result  = ctk.CTkLabel(affichage_frame, text= accuracy, font=("Garamone", 40), fg_color="transparent")
    result.pack(fill="x",padx=5, pady=20,side="left")

    loss_v = f"Loss: {parametres['train_loss']*100:.2f}%"
    loss  = ctk.CTkLabel(affichage_frame, text=loss_v, font=("Garamone", 40), fg_color="transparent")
    loss.pack(fill="x",padx=5, pady=20, side="right")

    les_entrys = {
        "nombre_de_couche": nb_couche,
        "batch_size": batch_size,
        "epoque": epoque,
        "verbose": verbose,
    }
