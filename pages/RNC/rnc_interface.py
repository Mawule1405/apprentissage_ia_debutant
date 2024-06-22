import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
from tensorflow.keras.models import load_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import json


import pages.SAVE_VALUE.diagramme_de_visualisation as psv


def switch(frame, page):
    for widget in frame.winfo_children():
        widget.destroy()

    page()

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
    entrainement = ttk.Button(btn_frame, text="Entraînement", command= lambda: switch(resultat_frame, entrainement_cnn))
    entrainement.pack(fill="x", padx=2, pady=2)

    courbe_de_perte = ttk.Button(btn_frame, text="Courbe de perte", command= lambda: psv.plot_loss_curve(history))
    courbe_de_perte.pack(fill="x", padx=2, pady=2)

    courbe_de_valdation = ttk.Button(btn_frame, text="Courbe de validation", command= lambda: psv.plot_validation_curve(history))
    courbe_de_valdation.pack(fill="x", padx=2, pady=2)

    matrice_de_confusion = ttk.Button(btn_frame, text="Matrice de confusion", command= lambda: psv.matrice_de_confusion_rnc() )
    matrice_de_confusion.pack(fill="x", padx=2, pady=2)

    images_mal_prédicte = ttk.Button(btn_frame, text="20 images mals prédictes", command= lambda: switch(resultat_frame, image_loss))
    images_mal_prédicte.pack(fill="x", padx=2, pady=2)

    
    def entrainement_cnn():
        entrainement_perceptron(resultat_frame, params)
    
    def image_loss():
        psv.images_mal_predictes_rnc(resultat_frame)


    entrainement_cnn()



def entrainement_perceptron(affichage_frame, parametres):

    """
    Interface d' entrainement du modèle
    @param   : zone d'affichage des widgets
    """


    ctk.CTkLabel(affichage_frame, text="ENTRAINEMENT D'UN CNN", font=("Garamone", 20)).pack(padx=20, pady= 20)
    
    style = ttk.Style()
    style.configure("Custom.TLabelframe", background="#f0f0f0", borderwidth=3, relief="solid")
    style.configure("Custom.TLabelframe.Label", font=("Arial", 12, "bold"))

    # Créer un Labelframe personnalisé
    setting = ttk.Labelframe(affichage_frame, text="Paramètre d'entraînement", style="Custom.TLabelframe", width=1200, height=300)
    setting.pack(fill="x", padx=10, pady=10)


    

    #Paramètre: la taille de données utilisé pendant une époque
    ctk.CTkLabel(setting, text="Taille de données par epoque").place(x=30, y=20)
    batch_size = ctk.CTkEntry(setting, placeholder_text=10, width= 250)
    batch_size.place(x=30, y=50)
    batch_size.insert(0, parametres["batch_size"])

    #Paramètre: Nombre d'époque
    ctk.CTkLabel(setting, text="Nombre d'époque").place(x=330, y=20)
    epoque = ctk.CTkEntry(setting, placeholder_text=40, width= 250)
    epoque.place(x=330, y=50)
    epoque.insert(0, parametres["epochs"])


    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="La verbose").place(x=630, y=20)
    verbose = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    verbose.place(x=630, y=50)
    verbose.insert(0, parametres["verbose"])

    #Paramètre: taille de filtre
    ctk.CTkLabel(setting, text="Taille du filtre").place(x=30, y=90)
    taille_filtre = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    taille_filtre.place(x=30, y=120)
    taille_filtre.insert(0, parametres["taille_filtre"][0])


    #Paramètre: Nombre de filtre
    ctk.CTkLabel(setting, text="Nombre de filtre").place(x=330, y=90)
    nb_filtre = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    nb_filtre.place(x=330, y=120)
    nb_filtre.insert(0, parametres["nombre_de_filtre"])


   #Paramètre: taille de pooling
    ctk.CTkLabel(setting, text="Taille de pooling").place(x=630, y=90)
    taille_pooling = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    taille_pooling.place(x=630, y=120)
    taille_pooling.insert(0, parametres["taille_pooling"][0])



    ctk.CTkButton(affichage_frame, text="Exécuter", command= lambda: valider_entries(les_entrys)).pack(pady=5)

    plot_in_frame(affichage_frame, parametres["train_loss"], parametres["val_loss"], parametres["train_accuracy"], parametres["val_accuracy"])


    les_entrys = {
        "nombre_de_filtre": nb_filtre,
        "batch_size": batch_size,
        "epochs": epoque,
        "verbose": verbose,
        "taille_filtre": taille_filtre,
        "taille_pooling": taille_pooling
    }





def plot_in_frame(parent_frame, train_loss, val_loss, train_accuracy, val_accuracy):

    # Créer une figure de Matplotlib
    fig = Figure(figsize=(14, 6))

    # Courbe de perte
    ax1 = fig.add_subplot(121)
    ax1.plot(train_loss, label='Perte d\'entraînement')
    ax1.plot(val_loss, label='Perte de validation')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Perte')
    ax1.set_title('Courbe de perte')
    ax1.legend()

    # Courbe de précision
    ax2 = fig.add_subplot(122)
    ax2.plot(train_accuracy, label='Précision d\'entraînement')
    ax2.plot(val_accuracy, label='Précision de validation')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Précision')
    ax2.set_title('Courbe de précision')
    ax2.legend()

    # Intégrer la figure dans le Tkinter Frame
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



def valider_entries(entries_dict):
    good_value= dict()
    all_valid = True  # Variable pour suivre si toutes les valeurs sont valides

    # Parcourir chaque entrée dans le dictionnaire
    for key, entry_widget in entries_dict.items():
        value = entry_widget.get()  # Obtenir la valeur de l'entry

        # Vérifier si la valeur est un entier
        try:
            int_value = int(value)
            good_value[key] = int_value
            
            # Si c'est un entier, configurer la bordure en noir
            entry_widget.configure(border_color="black")
        except ValueError:
            # Si ce n'est pas un entier, configurer la bordure en rouge
            entry_widget.configure(border_color="red")
            all_valid = False  # Marquer comme non valide
            continue  # Passer à l'entry suivante

    # Si toutes les valeurs sont valides, retourner True
    if all_valid:
        
        return good_value
    else:
        # Sinon, afficher un message d'avertissement
        messagebox.showwarning("Erreur de validation", "Veuillez entrer des nombres entiers valides.")
        return False