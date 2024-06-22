import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json



import pages.SAVE_VALUE.diagramme_de_visualisation as psv
import pages.SAVE_VALUE.training_progress_bar as pstp
import pages.RNS.reseau_de_neuronne_sequentielle as prns

def switch(frame, page):
    for widget in frame.winfo_children():
        widget.destroy()

    page()


def affichage_resultat(frame):

    """
    
    """
    params = None
    history = None

    model = load_model("pages/RNS/model_de_rns.keras")
    with open('pages/RNS/resultats_rns.json', 'r') as json_file:
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
    entrainement = ctk.CTkButton(btn_frame, text="Entraînement", command= lambda: switch(resultat_frame, entrainement_rns),
                                    fg_color='transparent', border_color="#000", border_width=1, text_color="#000", font = ('Garamone', 14), height=35)
    entrainement.pack(fill="x", padx=2, pady=2)

    courbe_de_perte = ctk.CTkButton(btn_frame, text="Courbe de perte", command= lambda: psv.plot_loss_curve(history),
                                    fg_color='transparent', border_color="#000", border_width=1, text_color="#000", font = ('Garamone', 14), height=35)
    courbe_de_perte.pack(fill="x", padx=2, pady=2)

    courbe_de_valdation = ctk.CTkButton(btn_frame, text="Courbe de validation", command= lambda: psv.plot_validation_curve(history),
                                    fg_color='transparent', border_color="#000", border_width=1, text_color="#000", font = ('Garamone', 14), height=35)
    courbe_de_valdation.pack(fill="x", padx=2, pady=2)

    matrice_de_confusion = ctk.CTkButton(btn_frame, text="Matrice de confusion", command= lambda: psv.matrice_de_confusion_rnc() ,
                                    fg_color='transparent', border_color="#000", border_width=1, text_color="#000", font = ('Garamone', 14), height=35)
    matrice_de_confusion.pack(fill="x", padx=2, pady=2)

    images_mal_prédicte = ctk.CTkButton(btn_frame, text="Les images males prédictes", command= lambda: switch(resultat_frame, image_loss),
                                    fg_color='transparent', border_color="#000", border_width=1, text_color="#000", font = ('Garamone', 14), height=35)
    images_mal_prédicte.pack(fill="x", padx=2, pady=2)

    
    def entrainement_rns():
        entrainement_perceptron(resultat_frame, params)
    
    def image_loss():
        psv.images_mal_predictes(model, resultat_frame)


    entrainement_rns()




def entrainement_perceptron(affichage_frame, parametres):

    """
    Interface d' entrainement du modèle
    @param   : zone d'affichage des widgets
    """

    ctk.CTkLabel(affichage_frame, text="ENTRAINEMENT D'UN RESEAU DE NEURONNE SEQUENTIEL", font=("Garamone", 20)).pack(padx=20, pady= 20)
    
    setting = ttk.Labelframe(affichage_frame, text="Paramètre d'entrainement" , width=800, height=300, relief="solid", border=3)
    setting.pack(fill= "x", padx= 10)

    

    #Paramètre: la taille de données utilisé pendant une époque
    ctk.CTkLabel(setting, text="Batch size").place(x=30, y=20)
    batch_size = ctk.CTkEntry(setting, placeholder_text=10, width= 250)
    batch_size.place(x=30, y=50)
    batch_size.insert(0, parametres["batch_size"])


    #Paramètre: Nombre d'époque
    ctk.CTkLabel(setting, text="Nombre d'époque").place(x=630, y=20)
    epoque = ctk.CTkEntry(setting, placeholder_text=40, width= 250)
    epoque.place(x=630, y=50)
    epoque.insert(0, parametres["epochs"])


    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="La verbose").place(x=30, y=90)
    verbose = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    verbose.place(x=30, y=120)
    verbose.insert(0, parametres["verbose"])

    #Paramètre: la verbose
    ctk.CTkLabel(setting, text="Le nombre de neuronnes").place(x=630, y=90)
    nb_neuronne = ctk.CTkEntry(setting, placeholder_text=0, width= 250)
    nb_neuronne.place(x=630, y=120)
    nb_neuronne.insert(0, parametres["nombre_de_neuronne"])

    progress= pstp.TrainingProgress(affichage_frame)
    ctk.CTkButton(affichage_frame, text="Exécuter", command= lambda: valider_entries(les_entrys, affichage_frame, progress)).pack(pady=5)

    plot_in_frame(affichage_frame, parametres["train_loss"], parametres["val_loss"], parametres["train_accuracy"], parametres["val_accuracy"])

    les_entrys = {
        "nombre_de_neuronne": nb_neuronne,
        "batch_size": batch_size,
        "epoque": epoque,
        "verbose": verbose
    }




def plot_in_frame(parent_frame, train_loss, val_loss, train_accuracy, val_accuracy):

    #Création d'une sous frame
    frame = ctk.CTkFrame(parent_frame)
    frame.pack()

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
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



def valider_entries(entries_dict, affichage_frame, progress):
    good_value= dict()
    all_valid = True  # Variable pour suivre si toutes les valeurs sont valides
    progress.reset_progress()

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
        
        
        prns.rns(nombre_de_neuronne= good_value["nombre_de_neuronne"],
                 batch_size=good_value["batch_size"],
                 verbose= good_value["verbose"],
                 epochs= good_value["epoque"],
                 progress_bar= progress )
        
        with open('pages/RNS/resultats_rns.json', 'r') as json_file:
            data = json.load(json_file)
            params = data['params']
            
        
        #vide la zone d'affichage
        for widget in affichage_frame.winfo_children():
           if isinstance(widget, ctk.CTkFrame):
               widget.destroy()
        
        plot_in_frame(affichage_frame, params["train_loss"], params["val_loss"], params["train_accuracy"], params["val_accuracy"])

    else:
        # Sinon, afficher un message d'avertissement
        messagebox.showwarning("Erreur de validation", "Veuillez entrer des nombres entiers valides.")
        return False