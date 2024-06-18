
import customtkinter as ctk

def build_acceuil(conteneur : ctk.CTkFrame):
    ctk.CTkLabel(conteneur, text="INSTITUT AFRICAIN D'INFORMATIQUE", fg_color="transparent", text_color="#000",
                 font=("Garamone", 35, "bold")).pack(pady=5)
    ctk.CTkLabel(conteneur, text=" Etablissement d'Enseignement Supérieurs", fg_color="transparent", text_color="#000"
                 , font= ("Garamone", 25, "bold")).pack(pady=5)
    ctk.CTkLabel(conteneur, text="Tel: (+241) 07 70 55 00     BP: 2263//Libreville-Gabon", fg_color="transparent", text_color="#000",
                 font=("Garamone",15, "bold")).pack(pady=5)
    
    ctk.CTkLabel(conteneur, text="TP d'Intelligence Artificielle", fg_color="transparent", text_color="#2e538d",
                 font=("Garamone", 30, "bold")).pack(pady=15)
    
    titre = ctk.CTkFrame(conteneur, width=600, height=230, border_color="#ff8", border_width=5,
                        fg_color="transparent")
    titre.pack(pady= 10)

    ctk.CTkLabel(titre, text="TECHNIQUES DE DEVELOPPEMENT DES MODELES \nD'INTELLIGENCE ARTIFICIELLE",
                 font=("Garamone", 40, "bold")).pack(pady=15, padx=10)
    
    ctk.CTkLabel(titre, text="Enseignant: Dr. NOUSSI Roger", font=("Garamone", 25) ).pack(pady=15)

    etudiants_conteneur = ctk.CTkFrame(conteneur, fg_color="transparent", border_width=0)
    etudiants_conteneur.pack(side="bottom", pady=20)

    ctk.CTkLabel(etudiants_conteneur, text="Réalisé par: AGBOSSA Yao et HELOU Komlan Mawulé", font=("Garamone", 20, "bold")).pack(pady= 10)
    ctk.CTkLabel(etudiants_conteneur, text="Année Scolaire: 2023 2024", font=("Garamone", 20, "bold")).pack(pady= 10)


    
