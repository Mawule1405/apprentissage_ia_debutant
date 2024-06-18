import customtkinter as ctk
from tkinter import ttk
import tkinter as tk

#Importation de mes modules
import pages.PORTE_ET.interface_porte_et as ppiet
import pages.PORTE_OU.interface_porte_ou as ppou
import pages.PORTE_XOR.interface_porte_xor as ppxor
import pages.Acceuil.accueil as paa
import pages.RNS.rns_interface as prnsi
import pages.RNC.rnc_interface as prnci

def switch(frame, page):
    for widget in frame.winfo_children():
        widget.destroy()

    page()

#interface utilisateur
fen  = ctk.CTk(fg_color="#ddd")
fen.geometry("1200x700+0+0")
fen.title("TP d'Intelligence Artificielle")


#mise en place des menus
menubar = tk.Menu(selectcolor="red", font=("Garamone", 20), borderwidth=2)

"""Premier niveau"""
accueil_menu = tk.Menu(menubar, tearoff=0)
ps_menu = tk.Menu(menubar, tearoff=0)
pmc_menu = tk.Menu(menubar, tearoff=0)

"""Deuxième niveau"""
menubar.add_cascade(label="Accueil", menu= accueil_menu)
menubar.add_cascade(label="Perceptron simple", menu = ps_menu)
menubar.add_cascade(label="Perceptron multicouches", menu = pmc_menu)


"""Commande perceptron simple"""
accueil_menu.add_command(label="Accueil", font= ("Garamone", 13), command= lambda: switch(conteneur, construire_accueil))
accueil_menu.add_command(label="Quitter",  font=("Garamone", 13),  command= lambda: fen.destroy())

ps_menu.add_command(label="Perceptron ET",  font=("Garamone", 13),  command= lambda : ppiet.build_percetron_and())
ps_menu.add_command(label="Perceptron OU",  font=("Garamone", 13),  command= lambda: ppou.build_percetron_or())


"""Commande perceptron multicouches"""
pmc_menu.add_command(label="Perceptron XOR",  font=("Garamone", 13),  command= lambda: ppxor.build_percetron_xor())
pmc_menu.add_command(label="SNN",  font=("Garamone", 13),  command= lambda: switch(conteneur, construire_rns))
pmc_menu.add_command(label="CNN",  font=("Garamone", 13),  command= lambda:switch(conteneur, construire_rnc))
fen.config(menu= menubar)

#Zone d'affichage des résultats de SNN et CNN
conteneur = ctk.CTkFrame(fen, fg_color="transparent", corner_radius=0)
conteneur.pack(fill="both",padx=5, pady=5)
paa.build_acceuil(conteneur)


def construire_accueil():
    paa.build_acceuil(conteneur)

def construire_rns():
    prnsi.affichage_resultat(conteneur)

def construire_rnc():
    prnci.affichage_resultat(conteneur)
fen.mainloop()