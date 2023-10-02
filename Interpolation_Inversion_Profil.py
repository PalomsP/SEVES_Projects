# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:10:21 2023

@author: Pierre
"""
from math import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import os
import numpy as np
import pandas as pd
import scipy.interpolate as sp
import fonctions as fc
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
#%% Récupération des fichiers dans le dossier
#Ce programme permet de récupérer les fichiers .dat et .bln, et les afficher,
#classés par ordre alphabétique, il suffit de les afficher dans la console, 
#copier le tout et le coller dans le fichier excel, que l'on transforme 
#ensuite en .csv pour le récupérer en dataframe.

rep = "C:/Users/33671/Desktop/PIERRE_PALOMAR/Panneaux_Wenner_PD_comparaison"  # chemin du dossier à parcourir
fc.list_fichier(rep)
colonne_X = 'X-location'
colonne_Z = 'Z-location'


#%% Récupération des données excel
nom_rep="C:/Users/33671/Desktop/PIERRE_PALOMAR/Data_Panneaux_Bitkine"
tableau_forage = "CUTOFF005_EXTENDED.csv"

dt_tableau = fc.get_tableau_forage(tableau_forage, nom_rep)

#%%
rep="C:/Users/33671/Desktop/PIERRE_PALOMAR/Data_Panneaux_Bitkine"
os.chdir(nom_rep)
os.listdir()
os.getcwd()

oids =109
Fichier_Panneau,fichier_blanc = fc.get_nom_panneau(dt_tableau, oids, rep)
fig = fc.traitement_forage_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)




#%%
figs = []

nom_rep="C:/Users/33671/Desktop/PIERRE_PALOMAR/Data_Panneaux_Bitkine"
os.chdir(nom_rep)
os.listdir()
os.getcwd()


for oids in dt_tableau["OID"].to_numpy() :
    print(oids)
    Fichier_Panneau,fichier_blanc = fc.get_nom_panneau(dt_tableau, oids, nom_rep)
    fig = fc.traitement_forage_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)
    figs.append(fig)

with PdfPages('CUTOFF005_EXTENDED.pdf') as pdf:
    for fig in figs :
        pdf.savefig(fig, bbox_inches='tight')  # regarder dpi =

#%%

nom_rep="C:/Users/33671/Desktop/PIERRE_PALOMAR/Data_Panneaux_Bitkine"
df = pd_litho_total(dt_tableau, nom_rep,colonne_X,colonne_Z) 
print(df)

#%%

df2 = pd_horizon_total(dt_tableau, nom_rep,colonne_X,colonne_Z) 
print(df2)

# Exportation vers un fichier Excel
nom_fichier = 'tab_resistivite_horizons.xlsx'
df2.to_excel(nom_fichier, index=False)  # Spécifiez index=False si vous ne voulez pas inclure l'index dans le fichier Excel

#%%
histogramme_resistivites(df2)

