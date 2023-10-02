# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:58:43 2023

@author: Pierre
"""

from math import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colors
from matplotlib import cm
import os
import numpy as np
import pandas as pd
import scipy.interpolate as sp
import fonctions as fc
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import NullFormatter

rep = "C:/Users/33671/Desktop/PIERRE_PALOMAR/Panneaux_Wenner_PD_comparaison"  # chemin du dossier à parcourir
fc.list_fichier(rep)
colonne_X = 'X-location'
colonne_Z = 'Z-location'


#%% Récupération des données excel
nom_rep="C:/Users/33671/Desktop/PIERRE_PALOMAR/Data_Panneaux_Bitkine"
tableau_forage = "CUTOFF005_total.csv"

dt_tableau = fc.get_tableau_forage(tableau_forage, nom_rep)
#%%

df2 = fc.pd_horizon_total(dt_tableau, nom_rep,colonne_X,colonne_Z) 
df3 = fc.pd_horizon_total_resistivite_moy(dt_tableau, nom_rep,colonne_X,colonne_Z) 
# Exportation vers un fichier Excel
# nom_fichier = 'tab_resistivite_horizons.xlsx'
# df2.to_excel(nom_fichier, index=False)  # Spécifiez index=False si vous ne voulez pas inclure l'index dans le fichier Excel
#%%

Resistivite_RE = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'RE'))
Resistivite_HA = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'HA'))
Resistivite_HF = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'HF'))
Resistivite_S = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'S'))


log_Resistivite_RE = np.log10(Resistivite_RE)
log_Resistivite_HA = np.log10(Resistivite_HA)
log_Resistivite_HF = np.log10(Resistivite_HF)
log_Resistivite_S = np.log10(Resistivite_S)

print(Resistivite_RE.shape[0], '\n')
print(Resistivite_HA.shape[0], '\n')
print(Resistivite_HF.shape[0], '\n')
print(Resistivite_HF.shape[0], '\n')


# Tracer les histogrammes de résistivité pour chaque horizon
plt.hist(log_Resistivite_RE, bins=30, alpha=1, label='Recouvrement', histtype = 'barstacked')
plt.hist(log_Resistivite_HA, bins=30, alpha=0.8, label='HA', histtype = 'barstacked')
plt.hist(log_Resistivite_HF, bins=30, alpha=0.4, label='HF',  histtype = 'barstacked')
plt.hist(log_Resistivite_S, bins=30, alpha=0.3, label='Socle',  histtype = 'barstacked')

# Ajouter des légendes, un titre et des étiquettes d'axes
plt.legend()
plt.title('Histogrammes de résistivité pour chaque horizon')
# ticks = np.logspace(np.log10(1), np.log10(10000), num=5)
# plt.xticks(ticks, ticks)
# plt.xscale('log')
plt.xlabel('Résistivité')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()

#%%

Moy_Resistivite_RE = Resistivite_RE.mean()
Moy_Resistivite_HA = Resistivite_HA.mean()
Moy_Resistivite_HF = Resistivite_HF.mean()
Moy_Resistivite_S = Resistivite_S.mean()

print(Moy_Resistivite_RE,Moy_Resistivite_HA, Moy_Resistivite_HF, Moy_Resistivite_S)

Median_Resistivite_RE = np.median(Resistivite_RE)
Median_Resistivite_HA = np.median(Resistivite_HA)
Median_Resistivite_HF = np.median(Resistivite_HF)
Median_Resistivite_S = np.median(Resistivite_S)

print(Median_Resistivite_RE,Median_Resistivite_HA, Median_Resistivite_HF, Median_Resistivite_S)

#%%
##############################################################################"
# Créer une matrice carrée de sous-graphiques
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Tracer les histogrammes pour chaque horizon sur les sous-graphiques correspondants
axes[0, 0].hist(log_Resistivite_RE, bins=20, color = 'blue', label='Recouvrement', histtype='barstacked')
axes[0, 1].hist(log_Resistivite_HA, bins=20, color = 'orange', label='HA', histtype='barstacked')
axes[1, 0].hist(log_Resistivite_HF, bins=20, color = 'green', label='HF', histtype='barstacked')
axes[1, 1].hist(log_Resistivite_S, bins=20, color = 'red', label='Socle', histtype='barstacked')

# Spécifier les mêmes limites pour l'axe des x sur tous les sous-graphiques
xmin = min(min(log_Resistivite_RE), min(log_Resistivite_HA), min(log_Resistivite_HF), min(log_Resistivite_S))
xmax = max(max(log_Resistivite_RE), max(log_Resistivite_HA), max(log_Resistivite_HF), max(log_Resistivite_S))

for ax in axes.flat:
    ax.set_xlim(xmin, xmax+0.5)
   
# Ajouter des légendes, un titre et des étiquettes d'axes pour chaque sous-graphique
axes[0, 0].legend()
axes[0, 0].set_title('Recouvrement')
axes[0, 0].set_xlabel('Résistivité')
axes[0, 0].set_ylabel('Fréquence')

axes[0, 1].legend()
axes[0, 1].set_title('HA')
axes[0, 1].set_xlabel('Résistivité')
axes[0, 1].set_ylabel('Fréquence')

axes[1, 0].legend()
axes[1, 0].set_title('HF')
axes[1, 0].set_xlabel('Résistivité')
axes[1, 0].set_ylabel('Fréquence')

axes[1, 1].legend()
axes[1, 1].set_title('Socle')
axes[1, 1].set_xlabel('Résistivité')
axes[1, 1].set_ylabel('Fréquence')

# Ajuster les espacements entre les sous-graphiques
plt.tight_layout()

# Afficher le graphique
plt.show()

#%%

###########################

plt.subplot()
#Tracer les histogrammes de résistivité pour chaque horizon
plt.hist(Resistivite_RE, bins=50, alpha=0.3, label='Recouvrement', histtype = 'barstacked')
plt.hist(Resistivite_HA, bins=50, alpha=0.3, label='HA', histtype = 'barstacked')
plt.hist(Resistivite_HF, bins=50, alpha=0.3, label='HF',  histtype = 'barstacked')
plt.hist(Resistivite_S, bins=20, alpha=0.3, label='Socle',  histtype = 'barstacked')

# Ajouter des légendes, un titre et des étiquettes d'axes
plt.legend()
plt.title('Histogrammes de résistivité pour chaque horizon')
# ticks = np.logspace(np.log10(1), np.log10(10000), num=5)
# plt.xticks(ticks, ticks)
plt.xlabel('Résistivité')
plt.xscale('log')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()



#%%






#%%

df2 = fc.pd_litho_total(dt_tableau, nom_rep,colonne_X,colonne_Z) 
print(df2)

# Exportation vers un fichier Excel
# nom_fichier = 'tab_resistivite_horizons.xlsx'
# df2.to_excel(nom_fichier, index=False)  # Spécifiez index=False si vous ne voulez pas inclure l'index dans le fichier Excel

#%%
# dlargile = df2.loc[(df2['Lithologie']== 'ARGILE')]
# dlargilesableuse = df2.loc[(df2['Lithologie']== 'ARGILE SABLEUSE')]
# dlsableargileux = df2.loc[(df2['Lithologie']== 'SABLE ARGILEUX')]
                   
#dl2 = dl.loc[(dl['Type_lithologie']=='FRACTURE')]
#print(dl, dl2)
# Resistivite_Argile = fc.gamme_resistivite_litho(df2, 'Lithologie', 'ARGILE')
# Resistivite_Argile = np.sort(Resistivite_Argile)
# Resistivite_Argile_Sableuse = fc.gamme_resistivite_litho(df2, 'Lithologie', 'ARGILE SABLEUSE')
# Resistivite_Argile_Sableuse = np.sort(Resistivite_Argile_Sableuse)
# Resistivite_Sable_Argileux = fc.gamme_resistivite_litho(df2, 'Lithologie', 'SABLE ARGILEUX')
# Resistivite_Sable_Argileux = np.sort(Resistivite_Sable_Argileux)

Resistivite_Argile_Sable_tot = np.hstack([np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'ARGILE')), np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'ARGILE SABLEUSE')), np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'SABLE ARGILEUX')), np.array(fc.gamme_resistivite_litho(df2, 'Lithologie','SABLE'))])
#Resistivite_Argile_tot = np.sort(Resistivite_Argile_tot)
Resistivite_Granite_tot = np.hstack([np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'GRANITE')),np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'MICROGRANITE')), np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'MONZOGRANITE'))])
Resistivite_Diorite_tot = np.hstack([np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'DIORITE')), np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'GRANODIORITE')),np.array(fc.gamme_resistivite_litho(df2, 'Lithologie', 'DOLERITE')),])



log_Resistivite_Argile_Sable_tot = np.log10(Resistivite_Argile_tot)
log_Resistivite_Granite_tot = np.log10(Resistivite_Granite_tot)
log_Resistivite_Diorite_tot = np.log10(Resistivite_Diorite_tot)





plt.hist(log_Resistivite_Argile_Sable_tot, bins=30, alpha=1,color = 'black', label='Argile et Sable', histtype = 'barstacked')
plt.hist(log_Resistivite_Granite_tot, bins=30, alpha=0.5, label='Granite à biotites', histtype = 'barstacked')
plt.hist(log_Resistivite_Diorite_tot, bins=30, alpha=0.6, label='Diorite, Dolérite, Granodiorite', histtype = 'barstacked')

# Ajouter des légendes, un titre et des étiquettes d'axes
plt.legend(fontsize = 7.5)
plt.title('Histogrammes de résistivité pour chaque lithologie')
# ticks = np.logspace(np.log10(1), np.log10(10000), num=5)
# plt.xticks(ticks, ticks)
plt.xlabel('Log(Résistivité)')
# plt.xscale('log')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()

#%% 
n = Resistivite_Argile_tot.shape[0]
i = 0
while Resistivite_Argile_tot[i]<50 :
    i = i+1
print(i/n)

#%%

Resistivite_RE = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'RE'))
Resistivite_HA = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'HA'))
Resistivite_HF = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'HF'))
Resistivite_S = np.array(fc.gamme_resistivite_litho(df2, 'Horizon', 'S'))


log_Resistivite_RE = np.log10(Resistivite_RE)
log_Resistivite_HA = np.log10(Resistivite_HA)
log_Resistivite_HF = np.log10(Resistivite_HF)
log_Resistivite_S = np.log10(Resistivite_S)

# Tracer les histogrammes de résistivité pour chaque horizon
plt.hist(log_Resistivite_RE, bins=20, alpha=1, label='Recouvrement', histtype = 'barstacked')
plt.hist(log_Resistivite_HA, bins=20, alpha=0.7, label='HA', histtype = 'barstacked')
plt.hist(log_Resistivite_HF, bins=20, alpha=0.5, label='HF',  histtype = 'barstacked')
plt.hist(log_Resistivite_S, bins=20, alpha=0.3, label='Socle',  histtype = 'barstacked')

# Ajouter des légendes, un titre et des étiquettes d'axes
plt.legend()
plt.title('Histogrammes de résistivité pour chaque horizon')
# ticks = np.logspace(np.log10(1), np.log10(10000), num=5)
# plt.xticks(ticks, ticks)
plt.xlabel('Résistivité')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()

plt.subplot()
# Tracer les histogrammes de résistivité pour chaque horizon
plt.hist(Resistivite_RE, bins=50, alpha=1, label='Recouvrement', histtype = 'barstacked')
plt.hist(Resistivite_HA, bins=50, alpha=0.7, label='HA', histtype = 'barstacked')
plt.hist(Resistivite_HF, bins=50, alpha=0.5, label='HF',  histtype = 'barstacked')
plt.hist(Resistivite_S, bins=20, alpha=0.3, label='Socle',  histtype = 'barstacked')

# Ajouter des légendes, un titre et des étiquettes d'axes
plt.legend()
plt.title('Histogrammes de résistivité pour chaque horizon')
# ticks = np.logspace(np.log10(1), np.log10(10000), num=5)
# plt.xticks(ticks, ticks)
plt.xlabel('Résistivité')
plt.xscale('log')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()








