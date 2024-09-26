# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:38:10 2023

@author: Pierre
"""
from math import * # utile pour floor
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib import colors
# from matplotlib import cm
import os
import numpy as np
import pandas as pd
# import scipy.interpolate as sp
from scipy.interpolate import griddata
# from scipy.interpolate import RegularGridInterpolator
# from matplotlib.ticker import NullFormatter
from matplotlib.backends.backend_pdf import PdfPages

def list_fichier(rep):
    """
    Ce programme permet de récupérer les fichiers .dat et .bln, et les afficher,
    classés par ordre alphabétique, il suffit de les afficher dans la console,
    copier le tout et le coller dans le fichier excel, que l'on transforme
    ensuite en .csv pour le récupérer en dataframe.
    """

    bln_files = []  # liste pour stocker les noms des fichiers .bln
    dat_files = []  # liste pour stocker les noms des fichiers .dat

    # Parcours des fichiers dans le répertoire
    for filename in sorted(os.listdir(rep), key=lambda x: x.lower()):
        if filename.endswith(".bln"):
            bln_files.append(filename)  # Ajout du nom du fichier .bln à la liste
        elif filename.endswith(".dat"):
            dat_files.append(filename)  # Ajout du nom du fichier .dat à la liste

    # Affichage des noms des fichiers .bln triés par ordre alphabétique
    print("Fichiers .bln :")
    for bln_file in sorted(bln_files, key=lambda x: x.lower()):
        print(bln_file)

    # Affichage des noms des fichiers .dat triés par ordre alphabétique
    print("\nFichiers .dat :")
    for dat_file in sorted(dat_files, key=lambda x: x.lower()):
        print(dat_file)

    return dat_file, bln_file

## Le programme suivant prend en compte que les nom des fichier sont stocker dans un tableau (ce qui me permet de les corréler au données ce forages de manière automatique)
## Il n'est pas indidpensable si l'on a déjà les noms des fichiers et qu'on souhaite en représenter qu'un seul
def get_tableau_forage(tableau_forage, nom_rep):
    """
    Ce programme permet de charger un fichier CSV contenant un tableau de forage,
    en spécifiant le séparateur de colonnes et l'entête, puis de supprimer les lignes
    ayant une valeur manquante dans la colonne 'OID'. Enfin, il renvoie le DataFrame
    contenant le tableau de forage.
    """
    # Changement du répertoire de travail
    os.chdir(nom_rep)
    # Affichage du contenu du répertoire (facultatif)
    os.listdir()
    # Affichage du répertoire de travail (facultatif)
    os.getcwd()
    # Lecture du fichier CSV avec pandas, en spécifiant le séparateur de colonnes et l'entête
    dt_tableau = pd.read_csv(tableau_forage, sep=";", header=0)
    # Suppression des lignes ayant une valeur manquante dans la colonne 'OID'
    dt_tableau.dropna(subset=['OID'], inplace=True)

    return dt_tableau

def get_nom_panneau(dt_tableau: pd.DataFrame, oid: int, nom_rep: str) -> tuple[str, str]:
    """
    Ce programme permet d'obtenir les noms des fichiers de panneau et de fichier blanc
    correspondant à un OID spécifié dans un DataFrame. Il change également le répertoire
    de travail et affiche les informations relatives au répertoire.
    
    :param dt_tableau: DataFrame contenant les données du tableau
    :param oid: OID spécifié
    :param nom_rep: Chemin du répertoire
    :return: Tuple contenant les noms de fichier de panneau et de fichier blanc
    """

    # Changement du répertoire de travail en utilisant le chemin spécifié
    os.chdir(nom_rep)

    # Affichage du contenu du répertoire (facultatif)
    os.listdir()

    # Affichage du répertoire de travail (facultatif)
    os.getcwd()

    # Extraction du nom du fichier de panneau correspondant à l'OID spécifié
    Fichier_Panneau = dt_tableau.loc[dt_tableau['OID'] == oid, 'FICHIER_PANNEAU']

    # Extraction du nom du fichier blanc correspondant à l'OID spécifié
    fichier_blanc = dt_tableau.loc[dt_tableau['OID'] == oid, 'FICHIER_BLANC']
    # Conversion des valeurs extraites en tant que type str et renvoi en tant que tuple
    return Fichier_Panneau.item(), fichier_blanc.item()

def get_data(nom_rep: str, fichier_Panneau: str) -> pd.DataFrame:
    """
    Ce programme permet de récupérer les données d'un fichier .csv et les renvoie
    dans un DataFrame. Le fichier doit contenir des séparations de type espace ou tabulation.
    
    :param nom_rep: Chemin du répertoire
    :param fichier_Panneau: Nom du fichier .csv à lire
    :return: DataFrame contenant les données du fichier
    """
    # Changement du répertoire de travail en utilisant le chemin spécifié
    os.chdir(nom_rep)
    # Affichage du contenu du répertoire (facultatif)
    os.listdir()
    # Affichage du répertoire de travail (facultatif)
    os.getcwd()
    # Lecture du fichier .csv avec des séparations de type espace ou tabulation
    dt = pd.read_csv(fichier_Panneau, sep='\s+', header=0)
    # Retourne le DataFrame contenant les données du fichier
    return dt
    

def mise_en_forme(Fichier_Panneau): 
    """ Supprime les guillemets et les virgules en début de 
    fichier pour les transformer en espace, permet l'utilisation de la
    fonction get_data"""
    fichier  = open(Fichier_Panneau, 'r')
    a = fichier.read()
    fichier.close()
    
    ## Cette partie permet la suppression des guillemets
    characters = '"'
    a = ''.join( x for x in a if x not in characters)
    
    ## Celle-ci le remplacement des virgule par des espaces
    a = a.replace(',', ' ')
    
    ## On réécrit tout ça dans le fichier d'origine
    fichier  = open(Fichier_Panneau, 'w')
    fichier.write(a)
    fichier.close()
    

# def show_panneau(X, Z, Resistivite):        
#     fig, ax = plt.subplots(figsize = (20,4))
#     levels = np.logspace(1.0, 3*np.log(3), num=16)
#     cs = ax.contourf(X, Z, Resistivite, norm=colors.LogNorm(vmin=10, vmax=3000, clip=False),vmin=10, vmax=3000, levels = levels, cmap = 'rainbow')
#     cbar = fig.colorbar(cs)
#     return (cs)
    
def get_values_array(donnees, colonne_X, colonne_Z):
    """ Permet d'extraire les données dufichier.dat avec les résistivités et
    les retourne sous la forme de 3 vecteurs X,Z,R"""
    Xt = []
    for i in range(0, len(donnees[colonne_X])): 
        Xt.append(donnees[colonne_X][i])
    Xtot = np.array(Xt) 
    
    Zt = []
    for i in range(0, len(donnees[colonne_Z])): 
        Zt.append(donnees[colonne_Z][i])
    Ztot = np.array(Zt)
    
    Rt = []
    for i in range(0, len(donnees['Resistivity'])): 
        Rt.append(donnees['Resistivity'][i])
    Rtot = np.array(Rt)
    return (Xtot, Ztot, Rtot)

def interpol_2(X, Z, R, grid_X, grid_Z, method='linear'):
    """ Permet de réaliser l'interpolation des données sur une grille de maille 1m
    Il est nécessaire d'entrer la grille voulue en paramètre (grid_X, grid_Z créés avec Meshgrid)"""
    points = np.column_stack((X, Z))
    values = R
    grid_points = np.column_stack((grid_X.ravel(), grid_Z.ravel()))
    R_interpol = griddata(points, values, grid_points, method=method)
    R_interpol = R_interpol.reshape(grid_X.shape)
    return R_interpol


def interpol(donnees: pd.DataFrame, colonne_X: str, colonne_Z: str, methode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Réalise l'interpolation des données suivant la méthode choisie parmi 'nearest', 'linear', 'cubic'.
    Les colonne_X et colonne_Z sont les noms des colonnes en chaîne de caractères.
    Le programme retourne les abscisses (X_interpol), les ordonnées (Z_interpol) et le panneau interpolé (R_interpol).
    
    :param donnees: DataFrame contenant les données
    :param colonne_X: Nom de la colonne pour les abscisses
    :param colonne_Z: Nom de la colonne pour les ordonnées
    :param methode: Méthode d'interpolation ('nearest', 'linear', 'cubic')
    :return: Tuple contenant les abscisses, les ordonnées et le panneau interpolé
    """
    # Définition des bornes de l'interpolation
    min_X = floor(donnees[colonne_X].min())
    max_X = ceil(donnees[colonne_X].max())
    min_Z = floor(donnees[colonne_Z].min())
    max_Z = ceil(donnees[colonne_Z].max())
    
    # Génération des valeurs d'interpolation pour les abscisses et les ordonnées
    X_interpol = np.linspace(min_X, max_X, max_X - min_X + 1)
    Z_interpol = np.linspace(min_Z, max_Z, max_Z - min_Z + 1)
    Z_interpol = np.flip(Z_interpol)
    
    # Création de la grille d'interpolation avec meshgrid
    grid_X, grid_Z = np.meshgrid(X_interpol, Z_interpol)
    
    # Extraction des valeurs X, Z et R à partir des données
    Xtot, Ztot, Rtot = get_values_array(donnees, colonne_X, colonne_Z)
    
    # Interpolation des valeurs R avec griddata
    #R_interpol = griddata((Xtot, Ztot), Rtot, (grid_X, grid_Z), method=methode,options={'Qhull':'Qz'})
    R_interpol = interpol_2(Xtot, Ztot, Rtot, grid_X, grid_Z, method=methode)
    # Retourne les abscisses, les ordonnées et le panneau interpolé
    return X_interpol, Z_interpol, R_interpol


def filtre_blanc(R_interpol, X_interpol, Z_interpol, blanc):
    ''' Cette fonction prend en entrée le panneau interpolé et lui applique 
    le filtre blanc permettant de ne pas représenter les valeurs indésirable''' 
    nZ, nX = R_interpol.shape
    i_b,j_b = blanc.shape
    i_median=0
    R_interpol_filtre = R_interpol.copy()
    while blanc[i_median,0] <= blanc[i_median+1,0] :
        i_median+=1    
    for jr in range(nX-1) :
        for ib in range(i_median) :
            if (blanc[ib,0]<= X_interpol[jr] and blanc[ib+1,0]>= X_interpol[jr]):
                d = blanc[ib+1,0]-blanc[ib,0]
                ib_interpol = blanc[ib+1,1]*(X_interpol[jr]-blanc[ib,0])/d + blanc[ib+1,1]*(blanc[ib,0]-X_interpol[jr])/d
                for ir in range(nZ-1): 
                    if Z_interpol[ir]>=ib_interpol :
                        R_interpol_filtre[ir,jr] = np.nan
        for ib in range(i_median, i_b-1) :
            if (X_interpol[jr] <= blanc[ib,0] and X_interpol[jr] >= blanc[ib+1,0]):
                d = abs(blanc[ib+1,0]-blanc[ib,0])
                ib_interpol = blanc[ib+1,1]*(abs(X_interpol[jr]-blanc[ib,0]))/d + blanc[ib,1]*abs(blanc[ib+1,0]-X_interpol[jr])/d           
                for ir in range(nZ-1): 
                    if Z_interpol[ir]<=ib_interpol :
                        R_interpol_filtre[ir,jr] = np.nan
    return(R_interpol_filtre)


def Panneau(X_interpol, Z_interpol, R_interpol, dt_tableau, oid):
    ## Récupération de l'indice dans le dataframe correspondant à la ligne de notre panneau
    indice = dt_tableau[dt_tableau['OID'] == oid].index.to_numpy()
    ## Création de la figure et des axes avec le même ordonné et un ratio de 1 pour 7 entre les deux
    fig, ax3 = plt.subplots(figsize=(20, 4), dpi = 100) 

    # Plafonnage des valeurs à 3000 
    R_interpol[R_interpol > 4000] = 4000
    R_interpol[R_interpol < 10] = 10
    #Affichage du panneau en échelle log, et de la colorbar
    norm = colors.LogNorm(vmin=10, vmax=4000)
    #levels_1 = np.logspace(1.0, np.log10(4000), num=16)
    levels_1 = np.logspace(1, 4, num=20, base=10) 
    #   Affichage de l'interpolation avec pcolormesh
    # pan = ax3.pcolormesh(X_interpol, Z_interpol, R_interpol, norm=norm, cmap='rainbow')
    
    pan = ax3.contourf(X_interpol, Z_interpol, R_interpol, norm=colors.LogNorm(vmin=10, vmax=4000, clip=False),vmin=10,vmax=4000, levels = levels_1, cmap = 'rainbow')
    cb = fig.colorbar(pan, ax=ax3,ticks=[10, 100, 1000, 3000])
    cb.ax.set_yticklabels(['10', '100', '1000', '3000'])
    ## Mise en page du panneau, des labels, et des ticks
    cb.set_label('Résistivité (Ohm.m)',labelpad=25 ,rotation = 270, fontsize = 14)
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Profondeur (m)', labelpad=25, rotation=-90)
    ax3.tick_params(axis='y', which='both', width=1, length=5)
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.label.set_size(14)
    ax3.yaxis.label.set_size(14)
    
    ## Titre selon de le nom du village prospecté récupérer dans le dataframe
    fig.suptitle(dt_tableau.loc[indice[0], 'NOM_VILLAG'], fontsize=22, x=0.5, y=1.05)
    
    # ## Affichage des débits 
    # if not np.isnan(dt_tableau.loc[indice[0], 'Q_DEVELOPP']) : 
    #     Qd = "Qd = " + str(dt_tableau.loc[indice[0], 'Q_DEVELOPP'])+" m3/h"
    #     ax3.text(0.4, 1.09, Qd, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'blue')
    # else : 
    #     Qd = "Qd non renseigné"
    #     ax3.text(0.4, 1.09, Qd, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'blue')
    
    # if not np.isnan(dt_tableau.loc[indice[0], 'Q_MAX_PALI']) : 
    #     Qmax_Paliers = "Qmax_Paliers = " + str(dt_tableau.loc[indice[0], 'Q_MAX_PALI'])+" m3/h"
    #     ax3.text(0.1, 1.1, Qmax_Paliers, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'green')
    # else : 
    #     Qmax_Paliers = "Qmax_Paliers non renseigné"
    #     ax3.text(0.1, 1.1, Qmax_Paliers, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'green')
    
    # ## Affichage rabattement et débit spécifiques
    # if not np.isnan(dt_tableau.loc[indice[0], 'RABATTEMEN']) : 
    #     text_rabattement = "Rabattement = " + str(dt_tableau.loc[indice[0], 'RABATTEMEN'])+" m" + '  Q_specifiq = '+ str(dt_tableau.loc[indice[0], 'Q_SPECIFIQUE'])+' m3/m'
    #     ax3.text(0.9, 1.1, text_rabattement, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 11, color = 'red')
    # else : 
    #     text_rabattement  = "Rabattement non renseigné"
    #     ax3.text(0.9, 1.1, text_rabattement, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 11, color = 'red')
    
    # ## Ajout du RMS factor
    # RMS = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'RMS_FACTOR'])
    # text_RMS = 'Erreur absolue : ' + str(float(RMS))+'%'
    # ax3.text(0.9, 0.05, text_RMS, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
  
    ## Ajout altitude du forage 
    # Z = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'Z'])
    # text_Z = 'Altitude du forage : ' + str(int(Z))+'m'
    # ax3.text(0.1, -0.15, text_Z, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
    
    ## retour pour impression dans un PDF
    return(fig)

def traitement_forage_panneau(nom_rep, Fichier_Panneau, fichier_blanc, colonne_X,colonne_Z, oid, dt_tableau):
    ''' Mets en forme un panneau et un forage au travers d'une figure
    représentant le panneau, le forage, le log de résistivité et d'autres 
    infos'''
    
    ## Retrait des guillemets qui faussent la lecture des fichiers et 
    ## transformation des virgules en espaces
    mise_en_forme(Fichier_Panneau)
    
    ##  Obtention des données dans un dataframe
    donnees = get_data(nom_rep,Fichier_Panneau)
    ## Interpolation linéaire des données sur une grille d'interpolation de 1m
    X_interpol, Z_interpol, R_interpol_linear = interpol(donnees, colonne_X, colonne_Z, 'linear')
    ## Affichage du panneau sans le blanc incorporé (à décomment si nécessaire)
    #fc.show_panneau(X_interpol, Z_interpol, R_interpol_linear)
    
    ## récupération du blanc
    mise_en_forme(fichier_blanc)
    dt_blanc = get_data(nom_rep,fichier_blanc)
    dt_blanc.head()
    blanc = dt_blanc.to_numpy()
    
    ## Application du blanc au panneau interpolé précédemment.
    R_interpol_linear_filtre = filtre_blanc(R_interpol_linear, X_interpol, Z_interpol, blanc)
    
    ## Représentation du panneau
    fig = Panneau(X_interpol, Z_interpol, R_interpol_linear_filtre, dt_tableau, oid)
    return(fig)

 
###################################################################################################################
## MAIN 

rep = u"C:\\test_OG"  # chemin du dossier à parcourir
list_fichier(rep)
colonne_X = 'X-location'
colonne_Z = 'Z-location'


#%% Récupération des données excel
nom_rep=u"C:\\test_OG"
tableau_forage = "test_OG.csv"

dt_tableau = get_tableau_forage(tableau_forage, nom_rep)
 
#%%
figs = []

nom_rep=u"C:\\test_OG"
os.chdir(nom_rep)
os.listdir()
os.getcwd()


for oids in dt_tableau["OID"].to_numpy() :
    print(oids)
    Fichier_Panneau,fichier_blanc = get_nom_panneau(dt_tableau, oids, nom_rep)
    fig = traitement_forage_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)
    figs.append(fig)

with PdfPages('sortiePDF.pdf') as pdf:
    for fig in figs :
        pdf.savefig(fig, bbox_inches='tight')  # regarder dpi =
