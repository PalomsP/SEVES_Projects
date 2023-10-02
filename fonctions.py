# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:38:10 2023

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
    

def show_panneau(X, Z, Resistivite):        
    fig, ax = plt.subplots(figsize = (20,4))
    levels = np.logspace(1.0, 3*np.log(3), num=16)
    cs = ax.contourf(X, Z, Resistivite, norm=colors.LogNorm(vmin=10, vmax=3000, clip=False),vmin=10, vmax=3000, levels = levels, cmap = 'rainbow')
    cbar = fig.colorbar(cs)
    return (cs)
    
def get_values_array(donnees, colonne_X, colonne_Z):
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
    Xtot, Ztot, Rtot = fc.get_values_array(donnees, colonne_X, colonne_Z)
    
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


def get_log(X, R, x_f):
    i_R, j_R = R.shape
    logf = np.zeros(i_R)
    for i,x in enumerate(X) :
        if x_f >= x and x_f < X[i+1] :
            index = np.where(X == x)[0]
            écart_forage = x_f-x
            écart_X= X[i+1]-x
            logf = R[:,index]*(X[index+1]-x_f)/écart_X + R[:,index+1]*écart_forage/écart_X
    return(np.hstack(logf))

def arr_eau(dt_tableau) :
    """ Retourne le tableau des arrivée d'eau sous forme d'une liste de liste"""
    head = ['PROF_VE_1','PROF_VE_2','PROF_VE_3','PROF_VE_4','PROF_VE_5','PROF_VE_6','PROF_VE_7','PROF_VE_8','PROF_VE_9' ,'PROF_VE_10'  ]
    arr_eau = []
    for oid in dt_tableau['OID']:
        liste_arr_eau = []
        for prof in head : 
            if dt_tableau.loc[dt_tableau['OID'] == oid, prof].notnull().all():
               liste_arr_eau.append(-float(dt_tableau.loc[dt_tableau['OID'] == oid, prof]))
        arr_eau.append(liste_arr_eau)
    return(arr_eau)

def debit_arr_eau(dt_tableau) :
    """ Retourne le tableau des débits des arrivées d'eau sous forme d'une liste de liste"""
    head = ['DEBIT_VE_1','DEBIT_VE_2','DEBIT_VE_3','DEBIT_VE_4','DEBIT_VE_5','DEBIT_VE_6','DEBIT_VE_7','DEBIT_VE_8','DEBIT_VE_9','DEBIT_VE_10' ]
    debit_arr_eau = []
    for oid in dt_tableau['OID']:
        liste_debit_arr_eau = []
        for debit in head : 
            if dt_tableau.loc[dt_tableau['OID'] == oid, debit].notnull().all():
               liste_debit_arr_eau.append(float(dt_tableau.loc[dt_tableau['OID'] == oid, debit]))
        debit_arr_eau.append(liste_debit_arr_eau)
    return(debit_arr_eau)


def get_repere(donnees, colonne_X, colonne_Z) : 
    """ Récupère l'axe des abscisses et des profondeurs du dataframe et le
    retourne sous forme de 2 vecteurs ainsi que la longueur de ceux ci"""
    n_X = 0
    visited = []
    for i in range(0, len(donnees[colonne_X])):
        if donnees[colonne_X][i] not in visited: 
            visited.append(donnees[colonne_X][i])
            n_X += 1
    X = np.sort(np.array(visited)) 
    n_Z = 0
    visited = []
    for i in range(0, len(donnees[colonne_Z])):
        if donnees[colonne_Z][i] not in visited: 
            visited.append(donnees[colonne_Z][i])
            n_Z += 1
    Z = np.flip(np.sort(np.array(visited)))
    return (X, Z, n_X, n_Z)



def pseudo_section(donnees, colonne_X, colonne_Z):
    X, Z, n_X, n_Z = get_repere(donnees, colonne_X, colonne_Z)
    R = np.empty((n_Z,n_X))
    R[:] = np.nan
    for x in X :
        df = donnees[(donnees[colonne_X] == x)]
        for z in Z : 
            j = np.where(X==x)[0]
            j = j[0]
            i = np.where(Z==z)[0]
            i = i[0]
            r = df[df[colonne_Z]==z]
            if not r.empty :
                rx = r['Resistivity'].values
                R[i][j] = float(rx)
    return(R)

def prof_litho(dt_tableau) :
    """ Retourne le tableau des profondeurs des lithos sous forme d'une liste de liste"""
    ##head est la liste des en-têtes dans lesquels se trouvent les valeurs de profondeurs
    head = ['PROF_LITHO','PROF_LIT_1','PROF_LIT_2','PROF_LIT_3','PROF_LIT_4','PROF_LIT_5','PROF_LIT_6','PROF_LIT_7','PROF_LIT_8','PROF_LIT_9','PROF_LIT10' ]
    prof_litho = [] # Création d'une liste vide qui va contenir la liste des listes des profondeurs
    for oid in dt_tableau['OID']: #parcours les forages
        liste_prof_litho = [0] # Je rajoute 0 afin de pouvoir exploiter chaque liste par la suite
        for prof in head : #parcours les colonnes de profondeurs
            if dt_tableau.loc[dt_tableau['OID'] == oid, prof].notnull().all():#test si non nul pour éviter les erreurs
               liste_prof_litho.append(-float(dt_tableau.loc[dt_tableau['OID'] == oid, prof]))# ajoute chaque profondeur dans la liste
        prof_litho.append(liste_prof_litho) #ajoute la liste des profondeurs du forage correspondant à la liste de toutes les profondeurs de touys les forages
    return(prof_litho)

def type_litho(dt_tableau) :
    """ Retourne le tableau des type de litho sous forme d'une liste de liste"""
    head = ['TYPE_LITHO','TYPE_LIT_1','TYPE_LIT_2','TYPE_LIT_3','TYPE_LIT_4','TYPE_LIT_5','TYPE_LIT_6','TYPE_LIT_7','TYPE_LIT_8','TYPE_LIT_9','TYPE_LIT10' ]
    type_litho = []
    ## Formulation différente  que précédemment qui utilise iterrows()
    for index, row in dt_tableau.iterrows():
        list_temp = []
        for h0 in head :
            list_temp.append(row[h0])
        type_litho.append(list_temp)
    return(type_litho)

def lithologie(dt_tableau) :
    """ Retourne le tableau des arrivée d'eau sous forme d'une liste de liste"""
    head = ['LITHO_1','LITHO_2','LITHO_3','LITHO_4','LITHO_5','LITHO_6','LITHO_7','LITHO_8','LITHO_9','LITHO_10', 'LITHO_11' ]
    lithologie = []

    # Loop over the rows of the dataframe
    for index, row in dt_tableau.iterrows():
        list_temp = []
        for h0 in head :
            list_temp.append(row[h0])
        lithologie.append(list_temp)
    return(lithologie)

# Création de la figure


def litho(profondeurs, lithologies, ax, type_lithologies) : 
    '''Trace sur un axe donné le log de forages correspondant'''
    # Ajout des couches lithologiques sous forme de rectangles colorés
    couleurs = {'TERRE VEGETALE': 'Brown', 'ARGILE': 'PaleGoldenrod', 'ARGILE SABLEUSE': 'PapayaWhip','SABLE ARGILEUX': 'LightYellow', 'SABLE':'Yellow', 'DIORITE': 'lightgray', 'GRANITE' : 'PeachPuff','MICROGRANITE': 'rosybrown','GRANODIORITE':'khaki', 'MONZOGRANITE':'#F5DEB3', 'AMPHIBOLITE' : 'DarkGray', 'DOLERITE' : 'gray'}
    signe = {'RECOUVREMENT':'oo' , 'ALTERE':'..' , 'FRACTURE' : 'x---' ,'SAIN':'' , 'ALTERE / FRACTURE' :'x--..' , 'ALTERE / MICROFISSURE' :'..x--' ,'MICROFISSURE' : 'x---' , 'TRES ALTERE' : '...' , 'SAIN / MICROFISSURE' : 'x---', 'FISSURE' :'x---' }
    ylim = -76
    for i in range(len(profondeurs) - 1):
        litho = lithologies[i]
        couleur = couleurs.get(litho, 'white')
        rect = plt.Rectangle((0, profondeurs[i]), 2, profondeurs[i + 1] - profondeurs[i], color=couleur)
        ax.add_patch(rect)
        
        # Ajout du nom de la lithologie à côté de la couche
        if profondeurs[i+1] >= ylim :
            ax.text(2.3, (profondeurs[i] + profondeurs[i+1])/2, litho, ha='left', va='center', fontsize = 7)
        elif profondeurs[i]>=ylim and profondeurs[i+1]<=ylim : 
            ax.text(2.3, profondeurs[i] +1 , litho, ha='left', va='center', fontsize = 7)
    for i in range(len(profondeurs)-1):
        x = 0
        y = profondeurs[i]
        width = 2
        height = profondeurs[i+1] - profondeurs[i]
        type_litho = type_lithologies[i]
        hatch_litho = signe.get(type_litho)        
        ax.add_patch(plt.Rectangle((x, y), width, height, color='black', fill=False, hatch = hatch_litho, lw=0.5))
    # Configuration de l'axe y
    ax.set_ylim(ylim, profondeurs[0])
    #ax.set_ylabel('Profondeur (m)')
    
    # Configuration de l'axe x
    ax.set_xlim(0, 8)
    
    # Suppression des ticks sur l'axe x
    ax.set_xticks([])

def resistivite_moyenne_prof(Z,logf, profondeurs):
    '''Fait la liste des moyenne des résistivité des litho a portir du log de résistivité '''
    np = len(profondeurs)
    nz = Z.shape[0]
    resistivite_moyenne_prof = []
    for i in range(np-1):
        S = 0
        d=0
        for j in range(nz):  
            if Z[j]<=profondeurs[i] and Z[j]>=profondeurs[i+1] and profondeurs[i]!=profondeurs[i+1] and not isnan(logf[j]):
                S += float(logf[j])
                d+=1
        if d!=0 :
            moyenne = S/d
            resistivite_moyenne_prof.append(moyenne)
        if d==0:
            resistivite_moyenne_prof.append(nan)
    return(resistivite_moyenne_prof)


def get_log_resistivite_panneau(nom_rep, Fichier_Panneau, fichier_blanc, colonne_X,colonne_Z, oid, dt_tableau):
    
    ## Retrait des guillemets qui faussent la lecture des fichiers et 
    ## transformation des virgules en espaces
    fc.mise_en_forme(Fichier_Panneau)
    
    ##  Obtention des données dans un dataframe
    donnees = fc.get_data(nom_rep,Fichier_Panneau)
    ## Interpolation linéaire des données sur une grille d'interpolation de 1m
    X_interpol, Z_interpol, R_interpol_linear = fc.interpol(donnees, colonne_X, colonne_Z, 'linear')
    ## Affichage du panneau sans le blanc incorporé (à décomment si nécessaire)
    #fc.show_panneau(X_interpol, Z_interpol, R_interpol_linear)
    
    ## récupération du blanc
    fc.mise_en_forme(fichier_blanc)
    dt_blanc = fc.get_data(nom_rep,fichier_blanc)
    dt_blanc.head()
    blanc = dt_blanc.to_numpy()
    ## Application du blanc au panneau interpolé précédemment.
    R_interpol_linear_filtre = fc.filtre_blanc(R_interpol_linear, X_interpol, Z_interpol, blanc)
    ## Affichage du panneau filtré par le blanc (à decomment si nécessaire)
    ##fc.show_panneau(X_interpol, Z_interpol, R_interpol_linear_filtre)
    
    ## récupération de la position du log
    x_f = round(float(dt_tableau.loc[dt_tableau['OID'] == oid, 'POSITION_PROJETEE']),1)
    ## Récupération du log de résistivité à la position du forage
    logf=fc.get_log(X_interpol, R_interpol_linear_filtre, x_f)
    
    return(Z_interpol, logf)

def liste_resistivite_litho(Z,logf, profondeurs):
    '''Fait la liste des résistivité des litho a portir du log de résistivité '''
    profondeurs_t= profondeurs
    np = len(profondeurs_t)
    nz = Z.shape[0]
    liste_resistivite_total= []
    for i in range(np-1):
        liste_resistivite =[]
        for j in range(nz):  
            if Z[j]<=profondeurs_t[i] and Z[j]>=(profondeurs_t[i+1]-1) :
                liste_resistivite.append(logf[j])
        liste_resistivite_total.append(liste_resistivite)
    return(liste_resistivite_total)   

def liste_resistivite_horizons(Z,logf, profondeurs):
    '''Fait la liste des résistivité des litho a portir du log de résistivité'''
    profondeurs_t= profondeurs
    print(profondeurs_t)
    np = len(profondeurs_t)
    nz = Z.shape[0]
    liste_resistivite_total= []
    for i in range(np-1):
        liste_resistivite =[]
        for j in range(nz):  
            if Z[j]<=profondeurs_t[i] and Z[j]>=(profondeurs_t[i+1]-1) :
                liste_resistivite.append(logf[j])
        liste_resistivite_total.append(liste_resistivite)
    return(liste_resistivite_total)    

def add_line_pd_litho(df,lithologie_liste, type_lithologie_liste, resistivite_liste, profondeurs,oid): 
    epaisseurs = []
    for i in range(len(profondeurs)-1):
        epaisseurs.append(profondeurs[i]-profondeurs[i+1])
    del profondeurs[0]
    
    n = len(resistivite_liste)
    lithologie_liste = lithologie_liste[:n]
    type_lithologie_liste = type_lithologie_liste[:n]
    profondeurs = profondeurs[:n]
    epaisseurs = epaisseurs[:n]
    print(lithologie_liste, '\n', type_lithologie_liste, '\n', profondeurs, '\n', epaisseurs, '\n', resistivite_liste)
    donnees = {'OID' : oid, 'Lithologie': lithologie_liste, 'Type_lithologie': type_lithologie_liste, 'Resistivite': resistivite_liste, 'Profondeurs_bas':profondeurs, 'Epaisseur' : epaisseurs}

    nouvelle_ligne = pd.DataFrame(donnees)
    # Ajouter la nouvelle ligne au DataFrame existant
    df = pd.concat([df, nouvelle_ligne], ignore_index=True)
    return(df)




def pd_litho_total(dt_tableau, nom_rep, colonne_X,colonne_Z) : 
    head = {'OID':[],'Lithologie': [], 'Type_lithologie': [], 'Resistivite': [], 'Profondeurs_bas':[], 'Epaisseur' : []}
    df = pd.DataFrame(head)
    for oids in dt_tableau["OID"].to_numpy() :
        print(oids)
        Fichier_Panneau,fichier_blanc = fc.get_nom_panneau(dt_tableau, oids, nom_rep)
        Z_interpol, logf= fc.get_log_resistivite_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)
        profondeurs = prof_litho(dt_tableau.loc[dt_tableau['OID']==oids])[0]
        lithologies = lithologie(dt_tableau.loc[dt_tableau['OID']==oids])[0]
        type_lithologie = type_litho(dt_tableau.loc[dt_tableau['OID']==oids])[0]
        resistivite_liste = liste_resistivite_litho(Z_interpol,logf, profondeurs)
        df=add_line_pd_litho(df,lithologies, type_lithologie, resistivite_liste, profondeurs,oids)
    return(df)
    
def prof_horizons(dt_tableau) :
    """ Retourne le tableau des profondeurs des horizons
    avec d'abord le recouvrement, puisl'horizon altéré, puis
    l'horizon fissuré sous forme d'une liste de liste"""
    head = ['PROF_RE','PROF_HA','PROF_HF','PROFONDEUR']
    prof_horizons = []
    for oid in dt_tableau['OID']:
        liste_prof_horizon = [0]
        for prof in head : 
            if dt_tableau.loc[dt_tableau['OID'] == oid, prof].notnull().all():
               liste_prof_horizon.append(-float(dt_tableau.loc[dt_tableau['OID'] == oid, prof]))
        prof_horizons.append(liste_prof_horizon)
    return(prof_horizons)

def add_line_pd_horizon(df,resistivite_liste, profondeurs,oid): 
    epaisseurs = []
    profondeurs = profondeurs
    for i in range(len(profondeurs)-1):
        epaisseurs.append(profondeurs[i]-profondeurs[i+1])
    del profondeurs[0]
    
    n = len(resistivite_liste)
    print(resistivite_liste, '\n',profondeurs, '\n', epaisseurs)
    horizons = ['RE', 'HA', 'HF','S']
    liste_oid = [oid, oid, oid, oid]
    donnees = {'OID' : liste_oid, 'Horizon': horizons, 'Resistivite': resistivite_liste, 'Profondeurs_bas':profondeurs, 'Epaisseur' : epaisseurs}
    nouvelle_ligne = pd.DataFrame(donnees)
    # Ajouter la nouvelle ligne au DataFrame existant
    df = pd.concat([df, nouvelle_ligne], ignore_index=True)
    return(df)
    
def pd_horizon_total(dt_tableau, nom_rep, colonne_X,colonne_Z) : 
    '''crée un dataframe contenant les valeurs de résistivité des horizons pour tous les sites'''
    head = {'OID':[],'Horizon': [],'Resistivite': [], 'Profondeurs_bas':[], 'Epaisseur' : []}
    df = pd.DataFrame(head)
    for oids in dt_tableau["OID"].to_numpy() :
        Fichier_Panneau,fichier_blanc = fc.get_nom_panneau(dt_tableau, oids, nom_rep)
        Z_interpol, logf= fc.get_log_resistivite_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)
        profondeurs = prof_horizons(dt_tableau.loc[dt_tableau['OID']==oids])[0]
        resistivite_liste = liste_resistivite_horizons(Z_interpol,logf, profondeurs)
        df=add_line_pd_horizon(df, resistivite_liste, profondeurs,oids)
    return(df)

def pd_horizon_total_resistivite_moy(dt_tableau, nom_rep, colonne_X,colonne_Z) : 
    '''crée un dataframe contenant les valeurs de résistivité des horizons pour tous les sites'''
    head = {'OID':[],'Horizon': [],'Resistivite': [], 'Profondeurs_bas':[], 'Epaisseur' : []}
    df = pd.DataFrame(head)
    for oids in dt_tableau["OID"].to_numpy() :
        Fichier_Panneau,fichier_blanc = fc.get_nom_panneau(dt_tableau, oids, nom_rep)
        Z_interpol, logf= fc.get_log_resistivite_panneau(nom_rep,Fichier_Panneau,fichier_blanc, colonne_X,colonne_Z,oids, dt_tableau)
        profondeurs = prof_horizons(dt_tableau.loc[dt_tableau['OID']==oids])[0]
        resistivite_liste = resistivite_moyenne_prof(Z_interpol,logf, profondeurs)
        df=add_line_pd_horizon(df, resistivite_liste, profondeurs,oids)
    return(df)

def profondeurs_interbornes_de_resistivite(Rmin, Rmax, Z, logf):
    n = Z.shape[0]
    Pmin = 1
    Pmax = 1
    if logf[logf<Rmin].shape[0] == 0:
        Pmin = 0
    if logf[logf>Rmin].shape[0] == 0:
        Pmin = np.max(Z)
    if logf[logf<Rmax].shape[0] == 0:
        Pmax = 0
    if logf[logf>Rmax].shape[0] == 0:
        Pmax = np.max(Z)
        
    if (logf[logf<Rmin].shape[0] == 0)and(logf[logf>Rmax].shape[0] == 0):
        Pmin =0
        Pmax=np.max(Z)
    
    i = 0    
    for i in range(n-1):
        if (logf[i]<Rmin and logf[i+1]>Rmin) or (logf[i]>Rmin and logf[i+1]<Rmin):
            Pmin = Z[i]
            print('Pmin=', Pmin)
        if (logf[i]<Rmax and logf[i+1]>Rmax) or (logf[i]>Rmax and logf[i+1]<Rmax):
            Pmax = Z[i]
            print('Pmax=', Pmax)
            
    dif = Pmin-Pmax
    return(Pmin, Pmax, dif)   


def log_et_panneau(X_interpol, Z_interpol, R_interpol, logf, dt_tableau, oid):
    ''' Prend en entrée les axes interpolés et la matrice des résistivités,
    le log de résistivité obtenu à l'emplacement du forage et les données du
    tableau des forages obtenus précédemment et renvoie la représentation finale
    du site avec le forage identifier par un OID donné
    '''
    ## Récupération des données dans le dataframe
    x_f = round(float(dt_tableau.loc[dt_tableau['OID'] == oid, 'POSITION_PROJETEE']),1)
    Z = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'Z'])
    p_f = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'PROFONDEUR'])
    niv_w = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'NIVEAU_STA'])
    p_mft = -float(dt_tableau.loc[dt_tableau['OID'] == oid, 'PROF_TOP_MFT'])
    site = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'SITE_GEOPH'])
    RMS = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'RMS_FACTOR'])
    cutoff = float(dt_tableau.loc[dt_tableau['OID'] == oid, 'CUT_OFF_FACTOR'])
    prof_eau = arr_eau(dt_tableau)
    debit_arrivees_eau = debit_arr_eau(dt_tableau)
    indice = dt_tableau[dt_tableau['OID'] == oid].index.to_numpy()
    profondeurs = prof_litho(dt_tableau.loc[dt_tableau['OID']==oid])[0]
    lithologies = lithologie(dt_tableau.loc[dt_tableau['OID']==oid])[0]
    type_lithologie = type_litho(dt_tableau.loc[dt_tableau['OID']==oid])[0]
    ## Récupération des moyenne de résistivité pour chaque litho
    r_litho = resistivite_moyenne_prof(Z_interpol, logf, profondeurs)
    ## Création de la figure et des axes avec le même ordonné et un ratio de 1 pour 7 entre les deux
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4), sharey=True, gridspec_kw={'width_ratios': [1,1,8]}, dpi = 100) 
    
    ## Tracé du premier axe : log de résistivité en échelle log
    log = ax1.semilogx(logf,Z_interpol)
    ax1.set_xlim(right=10000, left = 10) ## Fixation des bornes des abscisses
    ## Mise en paeg des labels et des ticks de l'axe1
    ax1.set_xlabel('Résistivité (Ohm.m)')
    ax1.set_ylabel('Profondeur (m)')
    ax1.tick_params(axis='y', which='both', width=1, length=5)
    # Configuration des ticks sur l'axe des abscisses
    ax1.set_xscale('log')  # Utilisation d'une échelle logarithmique
    ax1.set_xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900, 1000,2000,3000,4000,5000,6000,7000,8000,9000, 10000])  # Définition des valeurs des ticks
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.label.set_size(14)
    ax1.yaxis.label.set_size(14)
    for i in range(2, 4):
        power = 10 ** i
        ax1.axvline(power, color='gray', linestyle='--')
    # Ajout des lignes horizontales au niveau des ticks
    # yticks = ax1.get_yticks()
    # for ytick in yticks:
    #     ax1.axhline(ytick, color='gray', linestyle='--')
    
    # Ajout des lignes horizontales au niveau des profondeurs de litho et niveau statique
    for prof in profondeurs:
        ax1.axhline(prof, color='gray', linestyle='--')
    
    ax1.axhline(-niv_w, color='blue', linestyle='--')
    
    for i in range(len(r_litho)):
        if not np.isnan(r_litho[i]):
            r = str(round(r_litho[i]))+' Ohm.m'
            ax1.text(12, (profondeurs[i]+profondeurs[i+1])/2 , r, ha='left', va='center', fontsize = 6, color = 'darkblue')
    ## Tracé ax2
    litho(profondeurs, lithologies, ax2, type_lithologie)
    
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
    
    ylim = 76
    ## Tracé du forage et légende du forage
    ax3.axvline(x_f,ymin = 1-p_f/ylim, color='black', linestyle='-', linewidth=2)
    text = 'Forage x='+str(x_f)+'m'
    ax3.text(x_f/232, 1.02, text, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 14)
    if p_f>ylim : 
        legend = 'Profondeur forage = \n' + str(p_f) + ' m'
        ax2.text(0.5, -0.15, legend, ha='center', va='bottom', transform=ax2.transAxes, fontsize = 10)
    ## Ajout filigramme sur méthode de prospection et site du profil
    # ax3.text(0.9, -0.15, 'Configuration : pôle-dipôle  48 électrodes espacées de 5m', ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
    text_site = 'Site du profil : ' + str(int(site))
    ax3.text(0.1, 0.05, text_site, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
    ## Ajustement de l'écart entre les axes
    fig.subplots_adjust(wspace=0.05)
    
    ## Ajout haut du Marteau Fond de Trou
    ax2.scatter(2.2,p_mft, s=300, color='red', marker='_', zorder = 10)
    #ax2.scatter(2.1,p_mft+2, s=300, color='red', marker='|', zorder = 10) 
    ax2.text(7, p_mft-1, 'Top MFT', ha='center', va='bottom', fontsize = 6, color = 'red')
    
    ## Ajout des profondeurs cractéristiques
    # Rmin = 150
    # Rmax = 2000
    # Pmin, Pmax, dif = profondeurs_interbornes_de_resistivite(Rmin, Rmax, Z_interpol, logf)
    # if Pmin!=1 and Pmax!=1:
    #     ax1.scatter(100,Pmin, s=150, color='blue', marker='_', zorder = 10)
    #     ax1.scatter(100,Pmax, s=150, color='blue', marker='_', zorder = 10)
    
    # Rmin = 0
    # Rmax = 400
    # Pmin, Pmax, dif = profondeurs_interbornes_de_resistivite(Rmin, Rmax, Z_interpol, logf)
    # if Pmin!=1 and Pmax!=1:
        # ax1.scatter(100,Pmin, s=150, color='red', marker='_', zorder = 10)
        # ax1.scatter(100,Pmax, s=150, color='red', marker='_', zorder = 10)
        
    ## Ajout du niveau statique
    ax2.scatter(0.5,-niv_w, s=150, color='blue', marker='_', zorder = 10)
    ax2.scatter(0.5,-niv_w+1.6, s=120, color='blue', marker='v', zorder = 10)    
    ax3.scatter(x_f+1.5,-niv_w, s=150, color='blue', marker='_', zorder = 10)
    ax3.scatter(x_f+2,-niv_w+1.6, s=120, color='blue', marker='v', zorder = 10)
    
    ## Ajout des arrivées d'eau
    X_arr_eau = []
    for prof in prof_eau[indice[0]] :
        X_arr_eau.append(x_f+1)
    ax3.scatter(X_arr_eau, prof_eau[indice[0]], s=100, facecolors='blue', marker='_', zorder = 9)
    for i in range(len(debit_arrivees_eau[indice[0]])):
        if prof_eau[indice[0]][i] >= -ylim :
            ax3.text(X_arr_eau[i]+2, prof_eau[indice[0]][i], str(debit_arrivees_eau[indice[0]][i]), ha='center', va='bottom', fontsize = 7, color = 'blue')
    

    
    ## Titre selon de le nom du village prospecté récupérer dans le dataframe
    fig.suptitle(dt_tableau.loc[indice[0], 'NOM_VILLAG'], fontsize=22, x=0.5, y=1.05)
    
    ## Affichage des débits 
    if not np.isnan(dt_tableau.loc[indice[0], 'Q_DEVELOPP']) : 
        Qd = "Qd = " + str(dt_tableau.loc[indice[0], 'Q_DEVELOPP'])+" m3/h"
        ax3.text(-0.1, 1.09, Qd, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'blue')
    else : 
        Qd = "Qd non renseigné"
        ax3.text(-0.1, 1.09, Qd, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'blue')
    
    if not np.isnan(dt_tableau.loc[indice[0], 'Q_MAX_PALI']) : 
        Qmax_Paliers = "Qmax_Paliers = " + str(dt_tableau.loc[indice[0], 'Q_MAX_PALI'])+" m3/h"
        ax3.text(-0.3, 1.1, Qmax_Paliers, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'green')
    else : 
        Qmax_Paliers = "Qmax_Paliers non renseigné"
        ax3.text(-0.3, 1.1, Qmax_Paliers, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10, color = 'green')
    
    ## Affichage rabattement et débit spécifiques
    if not np.isnan(dt_tableau.loc[indice[0], 'RABATTEMEN']) : 
        text_rabattement = "Rabattement = " + str(dt_tableau.loc[indice[0], 'RABATTEMEN'])+" m" + '  Q_specifiq = '+ str(dt_tableau.loc[indice[0], 'Q_SPECIFIQUE'])+' m3/m'
        ax3.text(0.9, 1.1, text_rabattement, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 11, color = 'red')
    else : 
        text_rabattement  = "Rabattement non renseigné"
        ax3.text(0.9, 1.1, text_rabattement, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 11, color = 'red')
    
    ## Ajout du RMS factor
    text_RMS = 'Erreur absolue : ' + str(float(RMS))+'%'
    ax3.text(0.9, 0.05, text_RMS, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
  
    ## Ajout altitude du forage 
    text_RMS = 'Altitude du forage : ' + str(int(Z))+'m'
    ax3.text(0.1, -0.15, text_RMS, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
    
    ## Ajout du cut-off factor
    text_cutoff = 'Cut-Off Factor : ' + str(float(cutoff))
    ax3.text(0.9, -0.15, text_cutoff, ha='center', va='bottom', transform=ax3.transAxes, fontsize = 10)
  
    ## retour pour impression dans un PDF
    return(fig)

def traitement_forage_panneau(nom_rep, Fichier_Panneau, fichier_blanc, colonne_X,colonne_Z, oid, dt_tableau):
    ''' Mets en forme un panneau et un forage au travers d'une figure
    représentant le panneau, le forage, le log de résistivité et d'autres 
    infos'''
    
    ## Retrait des guillemets qui faussent la lecture des fichiers et 
    ## transformation des virgules en espaces
    fc.mise_en_forme(Fichier_Panneau)
    
    ##  Obtention des données dans un dataframe
    donnees = fc.get_data(nom_rep,Fichier_Panneau)
    ## Interpolation linéaire des données sur une grille d'interpolation de 1m
    X_interpol, Z_interpol, R_interpol_linear = fc.interpol(donnees, colonne_X, colonne_Z, 'linear')
    ## Affichage du panneau sans le blanc incorporé (à décomment si nécessaire)
    #fc.show_panneau(X_interpol, Z_interpol, R_interpol_linear)
    
    ## récupération du blanc
    fc.mise_en_forme(fichier_blanc)
    dt_blanc = fc.get_data(nom_rep,fichier_blanc)
    dt_blanc.head()
    blanc = dt_blanc.to_numpy()
    ## Application du blanc au panneau interpolé précédemment.
    R_interpol_linear_filtre = fc.filtre_blanc(R_interpol_linear, X_interpol, Z_interpol, blanc)
    ## Affichage du panneau filtré par le blanc (à decomment si nécessaire)
    ##fc.show_panneau(X_interpol, Z_interpol, R_interpol_linear_filtre)
    
    ## récupération de la position du log
    x_f = round(float(dt_tableau.loc[dt_tableau['OID'] == oid, 'POSITION_PROJETEE']),1)
    ## Récupération du log de résistivité à la position du forage
    logf=fc.get_log(X_interpol, R_interpol_linear_filtre, x_f)
    
    ## Représentation du log et du panneau avec les arrivées d'eau, la 
    # position du forage etc.
    fig = fc.log_et_panneau(X_interpol, Z_interpol, R_interpol_linear_filtre, logf, dt_tableau, oid)
    return(fig)


def histogramme_resistivites(data):
    # Filtrer le DataFrame par type d'horizon
    horizon_1 = data[data['Horizon'] == 'RE']
    horizon_2 = data[data['Horizon'] == 'HA']
    horizon_3 = data[data['Horizon'] == 'HF']
    
    # Déterminer la plage des valeurs de résistivité pour tous les horizons
    resistivity_range = (data['Resistivite'].min(), data['Resistivite'].max())
    
    # Tracer les histogrammes de résistivité pour chaque horizon
    plt.hist(horizon_1['Resistivite'], bins=2, alpha=1, label='Recouvrement', histtype = 'barstacked')
    plt.hist(horizon_2['Resistivite'], bins=10, alpha=0.6, label='HA', histtype = 'barstacked')
    plt.hist(horizon_3['Resistivite'], bins=20, alpha=0.4, label='HF',  histtype = 'barstacked')
    
    # Ajouter des légendes, un titre et des étiquettes d'axes
    plt.legend()
    plt.title('Histogrammes de résistivité pour chaque horizon')
    plt.xlabel('Résistivité')
    plt.ylabel('Fréquence')
    
    # Afficher le graphique
    plt.show()

def gamme_resistivite_litho(df,colonne,litho):
    dl = df.loc[df[colonne]== litho]
    Resistivites = []
    for index, row in dl.iterrows():
        Resistivites += row['Resistivite']
    Resistivites_ss_nan = [x for x in Resistivites if not isnan (x)]
    Resistivites_ss_nan_ss_0 = [x for x in Resistivites_ss_nan if not x==0.0]
    # Resistivites_ss_nan_ss_0_ss_abherente = [x for x in Resistivites_ss_nan_ss_0 if x<=10000]
    #return Resistivites_ss_nan_ss_0_ss_abherente   
    return Resistivites_ss_nan_ss_0   

 
