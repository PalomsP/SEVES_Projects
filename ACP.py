# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:05:56 2023

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

#Dossier de travail
rep = "C:/Users/33671/Desktop/PIERRE_PALOMAR"  # chemin du dossier à parcourir
mon_fichier = "ACP_Bitkine_Param_Nouradine.csv"

#modification du dossier de travail
import os
os.chdir(rep)
#librairie pandas
import pandas
#version
print(pandas.__version__) # 0.23.0
#chargement de la première feuille de données
#X = pandas.read_excel("autos_acp_pour_python.xlsx",sheet_name=0,header=0,index_col=0)
X = pandas.read_csv(mon_fichier, sep=";",header=0,index_col=0)
print(X)
#dimension
print(X.shape) # (18, 6)
#nombre d'observations
n = X.shape[0]
#nombre de variables
p = X.shape[1]
#affichage des données
print(X)

#%%
#scikit-learn
import sklearn
#vérification de la version
print(sklearn.__version__) # 0.19.1

#classe pour standardisation
from sklearn.preprocessing import StandardScaler
#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(X)
print(Z)

#%%
#vérification - librairie numpy
import numpy
#moyenne
print(numpy.mean(Z,axis=0))

#écart-type
print(numpy.std(Z,axis=0,ddof=0))

#%%
#classe pour l'ACP
from sklearn.decomposition import PCA
#instanciation
acp = PCA(svd_solver='full')

#affichage des paramètres
print(acp)

#calculs
coord = acp.fit_transform(Z)
print(coord)
print((coord[0,0],coord[0,1]))
#nombre de composantes calculées
print(acp.n_components_) # 6

#variance expliquée
print(acp.explained_variance_)

#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print(eigval)

#ou bien en passant par les valeurs singulières
print(acp.singular_values_**2/n)

#proportion de variance expliquée
print(acp.explained_variance_ratio_)

#%%
#scree plot
plt.plot(numpy.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

#cumul de variance expliquée
plt.plot(numpy.arange(1,p+1),numpy.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()

#%%

#seuils pour test des bâtons brisés
bs = 1/numpy.arange(p,0,-1)
bs = numpy.cumsum(bs)
bs = bs[::-1]

#test des bâtons brisés
print(pandas.DataFrame({'Val.Propre':eigval,'Seuils':bs}))


#%%
#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6) #même limites en abscisse
axes.set_ylim(-6,6) #et en ordonnée
#placement des étiquettes des observations
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
#affichage
plt.show()

#contribution des individus dans l'inertie totale
di = numpy.sum(Z**2,axis=1)
print(pandas.DataFrame({'ID':X.index,'d_i':di}))

#qualité de représentation des individus - COS2
cos2 = coord**2
for j in range(p):
    cos2[:,j] = cos2[:,j]/di
print(pandas.DataFrame({'id':X.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]}))

#vérifions la théorie - somme en ligne des cos2 = 1
print(numpy.sum(cos2,axis=1))

#%%
#contributions aux axes
ctr = coord**2
for j in range(p):
 ctr[:,j] = ctr[:,j]/(n*eigval[j])

print(pandas.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]}))

#vérifions la théorie
print(numpy.sum(ctr,axis=0))

#le champ components_ de l'objet ACP
print(acp.components_)

#racine carrée des valeurs propres
sqrt_eigval = numpy.sqrt(eigval)

#corrélation des variables avec les axes
corvar = numpy.zeros((p,p))
for k in range(p):
 corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

#afficher la matrice des corrélations variables x facteurs
print(corvar)

#%%
#on affiche pour les deux premiers axes
print(pandas.DataFrame({'id':X.columns,'COR_1':corvar[:,0],'COR_2':corvar[:,1]}))

#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(p):
    plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage
plt.show()

#%%
#cosinus carré des variables
cos2var = corvar**2
print(pandas.DataFrame({'id':X.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))

#contributions
ctrvar = cos2var
for k in range(p):
 ctrvar[:,k] = ctrvar[:,k]/eigval[k]
#on n'affiche que pour les deux premiers axes
print(pandas.DataFrame({'id':X.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))




############################################################

#%% Variables supplémentaire
#importation des variables supplémentaires
varSupp = pandas.read_excel("autos_acp_pour_python.xlsx",sheet_name=2,header=0,index_col=0)
print(varSupp)

#variables supplémentaires quanti
vsQuanti = varSupp.iloc[:,:2].values
print(vsQuanti)

#corrélation avec les axes factoriels
corSupp = numpy.zeros((vsQuanti.shape[1],p))
for k in range(p):
    for j in range(vsQuanti.shape[1]):
        corSupp[j,k] = numpy.corrcoef(vsQuanti[:,j],coord[:,k])[0,1]
 
#affichage des corrélations avec les axes
print( corSupp)

#cercle des corrélations avec les var. supp
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#variables actives
for j in range(p):
 plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))
 
#variables illustratives
for j in range(vsQuanti.shape[1]):
 plt.annotate(varSupp.columns[j],(corSupp[j,0],corSupp[j,1]),color='g')
 
#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage
plt.show()


#%% pour rajouter les classes

#traitement de var. quali supplémentaire
vsQuali = varSupp.iloc[:,2]
print(vsQuali)


#modalités de la variable qualitative
modalites = numpy.unique(vsQuali)
print(modalites)

#liste des couleurs
couleurs = ['r','g','b']
#faire un graphique en coloriant les points
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6)
axes.set_ylim(-6,6)
#pour chaque modalité de la var. illustrative
for c in range(len(modalites)):
     #numéro des individus concernés
     numero = numpy.where(vsQuali == modalites[c])
     #les passer en revue pour affichage
     for i in numero[0]:
         plt.annotate(X.index[i],(coord[i,0],coord[i,1]),color=couleurs[c])
 
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
#affichage
plt.show()