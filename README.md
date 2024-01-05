# SEVES_Interpolation_without_log

Le programme permettant d'afficher les panneaux interpolés se compose de 1 seul fichier : il se compose de l'esemble des fonction utilisées et du main permettant d'appeler l'ensemble des fonctions, itérer l'opération sur tous les panneaux et les afficher dans un PDF final.

### Exemple de représentation obtenue avec ce programme : 
![image](https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/cf6d7075-4ea9-47c5-8cc1-f8e1a41a16f3)


# Notice de préparation du excel

Pour l'usage de ce programme, il est nécessaire de produire un tableau excel de l'ensemble des forages à représenter avec les données rassemblées dans les colonnes des noms ci-dessous. 

Un fois l'excel finalisé, il faut l'exporter au format CSV.

A noter que les éléments en gras, notés INDISPENSABLE sont nécessaires pour le bon fonctionnement du programme, si inconnus, mettre une valeurs par défaut pour que le programme ne plante pas. (L'ensemble des nombres décimaux sont notés avec des points et nondes virgules) :

### OID : identifiant unique du forage (à choisir arbitrairement mais doit être unique) INDISPENSABLE

Les deux variables suivantes sont des fichiers qui ont été extraits de RES2DINV après inversion (**mettres les fichiers ensembles dans un même dossier**).
### FICHIER_PANNEAU : fichier type ____modres.dat qui contient les abscisses, les profondeurs, et les résistivités associées. INDISPENSABLE
### FICHIER_BLANC : fichier type ____modres.bln contient les coordonnées du cadre permettant d'exclure les valeurs des faibles zones de sensibilité. INDISPENSABLE

### NOM_VILLAG (INDISPENSABLE) : le nom du village

Q_DEVELOPP : le débit de développement

Q_MAX_PALI : le débit max par pallier

Q_SPECIFIQUE : le débit spécifique

RABATTEMEN : le rabattement

### Ci-dessous une image d'exemple de tableau excel : 
<img width="453" alt="image" src="https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/aa2b06c4-2155-410a-bb9e-955c8d28f68d">


# Utilisation du 'main'

Votre excel prêt et exporté au format CSV, il faut s'assurer que votre programme aille chercher les fichiers dans le bon répertoire.

Pour cela, il est impératif d'indiquer ce chemin dans le 'main', ainsi que le nom du fichier tel que le montre l'image suivante. Attention : le sens des slashs est inversé. Il est conseillé de mettre le CSV dans le même dossier que l'ensemble des .dat et des .bln.

<img width="402" alt="image" src="https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/5ee87aac-09d2-4d67-b645-94dbafbeda57">

Par ailleurs, ne pas oublier de renommer votre PDF de sortie.

# Tutoriel

Lancer le script dans le même répertoire que les fichiers *.csv, *.bln et *.dat fournis dans le répertoire tutoriel



