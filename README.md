# SEVES_Interpolation

Le programme permettant d'afficher les panneaux interpolés se compose de 2 fichiers : le fichiers contenant les fonctions, nommé 'fonctions.py', et le 'main', nomé 'Interpolation_Inversion_Profil.py', permettant d'appeler l'ensemble des fonctions, itérer l'opération sur tous les panneaux et les afficher dans un PDF final.

### Exemple de représentation obtenue avec ce programme : 
<img width="647" alt="image" src="https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/9b447a09-0e8d-4918-89e9-8805d3c63bbb">

# Notice de préparation du excel

Pour l'usage de ce programme, il est nécessaire de produire un tableau excel de l'ensemble des forages à représenter avec les données rassemblées dans les colonnes des noms ci-dessous. 

Un fois l'excel finalisé, il faut l'exporter au format CSV.

A noter que les éléments en gras, notés INDISPENSABLE sont nécessaires pour le bon fonctionnement du programme, si inconnus, mettre une valeurs par défaut pour que le programme ne plante pas. (L'ensemble des nombres décimaux sont notés avec des points et nondes virgules) :

### OID : identifiant unique du forage (à choisir arbitrairement mais doit être unique) INDISPENSABLE

Les deux variables suivantes sont des fichiers qui ont été extraits de RES2DINV après inversion (**mettres les fichiers ensembles dans un même dossier**).
### FICHIER_PANNEAU : fichier type ____modres.dat qui contient les abscisses, les profondeurs, et les résistivités associées. INDISPENSABLE
### FICHIER_BLANC : fichier type ____modres.bln contient les coordonnées du cadre permettant d'exclure les valeurs des faibles zones de sensibilité. INDISPENSABLE

Les profondeurs des arrivées d'eau (le programme est écrit pour aller jusqu'à 10, rajouter des colonnes dans le tableau excel s'il en manque) :

PROF_VE_1,
PROF_VE_2,
PROF_VE_3,
PROF_VE_4,
PROF_VE_5,
PROF_VE_6,
PROF_VE_7,
PROF_VE_8,
PROF_VE_9,
PROF_VE_10.

Les débits des arrivées d'eau (idem) :

DEBIT_VE_1,
DEBIT_VE_2,
DEBIT_VE_3,
DEBIT_VE_4,
DEBIT_VE_5,
DEBIT_VE_6,
DEBIT_VE_7,
DEBIT_VE_8,
DEBIT_VE_9,
DEBIT_VE_10.

### Les profondeurs des lithos (INDISPENSABLE (programme écrit pour 11 couches) ) : 

PROF_LITHO,
PROF_LIT_1,
PROF_LIT_2,
PROF_LIT_3,
PROF_LIT_4,
PROF_LIT_5,
PROF_LIT_6,
PROF_LIT_7,
PROF_LIT_8,
PROF_LIT_9,
PROF_LIT10.


### L'état des lithos (INDISPENSABLE) (programme écrit pour 11 couches) (parmis RECOUVREMENT, ALTERE, FRACTURE, SAIN, ALTERE / FRACTURE, ALTERE / MICROFISSURE, MICROFISSURE, TRES ALTERE, SAIN / MICROFISSURE, FISSURE, sinon rajouter dans le dictionnaire) :

TYPE_LITHO,
TYPE_LIT_1,
TYPE_LIT_2,
TYPE_LIT_3,
TYPE_LIT_4,
TYPE_LIT_5,
TYPE_LIT_6,
TYPE_LIT_7,
TYPE_LIT_8,
TYPE_LIT_9,
TYPE_LIT10.

### Les lithos (INDISPENSABLE) (programme écrit pour 11 couches) (parmis TERRE VEGETALE, ARGILE, ARGILE SABLEUSE, SABLE ARGILEUX, SABLE, DIORITE, GRANITE, MICROGRANITE, GRANODIORITE, MONZOGRANITE; AMPHIBOLITE, DOLERITE, sinon rajouter la litho dans le dictionnaire) : 

LITHO_1,
LITHO_2,
LITHO_3,
LITHO_4,
LITHO_5,
LITHO_6,
LITHO_7,
LITHO_8,
LITHO_9,
LITHO_10,
LITHO_11.

### NOM_VILLAG (INDISPENSABLE) : le nom du village

Q_DEVELOPP : le débit de développement

Q_MAX_PALI : le débit max par pallier

Q_SPECIFIQUE : le débit spécifique

RABATTEMEN : le rabattement

### POSITION_PROJETEE (INDISPENSABLE) : la position du forage le long du profil

### Z : l'altitude du forage INDISPENSABLE

### PROFONDEUR (INDISPENSABLE) : la profondeur du forage

NIVEAU_STA : le niveau statique dans le forage

PROF_TOP_MFT : profondeur à laquelle le marteau fond de trou a été équipé

### SITE_GEOPH (INDISPENSABLE): le site géophysique sur lequel le forage a été réalisé
### RMS_FACTOR (INDISPENSABLE): l'erreur relative à l'inversion
### CUT_OFF_FACTOR (INDISPENSABLE): le cut-off factor utilisé pour l'inversion


### Ci-dessous une image d'exemple de tableau excel : 

<img width="587" alt="image" src="https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/55b7a9e4-47ce-4f37-bbd3-c16554f91d36">

# Utilisation du 'main'

Votre excel prêt et exporté au format CSV, il faut s'assurer que votre programme aille chercher les fichiers dans le bon répertoire.

Pour cela, il est impératif d'indiquer ce chemin dans le 'main', ainsi que le nom du fichier tel que le montre l'image suivante. Attention : le sens des slashs est inversé. Il est conseillé de mettre le CSV dans le même dossier que l'ensemble des .dat et des .bln.

<img width="402" alt="image" src="https://github.com/PalomsP/SEVES_Interpolation_ERT/assets/144214305/5ee87aac-09d2-4d67-b645-94dbafbeda57">

Par ailleurs, ne pas oublier de renommer votre PDF de sortie.

