# Introduction

Bienvenue dans la documentation du laboratoire de régulation.

## Présentation de la structure de la doc

## Présentation du langage Python

## Installation d'Anaconda

1. Télécharger Anaconda : https://www.anaconda.com/products/individual-d

2. Suivre les instructions d'installation du fichier téléchargé.

:::{note}
Même si l'installation d'Anaconda est assez rapide, il faut lui laisser le temps d'installer toutes ses applications en arrière-plan lors de la première ouverture de celui-ci. Néanmoins, vous devriez être en mesure de suivre les prochaines étapes pendant que ces applications s'installent.
:::

### Créer un environnement propre au cours (Python 3.7)

La version de Python installée avec Anaconda n'est pas celle qu'il faut pour utiliser les packages utiles pour le laboratoire de régulation. Dès lors, il faut créer un environnement avec la bonne version de Python. Pour ce faire, il suffit de suivre les instructions suivantes:

1. Ouvrez l'onglet `Environments`.

    :::{image} images/Intro_AnacondaInstallation1.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

2. Poussez sur le bouton `Create`.

    :::{image} images/Intro_AnacondaInstallation2.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

3. Configurez le nouvel environnement en lui donnant un nom sans espaces ni caractères spéciaux et en choisissant la version **3.7** de Python.

    :::{image} images/Intro_AnacondaInstallation3.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

    :::{note}
    Si l'option `3.7` n'apparaît pas dans la liste déroulante, cela veut dire qu'Anaconda n'est pas encore prêt. Il faut donc attendre que tout s'installe correctement en arrière-plan. Attendez un peu et recommencez l'opération jusqu'à ce que l'option `3.7` apparaisse bien dans la liste déroulante.
    :::

4. Retournez dans le menu `Home`.

5. Installez l'application `CMD.exe Prompt`.

    :::{image} images/Intro_AnacondaInstallation4.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

6. Installer l'IDE `Spyder`.

    :::{image} images/Intro_AnacondaInstallation5.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

7. Ouvrez les paramètres d'Anaconda en allant dans `File->Preferences` ou via le raccourci `CTRL+P`.

8. Configurez votre environnement comme environnement par défaut en le sélectionnant dans la liste déroulante et en validant les modifications.

    :::{image} images/Intro_AnacondaInstallation6.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

### Installation du package Python Control

1. Ouvrez la console de commande d'Anaconda (*Anaconda Prompt*):

    :::{image} images/Intro_ConsoleCMD.png
    :alt: ConsoleCMD
    :width: 175px
    :align: center
    :::

2. Entrez la commande suivante:
    ```
    conda install -c conda-forge control
    ```

3. À la question `Proceed?`, entrez `y` et valider avec la touche `enter`.

4. Une fois l'installation terminée, vous pouvez fermer la console.

## Configurer Spyder

### Avoir les figures dans des fenêtres séparées

1. Ouvrez la fenêtre des préférences.

    :::{image} images/Intro_Spyder1.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

2. Ouvrez la section `IPyhton console`.

    :::{image} images/Intro_Spyder2.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

3. Sélectionnez l'onglet `Graphics`, sélectionnez l'option `Automatic` dans la liste déroulante de la sous-section `Graphics backend` et cliquez sur le bouton `Apply`.

    :::{image} images/Intro_Spyder3.png
    :alt: ConsoleCMD
    :width: 525px
    :align: center
    :::

## Télécharger le package Python du labo
Afin de pouvoir utiliser les fonctions utiles pour le laboratoire de régulation, il est nécessaire de télécharger la dernière version du package python `ReguLabFct` [ici](https://studenthelmobe-my.sharepoint.com/:u:/g/personal/a_vanlaethem_helmo_be/ERCKwgbolmtMmXe6-FW0OOkBLWEK8-u5zAFORvJ_39Es4g?e=K58QB8).

:::{note}
Il est possible que votre navigateur internet ou votre antivirus empêche le téléchargement en le considérant comme une menace. Vous pouvez forcer le téléchargement car, promis, ce n'est pas un virus 😉
:::

Une fois téléchargé, il faut le placer dans le répertoire où se trouveront vos scripts afin que ces derniers puissent l'utiliser.

## Télécharger la documentation du package ReguLabFct
Afin de facilité l'usage du package `ReguLabFct`, une documentation existe et peut être téléchargeable [ici](https://studenthelmobe-my.sharepoint.com/:f:/g/personal/a_vanlaethem_helmo_be/EvRgBmANcQNMmIhF2nm2kBcBcyePZxDX5ah4yrH_FcUjwg?e=RKFGXU)