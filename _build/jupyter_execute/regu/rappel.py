#!/usr/bin/env python
# coding: utf-8

# # Aide-mémoire
# 
# Site Jupyter Book: https://jupyterbook.org/intro.html

# ## Avertissements
# 
# :::{danger}
# Danger repéré.
# :::
# 
# :::{note}
# Info à transmettre.
# :::
# 
# :::{admonition} Attention
# :class: warning
# Chose à faire attention!
# :::
# 
# :::{admonition} Astuce
# :class: tip
# Astuce à donner.
# :::
# 
# :::{admonition} A voir aussi
# :class: seealso
# Chose à voir.
# :::

# ## Symboles Latex
# 
# - Liste des symboles Latex : https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols
# 
# - Ajouter des flèches: https://www.math-linux.com/latex-26/faq/latex-faq/article/latex-arrows

# ## Cacher - supprimer
# 
# https://jupyterbook.org/interactive/hiding.html?highlight=hide
# 
# 

# ## Ajouter une ligne blanche
# 
# &nbsp;

# ## Equations en accolades
# 
# **Seule**
# 
# $$
# \left\{
#     \begin{array}{ll}
#         \frac{2\zeta}{\omega_n}=\frac{3}{K+1}\\
#         \frac{1}{\omega_n^2}=\frac{2}{K+1}
#     \end{array}
# \right.
# $$
# 
# **Sur la même ligne:**
# 
# $$
# \begin{alignat*}{2}
# \left\{ \begin{aligned}
#     \begin{array}{ll}
#         \frac{3}{K+1}=\frac{0.9}{\omega_n} \\
#         \frac{1}{\omega_n^2}=\frac{2}{K+1}
#     \end{array}
# \end{aligned}\right.
# \Rightarrow
# \left\{ \begin{aligned}
#     \begin{array}{ll}
#         \mathbf{K=4.4} \\
#         \omega_n=1.66
#     \end{array}
# \end{aligned}\right.
# \end{alignat*}\
# $$
# 
# **Accolade à droite:**
# 
# $$
# \left. \begin{array}{r} 
#         p_1 \\
#         p_2
# \end{array} \right\}
# = \omega_n*(-\zeta \pm j\sqrt{1-\zeta^2})
# = \sigma \pm j\omega_d
# $$

# ## Commandes consoles
# 
# ### Lancement environnement
# 
#     source regu/bin/activate
# 
# ### Fermeture environnement
#     deactivate
# 
# ### Accès dossier
# 
#     cd OneDrive\ -\ student.helmo.be/Cours/MA1/MA1\ -\ Régulation/
# 
# ### Build
# 
#     jb build ReguLaboGramme/
# 
# ### Supprimer un fichier avec son contenu
# 
#     rm -rf dir-name
# 
# 
