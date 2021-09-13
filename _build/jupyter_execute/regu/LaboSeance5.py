#!/usr/bin/env python
# coding: utf-8

# # Séance 5

# ## Objectifs de la séance
# 
# - Tracé des différents diagrammes fréquentiels
# - Effet d’un correcteur proportionnel dans ces différents diagrammes => régulateur P
# - Construction asymptotiques du diagramme de Bode par décomposition de la fonction de transfert en fonction simple
# - Analyse de la stabilité/instabilité

# In[1]:


from IPython.display import Image, display, Markdown

from control import matlab as ml  # Python Control Systems Toolbox (compatibility with MATLAB)
import numpy as np              # Library to manipulate array and matrix
import matplotlib.pyplot as plt # Library to create figures and plots
import math # Library to be able to do some mathematical operations
import ReguLabFct as rlf # Library useful for the laboratory of regulation of HELMo Gramme


# ## Diagrammes fréquentiels
# 
# Représentez le système modélisé par un pôle simple
# $
#  H_{BO}(p)=\frac{1}{p+1}
# $
# dans les différents diagrammes pour visualiser l’effet d’un pôle : p=-1, $\tau$=1s et $\omega_n$=1rad/s

# In[2]:


H_BO = ml.tf(1, [1, 1])


# ### Nyquist (*cf. p. 4-6*)

# In[3]:


real, imag, freq = ml.nyquist(H_BO);


# :::{admonition} Attention
# :class: warning
# Attention, le graphique est tracé aussi pour des $\omega_n<0$ (en trait discontinu) ce qui n’a aucun sens physiquement! Il ne faut donc tenir compte **que** du tracé en trait plein.
# :::

# ### Bode

# ### Nichols

# In[ ]:





# In[ ]:




