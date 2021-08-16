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

from control.matlab import *  # Python Control Systems Toolbox (compatibility with MATLAB)
import numpy as np              # Library to manipulate array and matrix
import matplotlib.pyplot as plt # Library to create figures and plots
import math # Library to be able to do some mathematical operations
import ReguLabFct as rlf # Library useful for the laboratory of regulation of HELMo Gramme


# In[ ]:




