#!/usr/bin/env python
# coding: utf-8

# # Séance 3

# ## Objectifs de la séance
# 
# - Analyse d’un système en boucle fermée : feedback
# - Analyse de la réponse indicielle et valeurs « idéales » de la réponse à un échelon
# - Influence du correcteur proportionnel sur les caractéristiques temporelles : dépassement, temps de réponse, ... et sur les pôles du système en BF.

# In[1]:


from IPython.display import Image, display, Markdown

from control.matlab import *  # Python Control Systems Toolbox (compatibility with MATLAB)
import numpy as np              # Library to manipulate array and matrix
import matplotlib.pyplot as plt # Library to create figures and plots
import math # Library to be able to do some mathematical operations
import ReguLabFct as rlf # Library useful for the laboratory of regulation of HELMo Gramme

