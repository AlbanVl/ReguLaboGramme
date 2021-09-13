#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:39:27 2021

Solution Questionnaire A1 et A2

@author: Alban Van Laethem
"""

#%% Includes

from control import matlab as ml  # Python Control Systems Toolbox (compatibility with MATLAB)
import numpy as np              # Library to manipulate array and matrix
import matplotlib.pyplot as plt # Library to create figures and plots
import math # Library to be able to do some mathematical operations
import ReguLabFct as rlf # Library useful for the laboratory of regulation of Gramme

#clear
plt.close('all')
#reset -f

"""
K=1
wn=1

# Définition des coefficients d'amortissement
zetas =[0.1, 0.2, 0.3, 0.4, 0.42, 0.5, 0.6, 0.7, 0.8, 1, 1.41, 2, 6, 10]
g=[]

from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = cycle(prop_cycle.by_key()['color'])

# Calcule les différentes fonctions de transfert ainsi que la réponse indicielle
for i, zeta in enumerate(zetas): 
    n=K;
    d=[(1/wn)**2, 2*zeta/wn, 1];
    g.append(tf(n, d));
    h=g[i];
    poles, zeros = rlf.pzmap(h,NameOfFigure = "Pole Zero Map", sysName = zeta, color=next(colors));
plt.plot([poles.real[0], 0], [0, 0], 'k:') # Ligne horizontale passant par 0 pour marquer l'axe des imaginaire
"""

"""
h=tf(20, [2, 4, 5])
info = rlf.stepWithInfo(h, plot_DCGain=False, plot_overshoot=False, plot_rt=False, plot_st=True)
rlf.nichols(h)
rlf.printInfo(info)
"""

"""
ft = rlf.generateTfFromCharac(0.8, 5, 0.5)
yout, t = step(ft)
yout_noisy = rlf.addNoise(t, yout, variance=0.05)

# Input signal
x = [1]*len(t)
x[0] = 0

rlf.saveFT(t, yout_noisy, x, name="Ordre2")

c = rlf.TfCharact()
c.G = 2
__ = rlf.fineIdentification("Ordre2.csv", 1, 10, 0.2, c)
"""

"""
from scipy import signal

h=feedback(tf(6, [2, 3, 1]))

resolution = 10000
value_init = 10
value_fin = 30

__, t = step(h)
period = t[-1]*2

t = np.linspace(0, period, resolution+1) # resolution+1 because t begins at 0
delta_values = value_fin-value_init
moy_values = (value_fin+value_init)/2
sq = -delta_values/2*signal.square(2*np.pi*(1/period)*t)+moy_values # '-' to start from the bottom
sq[0] = 0
sq[-1] = value_fin

yout, t, __ = lsim(h, sq, t)

plt.figure("Step " + str(value_init) + " -> " + str(value_fin))
plt.plot(t, sq)
plt.plot(t, yout)
plt.legend(["U(t)", "Y(t)"])

yout_useful = yout[int(resolution/2):] # To limit the research after the step

peak = np.amax(yout_useful)
peak_indice = np.where(yout_useful == np.amax(yout_useful))
peak_indice = peak_indice[-1][-1]
peak_time = peak_indice*(period/resolution)

plt.plot(period/2+peak_time, peak, 'ko')

t_useful = np.linspace(0, period/2, int(resolution/2)+1)
plt.figure("Step " + str(value_init) + " -> " + str(value_fin) + " (FOCUS)")
plt.plot(t_useful, yout_useful)

info = rlf.step_info(t_useful, yout_useful)
"""

"""
G_BF =feedback( tf(6, [2, 3, 1]))

# Saut de 10° à 30°
import warnings # Package permettant de gérer l'affichage de messages d'erreurs
warnings.filterwarnings('ignore') # Pour éviter d'afficher un message d'erreur inutile
peak, peakTime, yout, t = rlf.stepFromTo(G_BF, 10, 30, focus=True, NameOfFigure="Steps comparaison")
warnings.resetwarnings() # Pour réactiver l'affichage d'erreurs
info = rlf.step_info(yout, t)
rlf.printInfo(info)
print("")

# Saut de 0° à 30°
warnings.filterwarnings('ignore') # Pour éviter d'afficher un message d'erreur inutile
peak, peakTime, yout, t = rlf.stepFromTo(G_BF, 0, 30, focus=True, NameOfFigure="Steps comparaison")
warnings.resetwarnings() # Pour réactiver l'affichage d'erreurs
info = rlf.step_info(yout, t)
rlf.printInfo(info)
print("")

# Saut de 29° à 30°
warnings.filterwarnings('ignore') # Pour éviter d'afficher un message d'erreur inutile
peak, peakTime, yout, t = rlf.stepFromTo(G_BF, 29, 30, focus=True, NameOfFigure="Steps comparaison")
warnings.resetwarnings() # Pour réactiver l'affichage d'erreurs
info = rlf.step_info(yout, t)
rlf.printInfo(info)
print("")

"""

H_BO = ml.tf(1, [1, 1, 1])
ml.nyquist(H_BO, labelFreq=100)