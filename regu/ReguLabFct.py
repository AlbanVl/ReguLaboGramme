# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:23:05 2021

Fonctions utiles pour utiliser python en régu

@author: Alban Van Laethem
"""

import math # Library to be able to do some mathematical operations
from control import matlab as ml # Python Control Systems Toolbox (compatibility with MATLAB)
# from control.matlab import *
# from control.freqplot import default_frequency_range
# import control.nichols as cn
import numpy as np # Library to manipulate array and matrix
import matplotlib.pyplot as plt # Library to create figures and plots
# import scipy as sp
from matplotlib.offsetbox import AnchoredText  # To print text inside a plot


# %% Object to store all the important informations provided by the step
class Info():

    """
    Object to store all the interesting informations provided by the step
    response.

    Attributes
    ----------

    RiseTime: float
        Time it takes for the response to rise from 10% to 90% of the
        steady-state response.

    SettlingTime: float
        Time it takes for the error e(t) = \|y(t) – yfinal\| between the
        response y(t) and the steady-state response yfinal to fall below 5% of
        the peak value of e(t).

    SettlingMin: float
        Minimum value of y(t) once the response has risen.

    SettlingMax: float
        Maximum value of y(t) once the response has risen.

    Overshoot: float
        Percentage overshoot, relative to yfinal.

    Undershoot: float
        Percentage undershoot.

    Peak: float
        Peak absolute value of y(t).

    PeakTime: float
        Time at which the peak value occurs.

    DCGain: float
        Low-frequency (DC) gain of LTI system.
    """

    DCGain = None
    RiseTime = None
    SettlingTime = None
    SettlingMin = None
    SettlingMax = None
    Overshoot = None
    Undershoot = None
    Peak = None
    PeakTime = None


# %% printInfo
def printInfo(info):
    """
    Print in alphabetical order the informations stored in the given Info
    object.

    Parameters
    ----------
    info: Info
        Object in which all the informations of the step response are stored.

    Returns
    -------
    None
    """

    # Transform into a dict to be able to iterate
    temp = vars(info)
    for item in sorted(temp):
        print(item, ':', temp[item])

    # print("RiseTime:", info.RiseTime)
    # print("SettlingTime:", info.SettlingTime)
    # print("SettlingMin:", info.SettlingMin)
    # print("SettlingMax:", info.SettlingMax)
    # print("Overshoot:", info.Overshoot)
    # print("Undershoot:", info.Undershoot)
    # print("Peak:", info.Peak)
    # print("PeakTime:", info.PeakTime)
    # print("DCGain:", info.DCGain)


# %% pzmap
# pzmap function reviewed by Alban Van Laethem
def pzmap(sys, plot=True, title='Pole Zero Map', NameOfFigure="",
          sysName="", color="b"):
    """
    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        Linear system for which poles and zeros are computed.
    plot: bool, optional
        If ``True`` a graph is generated with Matplotlib,
        otherwise the poles and zeros are only computed and returned.
    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.
    sysName: String, optional
        Name of the system to plot.

    Returns
    -------
    poles: array
        The systems poles
    zeros: array
        The system's zeros.
    """

    poles = sys.pole()
    zeros = sys.zero()

    if plot:
        if NameOfFigure == "":
            plt.figure()
        else:
            plt.figure(NameOfFigure)

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            plt.scatter(poles.real, poles.imag, s=50, marker='x',
                        label=sysName, facecolors=color)
        if len(zeros) > 0:
            plt.scatter(zeros.real, zeros.imag, s=50, marker='o',
                        label=sysName, facecolors=color, edgecolors='k')

        plt.title(title)
        plt.ylabel("Imaginary Axis (1/seconds)")
        plt.xlabel("Real Axis (1/seconds)")
        if sysName != '':
            plt.legend()

    # Return locations of poles and zeros as a tuple
    return poles, zeros


# %% Function stepWithInfo
def stepWithInfo(sys, info=None, T=None, SettlingTimeThreshold=0.05,
                 RiseTimeThresholdMin=.10, RiseTimeThresholdMax=.90,
                 resolution=10000, NameOfFigure="", sysName='',
                 linestyle='-', plot_st=True, plot_rt=True,
                 plot_overshoot=True, plot_DCGain=True):
    """
    Trace the step response and the interesting points and return those
    interesting informations.

    WARNING: Overshoot is in %!

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    info: Info, optional
        Object in which to store all the informations of the step response

    T: 1D array, optional
        Time vector.

    SettlingTimeThreshold: float, optional
        Threshold of the settling time.

    RiseTimeThresholdMin: float, optional
        Lower rise time threshold.

    RiseTimeThresholdMax: float, optional
        Upper rise time threshold.

    resolution: long, optional
        Number of points calculated to trace the step response.

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    sysName: String, optional
        Name of the system to plot.

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the step response (default is '-').

    plot_st: bool, optionnal
        Plot the settling time point if True (default is True).

    plot_rt: bool, optionnal
        Plot the rise time point if True (default is True).

    plot_overshoot: bool, optionnal
        Plot the overshoot point if True (default is True).

    plot_DCGain: bool, optionnal
        Plot the DC gain point if True (default is True).

    Returns
    -------
    info: Info
        Object in which all the informations of the step response are stored.
    """

    [yout, t] = step_(sys, T, resolution, NameOfFigure, sysName,
                      linestyle=linestyle)

    # Add the interestings points to the plot and store the informations in
    # info
    info = step_info(t, yout, info, SettlingTimeThreshold,
                     RiseTimeThresholdMin, RiseTimeThresholdMax, plot_st,
                     plot_rt, plot_overshoot, plot_DCGain)

    return info


# %% Fonction pour tracer les résultats du step
def step_(sys, T=None, resolution=10000, NameOfFigure="",
          sysName='', linestyle='-'):
    """
    Trace the step with the given parameters.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    T: 1D array, optional
        Time vector.

    resolution: long, optional
        Number of points calculated to trace the step response.

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    sysName: String, optional
        Name of the system to plot.

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the step response (default is '-').

    Returns
    -------
    yout: 1D array
        Response of the system.

    t: 1D array
        Time vector.
    """

    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)
    [yout, t] = ml.step(sys, T)
    # Pour modifier la résolution
    [yout, t] = ml.step(sys, np.linspace(t[0], t[-1], resolution))

    # Arrondi les valeurs à x décimales
    # yout = np.around(yout, decimals=6)
    # t = np.around(t, decimals=6)

    plt.plot(t, yout, label=sysName, linestyle=linestyle)
    plt.title("Step Response")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    if sysName != '':
        plt.legend()

    return [yout, t]

# %% Fonction step_info
def step_info(t, yout, info=None, SettlingTimeThreshold=0.05,
              RiseTimeThresholdMin=.10, RiseTimeThresholdMax=.90, plot_st=True,
              plot_rt=True, plot_overshoot=True, plot_DCGain=True):
    """
    Trace the interesting points of a given step plot.

    Parameters
    ----------
    t: 1D array
        Time vector.

    y: 1D array
        Response of the system.

    info: Info, optional
        Object in which to store all the informations of the step response

    SettlingTimeThreshold: float, optional
        Threshold of the settling time (default is 0.05).

    RiseTimeThresholdMin: float, optional
        Lower rise time threshold (default is 0.10).

    RiseTimeThresholdMax: float, optional
        Upper rise time threshold (default is 0.90).

    plot_st: bool, optionnal
        Plot the settling time point if True (default is True).

    plot_rt: bool, optionnal
        Plot the rise time point if True (default is True).

    plot_overshoot: bool, optionnal
        Plot the overshoot point if True (default is True).

    plot_DCGain: bool, optionnal
        Plot the DC gain point if True (default is True).

    Returns
    -------
    info: Info
        Object in which all the informations of the step response are stored.
    """

    # Creation of the object info if not given
    if info is None:
        info = Info()

    # Store the colour of the current plot
    color = plt.gca().lines[-1].get_color()

    # Calcul du dépassement en prenant la valeur max retourné par step et en la
    # divisant par la valeur finale
    osIndice = np.where(yout == np.amax(yout))  # renvoie un tuple d'array
    osIndice = osIndice[-1][-1] # lit le dernier indice répondant à la condition

    info.Peak = yout.max()
    info.Overshoot = (yout.max()/yout[-1]-1)*100
    info.PeakTime = float(t[osIndice])
    #print ("Overshoot:", info.Overshoot)

    if plot_overshoot:
        plt.plot([t[osIndice], t[osIndice]], [0, yout[osIndice]], 'k-.',
                 linewidth=0.5)  # Vertical
        plt.plot([t[0], t[osIndice]], [yout[osIndice], yout[osIndice]], 'k-.',
                 linewidth=0.5)  # Horizontale
        plt.plot(t[osIndice], yout[osIndice], color=color, marker='o')

    # Calcul du temps de montée en fonction du treshold (par défaut: de 10% à
    # 90% de la valeur finale)
    delta_values = yout[-1]-yout[0]
    RiseTimeThresholdMinIndice = next(i for i in range(0, len(yout)-1) if
                                      yout[i]-yout[0] > delta_values*RiseTimeThresholdMin)
    RiseTimeThresholdMaxIndice = next(i for i in range(0, len(yout)-1) if
                                      yout[i]-yout[0] > delta_values*RiseTimeThresholdMax)

    RiseTimeThreshold = [None] * 2
    RiseTimeThreshold[0] = t[RiseTimeThresholdMinIndice]-t[0]
    RiseTimeThreshold[1] = t[RiseTimeThresholdMaxIndice]-t[0]
    info.RiseTime = RiseTimeThreshold[1] - RiseTimeThreshold[0]
    #print ("RiseTime:", info.RiseTime)

    if plot_rt:
        plt.plot([t[RiseTimeThresholdMinIndice], t[RiseTimeThresholdMinIndice]],
                 [0, yout[RiseTimeThresholdMaxIndice]], 'k-.', linewidth=0.5)  # Limite gauche
        plt.plot([t[RiseTimeThresholdMaxIndice], t[RiseTimeThresholdMaxIndice]],
                 [0, yout[RiseTimeThresholdMaxIndice]], 'k-.', linewidth=0.5)  # Limite droite
        plt.plot([t[0], t[RiseTimeThresholdMaxIndice]],
                 [yout[RiseTimeThresholdMaxIndice],
                  yout[RiseTimeThresholdMaxIndice]], 'k-.', linewidth=0.5)  # Limite horizontale
        plt.plot(t[RiseTimeThresholdMaxIndice], yout[RiseTimeThresholdMaxIndice],
                 color=color, marker='o')

    # Calcul du temps de réponse à x% (5% par défaut)
    settlingTimeIndice = next(i for i in range(len(yout)-1, 1, -1) if
                              abs(yout[i]-yout[0])/delta_values > (1+SettlingTimeThreshold)
                              or abs(yout[i]-yout[0])/delta_values < (1-SettlingTimeThreshold))
    info.SettlingTime = t[settlingTimeIndice]-t[0]
    #print ("SettlingTime:", info.SettlingTime)

    if plot_st:
        plt.plot([0, max(t)], [yout[0]+delta_values*(1+SettlingTimeThreshold),
                               yout[0]+delta_values*(1+SettlingTimeThreshold)],
                 'k-.', linewidth=0.5)  # Limite haute
        plt.plot([0, max(t)], [yout[0]+delta_values*(1-SettlingTimeThreshold),
                               yout[0]+delta_values*(1-SettlingTimeThreshold)],
                 'k-.', linewidth=0.5)  # Limite basse
        plt.plot([t[settlingTimeIndice], t[settlingTimeIndice]],
                 [0, yout[settlingTimeIndice]], 'k-.', linewidth=0.5)  # Vertical
        plt.plot(t[settlingTimeIndice], yout[settlingTimeIndice], color=color,
                 marker='o')

    # Gain statique
    info.DCGain = yout[-1]
    if plot_DCGain:
        plt.plot([0, max(t)], [yout[-1], yout[-1]], 'k:', linewidth=0.5)
        plt.plot(t[-1], yout[-1], color=color, marker='o')
    #print ("DC gain:", info.DCGain)

    return info

# %% Step from a value to another

def stepFromTo(sys, value_init, value_fin, resolution=10000, focus=True, 
               NameOfFigure=""):
    """
    Trace the step when the input goes from a given value to another.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    value_init: float
        Initial value of the intput.

    value_fin: float
        Final value of the intput.

    resolution: long, optional
        Number of points calculated to trace the step response.

    focus: boolean, optional
        Plot the interresting part of the step as the standard step function (True by default).

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    Returns
    -------
    Peak: float
        Peak absolute value of y(t).

    PeakTime: float
        Time at which the peak value occurs.

    yout_useful: 1D array
        Response of the system for the interresting part of the step.

    t_useful: 1D array
        Time vector of the interresting part of the step.
    """

    from scipy import signal

    __, t = ml.step(sys) # To get an optimized siez of t
    T = t[-1]*2 # To double the size of t to be able to trace the step from 0 and after stabilisation from the initial value

    t = np.linspace(0, T, resolution+1) # resolution+1 because t begins at 0
    delta_values = value_fin-value_init
    moy_values = (value_fin+value_init)/2
    sq = -delta_values/2*signal.square(2*np.pi*(1/T)*t)+moy_values # '-' to start from the bottom
    sq[0] = 0 # To start from 0 as the output of the lsim function start from 0
    sq[-1] = value_fin # To avoid to come back to the inital value

    yout, t, __ = ml.lsim(sys, sq, t) # Calculus of the special step

    yout_useful = yout[int(resolution/2):] # To limit the research after the step

    peak = np.amax(yout_useful)
    peak_indice = np.where(yout_useful == np.amax(yout_useful))
    peak_indice = peak_indice[-1][-1]
    peak_time = peak_indice*(T/resolution)

    t_useful = np.linspace(0, T/2, int(resolution/2)+1) # Calculus of the time vector to focus on the interresting part of the step

    if NameOfFigure=="":
        plt.figure("Step " + str(value_init) + " -> " + str(value_fin))
    else:
        plt.figure(NameOfFigure)
    if focus:
        plt.plot(t_useful, yout_useful)
    else:
        plt.plot(t, sq) # Plot the input
        plt.plot(t, yout) # Plot the output
        plt.legend(["U(t)", "Y(t)"])
        plt.plot(T/2+peak_time, peak, 'ko')

    return peak, peak_time, t_useful, yout_useful

# %% Get the gain and the frequency at a given phase
def getValues(sys, phaseValue, mag=None, phase=None, omega=None,
              printValue=True, NameOfFigure=""):
    """
    Get the values of the gain and the frequency at a given phase of the
    system.

    Get the values of the gain and the frequency at a given phase from given
    arrays of gains, phases and frequencies.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    phaseValue: float
        Phase at which we want to get the gain and frequency values.

    mag: 1D array, optional
        Array of gains (not in dB).

    phase: 1D array, optional
        Array of phases.

    omega: 1D array, optional
        Array of frequencies (in radians).

    printValue: boolean, optional
        print values if True (by default).

    NameOfFigure: String, optional
        Name of the figure in which to plot.

    Returns
    -------
    mag: float
         The gain value for the given phase.

    omega: float
        The gain value in rad/sec for the given phase.
    """

    lowLimit = -2
    highLimit = 2
    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)

    if(np.all(mag is None) and np.all(phase is None) and np.all(omega is None)):
        # liste de fréquences afin d'augmenter la résolution de calcul (par défaut: 50 éléments)
        w = np.logspace(lowLimit, highLimit, 10000)
        mag, phase, omega = ml.bode(sys, w, dB=True, Hz=False, deg=True)
        phase = phase*180/math.pi  # Pour avoir les phases en degrés plutôt qu'en radians
        idx = (np.abs(phase-phaseValue)).argmin()
        while idx in (np.size(phase)-1, 0):
            if idx == 0:
                lowLimit -= 1
            else:
                highLimit += 1
            # liste de fréquences afin d'augmenter la résolution de calcul (par défaut: 50 éléments)
            w = np.logspace(lowLimit, highLimit, 10000)
            mag, phase, omega = ml.bode(sys, w, dB=True, Hz=False, deg=True)
            phase = phase*180/math.pi  # Pour avoir les phases en degrés plutôt qu'en radians
            idx = (np.abs(phase-phaseValue)).argmin()

    else:
        phase = phase*180/math.pi  # Pour avoir les phases en degrés plutôt qu'en radians
        idx = (np.abs(phase-phaseValue)).argmin()

    if printValue:
        mag_dB = 20*np.log10(mag[idx])  # Pour avoir les gains en dB
        print(f"Gain à {phaseValue}° = {mag_dB} dB")
        print(f"Fréquence à {phaseValue}° = {omega[idx]} rad/sec")

    return mag[idx], omega[idx]

# %% Compute reasonable defaults for axes


def default_frequency_range(syslist):
    """Compute a reasonable default frequency range for frequency
    domain plots.

    Finds a reasonable default frequency range by examining the features
    (poles and zeros) of the systems in syslist.

    Parameters
    ----------
    syslist: list of Lti
        List of linear input/output systems (single system is OK)

    Returns
    -------
    omega: array
        Range of frequencies in rad/sec

    Examples
    --------
    >>> from matlab import ss
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> omega = default_frequency_range(sys)
    """
    # This code looks at the poles and zeros of all of the systems that
    # we are plotting and sets the frequency range to be one decade above
    # and below the min and max feature frequencies, rounded to the nearest
    # integer.  It excludes poles and zeros at the origin.  If no features
    # are found, it turns logspace(-1, 1)

    # Find the list of all poles and zeros in the systems
    features = np.array(())

    # detect if single sys passed by checking if it is sequence-like
    if not getattr(syslist, '__iter__', False):
        syslist = (syslist,)

    for sys in syslist:
        try:
            # Add new features to the list
            features = np.concatenate((features, np.abs(sys.pole())))
            features = np.concatenate((features, np.abs(sys.zero())))
        except:
            pass

    # Get rid of poles and zeros at the origin
    features = features[features != 0]

    # Make sure there is at least one point in the range
    if features.shape[0] == 0:
        features = [1]

    # Take the log of the features
    features = np.log10(features)

    #! TODO: Add a check in discrete case to make sure we don't get aliasing

    # Set the range to be an order of magnitude beyond any features
    omega = np.logspace(np.floor(np.min(features))-1,
                        np.ceil(np.max(features))+1)

    return omega

# %% Function to trace Nichols as needed for the laboratory
# Nichols function reviewed by Alban Van Laethem


def nichols(sys_list, omega=None, grid=None, labels=[''], NameOfFigure="",
            data=False, ax=None, linestyle='-'):
    """Nichols plot for a system

    Plots a Nichols plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys_list: list of LTI, or LTI
        List of linear input/output systems (single system is OK)

    omega: array_like, optional
        Range of frequencies (list or bounds) in rad/sec

    grid: boolean, optional
        True if the plot should include a Nichols-chart grid. Default is True.

    labels: list of Strings
        List of the names of the given systems

    NameOfFigure: String, optional
        Name of the figure in which to plot.

    data: boolean, optional
        True if we must return x and y (default is False)

    ax: axes.subplots.AxesSubplot, optional
        The axe on which to plot

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the nichols graph (default is '-').

    Returns
    -------
    if data == True:
        x: 1D array
            Abscisse vector
        y: 1D array
            Ordinate vector
    """

    # Open a figure with the given name or open a new one
    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)

    ax = ax or plt.gca()

    # Get parameter values
    #grid = config._get_param('nichols', 'grid', grid, True)

    # If argument was a singleton, turn it into a list
    if not getattr(sys_list, '__iter__', False):
        sys_list = (sys_list,)

    # Select a default range if none is provided
    if omega is None:
        omega = default_frequency_range(sys_list)

    for index, sys in enumerate(sys_list):
        # Get the magnitude and phase of the system
        mag_tmp, phase_tmp, omega = sys.freqresp(omega)
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)

        # Convert to Nichols-plot format (phase in degrees,
        # and magnitude in dB)
        x = np.unwrap(np.degrees(phase), 360)
        y = 20*np.log10(mag)

        # Generate the plot
        if labels != ['']:
            ax.plot(x, y, label=labels[index], linestyle=linestyle)
        else:
            ax.plot(x, y, linestyle=linestyle)

    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Nichols Plot')

    # Mark the -180 point
    ax.plot([-180], [0], 'r+', label='_nolegend_')

    # Add grid
    if grid:
        ml.nichols_grid()

    # Add legend
    if labels != ['']:
        plt.legend()

    if data:
        return x, y

# %% Function to generate a second order transfer function based on its typical
# characteristics.


def generateTfFromCharac(G, wn, zeta):
    """
    Generate a second order transfer function based on its typical
    characteristics.

    Parameters
    ----------
    G: float
        Gain of the transfer function.

    wn: float
        Frequency of the transfer function.

    zeta: float
        Damping coefficient of the transfer function.

    Returns
    -------
    ft: TransferFunction
        The linear system with those charcteristics
    """

    ft = ml.tf([G], [1/wn**2, 2*zeta/wn, 1])
    return ft

# %% Function to add noise to a given signal


def addNoise(t, signal, variance=0.05, rndNb=None):
    """
    Add noise to a given signal.

    Parameters
    ----------
    t: 1D array
        Time vector.
    signal: 1D array
        Signal at which to add noise.
    variance: float, optional
        Variance of random numbers. The default is 0.05.
    rndNb: int, optional
        Seed for RandomState. The default is None.

    Returns
    -------
    signal_noisy: 1D array
        Noisy signal.
    """

    if rndNb is not None:
        np.random.seed(rndNb)  # To master the random numbers
    noise = np.random.normal(0, variance, len(signal))
    signal_noisy = signal + noise

    plt.figure()
    plt.plot(t, signal, label="Original")
    plt.plot(t, signal_noisy, label="With noise")

    return signal_noisy

# %% Save data into a csv file


def saveFT(t, y, x=None, name="data"):
    """
    Save the data of the transfert function into a csv file.

    Parameters
    ----------
    t: 1D array
        Time vector.

    y: 1D array
        Response of the system.

    x: 1D array, optional
        Input of the system (default = [0, 1, ..., 1])

    name: String
        Name of the csv file (default = "data").

    Returns
    -------
    None
    """

    if x is None:
        x = np.ones(len(t))
        x[0] = 0
    np.savetxt(name + ".csv",
               np.transpose([t, x, y]),
               delimiter=",",
               fmt='%s',
               header='Temps(s),Consigne,Mesure',
               comments='')

# %% Load data from a csv file


def loadFT(file="data.csv"):
    """
    Load the data of the transfert function from a given csv file.

    Parameters
    ----------
    file: String
        Name of the csv file (default = "data.csv").

    Returns
    -------
    None
    """

    # Reading of the data headers with a comma as delimiter
    head = np.loadtxt(file, delimiter=',', max_rows=1, dtype=np.str)
    # Reading of the data
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.str)

    # Printing of the headers
    print(head)

    # Data selections based on header and convert to float
    # The sign - adapts the input data to be positive
    t = np.asarray(data[:, 0], dtype=np.float, order='C').flatten()
    x = np.asarray(data[:, 1], dtype=np.float, order='C').flatten()
    y = np.asarray(data[:, 2], dtype=np.float, order='C').flatten()

    return [t, x, y]

# %% Function to get the class of a given system.


def getClass(sys):
    """
    Get the class of the given system.

    Parameters
    ----------
    sys: LTI
        System analysed.

    Returns
    -------
    sysClass: int
        Class of the given system.
    """

    __, den = ml.tfdata(sys)
    den = den[0][0]  # To have the array as it's a list with one array
    # Reverse the direction of loop because the smallest power is at the last
    # index
    for sysClass, item in enumerate(reversed(den)):
        if item != 0:
            return sysClass

# %% Function to get the order of a given system.


def getOrder(sys):
    """
    Get the order of the given system.

    Parameters
    ----------
    sys: LTI
        System analysed.

    Returns
    -------
    sysClass: int
        Order of the given system.
    """

    __, den = ml.tfdata(sys)
    den = den[0][0]  # To have the array as it's a list with one array
    sysOrder = len(den)-1
    return sysOrder

# %% PID Tuner to see the modifications of the PID parameters on a given system.


def pidTuner(H, Kp=1, Ki=0, Kd=0):
    """
    PID Tuner to see the modifications of the PID parameters on a given system.

    Parameters
    ----------
    H: LTI
        Transfert function of the system (open loop) to regulate.

    Kp: float, optionnal
        Proportionnal parameter of the PID controller (default = 1).

    Ki: float, optionnal
        Integral parameter of the PID controller (default = 0).

        Reminder: Ki = Kp/tI

    Kd: float, optionnal
        Derivative parameter of the PID controller (default = 0).

        Reminder: Kd = Kp*tD

    Returns
    -------
    None
    """

    from matplotlib.widgets import Slider, Button, RadioButtons

    # Create the figure
    fig = plt.figure("PID Tuner")
    axGraph = fig.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Frames to contain the sliders
    axcolor = 'lightgoldenrodyellow'
    axKp = plt.axes([0.125, 0.2, 0.775, 0.03], facecolor=axcolor)
    axKi = plt.axes([0.125, 0.15, 0.775, 0.03], facecolor=axcolor)
    axKd = plt.axes([0.125, 0.1, 0.775, 0.03], facecolor=axcolor)

    # Slider
    sKp = Slider(axKp, 'Kp', Kp/20, Kp*20, valinit=Kp)

    if Ki == 0:
        sKi = Slider(axKi, 'Ki', 0, 100, valinit=Ki)
    else:
        sKi = Slider(axKi, 'Ki', Ki/20, Ki*20, valinit=Ki)

    if Kd == 0:
        sKd = Slider(axKd, 'Kd', 0, 100, valinit=Kd)
    else:
        sKd = Slider(axKd, 'Kd', Kd/20, Kd*20, valinit=Kd)

    def update(val):
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        c = KpNew*ml.tf(1, 1) + KiNew*ml.tf(1, [1, 0]) + KdNew*ml.tf([1, 0], 1)
        Gbo = c*H
        if radio.value_selected == 'Step':
            axGraph.clear()
            plotStep(axGraph, Gbo)
        elif radio.value_selected == 'Nichols':
            axGraph.clear()
            plotNichols(axGraph, Gbo)

        fig.canvas.draw_idle()  # Refresh the plots
    sKp.on_changed(update)
    sKi.on_changed(update)
    sKd.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        sKp.reset()
        sKi.reset()
        sKd.reset()
    reset_button.on_clicked(reset)

    def plotNichols(ax, Gbo):
        nichols([Gbo_init, Gbo], NameOfFigure="PID Tuner", ax=ax)
        # Print infos inside the plot
        textInfo = getNicholsTextInfos(Gbo)
        at = AnchoredText(textInfo,
                          prop=dict(size=10), frameon=True,
                          loc='lower right',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axGraph.add_artist(at)

    def plotStep(ax, Gbo):
        Gbf = ml.feedback(Gbo)
        [Y_New, __] = ml.step(Gbf, T)
        ax.plot(T, np.linspace(1, 1, len(T)), linestyle=':',
                lw=1, color='grey')  # 1 line
        ax.plot(T, Y, label="Initial system", lw=1)  # Original
        ax.plot(T, Y_New, label="Modified system", lw=1)
        #l.set_ydata(Y_New)
        ax.set_title("Step Response")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (seconds)")

    # Print button
    printax = plt.axes([0.6, 0.025, 0.1, 0.04])
    print_button = Button(printax, 'Infos', color=axcolor, hovercolor='0.975')

    # Function to create a string with the usefull info fo nichols plot
    def getNicholsTextInfos(Gbo):
        # Extract the gain margin (Gm) and the phase margin (Pm)
        gm, pm, __, __ = ml.margin(Gbo)
        gm = 20*np.log10(gm)  # Conversion of gm in dB
        return """Phase Margin = {PM}°
Gain Margin = {GM} dB""".format(PM=pm, GM=gm)

    def printInfos(event):
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        print("")  # To let space before the informations
        print("Kp =", KpNew)
        print("Ki =", KiNew)
        print("Kd =", KdNew)
        c = KpNew*ml.tf(1, 1) + KiNew*ml.tf(1, [1, 0]) + KdNew*ml.tf([1, 0], 1)
        print("Corr =", c)
        Gbo = c*H
        print("Gbo =", Gbo)
        if radio.value_selected == 'Step':
            Gbf = ml.feedback(Gbo)
            [Y_New, __] = ml.step(Gbf, T)

            # To change the current axes to be the graphs's one
            plt.sca(axGraph)
            stepInfo = step_info(T, Y_New)
            # Printing of the step infos
            printInfo(stepInfo)

        elif radio.value_selected == 'Nichols':
            # Extract the gain margin (Gm) and the phase margin (Pm)
            gm, pm, __, __ = ml.margin(Gbo)
            print("Phase Margin =", pm, "°")
            gm = 20*np.log10(gm)  # Conversion of gm in dB
            print("Gain Margin =", gm, "dB")
            # Plotting
            if pm != math.inf:
                axGraph.plot([-180, -180+pm], [0, 0], 'k-', linewidth=1)
                axGraph.plot(-180+pm, 0, 'ko')
            if gm != math.inf:
                axGraph.plot([-180, -180], [-gm, 0], 'k-', linewidth=1)
                axGraph.plot(-180, -gm, 'ko')

    print_button.on_clicked(printInfos)

    # Radio button
    rax = plt.axes([0.905, 0.5, 0.09, 0.1], facecolor=axcolor)
    radio = RadioButtons(rax, ('Step', 'Nichols'), active=0)

    def changeGraph(label):
        # Get the new parameters values
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        c = KpNew*ml.tf(1, 1) + KiNew*ml.tf(1, [1, 0]) + KdNew*ml.tf([1, 0], 1)
        Gbo = c*H
        # Deleting of the graphs
        axGraph.clear()
        # Base for the original graph
        if label == 'Step':
            plotStep(axGraph, Gbo)
        elif label == 'Nichols':
            plotNichols(axGraph, Gbo)

        fig.canvas.draw_idle()  # To refresh the plots

    radio.on_clicked(changeGraph)

    # Declaration of the transfer function of the system in BO and BF with the
    # given control parameters
    c = Kp*ml.tf(1, 1) + Ki*ml.tf(1, [1, 0]) + Kd*ml.tf([1, 0], 1)
    Gbo_init = c*H
    Gbf_init = ml.feedback(Gbo_init)
    [Y, T] = ml.step(Gbf_init)

    # Plot the step
    plotStep(axGraph, Gbo_init)

    plt.show()

    # It's needed to return those variables to keep the widgets references or
    # they don't work.
    return sKp, print_button, reset_button, radio


# %% Object to store all the characteristics determined.
class TfCharact():

    """
    Object to store all the characteristics determined.

    Attributes
    ----------

    G: float
        Time it takes for the response to rise from 10% to 90% of the
        steady-state response.

    wn: float
        Time it takes for the error e(t) = \|y(t) – yfinal\| between the
        response y(t) and the steady-state response yfinal to fall below 5% of
        the peak value of e(t).

    zeta: float
        Minimum value of y(t) once the response has risen.
    """

    G = None
    wn = None
    zeta = None


# %% Tool to identify a system
def fineIdentification(file, G, wn, zeta, tfCharach):
    """
    Tool to identify a system from its measured step data output.

    Parameters
    ----------
    file: String
        Name of the csv file (default = "data.csv").

    G: float
        Gain of the transfer function.

    wn: float
        Frequency of the transfer function.

    zeta: float
        Damping coefficient of the transfer function.

    tfCharach: TfCharact
        Object to store the characteristics determined.

    Returns
    -------
    None
    """

    from scipy.signal import lti, lsim
    from matplotlib.widgets import Slider, Button

    # Load the data from the given csv file
    t, u, y = loadFT(file)

    # Déclaration de la fonction de transfert
    ft = lti([G], [1/wn**2, 2*zeta/wn, 1])
    t_ft, y_ft, _ = lsim(ft, U=u, T=t)  # Réponse à la consigne

    fig, __ = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    l, = plt.plot(t, y, label="Mesures")
    l, = plt.plot(t_ft, y_ft, lw=1, color='red')

    axcolor = 'lightgoldenrodyellow'
    axamp = plt.axes([0.125, 0.2, 0.775, 0.03], facecolor=axcolor)
    axfreq = plt.axes([0.125, 0.15, 0.775, 0.03], facecolor=axcolor)
    axdamp = plt.axes([0.125, 0.1, 0.775, 0.03], facecolor=axcolor)

    samp = Slider(axamp, 'G', G/10, G*10.0, valinit=G)
    sfreq = Slider(axfreq, 'Wn', 0, wn*2, valinit=wn)
    sdamp = Slider(axdamp, 'Dzeta', zeta/10, zeta*10.0, valinit=zeta)

    def update(val):
        amp = samp.val
        omega2 = sfreq.val
        m2 = sdamp.val
        ft = lti([amp], [1/omega2**2, 2*m2/omega2, 1])
        __, y_ft, _ = lsim(ft, U=u, T=t_ft)  # Réponse à la consigne
        l.set_ydata(y_ft)
        fig.canvas.draw_idle()
    samp.on_changed(update)
    sfreq.on_changed(update)
    sdamp.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        samp.reset()
        sfreq.reset()
        sdamp.reset()
    reset_button.on_clicked(reset)

    # Print button
    saveax = plt.axes([0.6, 0.025, 0.1, 0.04])
    save_button = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')

    def save(event):
        tfCharach.G = samp.val
        tfCharach.wn = sfreq.val
        tfCharach.zeta = sdamp.val
        print("\nCharacterictics saved!")  # To let space before the informations
        print("G =", tfCharach.G)
        print("wn =", tfCharach.wn)
        print("zeta =", tfCharach.zeta)

    save_button.on_clicked(save)

    plt.show()

    # It's needed to return those variables to keep the widgets references or
    # they don't work.
    return samp, save_button, reset_button
