'''
Irrfahrten und verwandte Zufälle - Programm zur Darstellung von Wegen.

April 2023, Paul Kappmeyer
'''
import numpy as np
from scipy.stats import binom
import matplotlib as mpl
import matplotlib.animation
import scipy.special

'''' ------------------------------- Generierung von Wegen: '''

def calc_y(a, s0):
    ''' Berechnet die y-Koordinaten anhand der Schrittrichtungen.
        Parameters:
            a (array-like): Schrittrichtungen
            s0 (int): Starthöhe der Irrfahrt 
        
        Returns:
            y (array-like): Höhe der Irrfahrt zum jeweiligen Zeipunkt'''
    # Länge des Weges
    n = len(a)
    # leeres Array der y-Koordinaten anlegen
    y = np.empty(n+1, dtype=int)
    # Starthöhe festsetzen
    y[0] = s0
    # folgenden Höhen setzen
    for i in range(1, n+1):
        y[i] = y[i-1] + a[i-1]
        
    return y


def generate_random_walk(n, s0, a, p=None):
    ''' Generiert einen Weg, wobei Länge, Starthöhe, die mögliche Schrittrichtungen
    und deren Wahrscheinlichkeiten als Parameter übergeben werden können. Standard-
    mäßig wird eine einfache symmetrische Irrfahrt generiert. Gibt den Weg als
    Tupel bestehend aus (x, y, a, n) zurück. 
        Parameters:
            n (int): Länge der Irrfahrt
            s0 (int): Starthöhe der Irrfahrt
            a (array-like): mögliche Schrittrichtungen (bspw. [-1, 1] oder [-1, 0, 1])
            p (array-like): Wahrscheinlichkeiten der möglichen Schrittrichtungen, default: None=Gleichverteilung
            
        Returns:
            x (array): Zeitpunkte, d.h. [0, 1, ..., n] 
            y (array): Höhe der Irrfahrt zum jeweiligen Zeitpunkt
            a (array): Schrittrichtung zum jeweiligen Zeitpunkt'''
    if n < 0:
        print("Die Länge des Weges muss nichtnegativ sein.")
        return ([0], [0], [0])
    # Zufällige Schrittrichtungen auswählen
    a_sim = np.random.choice(a, n, p=p)
    
    # Koordinatenpunkte der Irrfahrt generieren
    x = np.arange(0, n+1)
    y = calc_y(a_sim, s0)
    
    return (x, y, a_sim)


def generate_symmetric_walk(n, s0=0):
    ''' Generiert eine einfache symmetrische Irrfahrt der Länge n. 
        Parameters:
            n (int): Länge der Irrfahrt
            s0 (int): Starthöhe der Irrfahrt, default: 0
        
        Returns:
                x (array): Zeitpunkte, d.h. [0, 1, ..., n] 
                y (array): Höhe der Irrfahrt zum jeweiligen Zeitpunkt
                a (array): Schrittrichtung zum jeweiligen Zeitpunkt'''
    return generate_random_walk(n=n, s0=s0, a=[-1, 1], p=[0.5, 0.5])


def generate_asymmetric_walk(n, p_up, s0=0):
    ''' Generiert eine einfache asymmetrische Irrfahrt der Länge n.         
        Parameters:
                n (int): Länge der Irrfahrt
                p_up (float): Wahrscheinlichkeit für einen Aufwärtsschritt
                s0 (int): Starthöhe der Irrfahrt, default: 0
            
            Returns:
                    x (array): Zeitpunkte, d.h. [0, 1, ..., n] 
                    y (array): Höhe der Irrfahrt zum jeweiligen Zeitpunkt
                    a (array): Schrittrichtung zum jeweiligen Zeitpunkt'''
    return generate_random_walk(n=n, s0=s0, a=[-1, 1], p=[1 - p_up, p_up])


def generate_bridge(n, s0=0):
    ''' Generiert einen Brückenweg der Länge n. 
            Parameters:
                    n (int): Länge des Brückenweges
                    s0 (int): Starthöhe der Irrfahrt, default: 0
                
            Returns:
                    x (array): Zeitpunkte, d.h. [0, 1, ..., n] 
                    y (array): Höhe der Irrfahrt zum jeweiligen Zeitpunkt
                    a (array): Schrittrichtung zum jeweiligen Zeitpunkt'''
    # Zufällige Schrittrichtungen auswählen
    half_n = int(n/2)
    steps = np.concatenate((np.ones(half_n, dtype=int), -np.ones(half_n, dtype=int)))
    a = np.random.permutation(steps)

    # Koordinatenpunkte der Irrfahrt generieren
    x = np.arange(0, n+1)
    y = calc_y(a, s0)
    
    return (x, y, a)


def generate_walk_binom_steps(n, s0, num_steps):
    ''' Generiert einen Weg mit num_steps verschiedenen Schrittmöglichkeiten, wobei die Wahrscheinlichkeiten der 
        Schrittmöglichkeiten binomialverteilt sind, d.h.
    num_steps = 1:
        a = [0], p = [1]
    num_steps = 2:
        a = [-1, 1], p = [1/2, 1/2]
    num_steps = 3:
        a = [-1, 0 , 1], p = [1/4, 2/4, 1/4]
    etc.
    '''
    if (num_steps % 2) == 1:
        a_half = np.arange(1, (num_steps + 1) / 2, 1)
        a = np.concatenate( (np.flip(-a_half), [0], a_half) )
    else:
        a_half = np.arange(1, num_steps/2 + 1, 1)
        a = np.concatenate( (np.flip(-a_half), a_half) )
    
    p = binom.pmf(range(num_steps), num_steps-1, 1/2)
    
    return generate_random_walk(n, s0, a, p)


def bridge_to_nonnegative(bruecke):
    ''' Transformiert einen Brückenweg in einen nichtnegativen Weg (s.h. Hauptlemma). '''
    # Daten des Weges entpacken
    x, y, a = bruecke
        # Länge des Weges
    n = len(x)
    
    # Minimum des Weges
    m = min(y)
    # Ist der Weg bereits nichtnegativ?
    if m == 0:
        return (x, y, a)
    
    # Zeitpunkt des erstmaligen Erreichens des Minimums
    k0 = first_stay_at(bruecke, m) 

    # Neue Schrittrichtungen setzen
    a_new = np.concatenate( (a[k0:n], -np.flip(a[0:k0])) ) 
    
    x_new = x + k0
    y_new = calc_y(a_new, 0)    

    return x_new, y_new, a_new


'''' ------------------------------- Eigenschaften von Wegen: '''

def first_stay_at(walk, state):
    ''' Gibt den Zeitpunkt an, zu welchem die Irrfahrt erstmalig den Zustand
    state angenommen hat. '''
    x, y, a = walk
    try:
        return int(min(np.argwhere(y == state)))
    except ValueError:
        return np.NaN


def last_stay_at(walk, state):
    ''' Gibt den Zeitpunkt an, zu welchem die Irrfahrt letztmalig den Zustand
    state angenommen hat. '''
    x, y, a = walk
    try:
        return int(max(np.argwhere(y == state)))
    except ValueError:
        return np.NaN


def is_nonpositive(walk):
    ''' Prüft, ob ein Weg nichtpositv ist. '''
    x, y, a = walk
    return not np.any(y == 1)

def is_nonnegative(walk):
    ''' Prüft, ob ein Weg nichtnegativ ist. '''
    x, y, a = walk
    return not np.any(y == -1)


def is_positive(walk):
    ''' Prüft, ob ein Weg positiv ist. '''
    x, y, a = walk
    return (is_nonnegative(walk) and not np.any(y[1:] == 0))


def is_negative(walk):
    ''' Prüft, ob ein Weg negativ ist. '''
    x, y, a = walk
    return (is_nonpositive(walk) and not np.any(y[1:] == 0))


def is_bridge(walk):
    ''' Prüft, ob ein Weg ein Brückenweg ist. '''
    x, y, a = walk
    return y[-1] == 0


def count_NST(walk, count_trivial=False):
    ''' Zählt die Nullstellen eines Weges. 
    Parameters:
        walk: Weg als Tupel = (x, y, a) 
        count_trivial (boolean): ob, der Start in (0, 0) mitgezählt werden soll, default: False
        
    Returns:
        num_NST (int): Anzahl der Nulsstellen
    '''
    # Wegdaten entpacken
    x, y, a = walk
    # Nullstellen des Weges ohne den Startpunkt zählen
    num_NST = np.count_nonzero(y[1:] == 0)
    if count_trivial and y[0] == 0:
        num_NST += 1
    return num_NST


def last_NST(walk):
    ''' Gibt den Zeitpunkt der letzten NST eines Weges zurück. '''
    return last_stay_at(walk, 0)



'''' ------------------------------- Verteilung der Zufallsvariablen: '''

def distribution_last_NST(n):
    ''' Gibt die Verteilung der Zufallsvariable L_{2n} (Zeipunkt der letzten 
    Nullstelle einer Irrfahrt der Länge 2n) zurück. '''
    k = np.arange(n + 1, dtype=float)
    p = scipy.special.binom(2*k, k) * scipy.special.binom(2*(n - k), n-k) / 2**(2*n)
    return np.array([item for items in zip(p, [0] * 2 * n) for item in items])[:-1]


def expected_value_last_NST(n):
    ''' Gibt den Erwartungswert der Zufallsvariable L_{2n} (Zeitpunkt der letzten
    Nullstelle einer Irrfahrt der Länge 2n) zurück '''
    return n


def variance_last_NST(n):
    ''' Gibt die Varianz der Zufallsvariable L_{2n} (Zeitpunkt der letzten 
    Nullstelle einer Irrfahrt der Länge 2n) zurück. '''
    return scipy.special.binom(n + 1, 2)


def distribution_num_NST(n):
    '''' Gibt die Verteilung der Zufallsvariable N_{2n} (Anzahl der Nullstellen 
    einer Irrfahrt der Länge 2n) zurück. '''
    j = np.arange(n + 1, dtype=float)
    return scipy.special.binom(2*n - j, n) / (2**(2*n - j))


def distribution_maximum(n):
    ''' Gibt die Verteilung der Zufallsvariable M_n (Maximum einer Irrfahrt der
    Länge n) zurück. '''
    k = np.arange(0, n+1, dtype=float)
    return scipy.special.binom(n, np.floor((n + k + 1) / 2)) / 2**n


'''' ------------------------------- Plotting der Wege: '''

def plot_distribution(ax, distribution):
    ''' Stellt eine Verteilung als Säulendiagramm dar. '''
    n = len(distribution)
    x = np.arange(n)
    ax.bar(x, distribution)


def plot_walk(ax, walk, scaling='auto'):
    ''' Stellt einen Weg in einem Koordinatensystem dar. '''
    # Daten des Weges entpacken
    x, y, _ = walk
    # Länge des Weges
    n = len(walk[0])
    
    # Ploteinrichtung
    init_ax(ax)
    if scaling == 'standard':
        fit_ax(ax, n)
    
    # Plotte die Irrfahrt
    markersize = 10/np.sqrt(n)
    ax.plot(x, y, color="black", marker='o', markersize=markersize, linestyle='-')

def plot_walks(ax, walks, colors=None):
    ''' Stellt mehrere Wege in einem Koordinatensystem dar. '''
    # Ploteinrichtung
    init_ax(ax)
    
    if colors == None:
        colors = len(walks) * [None]
    
    for walk, color in zip(walks, colors):
        # Simulationsdaten entpacken
        x, y, _ = walk
        
        # Plotte die Irrfahrt
        ax.plot(x, y, color=color, marker='o', markersize=2, linestyle='--', alpha=0.4)


def animate_walk(fig, ax, walk, scaling='auto', adjust_window=False):
    ''' Stellt einen Weg animiert in einem Koordinatensystem dar, wobei sich der 
    Weg pro Animationsframe einen Zeitschritt 'weiterbewegt'. '''
    # Daten des Weges entpacken
    x, y, _ = walk
    # Länge des Weges
    n = len(walk[0])
    
    # Ploteinrichtung
    init_ax(ax)
    if scaling == 'standard':
        fit_ax(ax, n)
    if adjust_window:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
    
    # Erzeuge einen Plot
    markersize = 10/np.sqrt(n)
    plot, = ax.plot(x, y, color="black", marker='o', markersize=markersize, linestyle='-')

    def update(frame):
        # Achsen und Skalierungen anpassen
        if adjust_window and frame != 0:
            fit_ax(ax, frame)
        
        # Graph anpassen
        plot.set_data(x[0:frame+1], y[0:frame+1])
        return plot,

    # Erzeuge das Animationsobjekt.
    bilt_on = True
    if adjust_window:
        bilt_on = False
    
    ani = mpl.animation.FuncAnimation(fig, update, interval=50, blit=bilt_on, frames=n+1, repeat_delay=2000)
    return ani


def init_ax(ax):
    ''' Passt Achsen und Beschriftungen eines Axes-Objekts an. '''
    # Die obere und rechte Achse unsichtbar machen
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Die linke Diagrammachse auf den Bezugspunkt '0' der x-Achse legen
    ax.spines['left'].set_position(('data', 0))
    # Die untere Diagrammachse auf den Bezugspunkt '0' der y-Achse legen:
    ax.spines['bottom'].set_position(('data',0))
    
    # Titel und Achsenbeschriftung
    ax.set_title('Irrfahrten und verwandte Zufälle', color='gray')
    ax.set_xlabel('Zeitschritte', loc='right', color='gray')
    ax.set_ylabel('Höhe', color='gray')
    
    # Gitternetz einfügen
    ax.grid(True)


def fit_ax(ax, n):
    ''' Passt die Achsen eines Axes-Objekts an eine Irrfahrt der Länge n an. '''
    # Ticks der x-Achse festlegen
    step = int(n/10)

    if step == 0:
        step += 1
    xticks = np.arange(0, 2*n, step, dtype=int)
    ax.set_xticks(xticks)
    
    # Grenzen der x-Achse festlegen
    ax.set_xlim(0, n)
    
    # Ticks der y-Achse festlegen
    step = int(np.sqrt(n)/2)
    if step == 0:
        step += 1
    one_half = np.arange(0, n, step, dtype=int)
    yticks = np.concatenate((np.flip(-one_half), one_half[1:]))
    ax.set_yticks(yticks)
    
    # Grenzen der y-Achse festlegen
    ylim = 2.2425*np.sqrt(n)
    if ylim == 0:
        ylim += 1
    ax.set_ylim(-ylim, ylim)
