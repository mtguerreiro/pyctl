import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
Li = 3.4e-3
Lg = 1.8e-3
C = 20e-6

wg = 2*np.pi*50

# Reference
r = np.array([5, 0])

# Init conditions
x_i = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
u_i = np.array([0, 0])

# Discretization
fs = 5e3
dt = 1 / fs

# Optimization parameters
r_w = np.array([1e-6, 1e-6])
n_p = 40
n_c = 40

# --- System ---
Am = np.array([[0,      wg,     0,      0,      -1/Lg,  0],
               [-wg,    0,      0,      0,      0,      -1/Lg],
               [0,      0,      0,      wg,     1/Li,   0],
               [0,      0,      -wg,    0,      0,      1/Li],
               [1/C,    0,      -1/C,   0,      0,      wg],
               [0,      1/C,    0,      -1/C,   -wg,    0]])

Bm = np.array([[0,      0],
               [0,      0],
               [-1/Li,  0],
               [0,      -1/Li],
               [0,      0],
               [0,      0]])

Cm = np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0]])

Ad, Bd, Cd, _, _ = scipy.signal.cont2discrete((Am, Bm, Cm, 0), dt, method='zoh')

# Sim points
n = 50

# --- System ---
sys = ctl.mpc.System(Ad, Bd, Cd, n_p=n_p, n_c=n_c, r_w=r_w)

# --- Sim with receding horizon ---
data = sys.dmpc(x_i, u_i, r, n)

# --- Plots ---
#plt.ion()
plt.subplot(2,1,1)
plt.plot(data['u'], '--o')
plt.subplot(2,1,2)
plt.plot(data['y'], '--x')
