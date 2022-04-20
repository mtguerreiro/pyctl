import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
# Model parameters
R = 2.2
L = 47e-6
C = 560e-6

Rds = 15e-3
Rl = 15e-3
Rc = 60e-3

V_in = 12

# Discretization
fs = 100e3
dt = 1 / fs

# Optimization parameters
r_w = 0.0
n_p = 2
n_c = 2
n_r = 2

# Constraints
u_lim = [[0], [V_in]]

# Sim points
n = 200

# Reference
r = 5

# Initial conditions
x_i = [0, 0, 0]
u_i = 0

# --- System ---
# Model matrices
a11 = -(Rds + Rl) / L
a12 = -1 / L

a21 = (L - (Rds + Rl) * Rc * C) * R / ((R + Rc) * L * C)
a22 = -(R * Rc * C + L) / ((R + Rc) * L * C)

b11 = 1 / L
b21 = R * Rc / ((R + Rc) * L)
        
Am = np.array([[a11, a12],
               [a21, a22]])

Bm = np.array([[b11],
               [b21]])

Cm = np.array([[0, 1]])

Ad, Bd, Cd, _, _ = scipy.signal.cont2discrete((Am, Bm, Cm, 0), dt, method='zoh')

# --- System ---
sys = ctl.mpc.ConstrainedSystem(Ad, Bd, Cd, n_p=n_p, n_c=n_c, n_r=n_c, r_w=r_w, u_lim=u_lim)

# --- Sim with receding horizon ---
data = sys.dmpc(x_i, u_i, r, n)

# --- Plots ---
#plt.ion()
plt.plot(data['u'], '--o')
plt.plot(data['y'], '--x')
