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
r = [10, 0]

# Init conditions
x_i = [0, 0, 0, 0, 0, 0, 0, 0]
u_i = [0, 0, 162.5, 0]

u_i_unc = [0, 0]

# Discretization
fs = 5e3
dt = 1 / fs

# Optimization parameters
r_w = [0.0005, 0.0005, 100.0, 100.0]
n_p = 10
n_c = 10
n_r = 10

# Constraints
u_lim = [[-250, -250, 162.5, 0], [250, 250, 162.5, 0]]
x_lim = [[-200, -200, -15, -15, -10000, -10000], [200, 200, 15, 15, 10000, 10000]]
#I_max = 10

# --- System ---
Am = np.array([[0,      wg,     0,      0,      -1/Lg,  0],
               [-wg,    0,      0,      0,      0,      -1/Lg],
               [0,      0,      0,      wg,     1/Li,   0],
               [0,      0,      -wg,    0,      0,      1/Li],
               [1/C,    0,      -1/C,   0,      0,      wg],
               [0,      1/C,    0,      -1/C,   -wg,    0]])

Bm = np.array([[0,      0,      1/Lg,   0],
               [0,      0,      0,      1/Lg],
               [-1/Li,  0,      0,      0],
               [0,      -1/Li,  0,      0],
               [0,      0,      0,      0],
               [0,      0,      0,      0]])

Cm = np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0]])

Ad, Bd, Cd, _, _ = scipy.signal.cont2discrete((Am, Bm, Cm, 0), dt, method='zoh')

# Sim points
n = 25

# --- System ---
sys = ctl.mpc.ConstrainedSystem(Ad, Bd, Cd, n_p=n_p, n_c=n_c, n_r=n_r, r_w=r_w, x_lim=x_lim, u_lim=u_lim)
        
# --- Sim with receding horizon ---
data = sys.dmpc(x_i, u_i, r, n)


# --- Plots ---
t = dt * np.arange(n)
#plt.ion()
ax = plt.subplot(3,1,1)
plt.step(t / 1e-3, data['u'], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Control')
plt.grid()

plt.subplot(3,1,2, sharex=ax)
plt.step(t / 1e-3, data['y'], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Current')
plt.grid()

plt.subplot(3,1,3, sharex=ax)
plt.step(t / 1e-3, data['x_m'][:,[4, 5]], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Current')
plt.grid()

plt.tight_layout()