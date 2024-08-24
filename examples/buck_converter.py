import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

#plt.ion()

# --- Model ---
# Model parameters
R = 2.2
L = 47e-6
C = 560e-6

Rds = 15e-3
Rl = 15e-3
Rc = 60e-3

V_in = 16

# Discretization
fs = 100e3
dt = 1 / fs

# Optimization parameters
rw = 0.01
n_pred = 5
n_ctl = 5
n_cnt = 1

# Constraints
u_lim = [[0], [V_in]]
x_lim = [[-10, None], [10, None]] # or None

# Sim points
n = 100

# Reference
r = 8

# Initial conditions
xi = 0
ui = 0

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
sys = ctl.mpc.System(Ad, Bd, Cd, n_pred=n_pred, n_ctl=n_ctl, n_cnt=n_cnt, rw=rw, x_lim=x_lim, u_lim=u_lim)

# --- Sim with receding horizon ---
data = sys.sim(xi, ui, r, n)

# --- Plots ---
t = dt * np.arange(n)

ax = plt.subplot(3,1,1)
plt.title('Duty-cycle', fontsize=11)
plt.step(t / 1e-3, data['u'], where='post')
plt.ylabel('Control')
plt.gca().tick_params(labelbottom=False)
plt.grid()
plt.xlim([t[0]/1e-3, t[-1]/1e-3])

plt.subplot(3,1,2, sharex=ax)
plt.title('Inductor current', fontsize=11)
plt.step(t / 1e-3, data['xm'][:, 0], where='post')
plt.ylabel('Current (A)')
plt.gca().tick_params(labelbottom=False)
plt.grid()

plt.subplot(3,1,3, sharex=ax)
plt.title('Output voltage', fontsize=11)
plt.step(t / 1e-3, data['y'], where='post')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (ms)')
plt.grid()

plt.tight_layout()

plt.show()
