import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# --- Model ---
Li = 3.4e-3
Lg = 1.8e-3
C = 20e-6

wg = 2 * np.pi * 50

# Init conditions
xi = 0
ui = 0

# Grid voltage
ug = [162.5, 0.0]

# Grid-side current references
r1 = [10, 0]
r2 = [-5, -5]

# Discretization
fs = 5e3
dt = 1.0 / fs

# Optimization parameters
rw = [0.00025, 0.00025]
n_pred = 4
n_ctl = 4
n_cnt = 4

# Constraints
V_dc = 650
V_max = V_dc / np.sqrt(3)
u_lim = [[-V_max, -V_max], [V_max, V_max]]
x_lim = [[None, None, -15, -15, None, None], [None, None, 15, 15, None, None]]

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
Bu = Bd[:, :2]
Bv = Bd[:, 2:]

# Sim points
n = 50

# --- System ---
sys = ctl.mpc.System(Ad, Bu, Cd, n_pred=n_pred, n_ctl=n_ctl, n_cnt=n_cnt, rw=rw, x_lim=x_lim, u_lim=u_lim)

# --- Sim with receding horizon ---
# Reference - grid-side currents
r = [10, 0]
r = np.zeros((n, 2))
r[:int(n/2), :] = r1
r[int(n/2):, :] = r2

data = sys.sim(xi, ui, r, n, Bd=Bv, ud=ug)

# --- Plots ---
t = dt * np.arange(n)

plt.figure(figsize=(8,8))

ax = plt.subplot(4,1,1)
plt.step(t / 1e-3, data['u'], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Control signals', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()
plt.xlim([t[0]/1e-3, t[-1]/1e-3])

plt.subplot(4,1,2, sharex=ax)
plt.step(t / 1e-3, data['xm'][:,[2, 3]], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Inverter-side current', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()

plt.subplot(4,1,3, sharex=ax)
plt.step(t / 1e-3, data['xm'][:,[4, 5]], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Filter cap. voltage', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()

plt.subplot(4,1,4, sharex=ax)
plt.step(t / 1e-3, data['y'], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Grid-side current', fontsize=11)
plt.grid()

plt.tight_layout()

plt.show()
