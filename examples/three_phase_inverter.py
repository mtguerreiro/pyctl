import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
plt.ion()

# --- Model ---
Li = 3.4e-3
Lg = 1.8e-3
C = 20e-6

wg = 2 * np.pi * 50

# Init conditions
x0 = 0
u0 = 0

# Grid voltage
ug = [162.5, 0.0]

# Grid-side current references
r1 = [10, 0]
r2 = [-5, -5]

# Discretization
fs = 7.5e3
dt = 1.0 / fs

t_sim = 10e-3

# Optimization parameters
rw = [2e-5, 2e-5]
l_pred = 4
l_ctl = 4
l_cnt = 4

# Constraints
k = 0.875
V_dc = 650
u_lim = V_dc / np.sqrt(3)
ud_lim = k * u_lim
uq_lim = np.sqrt(1 - k ** 2) * u_lim
u_lim = [[-ud_lim, -uq_lim], [ud_lim, uq_lim]]

ii_lim = 15
x_lim = [[None, None, -ii_lim, -ii_lim, None, None], [None, None, ii_lim, ii_lim, None, None]]

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
n = int(t_sim / dt)

# --- System ---
sys = ctl.mpc.System(
    Ad, Bu, Cd,
    l_pred=l_pred, l_ctl=l_ctl, l_u_cnt=l_cnt, l_x_cnt=l_cnt,
    rw=rw,
    x_lim=x_lim, u_lim=u_lim
)

# --- Sim with receding horizon ---
# Reference - grid-side currents
r = [10, 0]
r = np.zeros((n, 2))
r[:int(n/2), :] = r1
r[int(n/2):, :] = r2

data = sys.sim(x0, u0, r, n, Bd=Bv, ud=ug, solver='hild')

# --- Plots ---
ud_lim_l = -ud_lim * np.ones(n)
ud_lim_u =  ud_lim * np.ones(n)

uq_lim_l = -uq_lim * np.ones(n)
uq_lim_u =  uq_lim * np.ones(n)

ii_lim_l = -ii_lim * np.ones(n)
ii_lim_u =  ii_lim * np.ones(n)

t = dt * np.arange(n)

plt.figure(figsize=(8,8))

# Control signals
ax = plt.subplot(4,1,1)
plt.step(t / 1e-3, data['u'], where='post')

plt.plot(t / 1e-3, ud_lim_l, 'C0--')
plt.plot(t / 1e-3, ud_lim_u, 'C0--')
plt.plot(t / 1e-3, uq_lim_l, 'C1--')
plt.plot(t / 1e-3, uq_lim_u, 'C1--')

plt.ylabel('Voltage (V)')
plt.title('Control signals', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()
plt.xlim([t[0]/1e-3, t[-1]/1e-3])

# Inverter-side currents
plt.subplot(4,1,2, sharex=ax)

plt.plot(t / 1e-3, ii_lim_l, 'k--')
plt.plot(t / 1e-3, ii_lim_u, 'k--')

plt.step(t / 1e-3, data['xm'][:,[2, 3]], where='post')

plt.ylabel('Current (A)')
plt.title('Inverter-side current', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()

# Filter cap. voltages
plt.subplot(4,1,3, sharex=ax)
plt.step(t / 1e-3, data['xm'][:,[4, 5]], where='post')
plt.ylabel('Voltage (V)')
plt.title('Filter cap. voltage', fontsize=11)
plt.gca().tick_params(labelbottom=False)
plt.grid()

# grid-side currents
plt.subplot(4,1,4, sharex=ax)
plt.step(t / 1e-3, data['y'], where='post')
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Grid-side current', fontsize=11)
plt.grid()

plt.tight_layout()
