import numpy as np
import pyctl as ctl


def aug(Am, Bm, Cm):
    r"""Determines the augmented model. For now, only one control signal and
    one output is supported.

    Parameters
    ----------
    Am : np.array
        An (n, n) numpy matrix.

    Bm : np.array
        An (n, 1) numpy matrix.

    Cm : np.array
        A (1, n) numpy matrix.

    Returns
    -------
    (A, B, C) : tuple
        A tuple containing the augmented matrices.
    
    """
    n = Am.shape[0]
    zeros_n = np.zeros((n, 1))

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Am
    A[-1, :n] = Cm @ Am
    A[-1, -1] = 1

    B = np.zeros((n + 1, 1))
    B[:n] = Bm
    B[-1] = Cm @ Bm

    C = np.zeros((1, n + 1))
    C[0, -1] = 1

    return (A, B, C)


def opt(A, B, C, x_i, r, r_w, n_p, n_c):
    r"""Provides the control vector miniming the expression

    .. :math:

        J = (R_s - Y)^T(R_s - Y) + \Delta U^T R \Delta U.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, 1) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (1, n) numpy matrix.

    x_i : :class:`np.array`
        Initial conditions of the states. An (n, 1) numpy matrix.

    r : :class:`int`, :class:`float`
        The set-point signal.

    r_w : :class:`int`, :class:`float`
        Weight of the control action.

    n_p : :class:`int`
        Length of prediction horizon.

    n_c : :class:`int`
        Length of the control window.

    Returns
    -------
    :class:`np.array`
        An (n_c, 1) numpy matrix containing the optimal control values.

    """
    x_i = x_i.reshape(-1, 1)
    R_s = r * np.ones((n_p, 1))
    R = r_w * np.eye(n_c)

    F = np.zeros((n_p, (C @ A).shape[1]))
    F[0, :] = C @ A
    for i in range(1, n_p):
        F[i, :] = F[i - 1, :] @ A
    
    Phi = np.zeros((n_p, n_c))
    Phi[0, 0] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[i, 0] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[i, j + 1] = Phi[i - 1, j]

    Phi_t = Phi.T

    DU = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t @ (R_s - F @ x_i)
    
    return DU


def predict_horizon(A, B, C, u, x_i, n_p):
    """Predicts the system's response for a given control action and a given
    horizon.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, 1) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (1, n) numpy matrix.

    u : :class:`np.array`
        The control values. An (n_c, 1) numpy matrix, where `n_c` is the
        number of control actions.

    x_i : :class:`np.array`
        Initial conditions of the states. An (n, 1) numpy matrix.

    n_p : :class:`int`
        Length of prediction horizon. Should be equal or greater than the
        number of control actions.
    
    Returns
    -------
    (x, y) : :class:`tuple`
        A tuple containing two numpy matrices. The first matrix contains the
        state values `x` and the second matrix contains the output `y`.
    
    """
    x_i = x_i.reshape(-1, 1)
    n_c = u.shape[0]

    x = np.zeros((n_p, x_i.shape[0]))
    y = np.zeros((n_p, C.shape[0]))

    x[0, :] = x_i.reshape(-1)
    y[0, :] = C @ x[0, :]
    for i in range(1, n_c):
        x[i, :] = A @ x[i - 1, :] + B @ u[i - 1]
        y[i, :] = C @ x[i, :]
        
    for i in range(n_c, n_p):
        x[i, :] = A @ x[i - 1, :]
        y[i, :] = C @ x[i, :]

    return (x, y)


def opt_matrices(A, B, C, n_p, n_c):
    r"""Computes the :math:`F` and :math:`Phi` matrices.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, 1) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (1, n) numpy matrix.

    n_p : :class:`int`
        Length of prediction horizon.

    n_c : :class:`int`
        Length of the control window.

    Returns
    -------
    (F, Phi) : :class:`tuple`
        A tuple, where the first item corresponds to the `F` matrix and the
        second item corresponds to the `Phi` matrix.
    
    """
    F = np.zeros((n_p, (C @ A).shape[1]))
    F[0, :] = C @ A
    for i in range(1, n_p):
        F[i, :] = F[i - 1, :] @ A

    Phi = np.zeros((n_p, n_c))
    Phi[0, 0] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[i, 0] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[i, j + 1] = Phi[i - 1, j]

    return (F, Phi)


class system:

    def __init__(self, Am, Bm, Cm, n_p=None, n_c=None, r_w=None):
        self.A, self.B, self.C = ctl.mpc.aug(Am, Bm, Cm)
        self.Am = Am
        self.Bm = Bm
        self.Cm = Cm

        self.n_p = n_p
        self.n_c = n_c

        self.r_w = r_w
        

    def model_matrices(self):

        return (self.Am, self.Bm, self.Cm)


    def aug_matrices(self):

        return (self.A, self.B, self.C)


    def set_predict_horizon(self, n_p):

        self.n_p = n_p


    def set_control_horizon(self, n_c):

        self.n_c = n_c
        

    def set_r_w(self, r_w):

        self.r_w = r_w

    
    def opt(self, x_ki, r_ki, r_w, n_p, n_c):

        A, B, C = self.aug_matrices()
        DU = ctl.mpc.opt(A, B, C, x_ki, r_ki, r_w, n_p, n_c) 

        return DU

    
    def predict_horizon(self, u, x_ki, n_p):

        A, B, C = self.aug_matrices()
        x, y = ctl.mpc.predict_horizon(A, B, C, u, x_ki, n_p)

        return (x, y)

    
    def sim(self, x_ki, u_0, r_ki, r_w, n, n_p, n_c):

        Am, Bm, Cm = self.model_matrices()
        n_x = Am.shape[0]
        n_y = Cm.shape[0]
        n_dx = n_x + n_y
        
        u = np.zeros((n, 1))
        x = np.zeros((n, n_x))
        y = np.zeros((n, n_y))
        dx = np.zeros((n, n_x + n_y))
        
        y[0, :] = x_ki[n_x:]
        x[0, :] = 1 / Cm * y[0, :] # This only works if C is 1-D (only one output)
        dx[0, :] = x_ki.reshape(-1)
        
        u_p = u_0

        for i in range(1, n):
            du = self.opt(dx[i - 1].T, r_ki, r_w, n_p, n_c)
            u[i - 1] = u_p + du[0]
            u_p = u[i - 1]
            
            x[i, :] = Am @ x[i - 1, :] + Bm @ u[i - 1]
            y[i, :] = Cm @ x[i, :]

            dx[i, :n_x] = x[i, :] - x[i - 1, :]
            dx[i, n_x:] = y[i, :]
        
        return (u, x, y, dx)

    
    def opt_cl_gains(self):

        A, B, C = self.aug_matrices()
        n_p = self.n_p
        n_c = self.n_c
        r_w = self.r_w
        
        R_s_bar = np.ones((n_p, 1))
        R = r_w * np.eye(n_c)
        
        F, Phi = ctl.mpc.opt_matrices(A, B, C, n_p, n_c)
        Phi_t = Phi.T

        K = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t
        K_y = K @ R_s_bar
        K_mpc = K @ F

        return (K_y[0].reshape(1, -1), K_mpc[0, :].reshape(1, -1))


    def sim_cl(self, x_i, u_0, r, n):

        Am, Bm, Cm = self.model_matrices()
        A, B, C = self.aug_matrices()
        n_x = A.shape[0]
        n_y = C.shape[0]
        
        u = np.zeros((n, 1))
        x = np.zeros((n, n_x))
        y = np.zeros((n, n_y))
        
        y[0, :] = C @ x_i
        x[0, :x_i.shape[0]] = x_i.reshape(-1)
        x[0, x_i.shape[0]:] = y[0, :]

        K_y, K_mpc = self.opt_cl_gains()
        
        A_u = A - B @ K_mpc
        B_u = B @ K_y
        for i in range(1, n):
            x[i] = A_u @ x[i - 1, :] + B_u @ r
            y[i] = C @ x[i, :]

        return (x, y)
            
