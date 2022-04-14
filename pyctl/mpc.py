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
        An (n, m) numpy matrix.

    Cm : np.array
        A (q, n) numpy matrix.

    Returns
    -------
    (A, B, C) : tuple
        A tuple containing the augmented matrices.
    
    """
    # Number of states
    n = Am.shape[0]

    # Number of inputs
    if Bm.ndim == 1:
        m = 1
    else:
        m = Bm.shape[1]

    # Number of outputs
    q = Cm.shape[0]

    A = np.zeros((n + q, n + q))
    A[:n, :n] = Am
    A[n:, :n] = Cm @ Am
    A[n:, n:] = np.eye(q)

    B = np.zeros((n + q, m))
    B[:n, :] = Bm
    B[n:, :] = Cm @ Bm

    C = np.zeros((q, n + q))
    C[:, n:] = np.eye(q)

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
        The `B` matrix of the augmented model. An (n, m) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (q, n) numpy matrix.

    x_i : :class:`np.array`
        Initial conditions of the augmented states. An (n, 1) numpy matrix.

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
        An (n_c * m, 1) numpy matrix containing the optimal control values.

    """
    if type(r_w) is int or type(r_w) is float:
        r_w = np.array([r_w])
            
    # Number of states
    n = A.shape[0]

    # Number of inputs
    if B.ndim == 1:
        m = 1
    else:
        m = B.shape[1]

    # Number of outputs
    q = C.shape[0]

    x_i = x_i.reshape(-1, 1)

    R_s_bar = np.tile(np.eye(q), (n_p, 1))
    # R_s = np.zeros((n_p * q, 1))
    # for i in range(n_p):
    #    R_s[ (q * i) : (q * (i + 1)), 0] = r

    R = np.zeros((n_c * m, n_c * m))
    for i in range(n_c):
        R[m * i : m * (i + 1), m * i : m * (i + 1)] = np.diag(r_w)
        
    F = np.zeros((n_p * q, A.shape[1]))
    F[:q, :] = C @ A
    for i in range(1, n_p):
        F[q * i : q * (i + 1), :] = F[q * (i - 1) : q * i, :] @ A

    Phi = np.zeros((n_p * q, n_c * m))
    Phi[:q, :m] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[ (q * i) : ( q * (i + 1) ), :m] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[ (q * i) : ( q * (i + 1) ), m * (j + 1) : m * (j + 2)] = Phi[ ( q * (i - 1) ) : (q * i), m * j : m * (j + 1)]

    Phi_t = Phi.T
    DU = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t @ (R_s_bar - F @ x_i)
    
    return DU


def predict_horizon(A, B, C, u, x_i, n_p):
    r"""Predicts the system's response for a given control action and a given
    horizon.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, m) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (q, n) numpy matrix.

    u : :class:`np.array`
        The control values. An (n_c, 1) numpy matrix, where `n_c` is the
        number of control actions.

    x_i : :class:`np.array`
        Initial conditions of the augmented states. An (n, 1) numpy matrix.

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
        The `B` matrix of the augmented model. An (n, m) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (q, n) numpy matrix.

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
    # Number of states
    n = A.shape[0]

    # Number of inputs
    if B.ndim == 1:
        m = 1
    else:
        m = B.shape[1]

    # Number of outputs
    q = C.shape[0]
    
    F = np.zeros((n_p * q, A.shape[1]))
    F[:q, :] = C @ A
    for i in range(1, n_p):
        F[q * i : q * (i + 1), :] = F[q * (i - 1) : q * i, :] @ A

    Phi = np.zeros((n_p * q, n_c * m))
    Phi[:q, :m] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[ (q * i) : ( q * (i + 1) ), :m] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[ (q * i) : ( q * (i + 1) ), m * (j + 1) : m * (j + 2)] = Phi[ ( q * (i - 1) ) : (q * i), m * j : m * (j + 1)]

    return (F, Phi)


class System:
    """A class to create a discrete-time system for model predictive control
    simulations.

    Parameters
    ----------
    Am : :class:`np.array`
        Model matrix :math:`A_m`. An (n, n) numpy matrix.

    Bm : :class:`np.array`
        Model matrix :math:`B_m`. An (n, m) numpy matrix.

    Cm : :class:`np.array`
        Model matrix :math:`C_m`. An (q, n) numpy matrix.

    n_p : :class:`bool`, :class:`int`
        Length of prediction horizon . Can be set later. By default, it is
        `None`.

    n_c : :class:`NoneType`, :class:`int`
        Length of control window. Can be set later. By default, it is `None`.

    r_w : :class:`NoneType`, :class:`int`
        Weight of control action. Can be set later. By default, it is `None`.

    Attributes
    ----------
    A : :class:`np.array`
        Augmented model matrix :math:`A`.
    
    B : :class:`np.array`
        Augmented model matrix :math:`B`.
    
    C : :class:`np.array`
        Augmented model matrix :math:`C`.

    Am : :class:`np.array`
        Model matrix :math:`A_m`.
    
    Bm : :class:`np.array`
        Model matrix :math:`B_m`.
    
    Cm : :class:`np.array`
        Model matrix :math:`C_m`.
    
    n_p : :class:`bool`, :class:`int`
        Length of prediction horizon.

    n_c : :class:`NoneType`, :class:`int`
        Size of control window.

    r_w : :class:`NoneType`, :class:`int`
        Weight of control action.
    
    """
    def __init__(self, Am, Bm, Cm, n_p=None, n_c=None, r_w=None):
        self.A, self.B, self.C = ctl.mpc.aug(Am, Bm, Cm)
        self.Am = Am
        self.Bm = Bm
        self.Cm = Cm

        self.n_p = n_p
        self.n_c = n_c

        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])
        self.r_w = r_w
        

    def model_matrices(self):
        r"""Helper function that returns the matrices :math:`A_m`, :math:`B_m`
        and :math:`C_m` of the plant model.

        Returns
        -------
        (Am, Bm, Cm) : :class:`tuple`
            A tuple containing the three model matrices.
        
        """
        return (self.Am, self.Bm, self.Cm)


    def aug_matrices(self):
        r"""Helper function that returns the matrices :math:`A`, :math:`B` and
        :math:`C_m` of the augmented model.

        Returns
        -------
        (A, B, C) : :class:`tuple`
            A tuple containing the three matrices.
        
        """
        return (self.A, self.B, self.C)


    def set_predict_horizon(self, n_p):
        r"""Sets the length of the predict horizon.

        Parameters
        ----------
        n_p : :class:`int`
            Length of prediction horizon.        
        
        """
        self.n_p = n_p


    def set_control_horizon(self, n_c):
        r"""Sets the length of the control window.

        Parameters
        ----------
        n_c : :class:`int`
            Length of control window.
        
        """
        self.n_c = n_c
        

    def set_r_w(self, r_w):
        r"""Sets the weight for optimization of the control vector.

        Parameters
        ----------
        r_w : :class:`int`, :class:`float`
            Weight.        
        
        """
        self.r_w = r_w

    
    def opt(self, x_i, r, r_w, n_p, n_c):
        r"""Obtains the control vector minimizing the cost function.

        Parameters
        ----------
        x_i : :class:`np.array`
            Initial conditions of the augmented states. An (n, 1) numpy matrix.

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
        A, B, C = self.aug_matrices()
        DU = ctl.mpc.opt(A, B, C, x_i, r, r_w, n_p, n_c) 

        return DU

    
    def predict_horizon(self, u, x_i, n_p):
        r"""Predicts the states and the output based on the control actions and
        based on the given horizon.

        Parameters
        ----------
        u : :class:`np.array`
            The control values. An (n_c, 1) numpy matrix, where `n_c` is the
            number of control actions.

        x_i : :class:`np.array`
            Initial conditions of the augmented states. An (n, 1) numpy matrix.

        n_p : :class:`int`
            Length of prediction horizon. Should be equal or greater than the
            number of control actions.

        Returns
        -------
        (x, y) : :class:`tuple`
            A tuple containing two numpy matrices. The first matrix contains
            the state values `x` and the second matrix contains the output
            `y`.
        
        """
        A, B, C = self.aug_matrices()
        x, y = ctl.mpc.predict_horizon(A, B, C, u, x_ki, n_p)

        return (x, y)

    
    def opt_cl_gains(self):
        r"""Computes the optimum gains :math:`K_y` and :math:`K_{mpc}`.

        Returns
        -------
        (K_y, K_mpc) : :class:`tuple`
            A tuple, containing two elements. The first element is the matrix
            K_y and the second element is the matrix K_mpc.

        """    
        A, B, C = self.aug_matrices()
        n_p = self.n_p
        n_c = self.n_c
        r_w = self.r_w

        # Number of states
        n = A.shape[0]

        # Number of inputs
        if B.ndim == 1:
            m = 1
        else:
            m = B.shape[1]

        # Number of outputs
        q = C.shape[0]

        R_s_bar = np.tile(np.eye(q), (n_p, 1))
        # R_s_bar = np.eye(n_p * q)
        #  R_s_bar = np.zeros((n_p * q, 1))
        #  for i in range(n_p):
        #      R_s_bar[ (q * i) : (q * (i + 1)), 0] = 1

        R = np.zeros((n_c * m, n_c * m))
        for i in range(n_c):
            R[m * i : m * (i + 1), m * i : m * (i + 1)] = np.diag(r_w)

        F, Phi = ctl.mpc.opt_matrices(A, B, C, n_p, n_c)
        Phi_t = Phi.T
        
        K = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t
        K_mpc = K @ F
        K_y = K @ R_s_bar

        return (K_y[:m], K_mpc[:m, :])
        #return (K_y[0].reshape(1, -1), K_mpc[0, :].reshape(1, -1))


    def dmpc(self, x_i, u_i, r, n):
        """Simulates the MPC closed-loop system.

        Parameters
        ----------
        x_i : :class:`np.array`
            The initial conditions. Should be a (n_x, 2) numpy matrix, where
            `n_x` is the number of states of the model.

        u_i : :class:`np.array`
            The value of the control action at u(-1).

        r : :class:`float`, :class:`np.array`
            The set-point.

        n : :class:`int`
            Length of simulation.

        Returns
        -------
        data : :class:`dict`
            A dictionary containing the simulation results. The key `u`
            contains the control actions, the key `x_m` contains the states
            and the key `y` contains the output.

        """
        if type(r) is int or type(r) is float:
            r = r * np.ones((n, 1))
        if type(r) is np.ndarray and r.ndim == 1:
            r = np.tile(r, (n, 1))
            
        Am, Bm, Cm = self.model_matrices()
        A, B, C = self.aug_matrices()

        n_xm = Am.shape[0]
        n_x = A.shape[0]
        n_y = C.shape[0]
        n_u = B.shape[1]
        
        x_m = np.zeros((n, n_xm))
        x = np.zeros((n, n_x))
        y = np.zeros((n, n_y))

        u = np.zeros((n, n_u))

        x_m[0] = x_i[:, 0]
        dx = x_i[:, 1]
        u[0] = u_i

        K_y, K_mpc = self.opt_cl_gains()
        K_x = K_mpc[:, :n_xm]

        for i in range(n - 1):
            # Updates the output and dx
            y[i] = Cm @ x_m[i]
            dx = x_m[i] - dx

            # Computes the control law for sampling instant i
            du = -K_y @ (y[i] - r[i]) + -K_x @ dx
            u[i] = u[i] + du

            # Applies the control law
            x_m[i + 1] = Am @ x_m[i] + Bm @ u[i]

            # Update variables for next iteration
            dx = x_m[i]
            u[i + 1] = u[i]


        # Updates last value of y
        y[n - 1] = Cm @ x_m[n - 1]

        results = {}
        results['u'] = u
        results['x_m'] = x_m
        results['y'] = y

        return results
