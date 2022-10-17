import numpy as np
import pyctl as ctl
import qpsolvers as qps
import pydsim.qp as pydqp
import sys

def aug(Am, Bm, Cm):
    r"""Determines the augmented model.

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
    if Cm.ndim == 1:
        q = 1
    else:
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

    r_w : :class:`int`, :class:`float`, :class:`np.array`
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
    if C.ndim == 1:
        q = 1
    else:
        q = C.shape[0]

    x_i = x_i.reshape(-1, 1)

    R_s_bar = reference_matrix(q, n_p)

    R = control_weighting_matrix(r_w, n_c)

    F, Phi = opt_matrices(A, B, C, n_p, n_c)

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
    if C.ndim == 1:
        q = 1
    else:
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


def control_weighting_matrix(r_w, n_c):
    r"""Computes the :math:`\bar{R}` matrix.

    Parameters
    ----------
    r_w : :class:`int`, :class:`list`, :class:`np.array`
        The weighting coefficients, as a 1-d numpy array, an integer or a
        list.

    n_c : :class:`int`
        Length of the control window.

    Returns
    -------
    R_bar : :class:`np.array`
        An (n_c * m, n_c * m) matrix, where `m` is the number of coefficients
        (system input signals).
    
    """
    if type(r_w) is int or type(r_w) is float:
        r_w = np.array([r_w])
    elif type(r_w) is list:
        r_w = np.array(r_w)

    m = r_w.shape[0]
    
    R_bar = np.zeros((n_c * m, n_c * m))
    for i in range(n_c):
        R_bar[m * i : m * (i + 1), m * i : m * (i + 1)] = np.diag(r_w)

    return R_bar


def reference_matrix(q, n_p):
    r"""Computes the :math:`\bar{R_s}` matrix.

    Parameters
    ----------
    q : :class:`int`
        Number of references (system outputs).
        
    n_p : :class:`int`
        Length of prediction horizon.

    Returns
    -------
    R_s_bar : :class:`np.array`
        An (q, n_p * q) matrix, where `q` is the number of references
        (system outputs).
    
    """
    R_s_bar = np.tile(np.eye(q), (n_p, 1))

    return R_s_bar


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
        
    r_w : :class:`NoneType`, :class:`int`, :class:`np.array`
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

    r_w : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weight of control action.
    
    """
    def __init__(self, Am, Bm, Cm, n_p=None, n_c=None, n_r=None, r_w=None):
        self.A, self.B, self.C = ctl.mpc.aug(Am, Bm, Cm)
        self.Am = Am
        self.Bm = Bm
        self.Cm = Cm

        self.n_p = n_p
        self.n_c = n_c

        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])
        self.r_w = r_w

        self.n_r = n_r
        

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
        r_w : :class:`int`, :class:`float`, :class:`np.array`
            Weight.        
        
        """
        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])
        self.r_w = r_w

    
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
        if C.ndim == 1:
            q = 1
        else:
            q = C.shape[0]

        R_s_bar = reference_matrix(q, n_p)

        R = control_weighting_matrix(r_w, n_c)

        F, Phi = ctl.mpc.opt_matrices(A, B, C, n_p, n_c)
        Phi_t = Phi.T
        
        K = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t
        K_mpc = K @ F
        K_y = K @ R_s_bar

        return (K_y[:m], K_mpc[:m, :])
                

    def dmpc(self, x_i, u_i, r, n, Bd=None, u_d=None):
        """Simulates the MPC closed-loop system.

        Parameters
        ----------
        x_i : :class:`np.array`
            The initial conditions. Should be an (n_x, 1) numpy matrix, where
            `n_x` is the number of states of the model.

        u_i : :class:`np.array`
            The value of the control action at u(-1).

        r : :class:`float`, :class:`np.array`
            The set-point.

        n : :class:`int`
            Length of simulation.

        Bd : :class:`np.array`
            An (p, p) numpy matrix, where `p` is the number of disturbances.
            By default, it is `None`.

        u_d : :class:`np.array`
            An (p, 1) or (p, n) numpy matrix, where `p` is the number of
            disturbances. If the second dimension is 1, the disturbance is
            considered to be constant during the entire period. Otherwise,
            it must contain `n` values to be used during the entire
            simulation. By default, it is `None`.

        Returns
        -------
        data : :class:`dict`
            A dictionary containing the simulation results. The key `u`
            contains the control actions, the key `x_m` contains the states
            and the key `y` contains the output.

        """
        if type(x_i) is int or type(x_i) is float or type(x_i) is list:
            x_i = np.array(x_i).reshape(1, -1)
        elif type(x_i) is np.ndarray:
            x_i = np.array(x_i).reshape(1, -1)
            
        if type(r) is int or type(r) is float:
            r = r * np.ones((n, 1))
        elif type(r) is list:
            r = np.array(r)
        
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
        #dx = x_i[:, 1]
        dx = 0
        u[0] = u_i

        K_y, K_mpc = self.opt_cl_gains()
        K_x = K_mpc[:, :n_xm]

        self.K_y = K_y
        self.K_x = K_x

        if Bd is None:
            Bd = np.zeros(Bm.shape)
            u_d = np.zeros(u.shape)
        else:
            if u_d.ndim == 1:
                u_d = np.tile(u_d, (n, 1))
                
        for i in range(n - 1):
            # Updates the output and dx
            y[i] = Cm @ x_m[i]
            dx = x_m[i] - dx

            # Computes the control law for sampling instant i
            du = -K_y @ (y[i] - r[i]) + -K_x @ dx
            u[i] = u[i] + du

            # Applies the control law
            x_m[i + 1] = Am @ x_m[i] + Bm @ u[i] + Bd @ u_d[i]

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


class ConstrainedModel:
    """A class to hold data from a constrained system.
    
    Parameters
    ----------
    Am : :class:`np.array`
        Model matrix :math:`A_m`. An (n, n) numpy matrix.

    Bm : :class:`np.array`
        Model matrix :math:`B_m`. An (n, m) numpy matrix.

    Cm : :class:`np.array`
        Model matrix :math:`C_m`. An (q, n) numpy matrix.

    n_p : :class:`int`
        Length of prediction horizon.

    n_c : :class:`int`
        Length of the control window.

    n_r : :class:`int`
        Length of constraint window. 
        
    r_w : :class:`int`, :class:`float`, :class:`np.array`, :class:`list`
        Weight of control action.

    x_lim : :class:`np.array`, :class:`list`
        Lower and upper bounds of the states.
        
    u_lim : :class:`np.array`, :class:`list`
        Lower and upper bounds of the control signals.
        
    """
    def __init__(self, Am, Bm, Cm, n_p, n_c, n_r, r_w, x_lim, u_lim):

        self.Am, self.Bm, self.Cm = Am, Bm, Cm
        self.A, self.B, self.C = aug(Am, Bm, Cm)
        
        self.n_p, self.n_c, self.n_r = n_p, n_c, n_r
        self.r_w = r_w 

        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])
        elif type(r_w) is list:
            r_w = np.array(r_w)

        if type(u_lim) is int or type(u_lim) is float:
            u_lim = np.array([u_lim])
        elif type(u_lim) is list:
            u_lim = np.array(u_lim)

        if type(x_lim) is int or type(x_lim) is float:
            x_lim = np.array([x_lim])
        elif type(x_lim) is list:
            x_lim = np.array(x_lim)
            
        self.u_lim = u_lim
        self.x_lim = x_lim

        self.const_matrices()
    

    def const_matrices(self):
        """Sets constant matrices, to be used later by the optimization.

        """
        A, B, C = self.A, self.B, self.C
        Am, Bm, Cm = self.Am, self.Bm, self.Cm
        n_p, n_c, n_r = self.n_p, self.n_c, self.n_r
        r_w = self.r_w

        x_lim = self.x_lim
        u_lim = self.u_lim
                
        # Number of inputs
        if B.ndim == 1:
            m = 1
        else:
            m = B.shape[1]

        # Number of outputs
        if C.ndim == 1:
            q = 1
        else:
            q = C.shape[0]
        
        R_bar = control_weighting_matrix(r_w, n_c)
        R_s_bar = reference_matrix(q, n_p)

        self.R_bar = R_bar
        self.R_s_bar = R_s_bar

        # Creates left-hand side inequality constraint, starting first with
        # control inequality constraints
        M = []
        if u_lim is not None:
            M_aux = np.tril( np.tile( np.eye(m), (n_r, n_c) ) )
            M_u = np.concatenate((-M_aux, M_aux))
            M = M_u

        # Now, the state inequality constraints
        if x_lim is not None:
            n_state_ineq = 0
            x_lim_new = [[], []]
            C_x = []
            
            for i, x_i in enumerate(x_lim[0]):
                if x_i is not None:
                    x_lim_new[0].append(x_lim[0][i])
                    x_lim_new[1].append(x_lim[1][i])
                    
                    n_state_ineq = n_state_ineq + 1
                    
                    cx = np.zeros((1, Am.shape[0]))
                    cx[0, i] = 1
                    
                    if C_x == []:
                        C_x = cx
                    else:
                        C_x = np.concatenate((C_x, cx))
                        
            F_x, Phi_x = opt_matrices(Am, Bm, C_x, n_r, n_c)
            self.F_x, self.Phi_x = F_x, Phi_x
            self.C_x = C_x
            
            self.x_lim = np.array(x_lim_new)

            M_x = np.concatenate((-Phi_x, Phi_x))

            if M == []:
                M = M_x
            else:
                M = np.concatenate((M, M_x))

        # If there were no constraints, creates an empty matrix
        if M == []:
            M = np.zeros((1, m * n_r ))
        
        # Saves M matrix only after creating all constraints
        self.M = M

        # QP matrices
        F, Phi = opt_matrices(A, B, C, n_p, n_c)
        self.F, self.Phi = F, Phi

        E_j = Phi.T @ Phi + R_bar
        E_j_inv = np.linalg.inv(E_j)
        self.E_j, self.E_j_inv = E_j, E_j_inv


    def dyn_matrices(self, xm, dx, xa, u_i, r):
        """Sets dynamic matrices, to be used later by the optimization.

        """
        n_r = self.n_r
        F, Phi = self.F, self.Phi
        R_s_bar = self.R_s_bar
       
        u_lim = self.u_lim
        x_lim = self.x_lim

        F_j = -Phi.T @ (R_s_bar @ r.reshape(-1, 1) - F @ xa.reshape(-1, 1))

        # Creates the right-hand side inequality vector, starting first with
        # the control inequality constraints
        y = []
        
        if u_lim is not None:
            u_min = np.tile(-u_lim[0] + u_i, n_r).reshape(-1, 1)
            u_max = np.tile( u_lim[1] - u_i, n_r).reshape(-1, 1)

            y = np.concatenate((u_min, u_max))

        # Now, the state inequality constraints
        if x_lim is not None:
            C_x = self.C_x
            F_x = self.F_x
            x_min = np.tile(-x_lim[0] + C_x @ xm, n_r).reshape(-1, 1) + F_x @ dx.reshape(-1, 1)
            x_max = np.tile( x_lim[1] - C_x @ xm, n_r).reshape(-1, 1) - F_x @ dx.reshape(-1, 1)

            if y == []:
                y = np.concatenate((x_min, x_max))
            else:
                y = np.concatenate((y, x_min, x_max))

        # If there were no constraints, creates a zero vector
        if y == []:
            y = np.zeros(1)

        return (F_j, y)


    def opt(self, xm, dx, xa, u_i, r):

        n_u = u_i.shape[0]
        
        F_j, y = self.dyn_matrices(xm, dx, xa, u_i, r)

        du = self.qp(F_j, y, method='cvx')

        return du[:n_u]

    
    def qp(self, F_j, y, method='hild'):
        """Solver the QP problem given by:

        .. :math:

            J = \Delta U^T E_J \Delta U^T +  \Delta U^T F_j,

        subject to:

        .. :math:

            M \Delta U \leq y.
            
        """
        E_j, E_j_inv = self.E_j, self.E_j_inv
        M = self.M
        F, Phi = self.F, self.Phi

        H_j = M @ E_j_inv @ M.T
        K_j = y + M @ E_j_inv @ F_j
        self.H_j = H_j
        self.K_j = K_j

        if method == 'hild':
            if self.x_lim is None and self.u_lim is None:
                du_opt = (-E_j_inv @ F_j).reshape(-1)
            else:
                lm, n_iters = pydqp.hild(H_j, K_j, n_iter=5000, ret_n_iter=True)
                lm = lm.reshape(-1, 1)
                du_opt = -E_j_inv @ (F_j + M.T @ lm)
                du_opt = du_opt.reshape(-1)

        elif method == 'cvx':
            du_opt = qps.cvxopt_solve_qp(E_j, F_j.reshape(-1), M, y.reshape(-1))

        elif method == 'quadprog':
            du_opt = qps.solve_qp(E_j, F_j.reshape(-1), M, y.reshape(-1))

        else:
            du_opt = 0

        return du_opt


class ConstrainedSystem:
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

    n_r : :class:`NoneType`, :class:`int`
        Length of constraint window. Can be set later. By default, it is
        `None`.
        
    r_w : :class:`NoneType`, :class:`int`, :class:`np.array`
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

    n_r : :class:`NoneType`, :class:`int`
        Size of constraint window.

    r_w : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weight of control action.
    
    """
    def __init__(self, Am, Bm, Cm, n_p=None, n_c=None, n_r=None, r_w=None, x_lim=None, u_lim=None):
        self.A, self.B, self.C = ctl.mpc.aug(Am, Bm, Cm)
        self.Am = Am
        self.Bm = Bm
        self.Cm = Cm

        self.n_p = n_p
        self.n_c = n_c
        self.n_r = n_r

        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])

        elif type(r_w) is list:
            r_w = np.array(r_w)
            
        self.r_w = r_w

        if type(u_lim) is int or type(u_lim) is float:
            u_lim = np.array([u_lim])
        
        elif type(u_lim) is list:
            u_lim = np.array(u_lim)

        if type(x_lim) is int or type(x_lim) is float:
            x_lim = np.array([x_lim])
        
        elif type(x_lim) is list:
            x_lim = np.array(x_lim)
            
        self.u_lim = u_lim
        self.x_lim = x_lim
        
        self.constr_model = ConstrainedModel(Am, Bm, Cm, n_p, n_c, n_r, r_w, x_lim, u_lim)
        

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


    def set_constraint_horizon(self, n_r):
        r"""Sets the length of the constraint window.

        Parameters
        ----------
        n_r : :class:`int`
            Length of constraint window.
        
        """
        self.n_r = n_r
        

    def set_r_w(self, r_w):
        r"""Sets the weight for optimization of the control vector.

        Parameters
        ----------
        r_w : :class:`int`, :class:`float`, :class:`np.array`
            Weight.        
        
        """
        if type(r_w) is int or type(r_w) is float:
            r_w = np.array([r_w])
        self.r_w = r_w
    

    def dmpc(self, x_i, u_i, r, n, Bd=None, u_d=None):
        """Simulates the MPC closed-loop system.

        Parameters
        ----------
        x_i : :class:`np.array`
            The initial conditions. Should be an (n_x, 1) numpy matrix, where
            `n_x` is the number of states of the model.

        u_i : :class:`np.array`
            The value of the control action at u(-1).

        r : :class:`float`, :class:`np.array`
            The set-point.

        n : :class:`int`
            Length of simulation.

        Bd : :class:`np.array`
            An (p, p) numpy matrix, where `p` is the number of disturbances.
            By default, it is `None`.

        u_d : :class:`np.array`
            An (p, 1) or (p, n) numpy matrix, where `p` is the number of
            disturbances. If the second dimension is 1, the disturbance is
            considered to be constant during the entire period. Otherwise,
            it must contain `n` values to be used during the entire
            simulation. By default, it is `None`.

        Returns
        -------
        data : :class:`dict`
            A dictionary containing the simulation results. The key `u`
            contains the control actions, the key `x_m` contains the states
            and the key `y` contains the output.

        """
        if type(u_i) is int or type(u_i) is float or type(u_i) is list:
            u_i = np.array(u_i)

        if type(x_i) is int or type(x_i) is float or type(x_i) is list:
            x_i = np.array(x_i).reshape(1, -1)
        elif type(x_i) is np.ndarray:
            x_i = np.array(x_i).reshape(1, -1)
            
        if type(r) is int or type(r) is float:
            r = r * np.ones((n, 1))
        elif type(r) is list:
            r = np.array(r)
        
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
        #dx = x_i[:, 1]
        dx = 0
        u[0] = u_i

        if Bd is None:
            Bd = np.zeros(Bm.shape)
            u_d = np.zeros(u.shape)
        else:
            if type(u_d) is int or type(u_d) is float or type(u_d) is list:
                u_d = np.array(u_d)
            if u_d.ndim == 1:
                u_d = np.tile(u_d, (n, 1))
        
        du = np.zeros((B.shape[1], 1)).reshape(-1) + u_i.reshape(-1)
        xa = np.zeros((A.shape[0], 1))
        for i in range(n - 1):
            # Updates the output and dx
            y[i] = Cm @ x_m[i]
            dx = x_m[i] - dx
            xa[:n_xm, 0] = dx
            xa[n_xm:, 0] = y[i]

            du = self.constr_model.opt(x_m[i], dx, xa, u[i], r[i])
            u[i] = u[i] + du
            
            # Applies the control law
            #v = x_m[i, 1]
            #if v >= 5.0:
            #    #u_d[i] = 20 / v * np.sin(2*np.pi*300*i*1/50e3)
            #    u_d[i] = 20 / v
            #else:
            #    u_d[i] = 0
            x_m[i + 1] = Am @ x_m[i] + Bm @ u[i] + Bd @ u_d[i]

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


    def export(self, file='.', scaling=1.0, Bd=None):

        def np_array_to_c(arr, arr_name):

            if arr.ndim == 1:
                n = arr.shape[0]
                m = 1
            else:
                if (arr.shape[0] == 1) or (arr.shape[1] == 1):
                    arr = arr.flatten()
                    n = arr.shape[0]
                    m = 1
                else:
                    n, m = arr.shape

            arr_str = np.array2string(arr, separator=',')
            arr_str = arr_str.replace('[', '{')
            arr_str = arr_str.replace(']', '}')

            if m == 1:
                arr_str = '{:}[{:}] = {:};'.format(arr_name, n, arr_str)
            else:
                arr_str = '{:}[{:}][{:}] = {:};'.format(arr_name, n, m, arr_str)

            return arr_str
    
        #np.set_printoptions(floatmode='unique')
        np.set_printoptions(threshold=sys.maxsize)

        Am, Bm, Cm = self.constr_model.Am, self.constr_model.Bm, self.constr_model.Cm

        n_s, n_as = self.constr_model.Am.shape[0], self.constr_model.A.shape[0]

        n_u = self.constr_model.Bm.shape[1]

        if Bd is None:
            n_d = 0
        else:
            n_d = Bd.shape[1]

        if self.constr_model.Cm.ndim != 1: n_y = self.constr_model.Cm.shape[0]
        else: n_y = 1

        n_p, n_c, n_r = self.n_p, self.n_c, self.n_r
        u_lim, x_lim = self.u_lim, self.x_lim

        Fj1 = -self.constr_model.Phi.T @ self.constr_model.R_s_bar
        Fj2 = self.constr_model.Phi.T @ self.constr_model.F

        Kj1 = self.constr_model.M @ self.constr_model.E_j_inv

        if x_lim is None:
            Fxp = np.zeros((1,1))
        else:
            Fxp = self.constr_model.F_x

        Ej = self.constr_model.E_j

        M = self.constr_model.M

        Hj = np.zeros(self.constr_model.H_j.shape, dtype=self.constr_model.H_j.dtype)
        Hj[:] = self.constr_model.H_j[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        DU1 = (-self.constr_model.E_j_inv)[:n_u, :]
        DU2 = (-self.constr_model.E_j_inv @ self.constr_model.M.T)[:n_u, :]
        n_lambda = DU2.shape[1]

        text = ''

        header = '/**\n'\
         ' * @file dmpc_matrices.h\n'\
         ' * @brief Header with data to run the DMPC algorithm.\n'\
         ' *\n'\
         ' * This file is generated automatically and should not be modified.\n'\
         ' *\n'\
         ' * The Hj matrix is already generated by flipping the sign and inverting its\n'\
         ' * diagonal elements, so that Hildreth\'s algorithm does not require any \n'\
         ' * divisions.\n'\
         ' *\n'\
         ' *  Originally created on: 22.04.2022\n'\
         ' *      Author: mguerreiro\n'\
         ' */\n'
        
        text = text + header

        def_guard = '\n#ifndef DMPC_MATRICES_H_\n'\
                    '#define DMPC_MATRICES_H_\n'
        text = text + def_guard

        def_prefix = 'DMPC_CONFIG'

        defines ='\n/* Scaling factor */\n'\
                  '#define {:}_SCALE\t\t\t{:f}f\n'.format(def_prefix, scaling)+\
                  '\n/* Number of model states and augmented states */\n'\
                  '#define {:}_NXM\t\t\t{:}\n'.format(def_prefix, n_s)+\
                  '#define {:}_NXA\t\t\t{:}\n'.format(def_prefix, n_as)+\
                  '\n/* Prediction, control and constraint horizon */\n'\
                  '#define {:}_NP\t\t\t{:}\n'.format(def_prefix, n_p)+\
                  '#define {:}_NC\t\t\t{:}\n'.format(def_prefix, n_c)+\
                  '#define {:}_NR\t\t\t{:}\n'.format(def_prefix, n_r)+\
                  '#define {:}_NLAMBDA\t\t{:}\n'.format(def_prefix, n_lambda)+\
                  '\n/* Number of inputs and outputs */\n'\
                  '#define {:}_NU\t\t\t{:}\n'.format(def_prefix, n_u)+\
                  '#define {:}_NY\t\t\t{:}\n'.format(def_prefix, n_y)+\
                  '\n/* Number of external disturbances */\n'\
                  '#define {:}_ND\t\t\t{:}\n'.format(def_prefix, n_d)+\
                  '\n/* Size of control vector */\n'\
                  '#define {:}_NC_x_NU\t\t{:}_NC * {:}_NU\n'.format(def_prefix, def_prefix, def_prefix)
        text = text + defines

        if u_lim is not None:
            idx = []
            for i, xi in enumerate(u_lim[0]):
                if xi != None:
                    idx.append(i)
            idx = np.array(idx)

        u_lim_sz = idx.shape[0]
        u_min_text = np_array_to_c(u_lim[0] / scaling, 'float {:}_U_MIN'.format(def_prefix, u_lim_sz)) + '\n'
        u_max_text = np_array_to_c(u_lim[1] / scaling, 'float {:}_U_MAX'.format(def_prefix, u_lim_sz)) + '\n'
        x_lim_idx_text = np_array_to_c(idx, 'uint32_t {:}_U_LIM_IDX'.format(def_prefix, u_lim_sz)) + '\n'
            
        constraints = '\n/* Input constraints */\n'+\
                      '#define {:}_NU_CTR\t\t{:}\n'.format(def_prefix, u_lim_sz)+\
                      u_min_text+\
                      u_max_text+\
                      x_lim_idx_text
        text = text + constraints

        if x_lim is not None:
            idx = []
            for i, xi in enumerate(x_lim[0]):
                if xi != None:
                    idx.append(i)
            idx = np.array(idx)

            x_lim_sz = idx.shape[0]
            x_min_text = np_array_to_c(x_lim[0][idx] / scaling, 'float {:}_XM_MIN'.format(def_prefix, x_lim_sz)) + '\n'
            x_max_text = np_array_to_c(x_lim[1][idx] / scaling, 'float {:}_XM_MAX'.format(def_prefix, x_lim_sz)) + '\n'
            x_lim_idx_text = np_array_to_c(idx, 'uint32_t {:}_XM_LIM_IDX'.format(def_prefix, x_lim_sz)) + '\n'
            
            constraints = '\n/* State constraints */\n'+\
                          '#define {:}_NXM_CTR\t\t{:}\n'.format(def_prefix, x_lim_sz)+\
                          x_min_text+\
                          x_max_text+\
                          x_lim_idx_text
        else:
            
            constraints = '\n/* State constraints */\n'+\
                          '#define {:}_NXM_CTR\t\t{:}\n'.format(def_prefix, 0)
            
        text = text + constraints

        idx = []
        if Cm.ndim != 1:
            Cm = np.sum(Cm, axis=0)
        for i, yi in enumerate(Cm):
            if np.abs(yi) > 0.5: idx.append(i)
        idx = np.array(idx)
        outputs_sz = idx.shape[0]
        outputs_idx_text = np_array_to_c(idx, 'uint32_t {:}_Y_IDX'.format(def_prefix, outputs_sz)) + '\n'
        outs = '\n/* Indexes of outputs */\n'+\
               outputs_idx_text
        text = text + outs
        
        matrices_prefix = 'DMPC_M'
        A_text = np_array_to_c(Am, 'float {:}_A'.format(matrices_prefix)) + '\n'
        if Bd is not None:
            B = np.concatenate((Bm, Bd), axis=1)
        else:
            B = Bm
        B_text = np_array_to_c(B, 'float {:}_B'.format(matrices_prefix)) + '\n'
        
        matrices ='\n/* A and B matrices for prediction */\n'+\
                  A_text+\
                  B_text
        text = text + matrices
        
        matrices_prefix = 'DMPC_M_'
        matrices = '\n/*\n * Matrices for QP solvers \n'\
                   ' *\n'\
                   ' * The matrices were generated considering the following problem:\n'\
                   ' *\n'\
                   ' * min (1/2) * DU\' * Ej * DU + DU\' * Fj\n'\
                   ' * DU\n'\
                   ' *\n'\
                   ' * s.t. M * DU <= gam\n'\
                   ' *\n'\
                   ' * The (1/2) term in from of DU\' * Ej * DU needs to be considered in the QP\n'\
                   ' * solver selected, or the solution will appear to be inconsistent.\n'\
                   ' * Note that the Fj and gam matrices are usually updated online, while Ej\n'\
                   ' * and M are static.\n'\
                   ' */\n'
        ej = np_array_to_c(Ej, 'float {:}Ej'.format(matrices_prefix)) + '\n\n'
        fj = 'float {:}Fj[{:}];\n\n'.format(matrices_prefix, n_c * n_u)
        m = np_array_to_c(M, 'float {:}M'.format(matrices_prefix)) + '\n\n'
        gam = 'float {:}gam[{:}];\n'.format(matrices_prefix, n_lambda)
        text = text + matrices + ej + fj + m + gam
        
        matrices = '\n /* Matrices for Hildreth\'s QP procedure */\n'
        fj1 = np_array_to_c(Fj1, 'float {:}Fj_1'.format(matrices_prefix)) + '\n\n'
        fj2 = np_array_to_c(Fj2, 'float {:}Fj_2'.format(matrices_prefix)) + '\n\n'
        fxp = np_array_to_c(Fxp, 'float {:}Fx'.format(matrices_prefix)) + '\n\n'
        kj1 = np_array_to_c(Kj1, 'float {:}Kj_1'.format(matrices_prefix)) + '\n\n'
        hj = np_array_to_c(Hj, 'float {:}Hj'.format(matrices_prefix)) + '\n\n'
        du1 = np_array_to_c(DU1, 'float {:}DU_1'.format(matrices_prefix)) + '\n\n'
        du2 = np_array_to_c(DU2, 'float {:}DU_2'.format(matrices_prefix)) + '\n\n'
        text = text + matrices + fj1 + fj2 + fxp + kj1 + hj + du1 + du2

        def_guard_end = '\n#endif /* DMPC_MATRICES_H_ */\n'
        text = text + def_guard_end

        if file is not None:
            with open(file, 'w') as efile:
                efile.write(text)
                
        #print(text)

        np.set_printoptions(threshold=1000)
        #np.set_printoptions(floatmode='maxprec_equal')
