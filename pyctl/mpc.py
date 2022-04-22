import numpy as np
import pyctl as ctl
import qpsolvers as qps
import pydsim.qp as pydqp

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
        dx = x_i[:, 1]
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
        
        A, B, C = self.A, self.B, self.C
        Am, Bm, Cm = self.Am, self.Bm, self.Cm
        n_p, n_c, n_r = self.n_p, self.n_c, self.n_r
        r_w = self.r_w

        x_lim = self.x_lim
##        x_lim_new = [[], []]
##
##        n_state_ineq = 0
##
##        for i in range(len(x_lim[0])):
##            if x_lim[0] is not None:
##                n_state_ineq = n_state_ineq + 1
                
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

        # Control inequality constraints
        M_aux = np.tril( np.tile( np.eye(m), (n_r, n_c) ) )
        M_u = np.concatenate((-M_aux, M_aux))
        #self.M = M_u

        # State inequality constraints
        #C_x = np.eye(Am.shape[0])
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
        self.C_x = np.tile(C_x, (n_r, 1))
        self.x_lim_new = np.array(x_lim_new)
        print(F_x.shape)
        print(Phi_x.shape)
        print(C_x.shape)
        M_x = np.concatenate((-Phi_x, Phi_x))

        self.M = np.concatenate((M_u, M_x))

        # QP matrices
        F, Phi = opt_matrices(A, B, C, n_p, n_c)
        self.F, self.Phi = F, Phi

        E_j = Phi.T @ Phi + R_bar
        E_j_inv = np.linalg.inv(E_j)
        self.E_j, self.E_j_inv = E_j, E_j_inv


    def dyn_matrices(self, xm, dx, xa, u_i, r):

        n_r = self.n_r
        
        F, Phi = self.F, self.Phi

        C_x = self.C_x

        R_s_bar = self.R_s_bar
       
        u_lim = self.u_lim
        x_lim = self.x_lim_new

        F_j = -Phi.T @ (R_s_bar @ r.reshape(-1, 1) - F @ xa.reshape(-1, 1))
        
        u_min = np.tile(-u_lim[0] + u_i, n_r).reshape(-1, 1)
        u_max = np.tile( u_lim[1] - u_i, n_r).reshape(-1, 1)

        print(self.F_x.shape)
        print(self.C_x.shape)
        print(dx.shape)
        x_min = np.tile(-x_lim[0] + C_x @ xm, n_r).reshape(-1, 1) + self.F_x @ C_x @ dx.reshape(-1, 1)
        x_max = np.tile( x_lim[1] - C_x @ xm, n_r).reshape(-1, 1) - self.F_x @ C_x @ dx.reshape(-1, 1)
        
        y = np.concatenate((u_min, u_max, x_min, x_max))

        return (F_j, y)


    def opt(self, xm, dx, xa, u_i, r):

        n_u = u_i.shape[0]
        
        F_j, y = self.dyn_matrices(xm, dx, xa, u_i, r)

        du = self.qp(F_j, y, method='cvx')

        return du[:n_u]

    
    def qp(self, F_j, y, method='hild'):

        E_j, E_j_inv = self.E_j, self.E_j_inv
        M = self.M
        F, Phi = self.F, self.Phi
        
        if method == 'hild':
            H_j = M @ E_j_inv @ M.T
            K_j = y + M @ E_j_inv @ F_j
            lm, n_iters = pydqp.hild(H_j, K_j, n_iter=500, ret_n_iter=True)
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
        dx = x_i[:, 1]
        u[0] = u_i

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
            x_m[i + 1] = Am @ x_m[i] + Bm @ u[i]

            # Update variables for next iteration
            dx = x_m[i]
            u[i + 1] = u[i]

            #print('\n---\n')

        # Updates last value of y
        y[n - 1] = Cm @ x_m[n - 1]

        results = {}
        results['u'] = u
        results['x_m'] = x_m
        results['y'] = y

        return results
