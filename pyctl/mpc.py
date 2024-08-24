import numpy as np
import pyctl as ctl
import qpsolvers as qps
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


def opt_unc_gains(A, B, C, n_pred, n_ctl, rw):
    r"""Computes the optimum gains `Ky` and `K_mpc` for the unconstrained
    closed-loop system.

    Parameters
    ----------
    A : :class:`np.array`
        :math:`A` matrix. An (n, n) numpy matrix.

    B : :class:`np.array`
        :math:`B` matrix. An (n + q, m) numpy matrix.

    C : :class:`np.array`
        :math:`C_m` matrix. A (q, n) numpy matrix.

    n_pred : :class:`int`
        Length of prediction horizon.

    n_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `n_ctl` is less than or equal to `n_pred`. For
        `n_ctl` less than `n_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `n_ctl = n_pred` is
        assumed.

    n_ctn : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints are enforced.
        Note that `n_ctn` is less than or equal to `n_ctl`. If set to `None`,
        `n_ctn = n_ctl` is assumed.
    
    rw : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weighting factor for the control inputs.If set to `None`, `rw` is set
        to zero.

    Returns
    -------
    (Ky, K_mpc) : :class:`tuple`
        A tuple, containing two elements. The first element is the vector
        `Ky` and the second element is the vector `K_mpc`.

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

    Rs_bar = reference_matrix(q, n_pred)

    R = control_weighting_matrix(rw, n_ctl)

    F, Phi = opt_matrices(A, B, C, n_pred, n_ctl)
    Phi_t = Phi.T
    
    K = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t
    K_mpc = K @ F
    Ky = K @ Rs_bar

    return (Ky[:m], K_mpc[:m, :])


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

    n_pred : :class:`int`
        Length of prediction horizon.

    n_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `n_ctl` is less than or equal to `n_pred`. For
        `n_ctl` less than `n_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `n_ctl = n_pred` is
        assumed.

    n_cnt : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints are enforced.
        Note that `n_cnt` is less than or equal to `n_ctl`. If set to `None`,
        `n_cnt = n_ctl` is assumed.
    
    rw : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weighting factor for the control inputs. If set to `None`, `rw` is set
        to zero.

    q : :class:`NoneType`,  :class:`int`, :class:`np.array`
        Weighting factor for frequency components of the control inputs.

    x_lim : :class:`NoneType`, :class:`np.array`, :class:`list`
        Lower and upper bounds of the states. If set to `None`, no constraints
        are imposed on the states. 
        
    u_lim : :class:`NoneType`, :class:`np.array`, :class:`list`
        Lower and upper bounds of the control signals. If set to`None`, no
        constraints and set on the input signals.

    """
    def __init__(self, Am, Bm, Cm, n_pred, n_ctl=None, n_cnt=None, rw=None, q=None, x_lim=None, u_lim=None):

        # System model and augmented model
        self.Am = Am; self.Bm = Bm; self.Cm = Cm
        
        self.A, self.B, self.C = aug(Am, Bm, Cm)

        # Prediction horizon
        self.n_pred = n_pred

        # Control horizon
        if n_ctl is None:
            self.n_ctl = n_pred
        else:
            self.n_ctl = n_ctl

        # Constraints horizon
        if n_cnt is None:
            self.n_cnt = self.n_ctl
        else:
            self.n_cnt = n_cnt

        # Weighting factor
        if Bm.ndim == 1:
            nu = 1
        else:
            nu = Bm.shape[1]
            
        if rw is None:
            rw = np.zeros(nu)
        
        elif type(rw) is int or type(rw) is float:
            rw = rw * np.ones(nu)

        self.rw = rw

        # Spectrum weighting factor
        if type(q) is None:
            q = 0.0
        
        if type(q) is int or type(q) is float:
            r_w = np.array([q])

        # Bounds
        if type(x_lim) is list:
            x_lim = np.array(x_lim)
        if type(u_lim) is list:
            u_lim = np.array(u_lim)
        
        self.x_lim = x_lim
        self.u_lim = u_lim

        # Gains for unconstrained problem
        n_xm = Am.shape[0]
        Ky, K_mpc = opt_unc_gains(self.A, self.B, self.C, \
                                  self.n_pred, self.n_ctl, self.rw)
        Kx = K_mpc[:, :n_xm]

        self.Ky = Ky
        self.Kx = Kx

        if (x_lim is not None) or (u_lim is not None):
            # Initializes static qp matrices
            self.gen_static_qp_matrices()

            # Creates Hildreth's static matrix        
            self.Hj = self.M @ self.Ej_inv @ self.M.T

    
    def gen_static_qp_matrices(self):
        """Sets constant matrices, to be used later by the optimization.

        """
        A = self.A; B = self.B; C = self.C
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        n_pred = self.n_pred; n_ctl = self.n_ctl; n_cnt = self.n_cnt
        rw = self.rw

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
        
        R_bar = control_weighting_matrix(rw, n_ctl)
        Rs_bar = reference_matrix(q, n_pred)

        self.R_bar = R_bar
        self.Rs_bar = Rs_bar

        # Creates left-hand side inequality constraint, starting first with
        # control inequality constraints
        M = None
        if u_lim is not None:
            M_aux = np.tril( np.tile( np.eye(m), (n_cnt, n_ctl) ) )
            Mu = np.concatenate((-M_aux, M_aux))
            M = Mu

        # Now, the state inequality constraints
        if x_lim is not None:
            n_state_ineq = 0
            x_lim_new = [[], []]
            Cx = None
            
            for i, xi in enumerate(x_lim[0]):
                if xi is not None:
                    x_lim_new[0].append(x_lim[0][i])
                    x_lim_new[1].append(x_lim[1][i])
                    
                    n_state_ineq = n_state_ineq + 1
                    
                    cx = np.zeros((1, Am.shape[0]))
                    cx[0, i] = 1
                    
                    if Cx is None:
                        Cx = cx
                    else:
                        Cx = np.concatenate((Cx, cx))
                        
            Fx, Phi_x = opt_matrices(Am, Bm, Cx, n_cnt, n_ctl)
            self.Fx = Fx; self.Phi_x = Phi_x
            self.Cx = Cx
            
            self.x_lim = np.array(x_lim_new)

            M_aux = np.tril( np.tile( np.eye(self.x_lim.shape[1]), (n_cnt, n_cnt) ) )
            self.Mx_aux = M_aux
            Mx = np.concatenate((-M_aux @ Phi_x, M_aux @ Phi_x))

            if M is None:
                M = Mx
            else:
                M = np.concatenate((M, Mx))

        self.M = M

        # QP matrices
        F, Phi = opt_matrices(A, B, C, n_pred, n_ctl)
        self.F = F; self.Phi = Phi

        Ej = Phi.T @ Phi + R_bar
        Ej_inv = np.linalg.inv(Ej)
        self.Ej = Ej; self.Ej_inv = Ej_inv


    def gen_dyn_qp_matrices(self, xm, dx, xa, ui, r):
        """Sets dynamic matrices, to be used later by the optimization.

        """
        n_cnt = self.n_cnt
        F = self.F; Phi = self.Phi
        Rs_bar = self.Rs_bar
       
        u_lim = self.u_lim
        x_lim = self.x_lim

        Fj = -Phi.T @ (Rs_bar @ r.reshape(-1, 1) - F @ xa.reshape(-1, 1))

        # Creates the right-hand side inequality vector, starting first with
        # the control inequality constraints
        y = None
        
        if u_lim is not None:
            u_min = np.tile(-u_lim[0] + ui, n_cnt).reshape(-1, 1)
            u_max = np.tile( u_lim[1] - ui, n_cnt).reshape(-1, 1)

            y = np.concatenate((u_min, u_max))

        # Now, the state inequality constraints
        if x_lim is not None:
            Mx_aux = self.Mx_aux
            Cx = self.Cx
            Fx = self.Fx
            M_Fx = Mx_aux @ Fx
            x_min = np.tile(-x_lim[0] + Cx @ xm, n_cnt).reshape(-1, 1) + M_Fx @ dx.reshape(-1, 1)
            x_max = np.tile( x_lim[1] - Cx @ xm, n_cnt).reshape(-1, 1) - M_Fx @ dx.reshape(-1, 1)

            if y is None:
                y = np.concatenate((x_min, x_max))
            else:
                y = np.concatenate((y, x_min, x_max))

        return (Fj, y)


    def opt(self, xm, dx, xa, ui, r, method='hild'):

        nu = ui.shape[0]
        
        Fj, y = self.gen_dyn_qp_matrices(xm, dx, xa, ui, r)

        du, n_iters = self.qp(Fj, y, method=method)

        return (du[:nu], n_iters)

    
    def qp(self, Fj, y, method='hild'):
        """Solver the QP problem given by:

        .. :math:

            J = \Delta U^T E_J \Delta U^T +  \Delta U^T F_j,

        subject to:

        .. :math:

            M \Delta U \leq y.
            
        """
        Ej = self.Ej; Ej_inv = self.Ej_inv

        if method == 'hild':
            M = self.M
            F = self.F; Phi = self.Phi
            
            Hj = self.Hj
            Kj = y + M @ Ej_inv @ Fj

            lm, n_iters = ctl.qp.hild(Hj, Kj, n_iter=250, ret_n_iter=True)
            lm = lm.reshape(-1, 1)
            du_opt = -Ej_inv @ (Fj + M.T @ lm)
            du_opt = du_opt.reshape(-1)

        elif method == 'cvx':
            du_opt = qps.cvxopt_solve_qp(Ej, Fj.reshape(-1), M, y.reshape(-1))
            n_iters = 0

        elif method == 'quadprog':
            du_opt = qps.solve_qp(Ej, Fj.reshape(-1), M, y.reshape(-1))
            n_iters = 0

        else:
            du_opt = 0
            n_iters = 0

        return (du_opt, n_iters)
    

    def sim(self, xi, ui, r, n, Bd=None, ud=None):
        """Simulates closed-loop system with the predictive controller.

        Parameters
        ----------
        xi : :class:`int`, :class:`float`, :class:`list`, :class:`np.array`
            Initial state conditions. Can be passed as a single value, list
            or array, where each element corresponds the initial condition
            of each state. If there are multiple states, and `xi` is a
            single element, or a list with a single element, the initial
            value of all states is set to `xi`.

        ui : :class:`np.array`
            Initial conditions for the control inputs. Can be passed as a
            single value, list or array, where each element corresponds the
            initial value of each input. If there are multiple states, and
            `xi` is a single element, or a list with a single element, the
            initial value of all states is set to `xi`.

        r : :class:`float`, :class:`np.array`
            The set-point. If a single value, it is assumed constant for the
            whole simulation. If a vector, each row is used at each step of
            the simulation.

        n : :class:`int`
            Number of points for the simulation.

        Bd : :class:`np.array`
            An (p, p) numpy matrix, where `p` is the number of disturbances.
            By default, it is `None`.

        ud : :class:`np.array`
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
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        A = self.A; B = self.B; C = self.C

        if type(xi) is int or type(xi) is float or type(xi) is list:
            xi = np.array(xi).reshape(-1)
        elif type(xi) is np.ndarray:
            xi = np.array(xi).reshape(-1)

        if type(ui) is int or type(ui) is float or type(ui) is list:
            ui = np.array(ui).reshape(-1)
        elif type(xi) is np.ndarray:
            ui = np.array(ui).reshape(-1)
            
        if type(r) is int or type(r) is float:
            r = r * np.ones((n, 1))
        elif type(r) is list:
            r = np.array(r)
        
        if type(r) is np.ndarray and r.ndim == 1:
            r = np.tile(r, (n, 1))
        
        n_xm = Am.shape[0]
        nx = A.shape[0]
        ny = C.shape[0]
        nu = B.shape[1]
        
        xm = np.zeros((n, n_xm))
        x = np.zeros((n, nx))
        y = np.zeros((n, ny))
        xa = np.zeros((A.shape[0], 1))

        u = np.zeros((n, nu))

        xm[0] = xi
        dx = xm[0]
        u[0] = ui

        Ky = self.Ky
        Kx = self.Kx

        if Bd is None:
            Bd = np.zeros(Bm.shape)
            ud = np.zeros(u.shape)
        else:
            if type(ud) is int or type(ud) is float or type(ud) is list:
                ud = np.array(ud)
            if ud.ndim == 1:
                ud = np.tile(ud, (n, 1))

        # Simulation
        for i in range(n - 1):
            # Updates the output and dx
            y[i] = Cm @ xm[i]
            dx = xm[i] - dx

            # Computes the control law for sampling instant i
            if (self.u_lim is None) and (self.x_lim is None):
                du = -Ky @ (y[i] - r[i]) + -Kx @ dx
            else:
                xa[:n_xm, 0] = dx
                xa[n_xm:, 0] = y[i]
                du, _ = self.opt(xm[i], dx, xa, u[i], r[i]) 
            
            u[i] = u[i] + du

            # Applies the control law
            xm[i + 1] = Am @ xm[i] + Bm @ u[i] + Bd @ ud[i]

            # Update variables for next iteration
            dx = xm[i]
            u[i + 1] = u[i]

        # Updates last value of y
        y[n - 1] = Cm @ xm[n - 1]

        results = {}
        results['u'] = u
        results['xm'] = xm
        results['y'] = y

        return results



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
        n_iters = np.zeros(n)

        u = np.zeros((n, n_u))

        x_m[0] = x_i[0][:n_xm]
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

            du, n_iters[i] = self.constr_model.opt(x_m[i], dx, xa, u[i], r[i])
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
        results['n_iters'] = n_iters
        
        return results


    def _export_np_array_to_c(self, arr, arr_name, fill=True):

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
            arr_txt = '{:}[{:}];'.format(arr_name, n)
        else:
            arr_txt = '{:}[{:}][{:}];'.format(arr_name, n, m)

        if fill is True:
            arr_txt = arr_txt[:-1] + ' = {:};'.format(arr_str)
            
        return arr_txt

    
    def _export_gen_header(self, scaling=1.0, Bd=None, ref='constant'):

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

        if ref == 'constant':
            Fj1 = -self.constr_model.Phi.T @ self.constr_model.R_s_bar
        else:
            Fj1 = -self.constr_model.Phi.T
        Fj2 = self.constr_model.Phi.T @ self.constr_model.F

        Kj1 = self.constr_model.M @ self.constr_model.E_j_inv

        if x_lim is None:
            Fxp = np.zeros((1,1))
        else:
            Fxp = self.constr_model.M_x_aux @ self.constr_model.F_x

        Ej = self.constr_model.E_j

        M = self.constr_model.M

        Hj = np.zeros(self.constr_model.H_j.shape, dtype=self.constr_model.H_j.dtype)
        Hj[:] = self.constr_model.H_j[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        #Hj_fxp = (Hj * (2 ** qbase)).astype(np.int64)
        
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
        include = '\n#include "stdint.h"\n'
        text = text + def_guard + include

        def_prefix = 'DMPC_CONFIG'

        if u_lim is not None:
            idx = []
            for i, xi in enumerate(u_lim[0]):
                if xi != None:
                    idx.append(i)
            idx = np.array(idx)

        u_lim_sz = idx.shape[0]
        u_min_text = self._export_np_array_to_c(u_lim[0] / scaling, 'extern float {:}_U_MIN'.format(def_prefix, u_lim_sz), fill=False) + '\n'
        u_max_text = self._export_np_array_to_c(u_lim[1] / scaling, 'extern float {:}_U_MAX'.format(def_prefix, u_lim_sz), fill=False) + '\n'
        x_lim_idx_text = self._export_np_array_to_c(idx, 'extern uint32_t {:}_U_LIM_IDX'.format(def_prefix, u_lim_sz), fill=False) + '\n'
            
        constraints = '\n/* Input constraints */\n'+\
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
            x_min_text = self._export_np_array_to_c(x_lim[0][idx] / scaling, 'extern float {:}_XM_MIN'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            x_max_text = self._export_np_array_to_c(x_lim[1][idx] / scaling, 'extern float {:}_XM_MAX'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            x_lim_idx_text = self._export_np_array_to_c(idx, 'extern uint32_t {:}_XM_LIM_IDX'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            
            constraints = '\n/* State constraints */\n'+\
                          x_min_text+\
                          x_max_text+\
                          x_lim_idx_text
        else:
           
            constraints = '\n/* State constraints */\n'+\
                          ' '
            
        text = text + constraints

        idx = []
        if Cm.ndim != 1:
            Cm = np.sum(Cm, axis=0)
        for i, yi in enumerate(Cm):
            if np.abs(yi) > 0.5: idx.append(i)
        idx = np.array(idx)
        outputs_sz = idx.shape[0]
        outputs_idx_text = self._export_np_array_to_c(idx, 'extern uint32_t {:}_Y_IDX'.format(def_prefix, outputs_sz), fill=False) + '\n'
        outs = '\n/* Indexes of outputs */\n'+\
               outputs_idx_text
        text = text + outs
        
        matrices_prefix = 'DMPC_M'
        A_text = self._export_np_array_to_c(Am, 'extern float {:}_A'.format(matrices_prefix), fill=False) + '\n'
        if Bd is not None:
            B = np.concatenate((Bm, Bd), axis=1)
        else:
            B = Bm
        B_text = self._export_np_array_to_c(B, 'extern float {:}_B'.format(matrices_prefix), fill=False) + '\n'
        
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
        ej = self._export_np_array_to_c(Ej, 'extern float {:}Ej'.format(matrices_prefix), fill=False) + '\n\n'
        fj = 'extern float {:}Fj[{:}];\n\n'.format(matrices_prefix, n_c * n_u)
        m = self._export_np_array_to_c(M, 'extern float {:}M'.format(matrices_prefix), fill=False) + '\n\n'
        gam = 'extern float {:}gam[{:}];\n'.format(matrices_prefix, n_lambda)
        text = text + matrices + ej + fj + m + gam
        
        matrices = '\n /* Matrices for Hildreth\'s QP procedure */\n'
        fj1 = self._export_np_array_to_c(Fj1, 'extern float {:}Fj_1'.format(matrices_prefix), fill=False) + '\n\n'
        fj2 = self._export_np_array_to_c(Fj2, 'extern float {:}Fj_2'.format(matrices_prefix), fill=False) + '\n\n'
        fxp = self._export_np_array_to_c(Fxp, 'extern float {:}Fx'.format(matrices_prefix), fill=False) + '\n\n'
        kj1 = self._export_np_array_to_c(Kj1, 'extern float {:}Kj_1'.format(matrices_prefix), fill=False) + '\n\n'
        hj = self._export_np_array_to_c(Hj, 'extern float {:}Hj'.format(matrices_prefix), fill=False) + '\n\n'
        #hj_fxp = self._export_np_array_to_c(Hj_fxp, 'int {:}Hj_fxp'.format(matrices_prefix)) + '\n\n'
        du1 = self._export_np_array_to_c(DU1, 'extern float {:}DU_1'.format(matrices_prefix), fill=False) + '\n\n'
        du2 = self._export_np_array_to_c(DU2, 'extern float {:}DU_2'.format(matrices_prefix), fill=False) + '\n\n'
        text = text + matrices + fj1 + fj2 + fxp + kj1 + hj + du1 + du2

        def_guard_end = '\n#endif /* DMPC_MATRICES_H_ */\n'
        text = text + def_guard_end

        return text


    def _export_gen_source(self, scaling=1.0, Bd=None, ref='constant'):

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

        if ref == 'constant':
            Fj1 = -self.constr_model.Phi.T @ self.constr_model.R_s_bar
        else:
            Fj1 = -self.constr_model.Phi.T
        Fj2 = self.constr_model.Phi.T @ self.constr_model.F

        Kj1 = self.constr_model.M @ self.constr_model.E_j_inv

        if x_lim is None:
            Fxp = np.zeros((1,1))
        else:
            Fxp = self.constr_model.M_x_aux @ self.constr_model.F_x

        Ej = self.constr_model.E_j

        M = self.constr_model.M

        Hj = np.zeros(self.constr_model.H_j.shape, dtype=self.constr_model.H_j.dtype)
        Hj[:] = self.constr_model.H_j[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        #Hj_fxp = (Hj * (2 ** qbase)).astype(np.int64)
        
        DU1 = (-self.constr_model.E_j_inv)[:n_u, :]
        DU2 = (-self.constr_model.E_j_inv @ self.constr_model.M.T)[:n_u, :]
        n_lambda = DU2.shape[1]

        text = ''

        header = '/**\n'\
         ' * @file dmpc_matrices.c\n'\
         ' * @brief Source with data to run the DMPC algorithm.\n'\
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

        inc = '\n#include \"dmpc_matrices.h\"\n'
        text = text + header + inc

        def_prefix = 'DMPC_CONFIG'

        if u_lim is not None:
            idx = []
            for i, xi in enumerate(u_lim[0]):
                if xi != None:
                    idx.append(i)
            idx = np.array(idx)

        u_lim_sz = idx.shape[0]
        u_min_text = self._export_np_array_to_c(u_lim[0] / scaling, 'float {:}_U_MIN'.format(def_prefix, u_lim_sz)) + '\n'
        u_max_text = self._export_np_array_to_c(u_lim[1] / scaling, 'float {:}_U_MAX'.format(def_prefix, u_lim_sz)) + '\n'
        x_lim_idx_text = self._export_np_array_to_c(idx, 'uint32_t {:}_U_LIM_IDX'.format(def_prefix, u_lim_sz)) + '\n'
            
        constraints = '\n/* Input constraints */\n'+\
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
            x_min_text = self._export_np_array_to_c(x_lim[0][idx] / scaling, 'float {:}_XM_MIN'.format(def_prefix, x_lim_sz)) + '\n'
            x_max_text = self._export_np_array_to_c(x_lim[1][idx] / scaling, 'float {:}_XM_MAX'.format(def_prefix, x_lim_sz)) + '\n'
            x_lim_idx_text = self._export_np_array_to_c(idx, 'uint32_t {:}_XM_LIM_IDX'.format(def_prefix, x_lim_sz)) + '\n'
            
            constraints = '\n/* State constraints */\n'+\
                          x_min_text+\
                          x_max_text+\
                          x_lim_idx_text
        else:
           
            constraints = '\n/* State constraints */\n'+\
                          ' '
            
        text = text + constraints

        idx = []
        if Cm.ndim != 1:
            Cm = np.sum(Cm, axis=0)
        for i, yi in enumerate(Cm):
            if np.abs(yi) > 0.5: idx.append(i)
        idx = np.array(idx)
        outputs_sz = idx.shape[0]
        outputs_idx_text = self._export_np_array_to_c(idx, 'uint32_t {:}_Y_IDX'.format(def_prefix, outputs_sz)) + '\n'
        outs = '\n/* Indexes of outputs */\n'+\
               outputs_idx_text
        text = text + outs
        
        matrices_prefix = 'DMPC_M'
        A_text = self._export_np_array_to_c(Am, 'float {:}_A'.format(matrices_prefix)) + '\n'
        if Bd is not None:
            B = np.concatenate((Bm, Bd), axis=1)
        else:
            B = Bm
        B_text = self._export_np_array_to_c(B, 'float {:}_B'.format(matrices_prefix)) + '\n'
        
        matrices ='\n/* A and B matrices for prediction */\n'+\
                  A_text+\
                  B_text
        text = text + matrices

        text = text + '\n/* Matrices for QP solvers */\n'
        matrices_prefix = 'DMPC_M_'
        ej = self._export_np_array_to_c(Ej, 'float {:}Ej'.format(matrices_prefix)) + '\n\n'
        fj = 'float {:}Fj[{:}];\n\n'.format(matrices_prefix, n_c * n_u)
        m = self._export_np_array_to_c(M, 'float {:}M'.format(matrices_prefix)) + '\n\n'
        gam = 'float {:}gam[{:}];\n'.format(matrices_prefix, n_lambda)
        text = text + ej + fj + m + gam
        
        matrices = '\n /* Matrices for Hildreth\'s QP procedure */\n'
        fj1 = self._export_np_array_to_c(Fj1, 'float {:}Fj_1'.format(matrices_prefix)) + '\n\n'
        fj2 = self._export_np_array_to_c(Fj2, 'float {:}Fj_2'.format(matrices_prefix)) + '\n\n'
        fxp = self._export_np_array_to_c(Fxp, 'float {:}Fx'.format(matrices_prefix)) + '\n\n'
        kj1 = self._export_np_array_to_c(Kj1, 'float {:}Kj_1'.format(matrices_prefix)) + '\n\n'
        hj = self._export_np_array_to_c(Hj, 'float {:}Hj'.format(matrices_prefix)) + '\n\n'
        #hj_fxp = self._export_np_array_to_c(Hj_fxp, 'int {:}Hj_fxp'.format(matrices_prefix)) + '\n\n'
        du1 = self._export_np_array_to_c(DU1, 'float {:}DU_1'.format(matrices_prefix)) + '\n\n'
        du2 = self._export_np_array_to_c(DU2, 'float {:}DU_2'.format(matrices_prefix)) + '\n'
        text = text + matrices + fj1 + fj2 + fxp + kj1 + hj + du1 + du2

        return text
    

    def _export_gen_defs(self, scaling=1.0, Bd=None, ref='constant'):

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

        if ref == 'constant':
            Fj1 = -self.constr_model.Phi.T @ self.constr_model.R_s_bar
        else:
            Fj1 = -self.constr_model.Phi.T
        Fj2 = self.constr_model.Phi.T @ self.constr_model.F

        Kj1 = self.constr_model.M @ self.constr_model.E_j_inv

        if x_lim is None:
            Fxp = np.zeros((1,1))
        else:
            Fxp = self.constr_model.M_x_aux @ self.constr_model.F_x

        Ej = self.constr_model.E_j

        M = self.constr_model.M

        Hj = np.zeros(self.constr_model.H_j.shape, dtype=self.constr_model.H_j.dtype)
        Hj[:] = self.constr_model.H_j[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        #Hj_fxp = (Hj * (2 ** qbase)).astype(np.int64)
        
        DU1 = (-self.constr_model.E_j_inv)[:n_u, :]
        DU2 = (-self.constr_model.E_j_inv @ self.constr_model.M.T)[:n_u, :]
        n_lambda = DU2.shape[1]
        
        text = ''

        header = '/**\n'\
         ' * @file dmpc_defs.h\n'\
         ' * @brief Header with definitions to aid the DMPC algorithm.\n'\
         ' *\n'\
         ' * This file is generated automatically and should not be modified.\n'\
         ' *\n'\
         ' *  Originally created on: 21.10.2022\n'\
         ' *      Author: mguerreiro\n'\
         ' */\n'
        
        text = text + header

        def_guard = '\n#ifndef DMPC_DEFS_H_\n'\
                    '#define DMPC_DEFS_H_\n'
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
        u_min_text = self._export_np_array_to_c(u_lim[0] / scaling, 'extern float {:}_U_MIN'.format(def_prefix, u_lim_sz), fill=False) + '\n'
        u_max_text = self._export_np_array_to_c(u_lim[1] / scaling, 'extern float {:}_U_MAX'.format(def_prefix, u_lim_sz), fill=False) + '\n'
        x_lim_idx_text = self._export_np_array_to_c(idx, 'extern uint32_t {:}_U_LIM_IDX'.format(def_prefix, u_lim_sz), fill=False) + '\n'
          
        constraints = '\n/* Input constraints */\n'+\
                      '#define {:}_NU_CTR\t\t{:}\n'.format(def_prefix, u_lim_sz)
        text = text + constraints

        if x_lim is not None:
            idx = []
            for i, xi in enumerate(x_lim[0]):
                if xi != None:
                    idx.append(i)
            idx = np.array(idx)

            x_lim_sz = idx.shape[0]
            x_min_text = self._export_np_array_to_c(x_lim[0][idx] / scaling, 'extern float {:}_XM_MIN'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            x_max_text = self._export_np_array_to_c(x_lim[1][idx] / scaling, 'extern float {:}_XM_MAX'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            x_lim_idx_text = self._export_np_array_to_c(idx, 'extern uint32_t {:}_XM_LIM_IDX'.format(def_prefix, x_lim_sz), fill=False) + '\n'
            
            constraints = '\n/* State constraints */\n'+\
                          x_min_text+\
                          x_max_text+\
                          x_lim_idx_text
        else:
           
            constraints = '\n/* State constraints */\n'+\
                          ' '
        if x_lim is not None:
            constraints = '\n/* State constraints */\n'+\
                          '#define {:}_NXM_CTR\t\t{:}\n'.format(def_prefix, x_lim_sz)
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
        outputs_idx_text = self._export_np_array_to_c(idx, 'uint32_t {:}_Y_IDX'.format(def_prefix, outputs_sz)) + '\n'

        def_guard_end = '\n#endif /* DMPC_DEFS_H_ */\n'
        text = text + def_guard_end

        return text

    
    def export(self, file_path='.', scaling=1.0, Bd=None, ref='constant'):
    
        #np.set_printoptions(floatmode='unique')
        #np.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(floatmode='unique', threshold=sys.maxsize)

        text_source = self._export_gen_source(scaling=scaling, Bd=Bd, ref=ref)
        text_header = self._export_gen_header(scaling=scaling, Bd=Bd, ref=ref)
        text_defs   = self._export_gen_defs(scaling=scaling, Bd=Bd, ref=ref)
        if file_path is not None:
            with open(file_path + 'dmpc_matrices.c', 'w') as efile:
                efile.write(text_source)
            with open(file_path + 'dmpc_matrices.h', 'w') as efile:
                efile.write(text_header)
            with open(file_path + 'dmpc_defs.h', 'w') as efile:
                efile.write(text_defs)
                
        #print(text)

        np.set_printoptions(floatmode='fixed', threshold=1000)
        #np.set_printoptions(threshold=1000)
        #np.set_printoptions(floatmode='maxprec_equal')
