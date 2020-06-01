import numpy as np
import pyctl as ctl


class System:
    """A class to create a discrete-time model from a continuous-time model.

    Parameters
    ----------
    A : :class:`np.array`
        Continuous-time model matrix :math:`A`. An (n, n) numpy matrix.
    
    B : :class:`np.array`
        Continuous-time model matrix :math:`B`. An (n, 1) numpy matrix.
    
    C : :class:`np.array`
        Continuous-time model matrix :math:`C`. An (1, n) numpy matrix.

    dt : :class:`float`, :class:`NoneType`
        Discretization time. If `None`, the system is considered to already
        be a discrete-time model. By default, it is `None`.

    Attributes
    ----------
    Ad : :class:`np.array`
        Discrete-time model matrix :math:`A_d`. An (n, n) numpy matrix.
    
    Bd : :class:`np.array`
        Discrete-time model matrix :math:`B_d`. An (n, n) numpy matrix.
    
    Cd : :class:`np.array`
        Discrete-time model matrix :math:`C_d`. An (n, n) numpy matrix.

    dt : :class:`float`, :class:`NoneType`
        Discretization time.
    
    """
    def __init__(self, A, B, C, dt=None):

        n = A.shape[0]
        self.dt = dt
        
        if dt is None:
            self.Ad = A
            self.Bd = B
            self.Cd = C
        else:
            self.Ad = np.eye(n) + dt * A
            self.Bd = dt * B
            self.Cd = C

    
    def discrete_model(self):
        """Returns the discrete-time model matrices.

        Returns
        -------
        (Ad, Bd, Cd) : :class:`tuple`
            A tuple containing the :math:`A_d`, :math:`B_d` and :math:`C_d`
            matrices.
        
        """
        return (self.Ad, self.Bd, self.Cd)

    
    def sim(self, x_i, r, n):

        if type(r) is float or type(r) is int:
            r = np.array([r])
        
        Ad, Bd, Cd = self.discrete_model()
        n_x = Ad.shape[0]
        n_y = Cd.shape[0]

        x = np.zeros((n, n_x))
        y = np.zeros((n, n_y))
        
        x[0, :] = x_i
        y[0, :] = Cd @ x_i
        
        for i in range(n - 1):
            x[i + 1] = Ad @ x[i] + Bd @ r
            y[i] = Cd @ x[i]

        # Updates last value of y
        y[n - 1] = Cd @ x[n - 1]

        return (x, y)

    
    def p_sim(self, x_i, r, k_p, n):

        if type(r) is float or type(r) is int:
            r = np.array([r])
        
        Ad, Bd, Cd = self.discrete_model()
        n_x = Ad.shape[0]
        n_y = Cd.shape[0]

        x = np.zeros((n, n_x))
        y = np.zeros((n, n_y))
        
        x[0, :] = x_i
        y[0, :] = Cd @ x_i
        
        for i in range(n - 1):
            # Measures the output
            y[i] = Cd @ x[i]
            
            # Computes control action
            e = r - y[i]
            u = k_p * e

            # Aplies control action
            x[i + 1] = Ad @ x[i] + Bd @ u

        # Updates last value of y
        y[n - 1] = Cd @ x[n - 1]

        return (x, y)
