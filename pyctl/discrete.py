import numpy as np
import pyctl as ctl


class System:

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
