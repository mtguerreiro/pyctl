import numpy as np

def extract_settling_overshoot(
    data: np.ndarray,
    v1: float,
    v2: float,
    settling_percentage: float = 2
    ):
    """
    Extracts the settling time and overshoot from a given 1D numpy array.
    
    Parameters:
        data (np.ndarray): The response signal.
        v1 (float): Initial value of the signal.
        v2 (float): Steady-state value of the signal.
        settling_percentage (float): The percentage criteria for settling time (e.g., 2% or 5%).
        
    Returns:
        tuple: (settling_time_index, overshoot_percentage)
    """
    
    steady_state_value = v2
    threshold = (settling_percentage / 100) * abs(steady_state_value - v1)  # Settling tolerance band
    lower_bound = steady_state_value - threshold
    upper_bound = steady_state_value + threshold
    
    # Find the index where the signal settles within specified percentage and stays there
    settling_time_index = None
    for i in range(len(data)):
        if np.all((data[i:] >= lower_bound) & (data[i:] <= upper_bound)):
            settling_time_index = i
            break
    
    # Calculate overshoot percentage
    peak_value = np.min(data) if v2 < v1 else np.max(data)
    overshoot_percentage = abs( (peak_value - steady_state_value) / (steady_state_value - v1) ) * 100
    
    return settling_time_index, overshoot_percentage


def bw_2nd_order_sys(ts, os, Hz=False):

    zeta = -np.log(os/100) / np.sqrt( np.pi**2 + np.log(os/100)**2 )

    bw = 4 / ts / zeta * np.sqrt( (1-2*zeta**2) + np.sqrt(4*zeta**4 - 4*zeta**2 + 2) )
    
    if Hz is True:
        bw = bw / 2 / np.pi

    return bw
