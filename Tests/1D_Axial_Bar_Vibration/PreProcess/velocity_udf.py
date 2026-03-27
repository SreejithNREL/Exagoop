# user_velocity.py
import math
import numpy as np

def velocity_field(x, y, z, t=0.0):
    """
    Example velocity field:    
    For 1D or 3D, unused components are ignored.
    """  
    L = 25 
    n = 1
    beta_n = (2*n - 1) * np.pi / (2*L) 
    v0 = 0.1

    vx = v0 * np.sin(beta_n * x)
    vy = 0.0
    vz = 0.0

    return vx, vy, vz
