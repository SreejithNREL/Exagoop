# user_temperature.py
import math

def temperature_field(x, y, z, t=0.0):
    """
    Example temperature field:
    A Gaussian hot spot centered at (0.5, 0.5).
    """
    xc, yc = 0.5, 0.5
    sigma = 0.1

    r2 = (x - xc)**2 + (y - yc)**2
    T = 300.0 + 200.0 * math.exp(-r2 / (2 * sigma * sigma))

    spheat = 1000.0
    thermcond = 0.5
    heatsrc = 0.0

    return T, spheat, thermcond, heatsrc

