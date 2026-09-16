"""Initial velocity field for the inertial compression test.

Uniaxial compression about the block centre x_c at nominal strain rate
EPS_DOT: v_x = -EPS_DOT * (x - x_c), v_y = v_z = 0. The generator calls
vel(x, y, z) for every particle.
"""
X_CENTRE = 0.5
EPS_DOT = 2.0


def vel(x, y, z):
    return -EPS_DOT * (x - X_CENTRE), 0.0, 0.0
