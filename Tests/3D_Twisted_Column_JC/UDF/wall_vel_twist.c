/*
 * Rigid-body rotation velocity for the top surface of the twisted column.
 *
 * Nguyen et al., Sect. 10.3.3, Eq. 10.45:
 *     v_x(t) = -omega * y(t),   v_y(t) = +omega * x(t),   v_z = 0
 * where x(t), y(t) are the CURRENT coordinates of the boundary node, so the
 * imposed field is a rigid rotation about the z axis through the origin.
 *
 * The book sets omega = omega_0 * n / T with omega_0 = 2*pi rad/ms, n the
 * number of rotations and T the final time. The reference run is n = 3 over
 * T = 3 ms, hence omega = 2*pi rad/ms -- one full turn per millisecond.
 *
 * Units: mm, ms  =>  omega in rad/ms, velocity in mm/ms.
 */
#include <math.h>

#define OMEGA 6.283185307179586 /* 2*pi rad/ms */

void wall_vel_twist(double x, double y, double z, double t, double vel[3])
{
    (void)z;
    (void)t;
    vel[0] = -OMEGA * y;
    vel[1] = OMEGA * x;
    vel[2] = 0.0;
}
