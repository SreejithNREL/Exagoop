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

#define OMEGA 6.283185307179586 /* 2*pi rad/ms (steady value) */

/*
 * Ramp: omega(t) = OMEGA * min(t / T_RAMP, 1).
 *
 * An impulsive start (omega = OMEGA from t = 0) launches a PLASTIC torsional
 * wave down the column at only ~120-200 mm/ms (sqrt(H_gamma/rho) with the
 * Johnson-Cook tangent modulus), so for the ~0.5 ms the front needs to reach
 * the base the whole imposed rotation is absorbed in the yielded region near
 * the grip: the twist is top-heavy, the top cell overheats, and a band of
 * near-perfectly-plastic material forms below the grip. Ramping omega over
 * T_RAMP (the front transit time) lets the twist rate become uniform along
 * the column before large rotations accumulate, which is the state the
 * book's figures show. Total rotation at t = 3 ms is then 2.75 turns
 * (compare with the book at equal rotation angle, or run to 3.25 ms).
 * Set T_RAMP = 0 to recover the impulsive start.
 */
#define T_RAMP 0.5 /* ms */

void wall_vel_twist(double x, double y, double z, double t, double vel[3])
{
    (void)z;
    double omega = OMEGA;
    if (T_RAMP > 0.0 && t < T_RAMP)
        omega = OMEGA * (t / T_RAMP);
    vel[0] = -omega * y;
    vel[1] = omega * x;
    vel[2] = 0.0;
}
