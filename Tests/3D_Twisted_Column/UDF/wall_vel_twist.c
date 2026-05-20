#include <math.h>

/* Angular velocity omega_0 = 2*pi rad/ms (Section 10.3.3, Nguyen et al. 2023)
 * Applied to the top face of the column. Velocity prescription follows Eq. 10.45:
 *   vx = -omega * y(t),  vy = +omega * x(t)
 * where x(t), y(t) are the current node positions (passed in as x, y).
 *
 * Nodes outside the column cross-section (|x|>W or |y|>W, W=5 mm) receive
 * zero velocity so that grid-buffer nodes do not contaminate particle
 * velocities through the B-spline shape function support. */
#define OMEGA (2.0 * M_PI)
#define W     (5.0)

void wall_vel_twist(double x, double y, double z, double t, double vel[3])
{
    if (x < -W || x > W || y < -W || y > W) {
        vel[0] = 0.0;
        vel[1] = 0.0;
        vel[2] = 0.0;
        return;
    }
    vel[0] = -OMEGA * y;
    vel[1] =  OMEGA * x;
    vel[2] =  0.0;
}
