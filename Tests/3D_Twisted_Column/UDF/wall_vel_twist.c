#include <math.h>

#define OMEGA 0.5

void wall_vel_twist(double x, double y, double z, double t, double vel[3])
{
    vel[0] = -OMEGA * y;
    vel[1] =  OMEGA * x;
    vel[2] =  0.0;
}
