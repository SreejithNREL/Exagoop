#include <cmath>

extern "C"
void udf_temp_heatflux(double x, double y, double z, double t, double* out)
{
    (void)x; (void)y; (void)z;

    const double q0    = 1.0;
    const double omega = 2.0 * M_PI;

    out[0] = q0 * (1.0 + std::sin(omega * t));
    out[1] = 0.0;
}
