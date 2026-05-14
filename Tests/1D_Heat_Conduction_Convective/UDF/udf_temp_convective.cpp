#include <cmath>

extern "C"
void udf_temp_convective(double x, double y, double z, double t, double* out)
{
    (void)x; (void)y; (void)z;

    const double h0    = 2.0;
    const double Tinf0 = 0.5;
    const double omega = 2.0 * M_PI;

    out[0] = h0 * (1.0 + 0.5 * std::cos(omega * t));
    out[1] = Tinf0 * std::sin(omega * t);
}
