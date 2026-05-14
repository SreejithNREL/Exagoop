#include <cmath>

extern "C"
void udf_temp_dirichlet(double x, double y, double z, double t, double* out)
{
    (void)y; (void)z;

    const double L     = 1.0;
    const double T_amp = 1.0;
    const double omega = 2.0 * M_PI;

    out[0] = T_amp * std::sin(M_PI * x / L) * std::cos(omega * t);
    out[1] = 0.0;
}
