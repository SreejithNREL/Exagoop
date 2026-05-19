#include <cmath>

extern "C"
void udf_temp_convective(double x, double y, double z, double t, double* out)
{
    (void)x; (void)y; (void)z;

    const double h0    = 2.0;
    const double Tinf0 = 0.0;

    out[0] = h0 ;
    out[1] = Tinf0;
}
