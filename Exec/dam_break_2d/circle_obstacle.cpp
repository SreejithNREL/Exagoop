#include <cmath>

#ifdef EXAGOOP_UDF_WINDOWS_EXPORT
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API
#endif

extern "C" EXAGOOP_API double levelset_phi(double x, double y, double z)
{
    const double cx = 0.25;
    const double cy = 0.07;
    const double r  = 0.05;   // radius 0.05m — see note below
    return (std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) - r);
}
