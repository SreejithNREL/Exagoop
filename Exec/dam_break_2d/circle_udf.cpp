#include <cmath>

#ifdef _WIN32
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API __attribute__((visibility("default")))
#endif

extern "C" EXAGOOP_API double levelset_phi(double x, double y, double z)
{
    // Circle/sphere centred at (0.25, 0.07) with radius 0.05
    // phi < 0 inside obstacle, phi > 0 outside
    const double cx = 0.25;
    const double cy = 0.07;
    const double r  = 0.05;
    return std::sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy)) - r;
}
