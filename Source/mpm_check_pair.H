#ifndef MPM_CHECK_PAIR_H_
#define MPM_CHECK_PAIR_H_
// clang-format off
#include <mpm_specs.H>
// clang-format on

struct CheckPair
{
    template <class P>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool operator()(const P& p1,
                                                        const P& p2) const
    {
        amrex::Real dsquared = 0.0;

        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Real diff = p1.pos(d) - p2.pos(d);
            dsquared += diff * diff;
        }

        amrex::Real search_radius =
            0.5 * (p1.rdata(realData::radius) + p2.rdata(realData::radius));


        return (dsquared <= 9.0 * search_radius * search_radius);
    }
};


#endif
