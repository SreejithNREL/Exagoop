#ifndef PARTICLEBASIS_H_
#define PARTICLEBASIS_H_
// clang-format off
#include<constants.H>
#include <AMReX.H>
#include<mpm_particle_container.H>
// clang-format on
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
spherical_gaussian(amrex::Real xi[AMREX_SPACEDIM],
                   amrex::Real xp[AMREX_SPACEDIM],
                   amrex::Real r0)
{
    amrex::Real r2 = 0.0;
#if (AMREX_SPACEDIM == 1)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]);
    // Normalization factor in 1D: (π^(-0.5)) * r0^(-1)
    return std::exp(-r2 / (r0 * r0)) * std::pow(PI, -0.5) * std::pow(r0, -1.0);

#elif (AMREX_SPACEDIM == 2)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]);
    // Normalization factor in 2D: (π^(-1)) * r0^(-2)
    return std::exp(-r2 / (r0 * r0)) * std::pow(PI, -1.0) * std::pow(r0, -2.0);

#elif (AMREX_SPACEDIM == 3)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]) +
         (xi[2] - xp[2]) * (xi[2] - xp[2]);
    // Normalization factor in 3D: (π^(-1.5)) * r0^(-3)
    return std::exp(-r2 / (r0 * r0)) * std::pow(PI, -1.5) * std::pow(r0, -3.0);
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
box_kernel(amrex::Real xi[AMREX_SPACEDIM],
           amrex::Real xp[AMREX_SPACEDIM],
           amrex::Real r0)
{
    amrex::Real r2 = 0.0;
#if (AMREX_SPACEDIM == 1)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]);

#elif (AMREX_SPACEDIM == 2)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]);

#elif (AMREX_SPACEDIM == 3)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]) +
         (xi[2] - xp[2]) * (xi[2] - xp[2]);
#endif

    return (r2 < r0 * r0) ? 1.0 : 0.0;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
levelset(amrex::Real xi[AMREX_SPACEDIM],
         amrex::Real xp[AMREX_SPACEDIM],
         amrex::Real r0,
         amrex::Real minval = -1.0,
         amrex::Real maxval = 1.0)
{
    amrex::Real r2 = 0.0;

#if (AMREX_SPACEDIM == 1)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]);

#elif (AMREX_SPACEDIM == 2)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]);

#elif (AMREX_SPACEDIM == 3)
    r2 = (xi[0] - xp[0]) * (xi[0] - xp[0]) +
         (xi[1] - xp[1]) * (xi[1] - xp[1]) +
         (xi[2] - xp[2]) * (xi[2] - xp[2]);
#endif

    amrex::Real f1 = (r2 < r0 * r0) ? minval : std::sqrt(r2);
    amrex::Real f2 = (f1 > 3.0 * r0) ? maxval : f1;

    return f2;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
hat1d(amrex::Real xi, amrex::Real xp, amrex::Real hatsize)
{
    amrex::Real zeta = xi - xp;
    amrex::Real halfhatsize = hatsize * HALF_CONST;
    amrex::Real funcval = zero;
    amrex::Real maxval = one / halfhatsize;
    amrex::Real slope = four / (hatsize * hatsize);

    if (amrex::Math::abs(zeta) < halfhatsize)
    {
        if (zeta > 0.0)
        {
            funcval = -slope * zeta + maxval;
        }
        else
        {
            funcval = slope * zeta + maxval;
        }
    }
    return (funcval);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
hatND(amrex::Real xi[AMREX_SPACEDIM],
      amrex::Real xp[AMREX_SPACEDIM],
      amrex::Real hatsize[AMREX_SPACEDIM])
{
    amrex::Real funcval = one;

#if (AMREX_SPACEDIM == 1)
    funcval *= hat1d(xi[0], xp[0], hatsize[0]);

#elif (AMREX_SPACEDIM == 2)
    funcval *= hat1d(xi[0], xp[0], hatsize[0]);
    funcval *= hat1d(xi[1], xp[1], hatsize[1]);

#elif (AMREX_SPACEDIM == 3)
    funcval *= hat1d(xi[0], xp[0], hatsize[0]);
    funcval *= hat1d(xi[1], xp[1], hatsize[1]);
    funcval *= hat1d(xi[2], xp[2], hatsize[2]);
#endif

    return funcval;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
bilin_interp(amrex::Real xp[AMREX_SPACEDIM],
		 amrex::IntVect iv,
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
         amrex::Array4<amrex::Real> nodaldata,
         int comp)
{
    amrex::Real value = zero;

#if (AMREX_SPACEDIM == 1)
    // Linear interpolation in 1D
    int i=iv[XDIR];
    amrex::Real alpha = (xp[0] - (plo[0] + i * dx[0])) / dx[0];
    value = (1.0 - alpha) * nodaldata(i, 0, 0, comp)
          + alpha * nodaldata(i + 1, 0, 0, comp);

#elif (AMREX_SPACEDIM == 2)
    // Bilinear interpolation in 2D
    int i=iv[XDIR];
    int j=iv[YDIR];
    amrex::Real alpha = (xp[0] - (plo[0] + i * dx[0])) / dx[0];
    amrex::Real beta  = (xp[1] - (plo[1] + j * dx[1])) / dx[1];

    value  = (1.0 - alpha) * (1.0 - beta) * nodaldata(i,     j,     0, comp);
    value += alpha         * (1.0 - beta) * nodaldata(i + 1, j,     0, comp);
    value += (1.0 - alpha) * beta         * nodaldata(i,     j + 1, 0, comp);
    value += alpha         * beta         * nodaldata(i + 1, j + 1, 0, comp);

//amrex::Print()<<"\n Value values in bilin_interp: "<<value;


#elif (AMREX_SPACEDIM == 3)
    // Trilinear interpolation in 3D (your original code)
    int i=iv[XDIR];
    int j=iv[YDIR];
    int k=iv[ZDIR];

    amrex::Real alpha = (xp[0] - (plo[0] + i * dx[0])) / dx[0];
    amrex::Real beta  = (xp[1] - (plo[1] + j * dx[1])) / dx[1];
    amrex::Real gama  = (xp[2] - (plo[2] + k * dx[2])) / dx[2];

    value  = (1.0 - alpha) * (1.0 - beta) * (1.0 - gama) * nodaldata(i,     j,     k,     comp);
    value += alpha         * (1.0 - beta) * (1.0 - gama) * nodaldata(i + 1, j,     k,     comp);
    value += (1.0 - alpha) * beta         * (1.0 - gama) * nodaldata(i,     j + 1, k,     comp);
    value += alpha         * beta         * (1.0 - gama) * nodaldata(i + 1, j + 1, k,     comp);

    value += (1.0 - alpha) * (1.0 - beta) * gama         * nodaldata(i,     j,     k + 1, comp);
    value += alpha         * (1.0 - beta) * gama         * nodaldata(i + 1, j,     k + 1, comp);
    value += (1.0 - alpha) * beta         * gama         * nodaldata(i,     j + 1, k + 1, comp);
    value += alpha         * beta         * gama         * nodaldata(i + 1, j + 1, k + 1, comp);
#endif

    return value;
}

/*
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
bilin_interp(amrex::Real xp[AMREX_SPACEDIM], amrex::IntVect iv,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
             amrex::Array4<amrex::Real> nodaldata, int comp) {
	int i,j,k;
	i=iv[0];
	j=iv[1];
	k=iv[2];


  amrex::Real alpha, beta, gama;
  amrex::Real value = zero;

  alpha = (xp[XDIR] - (plo[XDIR] + i * dx[XDIR])) / dx[XDIR];
  beta = (xp[YDIR] - (plo[YDIR] + j * dx[YDIR])) / dx[YDIR];
  gama = (xp[ZDIR] - (plo[ZDIR] + k * dx[ZDIR])) / dx[ZDIR];
  gama=0.0;

  value += (1.0 - alpha) * (1 - beta) * (1 - gama) * nodaldata(i, j, k, comp);
  value += alpha * (1 - beta) * (1 - gama) * nodaldata(i + 1, j, k, comp);
  value += (1.0 - alpha) * beta * (1 - gama) * nodaldata(i, j + 1, k, comp);
  value += alpha * beta * (1 - gama) * nodaldata(i + 1, j + 1, k, comp);

  value += (1.0 - alpha) * (1 - beta) * gama * nodaldata(i, j, k + 1, comp);
  value += alpha * (1 - beta) * gama * nodaldata(i + 1, j, k + 1, comp);
  value += (1.0 - alpha) * beta * gama * nodaldata(i, j + 1, k + 1, comp);
  value += alpha * beta * gama * nodaldata(i + 1, j + 1, k + 1, comp);

  return (value);
}
*/


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
cubicspline_1d_der(int shapefunctiontype, amrex::Real zi)
{
    amrex::Real value = 0.0;
    if (shapefunctiontype == 1) // Boundary node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi / 2.0 + 2 * zi + 2.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi / 2.0 + 1.0;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi / 2.0 - 1.0;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi / 2.0 + 2 * zi - 2.0;
        }
    }
    else if (shapefunctiontype == 2) // Near Boundary node
    {
        if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi - 2.0 * zi;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = 3.0 * zi * zi / 2.0 - 2.0 * zi;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi / 2.0 + 2 * zi - 2.0;
        }
    }
    else if (shapefunctiontype == 3) // Interior node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi / 2.0 + 2 * zi + 2.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -3.0 * zi * zi / 2.0 - 2.0 * zi;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = 3.0 * zi * zi / 2.0 - 2.0 * zi;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi / 2.0 + 2 * zi - 2.0;
        }
    }
    else if (shapefunctiontype == 4) // Interior node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi / 2.0 + 2 * zi + 2.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -3.0 * zi * zi / 2.0 - 2.0 * zi;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi - 2.0 * zi;
        }
    }
    else
    {
    	amrex::Abort("\n Incorrect shape function type in cubicspline_1d_der");
    }
    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
cubicspline_1d(int shapefunctiontype, amrex::Real zi, int dir)
{
    amrex::Real value = 0.0;
    if (shapefunctiontype == 1) // Boundary node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi * zi / 6.0 + zi * zi + 2.0 * zi + 4.0 / 3.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi * zi / 6.0 + zi + 1.0;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi * zi / 6.0 - zi + 1.0;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi * zi / 6.0 + zi * zi - 2.0 * zi + 4.0 / 3.0;
        }
        else
        {
#ifndef AMREX_USE_GPU
            amrex::Print() << "\n Error in shape func 1: " << zi;
#endif
        }
    }
    else if (shapefunctiontype == 2) // Near Boundary node
    {
        if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi * zi / 3.0 - zi * zi + 2.0 / 3.0;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi * zi / 2.0 - zi * zi + 2.0 / 3.0;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi * zi / 6.0 + zi * zi - 2.0 * zi + 4.0 / 3.0;
        }
        else
        {
#ifndef AMREX_USE_GPU
            amrex::Print() << "\n Error in shape func 2: " << zi
                           << " dir = " << dir;
#endif
        }
    }
    else if (shapefunctiontype == 3) // Interior node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi * zi / 6.0 + zi * zi + 2.0 * zi + 4.0 / 3.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi * zi / 2.0 - zi * zi + 2.0 / 3.0;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi * zi / 2.0 - zi * zi + 2.0 / 3.0;
        }
        else if (zi >= 1 && zi <= 2.0)
        {
            value = -zi * zi * zi / 6.0 + zi * zi - 2.0 * zi + 4.0 / 3.0;
        }
        else
        {
#ifndef AMREX_USE_GPU
            amrex::Print() << "\n Error in shape func 3: " << zi;
#endif
        }
    }
    else if (shapefunctiontype == 4) // Interior node
    {
        if (zi >= -2 && zi <= -1)
        {
            value = zi * zi * zi / 6.0 + zi * zi + 2.0 * zi + 4.0 / 3.0;
        }
        else if (zi >= -1 && zi <= 0)
        {
            value = -zi * zi * zi / 2.0 - zi * zi + 2.0 / 3.0;
        }
        else if (zi >= 0 && zi <= 1)
        {
            value = zi * zi * zi / 3.0 - zi * zi + 2.0 / 3.0;
        }
        else
        {
#ifndef AMREX_USE_GPU
            amrex::Print() << "\n Error in shape func 4: " << zi
                           << " dir= " << dir;
#endif
        }
    }
    else
    {
#ifndef AMREX_USE_GPU
        amrex::Print() << "\n Shapefunction = " << shapefunctiontype;
#endif
        amrex::Abort("\n Incorrect shape function type in cubicspline_1d");
    }
    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
cubic_interp(amrex::Real xp[AMREX_SPACEDIM],
			 IntVect iv,
			 IntVect min_index,
			 IntVect max_index,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
             amrex::Array4<amrex::Real> nodaldata,
             int comp,
             const int *lo,
             const int *hi)
{
    amrex::Real value = zero;


#if (AMREX_SPACEDIM == 1)

				 int i = iv[XDIR];
                 int lmin = min_index[XDIR];
                 int lmax = max_index[XDIR];

    for (int l = lmin; l < lmax; ++l) {
        int shapetypex;
        if (i + l == lo[0]) shapetypex = 1;
        else if (i + l == lo[0] + 1) shapetypex = 2;
        else if (i + l == hi[0] + 1) shapetypex = 1;
        else if (i + l == hi[0]) shapetypex = 4;
        else shapetypex = 3;

        amrex::Real rx = (xp[0] - (plo[0] + (i + l) * dx[0])) / dx[0];
        amrex::Real lval = cubicspline_1d(shapetypex, rx, 0);
        value += lval * nodaldata(i + l, 0, 0, comp);
    }

#elif (AMREX_SPACEDIM == 2)
    int i = iv[XDIR];
    int lmin = min_index[XDIR];
    int lmax = max_index[XDIR];
    int j = iv[YDIR];
    int mmin = min_index[YDIR];
    int mmax = max_index[YDIR];

    for (int m = mmin; m < mmax; ++m) {
        for (int l = lmin; l < lmax; ++l) {
            int shapetypex, shapetypey;
            // X direction
            if (i + l == lo[0]) shapetypex = 1;
            else if (i + l == lo[0] + 1) shapetypex = 2;
            else if (i + l == hi[0] + 1) shapetypex = 1;
            else if (i + l == hi[0]) shapetypex = 4;
            else shapetypex = 3;
            // Y direction
            if (j + m == lo[1]) shapetypey = 1;
            else if (j + m == lo[1] + 1) shapetypey = 2;
            else if (j + m == hi[1] + 1) shapetypey = 1;
            else if (j + m == hi[1]) shapetypey = 4;
            else shapetypey = 3;

            amrex::Real rx = (xp[0] - (plo[0] + (i + l) * dx[0])) / dx[0];
            amrex::Real ry = (xp[1] - (plo[1] + (j + m) * dx[1])) / dx[1];

            amrex::Real lval = cubicspline_1d(shapetypex, rx, 0);
            amrex::Real mval = cubicspline_1d(shapetypey, ry, 1);

            value += lval * mval * nodaldata(i + l, j + m, 0, comp);
        }
    }

#elif (AMREX_SPACEDIM == 3)
    int i = iv[XDIR];
    int lmin = min_index[XDIR];
    int lmax = max_index[XDIR];
    int j = iv[YDIR];
    int mmin = min_index[YDIR];
    int mmax = max_index[YDIR];
    int k = iv[ZDIR];
    int nmin = min_index[ZDIR];
    int nmax = max_index[ZDIR];
    for (int n = nmin; n < nmax; ++n) {
        for (int m = mmin; m < mmax; ++m) {
            for (int l = lmin; l < lmax; ++l) {
                int shapetypex, shapetypey, shapetypez;
                // X direction
                if (i + l == lo[0]) shapetypex = 1;
                else if (i + l == lo[0] + 1) shapetypex = 2;
                else if (i + l == hi[0] + 1) shapetypex = 1;
                else if (i + l == hi[0]) shapetypex = 4;
                else shapetypex = 3;
                // Y direction
                if (j + m == lo[1]) shapetypey = 1;
                else if (j + m == lo[1] + 1) shapetypey = 2;
                else if (j + m == hi[1] + 1) shapetypey = 1;
                else if (j + m == hi[1]) shapetypey = 4;
                else shapetypey = 3;
                // Z direction
                if (k + n == lo[2]) shapetypez = 1;
                else if (k + n == lo[2] + 1) shapetypez = 2;
                else if (k + n == hi[2] + 1) shapetypez = 1;
                else if (k + n == hi[2]) shapetypez = 4;
                else shapetypez = 3;

                amrex::Real rx = (xp[0] - (plo[0] + (i + l) * dx[0])) / dx[0];
                amrex::Real ry = (xp[1] - (plo[1] + (j + m) * dx[1])) / dx[1];
                amrex::Real rz = (xp[2] - (plo[2] + (k + n) * dx[2])) / dx[2];

                amrex::Real lval = cubicspline_1d(shapetypex, rx, 0);
                amrex::Real mval = cubicspline_1d(shapetypey, ry, 1);
                amrex::Real nval = cubicspline_1d(shapetypez, rz, 2);

                value += lval * mval * nval * nodaldata(i + l, j + m, k + n, comp);
            }
        }
    }
#endif

    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
basisval(const IntVect stencil,
         const IntVect cell,
         const amrex::Real xp[AMREX_SPACEDIM],
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& plo,
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
         const GpuArray<int, AMREX_SPACEDIM>& order_scheme_directional,
         const GpuArray<int, AMREX_SPACEDIM>& periodic,
         const int* lo,
         const int* hi)
{
    amrex::Real contrib[AMREX_SPACEDIM];
    // Directional indices assumed to be 0,1,2 for XDIR,YDIR,ZDIR
    // and consistent with IntVect indexing.

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const int i  = cell[d];
        const int l  = stencil[d];
        const int od = order_scheme_directional[d];

        // Local coordinate in cell i along dimension d
        amrex::Real r = (xp[d] - (plo[d] + i * dx[d])) / dx[d];

        if (od == 1) {
            // Linear shape in this direction: choose left/right half
            // l==0 refers to the "left" node, l==1 the "right" node
            contrib[d] = (l == 0) ? (one - r) : r;
        } else {
            // Higher-order (cubic spline) shape; compute type based on boundary/periodicity.
            int shapefunctype = 3; // interior default
            const int idx = i + l;

            if (periodic[d] == 0) {
                // Non-periodic: classify boundary shape
                // Types: 1 (ghost just outside lo/hi+1), 2 (near lo+1), 4 (near hi), 3 (interior)
                if (idx == lo[d] || idx == hi[d] + 1) {
                    shapefunctype = 1;
                } else if (idx == lo[d] + 1) {
                    shapefunctype = 2;
                } else if (idx == hi[d]) {
                    shapefunctype = 4;
                } else {
                    shapefunctype = 3;
                }
            } else {
                shapefunctype = 3; // periodic → interior shape
            }

            // Re-center r at the stencil location (i+l)
           r = (xp[d] - (plo[d] + (i + l) * dx[d])) / dx[d];
           

            contrib[d] = cubicspline_1d(shapefunctype, r, d);
        }
    }

    // Product over dimensions
    amrex::Real val = one;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        val *= contrib[d];
    }
    return val;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
basisvalder(const int dir,
            const IntVect stencil,
            const IntVect cell,
            const amrex::Real xp[AMREX_SPACEDIM],
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& plo,
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
            const GpuArray<int, AMREX_SPACEDIM>& order_scheme_directional,
            const GpuArray<int, AMREX_SPACEDIM>& periodic,
            const int* lo,
            const int* hi)
{
    amrex::Real dxinv[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        dxinv[d] = one / dx[d];
        //amrex::Print()<<"\n dxinv dx = "<<dxinv[d]<<" "<<dx[d];
    }

    // Product accumulator; we will multiply per-dimension contributions,
    // using the derivative in 'dir' and value factors in all other dims.
    amrex::Real prod = one;

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const int i  = cell[d];
        const int l  = stencil[d];
        const int od = order_scheme_directional[d];

        // Default: interior cubic shape
        int shapefunctype = 3;

        // Local coordinate r: for linear, centered at cell i; for cubic, at stencil (i+l)
        amrex::Real r;

        amrex::Real val_d = 0.0; // value contribution in dimension d
        amrex::Real der_d = 0.0; // derivative contribution in dimension d

        if (od == 1)
        {
            // Linear shape
            r = (xp[d] - (plo[d] + i * dx[d])) / dx[d];
            val_d = (l == 0) ? (one - r) : r;
            der_d = (l == 0) ? -dxinv[d] : dxinv[d];
        }
        else
        {
            // Cubic spline shape (order 3): classify boundary/periodic type
            const int idx = i + l;
            if (periodic[d] == 0) {
                if      (idx == lo[d]         || idx == hi[d] + 1) shapefunctype = 1;
                else if (idx == lo[d] + 1)                            shapefunctype = 2;
                else if (idx == hi[d])                                shapefunctype = 4;
                else                                                   shapefunctype = 3;
            } else {
                shapefunctype = 3; // periodic → interior
            }

            r = (xp[d] - (plo[d] + (i + l) * dx[d])) / dx[d];
            val_d = cubicspline_1d(shapefunctype, r, d);
            der_d = cubicspline_1d_der(shapefunctype, r) * dxinv[d];
        }

        // Multiply derivative in the requested direction; value otherwise
        prod *= (d == dir) ? der_d : val_d;
    }

    return prod;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
get_tensor(MPMParticleContainer::ParticleType &p,
           int start_index,
           amrex::Real tens[AMREX_SPACEDIM * AMREX_SPACEDIM])
{
    int ind = 0;
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        for (int j = i; j < AMREX_SPACEDIM; j++)
        {
            tens[i * AMREX_SPACEDIM + j] = p.rdata(start_index + ind);
            tens[j * AMREX_SPACEDIM + i] = p.rdata(start_index + ind);
            ind++;
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void get_deformation_gradient_tensor(
    MPMParticleContainer::ParticleType &p,
    int start_index,
    amrex::Real gradvp[AMREX_SPACEDIM][AMREX_SPACEDIM],
    amrex::Real dt)
{
    amrex::Real Lp[AMREX_SPACEDIM][AMREX_SPACEDIM];
    amrex::Real Fp[AMREX_SPACEDIM][AMREX_SPACEDIM];

    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        for (int j = 0; j < AMREX_SPACEDIM; j++)
        {
            if (i == j)
            {
                Lp[i][j] = 1.0 + gradvp[i][j] * dt;
            }
            else
            {
                Lp[i][j] = gradvp[i][j] * dt;
            }
        }
    }

    for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            for (int j = 0; j < AMREX_SPACEDIM; j++)
            {
            	Fp[i][j] = p.rdata(start_index + AMREX_SPACEDIM*i+j);

            }
        }

    // Perform matrix multiplication: Result = Lp * Fp
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
        	p.rdata(start_index + i*AMREX_SPACEDIM+j)=0.0;
            for (int k = 0; k < AMREX_SPACEDIM; ++k) {
            	p.rdata(start_index + i*AMREX_SPACEDIM+j) += Lp[i][k] * Fp[k][j];
            }
        }
    }
    
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
tensor_vector_pdt(amrex::Real tens[AMREX_SPACEDIM * AMREX_SPACEDIM],
                  amrex::Real vect[AMREX_SPACEDIM],
                  amrex::Real tensvect[AMREX_SPACEDIM])
{
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        tensvect[i] = zero;
        for (int j = 0; j < AMREX_SPACEDIM; j++)
        {
            tensvect[i] += tens[i * AMREX_SPACEDIM + j] * vect[j];
        }
    }
}

#endif
