"""
Reader for ExaGOOP particle plotfiles (AMReX binary particle format,
"Version_Two_Dot_Zero_double"), so large runs need no ASCII dumps.

    from exagoop_plotfile import read_particle_plotfile, plotfile_step
    data, fields = read_particle_plotfile("Solution/particle_files/<tag>/plt000129")
    # data: (N, ncols) float64 array with the SAME column layout as the ASCII
    #       matpnt dumps: pos(dim) | reals... | id | cpu | ints...
    # fields: name -> column index (as exagoop_columns.build_field_dict)

Layout per grid chunk in Level_0/DATA_xxxxx: `count` particles, all int data
first (int32: id, cpu, then the extra ints) followed by all real data
(float64: positions, then the extra reals).
"""
import os, glob
import numpy as np


def _read_header(path):
    with open(path) as f:
        lines = [l.strip() for l in f]
    it = iter(lines)
    version = next(it); dim = int(next(it))
    nreal = int(next(it)); real_names = [next(it) for _ in range(nreal)]
    nint = int(next(it)); int_names = [next(it) for _ in range(nint)]
    is_chk = int(next(it)); npart = int(next(it)); nextid = int(next(it)); finest = int(next(it))
    grids = []
    for lev in range(finest + 1):
        ng = int(next(it)); g = []
        for _ in range(ng):
            w, c, o = next(it).split(); g.append((int(w), int(c), int(o)))
        grids.append(g)
    assert "double" in version, version
    return dim, real_names, int_names, npart, grids


def read_particle_plotfile(pltdir):
    pdir = os.path.join(pltdir, "particles")
    dim, real_names, int_names, npart, grids = _read_header(os.path.join(pdir, "Header"))
    nint_tot = 2 + len(int_names); nreal_tot = dim + len(real_names)
    ncols = nreal_tot + nint_tot
    out = np.empty((npart, ncols)); k = 0
    for lev, g in enumerate(grids):
        files = {}
        for which, count, where in g:
            if count == 0: continue
            fn = os.path.join(pdir, f"Level_{lev}", f"DATA_{which:05d}")
            if fn not in files: files[fn] = open(fn, "rb")
            fh = files[fn]; fh.seek(where)
            ints = np.frombuffer(fh.read(4 * nint_tot * count), dtype=np.int32).reshape(count, nint_tot)
            reals = np.frombuffer(fh.read(8 * nreal_tot * count), dtype=np.float64).reshape(count, nreal_tot)
            out[k:k + count, :nreal_tot] = reals
            out[k:k + count, nreal_tot:] = ints
            k += count
        for fh in files.values(): fh.close()
    assert k == npart, (k, npart)
    fields = {f"pos{'xyz'[d]}": d for d in range(dim)}
    for i, n in enumerate(real_names): fields[n] = dim + i
    fields["id"] = nreal_tot; fields["cpu"] = nreal_tot + 1
    for i, n in enumerate(int_names): fields[n] = nreal_tot + 2 + i
    # convenience aliases matching exagoop_columns (first component of vectors)
    for base in ("stress", "strain", "strainrate", "deformation_gradient", "isv"):
        if f"{base}_0" in fields: fields[base] = fields[f"{base}_0"]
    return out, fields


def plotfile_step(pltdir):
    return int(os.path.basename(pltdir.rstrip("/"))[3:])


def plotfile_times(tag_dir, diag_file):
    """Map plt step numbers -> time using a Diagnostics file with 'iter,time,...' rows."""
    d = np.loadtxt(diag_file, skiprows=1)
    it, t = np.r_[0.0, d[:, 0]], np.r_[0.0, d[:, 1]]
    plts = sorted(glob.glob(os.path.join(tag_dir, "plt*")), key=plotfile_step)
    return {p: float(np.interp(plotfile_step(p), it, t)) for p in plts}
