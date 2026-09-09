"""
Column map for ExaGOOP ASCII particle dumps (mpm.write_ascii = 1, the
"matpnt_t*" files written by amrex::ParticleContainer::WriteAsciiFile).

Each particle row is:
    pos(AMREX_SPACEDIM) | realData[0 .. count-1] | id | cpu | intData[0 .. 2]

The realData layout below mirrors `struct realData` in Source/mpm_specs.H and
must be updated together with it. The plotfile Header (Solution/particle_files/
plt*/particles/Header) lists the same names in the same order and can be used
to cross-check (see `names_from_plotfile_header`).

Usage (from a test's PostProcess script):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Tools", "PostProcess"))
    from exagoop_columns import build_field_dict
    fields = build_field_dict(AMREX_SPACEDIM, USE_TEMP)
    T = data[:, fields["temperature"]]
"""

# Slot pool for per-model internal state variables; EXAGOOP_NISV in
# Source/constitutive_models.H. isv_0.. meaning depends on the particle's
# material: see Solution/materials.txt of the run.
EXAGOOP_NISV = 9

NCOMP_TENSOR = 6
NCOMP_FULLTENSOR = 9


def realdata_layout(use_temp=True, nisv=EXAGOOP_NISV):
    """Ordered list of (name, width) for realData, matching mpm_specs.H."""
    layout = [
        ("radius", 1),
        ("xvel", 1), ("yvel", 1), ("zvel", 1),
        ("xvel_prime", 1), ("yvel_prime", 1), ("zvel_prime", 1),
        ("strainrate", NCOMP_TENSOR),
        ("strain", NCOMP_TENSOR),
        ("stress", NCOMP_TENSOR),
        ("deformation_gradient", NCOMP_FULLTENSOR),
        ("volume", 1), ("mass", 1), ("density", 1),
        ("jacobian", 1), ("vol_init", 1),
    ]
    if use_temp:
        layout += [
            ("temperature", 1), ("specific_heat", 1), ("thermal_conductivity", 1),
            ("heat_flux", 3),          # always 3 slots reserved
            ("heat_source", 1),
        ]
    layout += [("isv", nisv)]
    return layout


def realdata_enum(use_temp=True, nisv=EXAGOOP_NISV):
    """name -> first realData slot index (the C++ realData:: enumerator)."""
    enum, k = {}, 0
    for name, width in realdata_layout(use_temp, nisv):
        enum[name] = k
        k += width
    enum["count"] = k
    return enum


def build_field_dict(dim, use_temp=True, nisv=EXAGOOP_NISV):
    """
    name -> 0-based column index in a matpnt ASCII row.
    Positions come first: posx, posy, posz (as many as `dim`); realData
    fields follow at (enum + dim). Multi-component fields map to their
    first component (stress -> stress_0; use stress+k for component k).
    Integer fields follow the reals: id, cpu, phase, rigid_body_id,
    material_indx.
    """
    fields = {f"pos{'xyz'[d]}": d for d in range(dim)}
    enum = realdata_enum(use_temp, nisv)
    count = enum.pop("count")
    for name, idx in enum.items():
        fields[name] = idx + dim
    for k in range(nisv):
        fields[f"isv_{k}"] = enum["isv"] + k + dim
    base = dim + count
    fields.update({"id": base, "cpu": base + 1,
                   "phase": base + 2, "rigid_body_id": base + 3,
                   "material_indx": base + 4})
    fields["ncols"] = base + 5
    return fields


def names_from_plotfile_header(header_path):
    """Real-field names as written in a plotfile particles/Header (only the
    fields that were flagged for output), for cross-checking."""
    with open(header_path) as f:
        lines = [l.strip() for l in f]
    nreal = int(lines[2])
    return lines[3:3 + nreal]


if __name__ == "__main__":
    import sys
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ut = (sys.argv[2].lower() != "false") if len(sys.argv) > 2 else True
    for k, v in build_field_dict(dim, ut).items():
        print(f"{k:22s} {v}")
