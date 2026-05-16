#!/usr/bin/env python3
"""
Generate an ASCII STL file of an axis-aligned box (rectangular cuboid).

The box is defined by its lower and upper corner coordinates. Each of the
6 faces is tessellated into 2 triangles with outward-pointing normals,
giving 12 triangles total.

Usage
-----
    python generate_box_stl.py --lo 0.1 0.0 0.0 --hi 0.3 0.4 0.2
    python generate_box_stl.py --lo 0.1 0.0 0.0 --hi 0.3 0.4 0.2 --output obstacle.stl
    python generate_box_stl.py --lo 0.1 0.0 0.0 --hi 0.3 0.4 0.2 --output obstacle.stl --name my_box

Arguments
---------
    --lo  X Y Z   Lower corner coordinates of the box (required)
    --hi  X Y Z   Upper corner coordinates of the box (required)
    --output      Output STL file path (default: box.stl in script directory)
    --name        Solid name embedded in the STL header (default: box)
"""

import argparse
import os
import sys


def make_box_triangles(lo, hi):
    """
    Return the 12 triangles (normal, v0, v1, v2) of an axis-aligned box.

    Vertex winding follows the right-hand rule so that normals point outward.

    Parameters
    ----------
    lo : list[float]   Lower corner [x0, y0, z0]
    hi : list[float]   Upper corner [x1, y1, z1]

    Returns
    -------
    list of (normal, v0, v1, v2) tuples, each being a 3-tuple of floats.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    # 8 corner vertices
    v = [
        (x0, y0, z0),  # 0: left  front bottom
        (x1, y0, z0),  # 1: right front bottom
        (x1, y1, z0),  # 2: right back  bottom
        (x0, y1, z0),  # 3: left  back  bottom
        (x0, y0, z1),  # 4: left  front top
        (x1, y0, z1),  # 5: right front top
        (x1, y1, z1),  # 6: right back  top
        (x0, y1, z1),  # 7: left  back  top
    ]

    triangles = [
        # Bottom face  z = z0   normal (0, 0, -1)
        ((0.0,  0.0, -1.0), v[0], v[2], v[1]),
        ((0.0,  0.0, -1.0), v[0], v[3], v[2]),

        # Top face     z = z1   normal (0, 0, +1)
        ((0.0,  0.0,  1.0), v[4], v[5], v[6]),
        ((0.0,  0.0,  1.0), v[4], v[6], v[7]),

        # Front face   y = y0   normal (0, -1, 0)
        ((0.0, -1.0,  0.0), v[0], v[1], v[5]),
        ((0.0, -1.0,  0.0), v[0], v[5], v[4]),

        # Back face    y = y1   normal (0, +1, 0)
        ((0.0,  1.0,  0.0), v[2], v[3], v[7]),
        ((0.0,  1.0,  0.0), v[2], v[7], v[6]),

        # Left face    x = x0   normal (-1, 0, 0)
        ((-1.0, 0.0,  0.0), v[0], v[4], v[7]),
        ((-1.0, 0.0,  0.0), v[0], v[7], v[3]),

        # Right face   x = x1   normal (+1, 0, 0)
        (( 1.0, 0.0,  0.0), v[1], v[2], v[6]),
        (( 1.0, 0.0,  0.0), v[1], v[6], v[5]),
    ]

    return triangles


def write_ascii_stl(filepath, triangles, solid_name="box"):
    """
    Write a list of triangles to an ASCII STL file.

    Parameters
    ----------
    filepath   : str              Output file path.
    triangles  : list             List of (normal, v0, v1, v2) tuples.
    solid_name : str              Name embedded in the STL solid header.
    """
    with open(filepath, "w") as f:
        f.write(f"solid {solid_name}\n")
        for normal, v0, v1, v2 in triangles:
            f.write(
                f"  facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}\n"
            )
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.8e} {v0[1]:.8e} {v0[2]:.8e}\n")
            f.write(f"      vertex {v1[0]:.8e} {v1[1]:.8e} {v1[2]:.8e}\n")
            f.write(f"      vertex {v2[0]:.8e} {v2[1]:.8e} {v2[2]:.8e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an ASCII STL file of an axis-aligned box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lo",
        nargs=3,
        type=float,
        required=True,
        metavar=("X0", "Y0", "Z0"),
        help="Lower corner coordinates of the box.",
    )
    parser.add_argument(
        "--hi",
        nargs=3,
        type=float,
        required=True,
        metavar=("X1", "Y1", "Z1"),
        help="Upper corner coordinates of the box.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "box.stl"),
        help="Output STL file path (default: box.stl next to this script).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="box",
        help="Solid name written into the STL header (default: box).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lo = args.lo
    hi = args.hi

    # Validate corner ordering
    for i, (l, h) in enumerate(zip(lo, hi)):
        if l >= h:
            print(
                f"Error: lo[{i}] = {l} must be strictly less than hi[{i}] = {h}.",
                file=sys.stderr,
            )
            sys.exit(1)

    triangles = make_box_triangles(lo, hi)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    write_ascii_stl(args.output, triangles, solid_name=args.name)

    print(f"STL written : {args.output}")
    print(f"Solid name  : {args.name}")
    print(f"Lower corner: ({lo[0]}, {lo[1]}, {lo[2]})")
    print(f"Upper corner: ({hi[0]}, {hi[1]}, {hi[2]})")
    print(f"Triangles   : {len(triangles)}")


if __name__ == "__main__":
    main()
