#!/usr/bin/env python3
"""
clean_mpm.py — Cleanup utility for the MPM build/test tree.

Checks that MPM_HOME is defined, then removes:
  • build_matrix_*.txt / .json    in $MPM_HOME
  • Per test-subfolder artifacts   in $MPM_HOME/Tests/<case>/
      – ExaGOOP{1,2,3}d.gnu.MPI.ex
      – Backtrace.*
      – Solution/   (recursive)
      – tmp_build_dir/  (recursive)
      – Diagnostics/    (recursive)
"""

import glob
import os
import shutil
import sys


# ── helpers ─────────────────────────────────────────────────────────
def _remove_file(path: str) -> None:
    """Remove a single file; log the action."""
    try:
        os.remove(path)
        print(f"  [removed file]  {path}")
    except FileNotFoundError:
        pass  # already gone — no-op
    except OSError as exc:
        print(f"  [ERROR]  could not remove {path}: {exc}", file=sys.stderr)


def _remove_tree(path: str) -> None:
    """Recursively remove a directory tree; log the action."""
    if not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path)
        print(f"  [removed dir]   {path}")
    except OSError as exc:
        print(f"  [ERROR]  could not remove {path}: {exc}", file=sys.stderr)


# ── top-level build-matrix cleanup ──────────────────────────────────
def clean_build_matrices(root: str) -> None:
    """Delete build_matrix_*.txt and build_matrix_*.json in *root*."""
    patterns = ["build_matrix_*.txt", "build_matrix_*.json"]
    for pat in patterns:
        for fpath in sorted(glob.glob(os.path.join(root, pat))):
            _remove_file(fpath)


# ── per-test-case cleanup ──────────────────────────────────────────
EXECUTABLE_NAMES = [
    "ExaGOOP1d.gnu.MPI.ex",
    "ExaGOOP2d.gnu.MPI.ex",
    "ExaGOOP3d.gnu.MPI.ex",
]

DIRS_TO_REMOVE = [
    "Solution",
    "tmp_build_dir",
    "Diagnostics",
]


def clean_test_case(case_dir: str) -> None:
    """Remove build artifacts from a single test-case directory."""
    # Named executables
    for name in EXECUTABLE_NAMES:
        _remove_file(os.path.join(case_dir, name))

    # Backtrace.* (glob)
    for fpath in sorted(glob.glob(os.path.join(case_dir, "Backtrace.*"))):
        _remove_file(fpath)

    # Directories (recursive delete)
    for dirname in DIRS_TO_REMOVE:
        _remove_tree(os.path.join(case_dir, dirname))


def clean_all_tests(root: str) -> None:
    """Iterate over every subfolder in <root>/Tests/ and clean it."""
    tests_dir = os.path.join(root, "Tests")
    if not os.path.isdir(tests_dir):
        print(f"[WARN] Tests directory not found: {tests_dir}", file=sys.stderr)
        return

    subdirs = sorted(
        entry
        for entry in os.listdir(tests_dir)
        if os.path.isdir(os.path.join(tests_dir, entry))
    )

    if not subdirs:
        print(f"[INFO] No subdirectories found in {tests_dir}")
        return

    for case_name in subdirs:
        case_path = os.path.join(tests_dir, case_name)
        print(f"\n── cleaning Tests/{case_name}/ ──")
        clean_test_case(case_path)


# ── entry point ────────────────────────────────────────────────────
def main() -> None:
    mpm_home = os.environ.get("MPM_HOME")

    if not mpm_home:
        print(
            "ERROR: MPM_HOME environment variable is not defined.\n"
            "       Set it to the root of your MPM tree and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve to absolute path for clarity in log output
    mpm_home = os.path.abspath(mpm_home)

    if not os.path.isdir(mpm_home):
        print(
            f"ERROR: MPM_HOME points to a non-existent directory:\n"
            f"       {mpm_home}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(mpm_home)
    print(f"[INFO] MPM_HOME = {mpm_home}\n")

    print("── cleaning build matrices ──")
    clean_build_matrices(mpm_home)

    clean_all_tests(mpm_home)

    print("\n[DONE] Cleanup complete.")


if __name__ == "__main__":
    main()

