import itertools
import subprocess
import os
import shutil
import json


def make_auto_tag_from_params(dim, npcx, order, flip, sus, cfl):
    import hashlib
    desc = (
        f"AVB_dim{dim}_nx25_ppc{npcx}_buff5_"
        f"ord{order}_alpha{flip}_sus{sus}_CFL{cfl}"
    )
    short_hash = hashlib.md5(desc.encode()).hexdigest()[:6]
    return f"{desc}_{short_hash}"


def get_executable_for_dim(dim):
    if dim == 3:
        return "./ExaGOOP3d.gnu.MPI.ex"
    elif dim == 2:
        return "./ExaGOOP2d.gnu.MPI.ex"
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


# -----------------------------
# User settings
# -----------------------------
MPM_HOME = "/Users/snadakka/Git_Repositories/ExaGOOP_Dev/Tests/1D_Axial_Bar_Vibration"
MAKEFILE_PATH = os.path.join(MPM_HOME, "GNUmakefile")
GEN_SCRIPT = "./Generate_MPs_and_InputFiles.sh"
#executable = get_executable_for_dim(dim)
ERROR_SCRIPT = "./PostProcess/Calculate_Error.py"
POSTPROC_SCRIPT1 = "./PostProcess/plot_energy.py"
POSTPROC_SCRIPT2 = "./PostProcess/plot_vel.py"
POSTPROC_SCRIPT3 = "./PostProcess/AnimateVelocity.py"


ERROR_TOL = 1e-1




# -----------------------------
# Parameter sweep definitions
# -----------------------------
dimensions = [3]
np_per_cell_x_values = [1]
order_schemes = [1]
alpha_pic_flip_values = [1.0]
stress_update_schemes = [1]
CFL_values = [0.1]

# -----------------------------
# Helper: modify DIM in GNUmakefile
# -----------------------------
def update_makefile_dim(dim):
    lines = []
    with open(MAKEFILE_PATH, "r") as f:
        for line in f:
            if line.startswith("DIM"):
                lines.append(f"DIM     = {dim}\n")
            else:
                lines.append(line)
    with open(MAKEFILE_PATH, "w") as f:
        f.writelines(lines)


import glob
import re
import os

def get_latest_time(folder):
    """
    Find the latest matpnt_tXXXXXX file in the given folder.
    Returns (time_value, filename).
    """
    files = glob.glob(os.path.join(folder, "matpnt_t*"))
    if not files:
        return None, None

    def extract_time(fname):
        m = re.search(r"matpnt_t([0-9.]+)", fname)
        return float(m.group(1)) if m else -1

    # Pick file with max time
    latest_file = max(files, key=extract_time)
    latest_time = extract_time(latest_file)

    return latest_time, latest_file



# -----------------------------
# Helper: update Generate script
# -----------------------------
def update_generate_script(params):
    with open(GEN_SCRIPT, "w") as f:
        f.write("python3 ./PreProcess/generate_particle_and_inputfiles.py \\\n")
        for key, val in params.items():
            f.write(f"    --{key} {val} \\\n")
        f.write("    --output_tag \"\"\n")

# -----------------------------
# Helper: run shell commands
# -----------------------------
def run_cmd(cmd):
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# -----------------------------
# Main sweep
# -----------------------------
results = []

for dim, npcx, order, flip, sus, cfl in itertools.product(
    dimensions,
    np_per_cell_x_values,
    order_schemes,
    alpha_pic_flip_values,
    stress_update_schemes,
    CFL_values
):

    print("\n======================================")
    print(f"Running case: dim={dim}, npcx={npcx}, order={order}, flip={flip}, sus={sus}, CFL={cfl}")
    print("======================================")

    # 1. Update makefile
    update_makefile_dim(dim)

    # 2. Compile
    run_cmd(f"cd {MPM_HOME} && make -j")

    # 3. Update Generate script
   
    params = {
    "dimension": dim,
    "no_of_cell_in_x": 25,
    "buffery": 5,
    "periodic": 0,
    "np_per_cell_x": npcx,
    "order_scheme": order,
    "alpha_pic_flip": flip,
    "stress_update_scheme": sus,
    "CFL": cfl
    }

    # Add output_tag here
    params["output_tag"] = make_auto_tag_from_params(dim, npcx, order, flip, sus, cfl)

    update_generate_script(params)

    # 4. Generate particles + input files
    run_cmd(f"bash {GEN_SCRIPT}")

    #5. Select executable based on dimension
    executable = get_executable_for_dim(dim)
    
    # 6. Run ExaGOOP
    run_cmd(f"mpirun -np 4 {executable} Inputs_1DAxialBarVibration.inp")
    
    # 7. Compute RMS error
    # Determine correct folder for this run
    output_folder = f"Solution/ascii_files/{params['output_tag']}"
    PicsFolder = f"Solution/ascii_files/{params['output_tag']}/Pics"
    MoviesFolder = f"Solution/ascii_files/{params['output_tag']}/Movies"
    
    pic_dir = os.path.join(".", "Solution/ascii_files/", params['output_tag'],"Pics")
    os.makedirs(pic_dir, exist_ok=True)
    
    movie_dir = os.path.join(".", "Solution/ascii_files/", params['output_tag'],"Movies")
    os.makedirs(movie_dir, exist_ok=True)
    
    # Find latest time
    latest_time, latest_file = get_latest_time(output_folder)
    
    if latest_time is None:
        print("[WARNING] No matpnt_t files found. Skipping RMS error.")
        rms = float("nan")
    else:
        print(f"[INFO] Latest time detected: {latest_time:.6f} from {os.path.basename(latest_file)}")
    
    # Compute RMS error at that time
    error_output = subprocess.check_output(
        f"python3 {ERROR_SCRIPT} {latest_time} --folder {output_folder}",
        shell=True,
        text=True
    )
    
    postproc1_output = subprocess.check_output(
        f"python3 {POSTPROC_SCRIPT1} ./Diagnostics/{params['output_tag']}/Total_Energies.dat {PicsFolder}/Energy.png",
        shell=True,
        text=True
    )
    
    postproc2_output = subprocess.check_output(
        f"python3 {POSTPROC_SCRIPT2} ./Diagnostics/{params['output_tag']}/VelComponents.dat {PicsFolder}/Velocity.png",
        shell=True,
        text=True
    )
    
    postproc3_output = subprocess.check_output(
        f"python3 {POSTPROC_SCRIPT3} Solution/ascii_files/{params['output_tag']} {MoviesFolder}/AxialVibratioBar.mp4",
        shell=True,
        text=True
    )
    
    rms = None
    for line in error_output.splitlines():
        if "RMS error" in line:
            rms = float(line.split()[-1])
            break
    
    if rms is None:
        print("[WARNING] RMS error could not be parsed.")
        rms = float("nan")
    
    
    
    # Extract RMS error from output
    for line in error_output.splitlines():
        if "RMS error" in line:
            rms = float(line.split()[-1])
            break
        
    
    
    # 7. Store result
    results.append({
        "dim": dim,
        "npcx": npcx,
        "order": order,
        "flip": flip,
        "sus": sus,
        "cfl": cfl,
        "rms": rms,
        "pass": rms < ERROR_TOL
    })

# -----------------------------
# Save results
# -----------------------------
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSweep complete. Results saved to sweep_results.json")

