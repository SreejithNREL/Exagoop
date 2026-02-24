#!/usr/bin/env python3
import itertools
import subprocess
import os
import shutil
import json
import glob
import re

def update_makefile_dim(MAKEFILE_PATH,dim):
    lines = []
    with open(MAKEFILE_PATH, "r") as f:
        for line in f:
            if line.startswith("DIM"):
                lines.append(f"DIM     = {dim}\n")
            else:
                lines.append(line)
    with open(MAKEFILE_PATH, "w") as f:
        f.writelines(lines)
        
def update_makefile_usetemp(MAKEFILE_PATH,usetemp):
    lines = []
    with open(MAKEFILE_PATH, "r") as f:
        for line in f:
            if line.startswith("USE_TEMP"):
                lines.append(f"USE_TEMP     = {usetemp}\n")
            else:
                lines.append(line)
    with open(MAKEFILE_PATH, "w") as f:
        f.writelines(lines)


# ============================================================
# Utility: Run shell commands
# ============================================================
def run_cmd(cmd):
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


# ============================================================
# Utility: Auto-tag generator (shared across tests)
# ============================================================
def make_auto_tag_from_params(desc_string):
    import hashlib
    short_hash = hashlib.md5(desc_string.encode()).hexdigest()[:6]
    return f"{desc_string}_{short_hash}"


# ============================================================
# Utility: Find latest matpnt_tXXXXXX file
# ============================================================
def get_latest_time(folder):
    files = glob.glob(os.path.join(folder, "matpnt_t*"))
    if not files:
        return None, None

    def extract_time(fname):
        m = re.search(r"matpnt_t([0-9.]+)", fname)
        return float(m.group(1)) if m else -1

    latest_file = max(files, key=extract_time)
    latest_time = extract_time(latest_file)
    return latest_time, latest_file


# ============================================================
# Utility: Copy master GNUmakefile into test folder
# ============================================================
def copy_gnumake_to_test(root, test_dir):
    src = os.path.join(root, "Build_Gnumake", "GNUmakefile")
    dst = os.path.join(root, "Tests", test_dir, "GNUmakefile")
    print(f"[COPY] {src} → {dst}")
    shutil.copy(src, dst)


def Run_ParameterSweep_1D_Axial_Bar_Vibration(cfg):
    # Parameter sweep setup
    dims = cfg["parameter_space"]["dimension"]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    flip_vals = cfg["parameter_space"]["alpha_pic_flip"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]
    cfl_vals = cfg["parameter_space"]["CFL"]

    for dim, npcx, order, flip, sus, cfl in itertools.product(
        dims, npcx_vals, order_vals, flip_vals, sus_vals, cfl_vals
    ):

        print(f"\n--- Case: dim={dim}, npcx={npcx}, ord={order}, flip={flip}, sus={sus}, CFL={cfl}")

        # Build auto-tag
        desc = f"{test_name}_dim{dim}_npcx{npcx}_ord{order}_flip{flip}_sus{sus}_CFL{cfl}"
        output_tag = make_auto_tag_from_params(desc)

        # Update generator script
        gen_script_path = os.path.join(test_dir, "Generate_MPs_and_InputFiles.sh")
        with open(gen_script_path, "w") as f:
            f.write(f"python3 {cfg['generator_script']} \\\n")
            f.write(f"    --dimension {dim} \\\n")
            f.write(f"    --no_of_cell_in_x 25 \\\n")
            f.write(f"    --buffery 5 \\\n")
            f.write(f"    --periodic 0 \\\n")
            f.write(f"    --np_per_cell_x {npcx} \\\n")
            f.write(f"    --order_scheme {order} \\\n")
            f.write(f"    --alpha_pic_flip {flip} \\\n")
            f.write(f"    --stress_update_scheme {sus} \\\n")
            f.write(f"    --CFL {cfl} \\\n")
            f.write(f"    --output_tag {output_tag}\n")

        # change gnumakefile
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        run_cmd(f"cd {test_dir} && make -j")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        if(dim==1):
            exe = "./ExaGOOP1d.gnu.MPI.ex"
        elif(dim==2):
            exe = "./ExaGOOP2d.gnu.MPI.ex"
        elif(dim==3):
            exe = "./ExaGOOP3d.gnu.MPI.ex"
            
        #exe = "./ExaGOOP3d.gnu.MPI.ex" if dim == 3 else "./ExaGOOP2d.gnu.MPI.ex"

        # Run simulation
        run_cmd(f"cd {test_dir} && mpirun -np 4 {exe} {cfg['input_file']}")

        # Post-processing
        ascii_folder = os.path.join(test_dir, "Solution", "ascii_files",output_tag)
        latest_time, _ = get_latest_time(ascii_folder)

        if latest_time is None:
            print("[WARNING] No output files found.")
            rms = float("nan")
        else:
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --time {latest_time} --folder {ascii_folder} --dim {dim}"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True)
            
            
            PicsFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Pics")
            MoviesFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Movies")   
            
            os.makedirs(PicsFolder, exist_ok=True)           
            os.makedirs(MoviesFolder, exist_ok=True)
            
            pp1_script = os.path.join(test_dir,cfg["postproc_scripts"][1])   
            input_file=os.path.join(test_dir,f"./Diagnostics/{output_tag}/Total_Energies.dat")
            pp1_cmd = f"python3 {pp1_script} {input_file} {PicsFolder}/Energy.png"            
            pp1_output = subprocess.check_output(pp1_cmd, shell=True, text=True)
            
            pp2_script = os.path.join(test_dir,cfg["postproc_scripts"][2])   
            input_file=os.path.join(test_dir,f"./Diagnostics/{output_tag}/VelComponents.dat")
            pp2_cmd = f"python3 {pp2_script} {input_file} {PicsFolder}/Velocity.png"            
            pp2_output = subprocess.check_output(pp2_cmd, shell=True, text=True)
            
            pp3_script = os.path.join(test_dir,cfg["postproc_scripts"][3])   
            input_file=os.path.join(test_dir,f"./Solution/ascii_files/{output_tag}")
            pp3_cmd = f"python3 {pp3_script} {input_file} {dim} {MoviesFolder}/AxialBar.mp4"            
            pp3_output = subprocess.check_output(pp3_cmd, shell=True, text=True)
            
            rms = None
            for line in error_output.splitlines():
                if "RMS error" in line:
                    rms = float(line.split()[-1])
                    break

        results.append({
            "test": test_name,
            "dim": dim,
            "npcx": npcx,
            "order": order,
            "flip": flip,
            "sus": sus,
            "cfl": cfl,
            "rms": rms,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_1D_HeatConduction(cfg):
    # Parameter sweep setup
    dim = cfg["parameter_space"]["dimension"][0]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    

    for npcx, order, sus in itertools.product(
        npcx_vals, order_vals, sus_vals
    ):

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}"
        output_tag = make_auto_tag_from_params(desc)

        # 1. Load template config
        with open(os.path.join(test_dir, "./Preprocess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus

        # Auto-tag       
        config["output_tag"] = output_tag
        
        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./Preprocess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        run_cmd(f"cd {test_dir} && make -j")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        if(dim==1):
            exe = "./ExaGOOP1d.gnu.MPI.ex"
        elif(dim==2):
            exe = "./ExaGOOP2d.gnu.MPI.ex"
        elif(dim==3):
            exe = "./ExaGOOP3d.gnu.MPI.ex"

        # Run simulation
        run_cmd(f"cd {test_dir} && mpirun -np 4 {exe} {cfg['input_file']}")

        # Post-processing
        ascii_folder = os.path.join(test_dir, "Solution", "ascii_files",output_tag)
        latest_time, _ = get_latest_time(ascii_folder)

        if latest_time is None:
            print("[WARNING] No output files found.")
            rms = float("nan")
        else:
            PicsFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Pics")
            os.makedirs(PicsFolder, exist_ok=True)           
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --time {latest_time} --fileloc {ascii_folder} --dim {dim} --outputpic {PicsFolder}/Temperature_x.png"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True)           
            
            rms = None
            for line in error_output.splitlines():
                if "RMS error" in line:
                    rms = float(line.split()[-1])
                    break

        results.append({
            "test": test_name,
            "dim": dim,
            "npcx": npcx,
            "order": order,            
            "sus": sus,            
            "rms": rms,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_2D_HeatConduction(cfg):
    # Parameter sweep setup    
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    

    for npcx, order, sus in itertools.product(
        npcx_vals, order_vals, sus_vals
    ):

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}"
        output_tag = make_auto_tag_from_params(desc)
        
        
        # 1. Load template config
        with open(os.path.join(test_dir, "./Preprocess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./Preprocess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),2)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        run_cmd(f"cd {test_dir} && make -j")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        exe = "./ExaGOOP2d.gnu.MPI.ex"

        # Run simulation
        run_cmd(f"cd {test_dir} && mpirun -np 6 {exe} {cfg['input_file']}")

        # Post-processing
        ascii_folder = os.path.join(test_dir, "Solution", "ascii_files",output_tag)
        latest_time, _ = get_latest_time(ascii_folder)

        if latest_time is None:
            print("[WARNING] No output files found.")
            rms = float("nan")
        else:
            PicsFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Pics")
            os.makedirs(PicsFolder, exist_ok=True)           
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --time {latest_time} --folder {ascii_folder} --outputpic {PicsFolder}/Temperature_x.png"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True)           
            
            rms = None
            for line in error_output.splitlines():
                if "RMS error" in line:
                    rms = float(line.split()[-1])
                    break

        results.append({
            "test": test_name,            
            "npcx": npcx,
            "order": order,            
            "sus": sus,            
            "rms": rms,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_Dambreak(cfg):
    # Parameter sweep setup    
    dims = cfg["parameter_space"]["dimension"]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    

    for dim, npcx, order, sus in itertools.product(
        dims,npcx_vals, order_vals, sus_vals
    ):

        print(f"\n--- Case:  npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}__dim{dim}_npcx{npcx}_ord{order}_sus{sus}"
        output_tag = make_auto_tag_from_params(desc)

        # Update generator script
        with open(os.path.join(test_dir, "./Preprocess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./Preprocess/config.json"), "w") as f:
            json.dump(config, f, indent=2)


        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        run_cmd(f"cd {test_dir} && make -j")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        exe = "./ExaGOOP3d.gnu.MPI.ex" if dim == 3 else "./ExaGOOP2d.gnu.MPI.ex"

        # Run simulation
        run_cmd(f"cd {test_dir} && mpirun -np 6 {exe} {cfg['input_file']}")

        # Post-processing
        ascii_folder = os.path.join(test_dir, "Solution", "ascii_files",output_tag)
        latest_time, _ = get_latest_time(ascii_folder)

        if latest_time is None:
            print("[WARNING] No output files found.")
            rms = float("nan")
        else:
            PicsFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Pics")
            MovieFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Movies")
            minmaxfile = os.path.join(test_dir,f"Diagnostics/{output_tag}/MinMaxPosition.dat")
            expdata = os.path.join(test_dir,f"ExperimentalData.dat")
            os.makedirs(PicsFolder, exist_ok=True)     
            os.makedirs(MovieFolder, exist_ok=True)           
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --folder {ascii_folder} --outputpic {PicsFolder}/WaterFront.png --expdata {expdata} --outputmovie {MovieFolder}/Water.mp4 --minmaxfile {minmaxfile}"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True)     
            

def Run_ParameterSweep_EDC(cfg):
    # Parameter sweep setup    
    dims = cfg["parameter_space"]["dimension"]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    

    for dim, npcx, order, sus in itertools.product(
        dims,npcx_vals, order_vals, sus_vals
    ):

        print(f"\n--- Case:  npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}__dim{dim}_npcx{npcx}_ord{order}_sus{sus}"
        output_tag = make_auto_tag_from_params(desc)

        # Update generator script
        with open(os.path.join(test_dir, "./Preprocess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./Preprocess/config.json"), "w") as f:
            json.dump(config, f, indent=2)


        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        run_cmd(f"cd {test_dir} && make -j")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        exe = "./ExaGOOP3d.gnu.MPI.ex" if dim == 3 else "./ExaGOOP2d.gnu.MPI.ex"

        # Run simulation
        run_cmd(f"cd {test_dir} && mpirun -np 4 {exe} {cfg['input_file']}")

        # Post-processing
        ascii_folder = os.path.join(test_dir, "Solution", "ascii_files",output_tag)
        latest_time, _ = get_latest_time(ascii_folder)

        if latest_time is None:
            print("[WARNING] No output files found.")
            rms = float("nan")
        else:
            PicsFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Pics")
            MovieFolder = os.path.join(test_dir,f"Solution/ascii_files/{output_tag}/Movies")    
            energyfile = os.path.join(test_dir,f"Diagnostics/{output_tag}/Total_Energies.dat")        
            os.makedirs(PicsFolder, exist_ok=True)     
            os.makedirs(MovieFolder, exist_ok=True)           
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --folder {ascii_folder} --outputpic {PicsFolder}/Energy.png --outputmovie {MovieFolder}/Disks.mp4 --energyfile {energyfile}"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True) 
            print("Done")    
              


# ============================================================
# PLACEHOLDER: Test Case Definitions
# ============================================================
TEST_CASES = {
    "1D_Axial_Bar_Vibration": {
        "generator_script": "./PreProcess/generate_particle_and_inputfiles.py",
        "input_file": "Inputs_1DAxialBarVibration.inp",
        "postproc_scripts": [
            "./PostProcess/Calculate_Error.py",
            "./PostProcess/plot_energy.py",
            "./PostProcess/plot_vel.py",
            "./PostProcess/AnimateVelocity.py"
        ],
        "parameter_space": {
            "dimension": [1,2,3],
            "np_per_cell_x": [1],
            "order_scheme": [1,2,3],
            "alpha_pic_flip": [1.0],
            "stress_update_scheme": [1],
            "CFL": [0.1]
        }
    },

    # ========================================================
    # ADD YOUR OTHER TEST CASES HERE
    # ========================================================
    "1D_Heat_Conduction": {
        "generator_script": "./PreProcess/generate_particle_and_inputfiles.py",
        "input_file": "Inputs_1DHeatConduction.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"            
        ],
        "parameter_space": {
            "dimension": [1],
            "np_per_cell_x": [1,2,4],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1]            
        }
    },
    
    "2D_Heat_Conduction": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_2DHeatConduction.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"            
        ],
        "parameter_space": {            
            "np_per_cell_x": [1],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1]            
        }
    },
    
    "Dam_Break": {
        "generator_script": "./PreProcess/generate_particle_and_inputfiles.py",
        "input_file": "Inputs_DamBreak.inp",
        "postproc_scripts": [
            "./PostProcess/plot_waterfront.py"            
        ],
        "parameter_space": {     
            "dimension": [2,3], 
            "no_of_cell_in_x": [100],      
            "np_per_cell_x": [1],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1]            
        }
    },
    
    "Elastic_disk_collision": {
        "generator_script": "./PreProcess/generate_particle_and_inputfiles.py",
        "input_file": "Inputs_ElasticDiskCollision.inp",
        "postproc_scripts": [
            "./PostProcess/plot_energy.py"            
        ],
        "parameter_space": {     
            "dimension": [2],        
            "np_per_cell_x": [4],            
            "order_scheme": [3],            
            "stress_update_scheme": [1]            
        }
    },

    # Add more test cases as needed...
}


# ============================================================
# Main Sweep Driver
# ============================================================
ROOT = "/Users/snadakka/Git_Repositories/ExaGOOP_Dev"
ERROR_TOL = 1e-1
results = []



sol_dir = "/Users/snadakka/Git_Repositories/ExaGOOP_Dev/Tests/1D_Axial_Bar_Vibration/Solution"




for test_name, cfg in TEST_CASES.items():

    print("\n====================================================")
    print(f"Running Test Case: {test_name}")
    print("====================================================")

    test_dir = os.path.join(ROOT, "Tests", test_name)
    
    #Deleting existing solution and diagnostics folders
    sol_dir = os.path.join(test_dir,"Solution")
    diagn_dir = os.path.join(test_dir,"Diagnostics")
    tmp_build_dir = os.path.join(test_dir,"tmp_build_dir")
    
    if os.path.exists(sol_dir):
        shutil.rmtree(sol_dir)
        print(f"Deleted: {sol_dir}")
    else:
        print(f"No Solution directory found at: {sol_dir}")
        
    if os.path.exists(diagn_dir):
        shutil.rmtree(diagn_dir)
        print(f"Deleted: {diagn_dir}")
    else:
        print(f"No Solution directory found at: {diagn_dir}") 
        
    if os.path.exists(tmp_build_dir):
        shutil.rmtree(tmp_build_dir)
        print(f"Deleted: {tmp_build_dir}")
    else:
        print(f"No Solution directory found at: {tmp_build_dir}") 
    
    # --- Delete ExaGOOP executables ---
    deleted_any = False
    
    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
    
        # Match pattern: ExaGOOP*.ex
        if (
            os.path.isfile(fpath)
            and fname.startswith("ExaGOOP")
            and fname.endswith(".ex")
            and os.access(fpath, os.X_OK)
        ):
            os.remove(fpath)
            print(f"Deleted executable: {fpath}")
            deleted_any = True
    
    if not deleted_any:
        print("No ExaGOOP executables found.")


    # Copy GNUmakefile
    copy_gnumake_to_test(ROOT, test_name)
    
    if(test_name=="1D_Axial_Bar_Vibration"):
        print('Nothing to do')                
        #Run_ParameterSweep_1D_Axial_Bar_Vibration(cfg)
    elif(test_name=="1D_Heat_Conduction"):
        print('Nothing to do')        
        #Run_ParameterSweep_1D_HeatConduction(cfg)
    elif(test_name=="2D_Heat_Conduction"):
        print('Nothing to do')        
        #Run_ParameterSweep_2D_HeatConduction(cfg)
    elif(test_name=="Dam_Break"):
        print('Nothing to do')        
        #Run_ParameterSweep_Dambreak(cfg)
    elif(test_name=="Elastic_disk_collision"):
        print('Nothing to do')
        Run_ParameterSweep_EDC(cfg)
        
    


# Save results
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll test cases complete.")

