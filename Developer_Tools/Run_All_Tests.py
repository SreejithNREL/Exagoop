#!/usr/bin/env python3
import argparse
import datetime
import itertools
import subprocess
import os
import shutil
import json
import glob
import re
import time
import tempfile
from pathlib import Path

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
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "1D_Axial_Bar_Vibration"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for dim, npcx, order, flip, sus, cfl, bwh, of in itertools.product(
        dims, npcx_vals, order_vals, flip_vals, sus_vals, cfl_vals,bwhs,output_formats
    ):
        
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: dim={dim}, npcx={npcx}, ord={order}, flip={flip}, sus={sus}, CFL={cfl}")

        # Build auto-tag
        desc = f"{test_name}_dim{dim}_npcx{npcx}_ord{order}_flip{flip}_sus{sus}_CFL{cfl}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        
        # 2. Modify config fields
        config["ppc"] = [npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["alpha_pic_flip"] = flip        
        config["CFL"] = cfl
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"
        config["output_tag"] = output_tag
        
        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # change gnumakefile
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]
            
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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_1D_HeatConduction(cfg):
    # Parameter sweep setup
    dim = cfg["parameter_space"]["dimension"][0]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "1D_Heat_Conduction"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for npcx, order, sus, bwh, of in itertools.product(
        npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"

        # Auto-tag       
        config["output_tag"] = output_tag
        
        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_1D_HeatConduction_HeatFlux(cfg):
    # Parameter sweep setup
    dim = cfg["parameter_space"]["dimension"][0]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "1D_Heat_Conduction_HeatFlux"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for npcx, order, sus, bwh, of in itertools.product(
        npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"

        # Auto-tag       
        config["output_tag"] = output_tag
        
        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            err_cmd = f"python3 {err_script} --time {latest_time} --fileloc {ascii_folder} --outputpic {PicsFolder}/Temperature_x.png"
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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_1D_HeatConduction_Convective(cfg):
    # Parameter sweep setup
    dim = cfg["parameter_space"]["dimension"][0]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "1D_Heat_Conduction_Convective"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for npcx, order, sus, bwh, of in itertools.product(
        npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"

        # Auto-tag       
        config["output_tag"] = output_tag
        
        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            err_cmd = f"python3 {err_script} --time {latest_time} --fileloc {ascii_folder} --outputpic {PicsFolder}/Temperature_x.png"
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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_2D_HeatConduction(cfg):
    # Parameter sweep setup    
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "2D_Heat_Conduction"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for npcx, order, sus, bwh, of in itertools.product(
        npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)
        
        
        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),2)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        dim = 2
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_2D_HeatConduction_Cylinder_Dirichlet(cfg):
    
    print("Running")
    # Parameter sweep setup    
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "2D_Heat_Conduction_Cylinder_Dirichlet"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for npcx, order, sus, bwh, of in itertools.product(
        npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case: npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)
        
        
        # 1. Load template config
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),2)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"TRUE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE USE_EB=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE USE_EB=TRUE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        #exe = "./ExaGOOP2d.gnu.MPI.ex"
        dim = 2
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": rms < ERROR_TOL if rms == rms else False
        })
        
def Run_ParameterSweep_Dambreak(cfg):
    # Parameter sweep setup    
    dims = cfg["parameter_space"]["dimension"]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]  
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "Dam_Break"
    test_dir = os.path.join(ROOT, "Tests", test_name)  

    for dim, npcx, order, sus, bwh, of in itertools.product(
        dims,npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case:  npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}__dim{dim}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # Update generator script
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
            print("Output format is ascii")
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"
            print("Output format is hdf5")

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)


        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        if(bwh==True):
            print("Building with HDF5")
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            print("Building without HDF5")
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        #exe = "./ExaGOOP3d.gnu.MPI.ex" if dim == 3 else "./ExaGOOP2d.gnu.MPI.ex"
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            
        #dim, npcx, order, sus, bwh, of
        results.append({
            "test": test_name,      
            "dim":dim,      
            "npcx": npcx,
            "order": order,            
            "sus": sus,
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": True
        })    
            

def Run_ParameterSweep_EDC(cfg):
    # Parameter sweep setup    
    dims = cfg["parameter_space"]["dimension"]
    npcx_vals = cfg["parameter_space"]["np_per_cell_x"]
    order_vals = cfg["parameter_space"]["order_scheme"]
    sus_vals = cfg["parameter_space"]["stress_update_scheme"]    
    
    bwhs = cfg["parameter_space"]["build_with_hdf"]    
    output_formats = cfg["parameter_space"]["output_format"]
    filename_prefix = cfg["parameter_space"]["filename_prefix"]           
   
    test_name = "Elastic_disk_collision"
    test_dir = os.path.join(ROOT, "Tests", test_name)

    for dim, npcx, order, sus, bwh, of in itertools.product(
        dims,npcx_vals, order_vals, sus_vals, bwhs,output_formats
    ):
        
        if(bwh==False and of=="hdf5"):
            continue      
        
        existing_mpm_h5 = os.path.join(test_dir,filename_prefix[0]+".h5")
        existing_mpm_dat = os.path.join(test_dir,filename_prefix[0]+".dat")  
        
        p = Path(existing_mpm_h5)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_h5)
            
        p = Path(existing_mpm_dat)
        if p.exists():
            p.unlink()
            print("Deleted existing file "+existing_mpm_dat)

        print(f"\n--- Case:  npcx={npcx}, ord={order}, sus={sus}")

        # Build auto-tag
        desc = f"{test_name}__dim{dim}_npcx{npcx}_ord{order}_sus{sus}_USEHDF{bwh}_OFORM{of}"
        output_tag = make_auto_tag_from_params(desc)

        # Update generator script
        with open(os.path.join(test_dir, "./PreProcess/config.json")) as f:
            config = json.load(f)

        # 2. Modify config fields
        config["ppc"] = [npcx, npcx]
        config["order_scheme"] = order
        config["stress_update_scheme"] = sus
        config["output_format"] = of
        if(of=="ascii"):
            config["materialpoint_filename"] = filename_prefix[0]+".dat"
            print("Output format is ascii")
        else:
            config["materialpoint_filename"] = filename_prefix[0]+".h5"
            print("Output format is hdf5")

        # Auto-tag       
        config["output_tag"] = output_tag

        # 3. Write updated config.json
        with open(os.path.join(test_dir, "./PreProcess/config.json"), "w") as f:
            json.dump(config, f, indent=2)


        # Change gnumake file
        update_makefile_dim(os.path.join(test_dir,"GNUmakefile"),dim)
        update_makefile_usetemp(os.path.join(test_dir,"GNUmakefile"),"FALSE")
        
        # Build executable
        if(bwh==True):
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=TRUE AMREX_USE_HDF5=TRUE")
        else:
            run_cmd(f"cd {test_dir} && make -j8 USE_HDF5=FALSE AMREX_USE_HDF5=FALSE")

        # Generate inputs
        run_cmd(f"cd {test_dir} && bash Generate_MPs_and_InputFiles.sh")

        # Select executable
        #exe = "./ExaGOOP3d.gnu.MPI.ex" if dim == 3 else "./ExaGOOP2d.gnu.MPI.ex"
        pattern = os.path.join(test_dir, f"ExaGOOP{dim}d.*.ex")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No executable found matching {pattern}")

        # If multiple matches exist, pick the first or apply your own logic
        exe = matches[0]

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
            energyfile = os.path.join(test_dir,f"Diagnostics/{output_tag}/Total_Energies.dat")        
            os.makedirs(PicsFolder, exist_ok=True)     
            os.makedirs(MovieFolder, exist_ok=True)           
            err_script = os.path.join(test_dir,cfg["postproc_scripts"][0])                     
            err_cmd = f"python3 {err_script} --folder {ascii_folder} --outputpic {PicsFolder}/Energy.png --outputmovie {MovieFolder}/Disks.mp4 --energyfile {energyfile}"
            error_output = subprocess.check_output(err_cmd, shell=True, text=True) 
            print("Done")    
            
        
        results.append({
            "test": test_name,      
            "dim":dim,      
            "npcx": npcx,
            "order": order,            
            "sus": sus,
            "built_with_hdf": bwh,
            "fileformat": of,
            "pass": True
        })
              


# ============================================================
# Build Matrix
# ============================================================

def Run_Build_Matrix(cfg, root=None, timeout=300, parallel_jobs=8, output_dir=None):
    """
    Tests all combinations of build configurations using GNUMake and CMake.

    Detects available tools (compilers, MPI, OpenMP, HDF5, GPU backends) and
    generates a build matrix spanning DIM 1/2/3, USE_EB, USE_TEMP, compiler,
    DEBUG, MPI, OMP, HDF5, and optional GPU backends.  Each combination is
    built in an isolated temporary directory with a configurable timeout.
    Results are written to a timestamped JSON file and a plain-text summary
    table.

    Parameters
    ----------
    cfg : dict
        Unused; kept for a uniform interface with the other Run_* functions.
    root : str, optional
        Repository root.  Defaults to the module-level ROOT.
    timeout : int, optional
        Per-combination build timeout in seconds (default 300).
    parallel_jobs : int, optional
        Number of parallel make jobs (default 8).
    output_dir : str, optional
        Directory for results files.  Defaults to *root*.
    """
    if root is None:
        root = ROOT
    if output_dir is None:
        output_dir = root

    # ------------------------------------------------------------------
    # 1. Tool detection
    # ------------------------------------------------------------------
    def _tool(name):
        return shutil.which(name) is not None

    def _hdf5_headers():
        search_dirs = [
            "/usr/include/hdf5/openmpi",
            "/usr/include",
            "/usr/local/include",
            "/opt/homebrew/include",
            "/opt/homebrew/opt/hdf5/include",
        ]
        for d in search_dirs:
            if os.path.exists(os.path.join(d, "hdf5.h")):
                return True
        try:
            r = subprocess.run(
                ["pkg-config", "--exists", "hdf5"],
                capture_output=True,
                timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _omp_available():
        omp_paths = [
            "/usr/local/lib/libomp.dylib",
            "/opt/homebrew/lib/libomp.dylib",
            "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
            "/usr/lib/libgomp.so.1",
        ]
        return any(os.path.exists(p) for p in omp_paths)

    has = {
        "gxx":   _tool("g++"),
        "clang": _tool("clang++"),
        "nvcc":  _tool("nvcc"),
        "hipcc": _tool("hipcc"),
        "icpx":  _tool("icpx"),
        "mpi":   _tool("mpicc"),
        "omp":   _omp_available(),
        "hdf5":  _hdf5_headers(),
    }

    print("\n=== Build-Matrix: Tool Detection ===")
    for tool, found in has.items():
        print(f"  {tool:8s}: {'FOUND' if found else 'not found'}")

    # ------------------------------------------------------------------
    # 2. Generate build combinations
    # ------------------------------------------------------------------
    def _cmake_bool(val):
        """Convert TRUE/FALSE string to CMake ON/OFF."""
        return "ON" if val == "TRUE" else "OFF"

    combos = []
    combo_id = 0

    for build_sys in ["gnumake", "cmake"]:
        for dim in [1, 2, 3]:
            for use_eb in ["TRUE", "FALSE"]:
                # AMReX does not support EB in 1D
                if dim == 1 and use_eb == "TRUE":
                    continue
                for use_temp in ["TRUE", "FALSE"]:
                    for use_hdf5 in ["TRUE", "FALSE"]:
                        # Skip HDF5=TRUE if headers not available
                        if use_hdf5 == "TRUE" and not has["hdf5"]:
                            continue

                        # Compiler variants: clang only for GNUMake and if available
                        compilers = ["gnu"]
                        if build_sys == "gnumake" and has["clang"]:
                            compilers.append("clang")

                        for comp in compilers:
                            base = dict(
                                id=combo_id,
                                build_sys=build_sys,
                                dim=dim,
                                use_eb=use_eb,
                                use_temp=use_temp,
                                comp=comp,
                                debug="FALSE",
                                use_mpi="TRUE" if has["mpi"] else "FALSE",
                                use_omp="FALSE",
                                use_hdf5=use_hdf5,
                                use_cuda="FALSE",
                                use_hip="FALSE",
                                use_sycl="FALSE",
                            )
                            combos.append(base)
                            combo_id += 1

                            # Extra variants only for one representative core
                            # combo to keep the total count manageable
                            is_representative = (
                                dim == 2
                                and use_eb == "FALSE"
                                and use_temp == "FALSE"
                                and use_hdf5 == "FALSE"
                                and comp == "gnu"
                            )

                            if is_representative:
                                # DEBUG=TRUE variant
                                c = base.copy()
                                c["id"] = combo_id
                                c["debug"] = "TRUE"
                                combos.append(c)
                                combo_id += 1

                                # USE_OMP=TRUE variant
                                if has["omp"]:
                                    c = base.copy()
                                    c["id"] = combo_id
                                    c["use_omp"] = "TRUE"
                                    combos.append(c)
                                    combo_id += 1

                            # GPU backends: DIM=3, EB=FALSE, TEMP=FALSE,
                            # HDF5=FALSE, gnu only — added once per core combo
                            if (dim == 3 and use_eb == "FALSE"
                                    and use_temp == "FALSE"
                                    and use_hdf5 == "FALSE"
                                    and comp == "gnu"):
                                if has["nvcc"]:
                                    c = base.copy()
                                    c["id"] = combo_id
                                    c["use_cuda"] = "TRUE"
                                    combos.append(c)
                                    combo_id += 1
                                if has["hipcc"]:
                                    c = base.copy()
                                    c["id"] = combo_id
                                    c["use_hip"] = "TRUE"
                                    combos.append(c)
                                    combo_id += 1
                                if has["icpx"]:
                                    c = base.copy()
                                    c["id"] = combo_id
                                    c["use_sycl"] = "TRUE"
                                    combos.append(c)
                                    combo_id += 1

    total = len(combos)
    print(f"\n=== Build-Matrix: {total} combinations to test ===\n")

    # ------------------------------------------------------------------
    # 3. Build each combination
    # ------------------------------------------------------------------
    gnumake_src = os.path.join(root, "Build_Gnumake", "GNUmakefile")
    results_bm = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, combo in enumerate(combos, 1):
        label_parts = [
            combo["build_sys"],
            f"DIM={combo['dim']}",
            f"EB={combo['use_eb']}",
            f"TEMP={combo['use_temp']}",
            f"COMP={combo['comp']}",
            f"DBG={combo['debug']}",
            f"MPI={combo['use_mpi']}",
            f"OMP={combo['use_omp']}",
            f"HDF5={combo['use_hdf5']}",
        ]
        if combo["use_cuda"] == "TRUE":
            label_parts.append("CUDA")
        if combo["use_hip"] == "TRUE":
            label_parts.append("HIP")
        if combo["use_sycl"] == "TRUE":
            label_parts.append("SYCL")
        label = " | ".join(label_parts)

        print(f"[{idx:3d}/{total}] {label}")

        build_dir = tempfile.mkdtemp(prefix="exagoop_bm_")
        status = "FAIL"
        elapsed = 0.0
        error_msg = ""
        captured_stdout = ""
        captured_stderr = ""
        captured_cmd = ""

        try:
            t0 = time.time()

            if combo["build_sys"] == "gnumake":
                dst_mf = os.path.join(build_dir, "GNUmakefile")
                shutil.copy(gnumake_src, dst_mf)
                make_flags = [
                    f"DIM={combo['dim']}",
                    f"USE_EB={combo['use_eb']}",
                    f"USE_TEMP={combo['use_temp']}",
                    f"COMP={combo['comp']}",
                    f"DEBUG={combo['debug']}",
                    f"USE_MPI={combo['use_mpi']}",
                    f"USE_OMP={combo['use_omp']}",
                    f"USE_HDF5={combo['use_hdf5']}",
                    f"AMREX_USE_HDF5={combo['use_hdf5']}",
                    f"USE_CUDA={combo['use_cuda']}",
                    f"USE_HIP={combo['use_hip']}",
                    f"USE_SYCL={combo['use_sycl']}",
                    f"MPM_HOME={root}",
                    f"AMREX_HOME={root}/Submodules/amrex",
                ]
                cmd = ["make", f"-j{parallel_jobs}"] + make_flags

            else:  # cmake
                cmake_flags = [
                    f"-DEXAGOOP_DIM={combo['dim']}",
                    f"-DEXAGOOP_ENABLE_EB={_cmake_bool(combo['use_eb'])}",
                    f"-DEXAGOOP_USE_TEMP={_cmake_bool(combo['use_temp'])}",
                    f"-DEXAGOOP_ENABLE_MPI={_cmake_bool(combo['use_mpi'])}",
                    f"-DEXAGOOP_ENABLE_OPENMP={_cmake_bool(combo['use_omp'])}",
                    f"-DEXAGOOP_USE_HDF5={_cmake_bool(combo['use_hdf5'])}",
                    f"-DEXAGOOP_ENABLE_CUDA={_cmake_bool(combo['use_cuda'])}",
                    f"-DEXAGOOP_ENABLE_HIP={_cmake_bool(combo['use_hip'])}",
                    f"-DEXAGOOP_ENABLE_SYCL={_cmake_bool(combo['use_sycl'])}",
                    f"-DCMAKE_BUILD_TYPE={'Debug' if combo['debug'] == 'TRUE' else 'Release'}",
                ]
                if combo["comp"] == "clang":
                    cmake_flags.append("-DCMAKE_CXX_COMPILER=clang++")
                # cmake + make chained via shell so we can run two commands
                # in the same cwd; use a list-compatible shell invocation
                cmd = ["sh", "-c",
                       "cmake " + " ".join(cmake_flags) + f" {root}"
                       + f" && make -j{parallel_jobs}"]

            captured_cmd = " ".join(cmd)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=build_dir,
            )
            elapsed = time.time() - t0
            captured_stdout = proc.stdout[-5000:]
            captured_stderr = proc.stderr[-5000:]

            if proc.returncode == 0:
                status = "PASS"
                print(f"         -> PASS  ({elapsed:.1f}s)")
            else:
                error_msg = captured_stderr or captured_stdout
                last_line = error_msg.splitlines()[-1] if error_msg else ""
                print(f"         -> FAIL  ({elapsed:.1f}s)")
                if last_line:
                    print(f"            {last_line}")

        except subprocess.TimeoutExpired:
            elapsed = float(timeout)
            status = "TIMEOUT"
            error_msg = f"Build timed out after {timeout}s"
            print(f"         -> TIMEOUT ({timeout}s)")

        except Exception as exc:
            status = "ERROR"
            error_msg = str(exc)
            print(f"         -> ERROR: {exc}")

        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

        combo["status"] = status
        combo["elapsed"] = round(elapsed, 2)
        combo["error_msg"] = error_msg
        combo["stdout"] = captured_stdout
        combo["stderr"] = captured_stderr
        combo["command"] = captured_cmd
        results_bm.append(combo)

    # ------------------------------------------------------------------
    # 4. Write results
    # ------------------------------------------------------------------
    json_path = os.path.join(output_dir, f"build_matrix_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results_bm, f, indent=2)

    n_pass    = sum(1 for r in results_bm if r["status"] == "PASS")
    n_fail    = sum(1 for r in results_bm if r["status"] == "FAIL")
    n_timeout = sum(1 for r in results_bm if r["status"] == "TIMEOUT")
    n_error   = sum(1 for r in results_bm if r["status"] == "ERROR")

    col_w = 8  # build_sys column width
    header = (
        f"{'#':>4}  {'SYS':{col_w}}  {'DIM':3}  {'EB':5}  {'TEMP':5}  "
        f"{'COMP':6}  {'DBG':5}  {'MPI':5}  {'OMP':5}  {'HDF5':5}  "
        f"{'GPU':5}  {'TIME':>8}  STATUS"
    )
    sep = "-" * len(header)

    lines = [
        f"ExaGOOP Build Matrix — {timestamp}",
        f"Total: {total}  PASS: {n_pass}  FAIL: {n_fail}  "
        f"TIMEOUT: {n_timeout}  ERROR: {n_error}",
        sep,
        header,
        sep,
    ]
    for r in results_bm:
        gpu = (
            "CUDA" if r["use_cuda"] == "TRUE" else
            "HIP"  if r["use_hip"]  == "TRUE" else
            "SYCL" if r["use_sycl"] == "TRUE" else "none"
        )
        lines.append(
            f"{r['id']:4d}  {r['build_sys']:{col_w}}  {r['dim']:3d}  "
            f"{r['use_eb']:5}  {r['use_temp']:5}  {r['comp']:6}  "
            f"{r['debug']:5}  {r['use_mpi']:5}  {r['use_omp']:5}  "
            f"{r['use_hdf5']:5}  {gpu:5}  {r['elapsed']:>7.1f}s  {r['status']}"
        )
    lines += [
        sep,
        f"PASS: {n_pass}/{total}  FAIL: {n_fail}  "
        f"TIMEOUT: {n_timeout}  ERROR: {n_error}",
    ]

    # Append last 30 lines of stderr for every non-passing result
    failure_details = []
    for r in results_bm:
        if r["status"] not in ("FAIL", "ERROR"):
            continue
        stderr_tail = "\n".join(r.get("stderr", "").splitlines()[-30:])
        failure_details += [
            "",
            f"--- #{r['id']} {r['build_sys']} DIM={r['dim']} "
            f"EB={r['use_eb']} TEMP={r['use_temp']} COMP={r['comp']} "
            f"HDF5={r['use_hdf5']} [{r['status']}] ---",
            f"COMMAND: {r.get('command', '')}",
            stderr_tail or r.get("error_msg", "(no stderr)"),
        ]

    summary = "\n".join(lines)
    if failure_details:
        summary += "\n\n=== FAILURE DETAILS (last 30 lines of stderr) ===\n"
        summary += "\n".join(failure_details)

    print("\n" + summary)

    txt_path = os.path.join(output_dir, f"build_matrix_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(summary + "\n")

    print(f"\nResults written to:\n  {json_path}\n  {txt_path}")
    return results_bm


# ============================================================
# PLACEHOLDER: Test Case Definitions
# ============================================================
TEST_CASES = {
    "1D_Axial_Bar_Vibration": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_1DAxialBarVibration.inp",
        "postproc_scripts": [
            "./PostProcess/Calculate_Error.py",
            "./PostProcess/plot_energy.py",
            "./PostProcess/plot_vel.py",
            "./PostProcess/AnimateVelocity.py"
        ],
        "parameter_space": {
            "dimension": [1],
            "np_per_cell_x": [1,2],
            "order_scheme": [1,2,3],
            "alpha_pic_flip": [1.0],            
            "stress_update_scheme": [1],
            "CFL": [0.1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]            
        }
    },

    # ========================================================
    # ADD YOUR OTHER TEST CASES HERE
    # ========================================================
    "1D_Heat_Conduction": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_1DHeatConduction.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"             
        ],
        "parameter_space": {
            "dimension": [1],
            "np_per_cell_x": [1.2],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]        
        }
    },
    
    "1D_Heat_Conduction_HeatFlux": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_1DHeatConduction_HeatFlux.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"             
        ],
        "parameter_space": {
            "dimension": [2],
            "np_per_cell_x": [1,2],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]        
        }
    },
    
    "1D_Heat_Conduction_Convective": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_1DHeatConduction_Convective.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"             
        ],
        "parameter_space": {
            "dimension": [2],
            "np_per_cell_x": [1,2],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]        
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
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]        
        }
    },
    
    "2D_Heat_Conduction_Cylinder_Dirichlet": {
        "generator_script": "./PreProcess/Generate_MPs_Inputfile_Generic.py",
        "input_file": "Inputs_2DHeat_Conduction_Cylinder_Dirichlet.inp",
        "postproc_scripts": [
            "./PostProcess/Plot_Temperature.py"            
        ],
        "parameter_space": {            
            "np_per_cell_x": [2],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]        
        }
    },
    
    "Dam_Break": {
        "generator_script": "./PreProcess/generate_particle_and_inputfiles.py",
        "input_file": "Inputs_DamBreak.inp",
        "postproc_scripts": [
            "./PostProcess/plot_waterfront.py"            
        ],
        "parameter_space": {     
            "dimension": [2], 
            "no_of_cell_in_x": [100],      
            "np_per_cell_x": [1],            
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]            
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
            "order_scheme": [1,2,3],            
            "stress_update_scheme": [1],
            "build_with_hdf": [False],
            "output_format": ["ascii","hdf5"],
            "filename_prefix": ["mpm_particles"]       
        }
    },

    # Add more test cases as needed...
}


# ============================================================
# Main Sweep Driver
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERROR_TOL = 1e-1
results = []

import json

# ANSI colors
BLACK_TICK = "✔"  
RED = "\033[91m"
RESET = "\033[0m"

def print_dynamic_test_table(json_file):
    # Load JSON
    with open(json_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No test results found.")
        return

    # Collect all keys across all entries
    all_keys = set()
    for entry in results:
        all_keys.update(entry.keys())

    # Ensure "test" is first and "pass" is last if present
    ordered_keys = []
    if "test" in all_keys:
        ordered_keys.append("test")

    for k in sorted(all_keys):
        if k not in ("test", "pass"):
            ordered_keys.append(k)

    if "pass" in all_keys:
        ordered_keys.append("pass")

    # Compute column widths
    col_widths = {}
    for key in ordered_keys:
        max_len = max(len(str(entry.get(key, ""))) for entry in results)
        col_widths[key] = max(max_len, len(key)) + 2

    # Print header
    header = ""
    for key in ordered_keys:
        header += f"{key:<{col_widths[key]}}"
    print(header)
    print("-" * len(header))

    # Print rows
    for entry in results:
        row = ""
        for key in ordered_keys:
            value = entry.get(key, "")

            if key == "pass":
                icon = "✔" if value else f"{RED}✘{RESET}"
                row += f"{icon:<{col_widths[key]}}"
            else:
                row += f"{str(value):<{col_widths[key]}}"

        print(row)


def _run_parameter_sweeps():
    for test_name, cfg in TEST_CASES.items():

        print("\n====================================================")
        print(f"Running Test Case: {test_name}")
        print("====================================================")

        test_dir = os.path.join(ROOT, "Tests", test_name)

        # Deleting existing solution and diagnostics folders
        sol_dir = os.path.join(test_dir, "Solution")
        diagn_dir = os.path.join(test_dir, "Diagnostics")
        tmp_build_dir = os.path.join(test_dir, "tmp_build_dir")

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

        if test_name == "1D_Axial_Bar_Vibration":
            print('Nothing to do')
            Run_ParameterSweep_1D_Axial_Bar_Vibration(cfg)
        elif test_name == "1D_Heat_Conduction":
            print('Nothing to do')
            Run_ParameterSweep_1D_HeatConduction(cfg)
        elif test_name == "1D_Heat_Conduction_HeatFlux":
            print('Nothing to do')
            Run_ParameterSweep_1D_HeatConduction_HeatFlux(cfg)
        elif test_name == "1D_Heat_Conduction_Convective":
            print('Nothing to do')
            Run_ParameterSweep_1D_HeatConduction_Convective(cfg)
        elif test_name == "2D_Heat_Conduction":
            print('Nothing to do')
            Run_ParameterSweep_2D_HeatConduction(cfg)
        elif test_name == "2D_Heat_Conduction_Cylinder_Dirichlet":
            print('Nothing to do')
            Run_ParameterSweep_2D_HeatConduction_Cylinder_Dirichlet(cfg)
        elif test_name == "Dam_Break":
            print('Nothing to do')
            Run_ParameterSweep_Dambreak(cfg)
        elif test_name == "Elastic_disk_collision":
            print('Nothing to do')
            Run_ParameterSweep_EDC(cfg)

    # Save results
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll test cases complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ExaGOOP test runner and build-matrix validator"
    )
    parser.add_argument(
        "--build-matrix",
        action="store_true",
        help=(
            "Run the full build-configuration matrix (GNUMake + CMake × "
            "DIM 1/2/3 × USE_EB × USE_TEMP × compiler × DEBUG × MPI × OMP × "
            "HDF5 × GPU backends) instead of the parameter-sweep tests."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Per-combination build timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        metavar="N",
        help="Parallel make jobs per build combination (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT,
        metavar="DIR",
        help="Directory for build-matrix result files (default: repo root).",
    )
    parser.add_argument(
        "--show-log",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After --build-matrix completes, print full stdout+stderr "
            "for result number N (0-based index into the results list)."
        ),
    )
    args = parser.parse_args()

    if args.build_matrix:
        bm_results = Run_Build_Matrix(
            cfg={},
            root=ROOT,
            timeout=args.timeout,
            parallel_jobs=args.jobs,
            output_dir=args.output_dir,
        )
        if args.show_log is not None:
            idx = args.show_log
            if 0 <= idx < len(bm_results):
                r = bm_results[idx]
                print(f"\n=== Result {idx}: {r['status']} ===")
                print("COMMAND:", r.get("command", ""))
                print("--- STDOUT ---")
                print(r.get("stdout", ""))
                print("--- STDERR ---")
                print(r.get("stderr", ""))
            else:
                print(f"--show-log {idx}: index out of range "
                      f"(0–{len(bm_results) - 1})")
    else:
        _run_parameter_sweeps()
        print_dynamic_test_table('sweep_results.json')


