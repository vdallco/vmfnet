import os
import subprocess

BSP_SRC_DIR = r"C:\\Program Files (x86)\\Steam\\steamapps\\common\\GarrysMod\\garrysmod\\maps"
VMF_OUT_DIR = r".\\vmf\\output"
BSPSRC_BAT = r"C:\\Users\\Cody\\Downloads\\bspsrc-windows\\bspsrc.bat"

os.makedirs(VMF_OUT_DIR, exist_ok=True)

for bsp_file in os.listdir(BSP_SRC_DIR):
    if bsp_file.lower().endswith(".bsp"):
        bsp_path = os.path.join(BSP_SRC_DIR, bsp_file)
        vmf_file = os.path.splitext(bsp_file)[0] + ".vmf"
        vmf_path = os.path.join(VMF_OUT_DIR, vmf_file)
        print(f"Decompiling {bsp_file} to {vmf_path}...")
        command = [BSPSRC_BAT, "-o", vmf_path, bsp_path]
        subprocess.run(command, shell=True, check=True)