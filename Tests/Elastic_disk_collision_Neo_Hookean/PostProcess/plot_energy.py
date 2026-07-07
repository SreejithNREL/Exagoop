import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np
import argparse

def read_matpnt(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())   # first line = number of particles
        data = np.loadtxt(f,skiprows=5)
    return data


def Animate(folder,moviefile):    
    files = sorted(glob.glob(f"{folder}/matpnt_t*"))
    
    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter([], [], s=1, c='blue')
    
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dambreak Flow")
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,
    
    def update(frame):
        data = read_matpnt(files[frame])
        # 2D format: phase, x, y, rad, dens, vx, vy, ...
        x = data[:,0]
        y = data[:,1]
        speed = np.sqrt(data[:,5]**2 + data[:,6]**2)
        scat.set_array(speed)
        scat.set_offsets(np.c_[x, y])
        ax.set_title(f"Dambreak Flow — Frame {frame}")
        return scat,
    
    ani = animation.FuncAnimation(fig, update, frames=len(files),
                                  init_func=init, blit=True, interval=50)
    
    ani.save(moviefile, fps=30, dpi=150)
    plt.close(fig)
    
def main():
    
    parser = argparse.ArgumentParser(description="2D heat conduction: numeric vs exact vs error.")    
    parser.add_argument("--folder", type=str, default="Solution/ascii_files",
                        help="Folder containing matpnt files")
    parser.add_argument("--outputpic", type=str, default="Solution/ascii_files/Pics",
                        help="Folder containing matpnt files")
    parser.add_argument("--outputmovie", type=str, default="Solution/ascii_files/Pics",
                        help="Folder containing matpnt files")
    parser.add_argument("--skiprows", type=int, default=5,
                        help="Number of metadata rows to skip")
    parser.add_argument("--energyfile", type=str, default=5,
                        help="Number of metadata rows to skip") 
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--N_terms", type=int, default=25)
    parser.add_argument("--Tcol", type=int, default=48,
                        help="Column index for temperature")
    args = parser.parse_args()
    
    Animate(args.folder,args.outputmovie)

    data=np.loadtxt(args.energyfile,skiprows=1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.grid('on')
    
    plt.plot(data[:,1],data[:,2],label='TKE',color='r')
    plt.plot(data[:,1],data[:,3],label='TSE',color='blue')
    plt.plot(data[:,1],data[:,4],label='TE',color='black')
    plt.xlabel("Time ")
    plt.ylabel("Energy ")
    lgd = ax.legend()  
    # saving the file.Make sure you 
    # use savefig() before show().
    plt.savefig(args.outputpic)

if __name__ == "__main__":
    main()