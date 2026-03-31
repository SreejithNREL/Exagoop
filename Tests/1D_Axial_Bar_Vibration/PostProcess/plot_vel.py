import numpy as np
import matplotlib.pyplot as plt
from sys import argv

data=np.loadtxt(argv[1],skiprows=1)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.grid('on')

modenumber=1
L = 25.0
E = 100.0
rho = 1.0
v0=0.1
beta_n = (2 * modenumber - 1.0) / 2 * np.pi / L
w_n = np.sqrt(E / rho) * beta_n
Vmex = v0 / (beta_n * L) * np.cos(w_n * data[:,1]);

plt.plot(data[:,1],data[:,2],label='MPM',color='r')
plt.plot(data[:,1],Vmex,label='Exact',color='blue')
plt.xlabel("Time ")
plt.ylabel("CM Velocity")
lgd = ax.legend()  
# saving the file.Make sure you 
# use savefig() before show().
plt.savefig(argv[2])
