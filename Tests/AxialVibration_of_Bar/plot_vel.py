import numpy as np
import matplotlib.pyplot as plt
from sys import argv

data=np.loadtxt('./Diagnostics/VelComponents.dat',skiprows=1)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.grid('on')

plt.plot(data[:,1],data[:,2],label='Exact',color='r')
#plt.plot(data[:,1],data[:,2],label='MPM',color='blue')
plt.xlabel("Time ")
plt.ylabel("CM Velocity")
lgd = ax.legend()  
# saving the file.Make sure you 
# use savefig() before show().
plt.savefig(argv[1])
