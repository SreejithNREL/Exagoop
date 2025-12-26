import numpy as np
import matplotlib.pyplot as plt
from sys import argv

data=np.loadtxt(argv[1],skiprows=1)
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
plt.savefig(argv[2])

