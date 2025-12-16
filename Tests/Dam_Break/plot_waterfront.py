import numpy as np
import matplotlib.pyplot as plt
from sys import argv

data=np.loadtxt('./Diagnostics/MinMaxPosition.dat',skiprows=1)
exp=np.loadtxt('ExperimentalData.dat')

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.grid('on')
H=0.2
g=9.81
plt.plot(data[:,1]/np.sqrt(H/g),data[:,3]/H,label='ExaGOOP',color='r')
plt.scatter(exp[:,0],exp[:,1],label='Experiments')
plt.legend()
plt.xlabel("Time ")
plt.xlim(0,1.5)
plt.ylabel("X* ")
plt.savefig('Waterfront.png')
plt.show()
