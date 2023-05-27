import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

e = np.genfromtxt("allcell.csv", delimiter=",", usecols=(0,1,2,3))

namelist = [r'$g_{11}$', r'$g_{22}$',r'$g_{33}$',]
fig,ax = plt.subplots(3,1, sharex=True)
for i in range(len(ax)):
    ax[i].loglog(e[:,0], e[:,i+1])
    ax[i].loglog(e[:,0], e[:,0]**-2, label = r'$N^{-2}$')
    ax[i].set_title(namelist[i])
    ax[i].set_ylabel("L2 Error")
    ax[i].legend()
ax[len(ax)-1].set_xlabel(r"$N$")

plt.show()
