import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
mu0 = 1.256e-6
#mu0=1.
MA=1e6

def Aphi_f(r,theta, a, I):
    """from Jackson in spherical coords
    Calculate A_phi from a loop with radius a, current I

    """
    ksq = 4*a*r*np.sin(theta)/(a**2+r**2 + 2*a*r*np.sin(theta))

    fac1 = mu0/(4*np.pi)

    num2 = 4*I*a
    denom2 = np.sqrt(a**2 + r**2 + 2*a*r*np.sin(theta))
    fac2 = num2/denom2

    num3 = (2-ksq)*ellipk(ksq) - 2*ellipe(ksq)
    denom3 = ksq
    fac3 = num3/denom3

    Aphi = fac1*fac2*fac3
    return Aphi


def psi_f(rho,rho0, z,z0, a, I):
    """rho,z: cylindrial coords
    rho0,z0: center position of loop
    a,I: loop radius, loop current
    Return psi(rho,z)
    """

    #convert to spherical coords
    rs = np.sqrt((rho-rho0)**2 + (z-z0)**2)
    theta = np.arccos((z-z0)/rs)
    #calculate A_phi from spherical coords
    Aphi = Aphi_f(rs,theta,a,I)

    #calculate psi = A_phi*rho
    psi = Aphi*rho
    return psi

#z_up = -3.0
#z_down = 3.0
#rshift0 = 0.0
#rshift_down = 0.5
#rshift_up = 0.5
#a0 = 2.0
#a_up = 2.0
#a_down = 2.0
#I0 = 10*MA
#I_down = 10*MA
#I_up = 10*MA

#Generate a set of Loops. Specify radius, current, radial center, vertical center
PF = []
#rcs = np.r_[0,0.5,0.5]
#zcs = np.r_[-3,3]
#Ics = np.r_[10,10,10]*MA
#acs = np.r_[2,2,2]

#last one is plasma current

#from STPP paper
rcs = np.r_[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
acs = np.r_[2.3, 2.3, 7.4, 7.4, 8.7, 8.7, 4.2]/1.2
zcs = np.r_[ 9.7, -9.7, -6.6, 6.6, 2.03, -2.03, 0.0]
Ics = np.r_[3.885,3.885, -8.0,-8.0, -4.375,-4.375, 50.0]*MA

#Modify like STEP paper
rcs = np.r_[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
acs = np.r_[1.8, 1.8, 5.3, 5.3, 7., 7., 3.1]
zcs = np.r_[ 6., -6., -5.5, 5.5, 2., -2., 0.0]
Ics = np.r_[3.885,3.885, -8.0,-8.0, -4.375,-4.375, 0.0]*MA*0.5
Ics[-1] = 17*MA

#Modify on my own:
rcs = np.r_[0, 0 ,0 ,0, 0, 0, 0, 0, 0]
Ics = np.r_[3.885,3.885, -8.0,-8.0, -4.375,-4.375,-4,-4, 0.0]*MA*0.5
Ics[-1] = 17*MA
acs = np.r_[2.4, 2.4, 5.3, 5.3, 6., 6., 6, 6, 3.1]
zcs = np.r_[ 5., -5., -5.5, 5.5, 2., -2.,0.5, -0.5, 0.0]*1


for i in range(len(acs)):
    PF.append({'r0':rcs[i],'z0' : zcs[i], 'a':acs[i], 'I':Ics[i]})


Rgrid = np.arange(0,8,0.05)
Zgrid = np.arange(-8.0,8.0,0.05)
#theta = np.arange(0,np.pi)
psiplot = np.zeros((len(Rgrid), len(Zgrid)))

for i,r in enumerate(Rgrid):
    for j,z in enumerate(Zgrid):
        for l in range(len(PF)):
            psiplot[i,j] += psi_f(r,PF[l]['r0'],z,PF[l]['z0'], PF[l]['a'],PF[l]['I'])





#levels  = np.arange(6.866,6.868,.001)
#6.868 is just inside
#6.867 is sep
#6.866 is just outside
levels = np.arange(0.16,0.19,0.001)
#levels = np.arange(0.0,0.5,0.01)
#levels = [-1,0,1]
X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T, levels = levels)
ax.clabel(CS, inline=True, fontsize=10, levels = levels)
fig.colorbar(CS)

#Calculate Geometry
R0 = acs[-1]
Ip = Ics[-1]/MA
xsep,ysep=1.76,4.0
Rmin = 0.38
Rmax = 4.73
a = (Rmax-Rmin)/2
R = a + Rmin
kappa = ysep/a
delta = (R0-xsep)/a
A=R/a

print("R0 = %1.2f m, a = %1.2f m, R0/a=%1.2f, kappa = %1.2f, delta = %1.2f "%(R0,a,A,kappa,delta))



title_string = r"$R = %1.2f,\, a = %1.2f ,\, R/a=%1.2f,$"%(R,a,A)+"\n" \
    +r"$\kappa = %1.2f,\, \delta = %1.2f,\, I_p = %1.2f MA$"%(kappa,delta,Ip)
#plt.title(r"$R_0 = %1.2f,\, a = %1.2f ,\, R_0/a=%1.2f,\,\n \kappa = %1.2f,\, \delta = %1.2f$"%(R0,a,A,kappa,delta))
plt.title(title_string)
plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.show()
