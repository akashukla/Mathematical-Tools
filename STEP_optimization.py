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

#Read the coil positions
coilpos = np.genfromtxt('data/STEP_upper_coils.csv',delimiter=',',skip_header=6, usecols=(0,1))
#Reflect about R axis to get lower coils too
coilpos = np.r_[coilpos,np.c_[coilpos[:,0], -coilpos[:,1]]]

#Read the control positions
conpos = np.genfromtxt('data/STEP_upper_controlpoints4.csv',delimiter=',',skip_header=6, usecols=(0,1))
#Reflect about R axis to get lower coils too
conpos = np.r_[conpos,np.c_[conpos[:,0], -conpos[:,1]]]

#Read the x point posision
xpos = np.genfromtxt('data/STEP_upper_xpoint.csv',delimiter=',',skip_header=6, usecols=(0,1))
xpostemp = np.zeros((2,2))
xpostemp[0] = xpos
xpostemp[1] = np.r_[xpos[0], -xpos[1]]
xpos = xpostemp

#Specify the currents for initial test
Is = np.r_[0,0,4,0,0,0,-8,-5]*MA/2
#duplicate for lower coils
Is = np.r_[Is,Is]

def psi_config(r,z,Is):
    psiret=0
    #Psi from the PF coils
    for l in range(len(Is)):
        psiret += psi_f(r,0,z,coilpos[l,1], coilpos[l,0],Is[l])
    #Psi from the plasma current
    psiret+= psi_f(r,0,z,0,3.1,16.5*MA)
    return psiret


#Optimize

from scipy.optimize import least_squares
delx = 0.0001*1.55
dely=0.0001*1.55
A=1/64
B=(1e-6)/64
def func(I_in):
    I_in = np.r_[I_in,I_in]
    #Error_X = (( psi_config(xpos[0,0]-delx,xpos[0,1],I_in) - psi_config(xpos[0,0]+delx,xpos[0,1],I_in))/delx)**2 \
    #        + (( psi_config(xpos[0,0],xpos[0,1]-dely,I_in) - psi_config(xpos[0,0],xpos[0,1]+dely,I_in))/dely)**2
    #Error_sep = np.sum((psi_config(xpos[0,0],xpos[0,1],I_in) - psi_config(conpos[:,0], conpos[:,1],I_in))**2)
    #Error_currents = np.sum(I_in**2)
    #return Error_X + A*Error_sep + B*Error_currents
    #return Error_X , A*Error_sep , B*Error_currents
    Error_X = ( psi_config(xpos[0,0]-delx,xpos[0,1],I_in) - psi_config(xpos[0,0]+delx,xpos[0,1],I_in))/delx, \
            ( psi_config(xpos[0,0],xpos[0,1]-dely,I_in) - psi_config(xpos[0,0],xpos[0,1]+dely,I_in))/dely
    Error_sep = psi_config(xpos[0,0],xpos[0,1],I_in) - psi_config(conpos[:,0], conpos[:,1],I_in)
    Error_currents = I_in
    return np.r_[Error_X,A*Error_sep,B*I_in]

def func_check(I_in):
    I_in = np.r_[I_in,I_in]
    Error_X = (( psi_config(xpos[0,0]-delx,xpos[0,1],I_in) - psi_config(xpos[0,0]+delx,xpos[0,1],I_in))/delx)**2 \
            + (( psi_config(xpos[0,0],xpos[0,1]-dely,I_in) - psi_config(xpos[0,0],xpos[0,1]+dely,I_in))/dely)**2
    Error_sep = np.sum((psi_config(xpos[0,0],xpos[0,1],I_in) - psi_config(conpos[:,0], conpos[:,1],I_in))**2)
    Error_currents = np.sum(I_in**2)
    return Error_X + A*Error_sep + B*Error_currents, Error_X,A*Error_sep,B*Error_currents




out = least_squares(func,x0=Is[0:len(Is)//2])



Rgrid = np.arange(0,8,0.05)
Zgrid = np.arange(-8.0,8.0,0.05)
#theta = np.arange(0,np.pi)
psiplot = np.zeros((len(Rgrid), len(Zgrid)))

for i,r in enumerate(Rgrid):
    for j,z in enumerate(Zgrid):
        #psiplot[i,j] = psi_config(r,z,Is)
        psiplot[i,j] = psi_config(r,z,np.r_[out.x,out.x])





levels = np.arange(0.0,2.3,0.01)

#choose levels based on psiSep
psisep = psi_config(xpos[:,0],xpos[:,1],np.r_[out.x,out.x])[0]
levels = np.arange(psisep-0.5,psisep+0.5,0.02)

X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T, levels = levels)
ax.clabel(CS, inline=True, fontsize=10, levels = levels)
fig.colorbar(CS)
ax.scatter(conpos[:,0],conpos[:,1])
ax.scatter(xpos[:,0],xpos[:,1])

# #Calculate Geometry
# R0 = acs[-1]
# Ip = Ics[-1]/MA
# xsep,ysep=1.76,4.0
# Rmin = 0.38
# Rmax = 4.73
# a = (Rmax-Rmin)/2
# R = a + Rmin
# kappa = ysep/a
# delta = (R-xsep)/a
# A=R/a
# 
# print("R0 = %1.2f m, a = %1.2f m, R0/a=%1.2f, kappa = %1.2f, delta = %1.2f "%(R0,a,A,kappa,delta))
# 
# 
# 
# title_string = r"$R = %1.2f,\, a = %1.2f ,\, R/a=%1.2f,$"%(R,a,A)+"\n" \
#     +r"$\kappa = %1.2f,\, \delta = %1.2f,\, I_p = %1.2f MA$"%(kappa,delta,Ip)
#plt.title(title_string)
plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.show()
