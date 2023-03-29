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
coilpos = np.genfromtxt('data/STEP_upper_coils_BSD.csv',delimiter=',',skip_header=6, usecols=(0,1))

#Add center coil
#coilpos = np.row_stack((coilpos,[0.1,0]))
#coilpos = np.row_stack((coilpos,[0.1,1]))
#coilpos = np.row_stack((coilpos,[0.1,3]))
coilpos = np.row_stack((coilpos,[0.1,2]))
coilpos = np.row_stack((coilpos,[0.1,0]))
coilpos = np.row_stack((coilpos,[0.1,0.75]))
coilpos = np.row_stack((coilpos,[0.1,0.9]))
coilpos = np.row_stack((coilpos,[1.2,8]))
#Reflect about R axis to get lower coils too
coilpos = np.r_[coilpos,np.c_[coilpos[:,0], -coilpos[:,1]]]

#Read the control positions
conpos = np.genfromtxt('data/STEP_upper_controlpoints_BSD2.csv',delimiter=',',skip_header=6, usecols=(0,1))
#conpos_addendum = np.genfromtxt('data/STEP_upper_controlpoints_BSD2_addendum.csv',delimiter=',',skip_header=6, usecols=(0,1))
conpos_addendum_plate = np.genfromtxt('data/STEP_upper_controlpoints_BSD2_addendum_plate.csv',delimiter=',',skip_header=6, usecols=(0,1))
plate_positions =  np.genfromtxt('data/STEP_upper_controlpoints_BSD2_plate_positions.csv',delimiter=',',skip_header=6, usecols=(0,1))
#conpos_addendum_plate = np.genfromtxt('data/STEP_upper_controlpoints_BSD2_addendum_innerplate.csv',delimiter=',',skip_header=6, usecols=(0,1))
#conpos = np.row_stack((conpos,conpos_addendum))
conpos = np.row_stack((conpos,conpos_addendum_plate))

#Try relaxing outer midplane
conpos[11][0]=4.35

#Reflect about R axis to get lower coils too
conpos = np.r_[conpos,np.c_[conpos[:,0], -conpos[:,1]]]


#Read the x point posision
xpos = np.genfromtxt('data/STEP_upper_xpoint.csv',delimiter=',',skip_header=6, usecols=(0,1))
xpostemp = np.zeros((2,2))
xpostemp[0] = xpos
xpostemp[1] = np.r_[xpos[0], -xpos[1]]
xpos = xpostemp

#Relax x point condition 
#xpos[:,0] = 2.0
#xpos[:,1] = 4.0

#Specify the currents for initial test
#LXD
#Is = np.r_[0,0,4,0,0,0,-8,-5]*MA/2
##Is = np.zeros(coilpos.shape[0]//2)
##duplicate for lower coils
#Is = np.r_[Is,Is]

#BSD
Is = np.r_[4,-4,-8,-5,-5,-5,-5,-5,2]*MA/2
#Is = np.zeros(coilpos.shape[0]//2)
#duplicate for lower coils
Is = np.r_[Is,Is]


#Dipole currents
I_mono = 16.5*MA
a_mono = 2.6
delR=0.1*a_mono
#dipole_fac=1/16
#I_dipole = I_mono*a_mono/delR*dipole_fac

#quadrapole
delZ = 0.1*a_mono

def psi_config(r,z,Is):
    psiret=0
    #Psi from the PF coils
    for l in range(len(Is)-2):
        psiret += psi_f(r,0,z,coilpos[l,1], coilpos[l,0],Is[l])
    #Psi from the plasma current
    psiret+= psi_f(r,0,z,0,a_mono,I_mono)
    #Try another plasma current
    #psiret+= psi_f(r,0,z,0,a_mono/2,I_mono)

    #Psi from dipole
    I_dipole=Is[-2]
    psiret+= psi_f(r,0,z,0,a_mono-delR,+I_dipole)
    psiret+= psi_f(r,0,z,0,a_mono+delR,-I_dipole)

    #Psi from quadrapole
    I_quad=Is[-1]
    psiret+= psi_f(r,0,z,0,a_mono,-2*I_quad)
    psiret+= psi_f(r,0,z,delZ,a_mono,I_quad)
    psiret+= psi_f(r,0,z,-delZ,a_mono,I_quad)

    return psiret


#Optimize

from scipy.optimize import least_squares
delx = 0.01*1.55
dely=0.01*1.55
A=5
#B=(1e-12)*1e4
#C=(1e-12)*10000
#D=(1e-12)*10000
B,C,D = 0,0,0
def func(I_in):
    I_temp = np.r_[I_in[:-2],I_in]
    #Error_X = (( psi_config(xpos[0,0]-delx,xpos[0,1],I_in) - psi_config(xpos[0,0]+delx,xpos[0,1],I_in))/delx)**2 \
    #        + (( psi_config(xpos[0,0],xpos[0,1]-dely,I_in) - psi_config(xpos[0,0],xpos[0,1]+dely,I_in))/dely)**2
    #Error_sep = np.sum((psi_config(xpos[0,0],xpos[0,1],I_in) - psi_config(conpos[:,0], conpos[:,1],I_in))**2)
    #Error_currents = np.sum(I_in**2)
    #return Error_X + A*Error_sep + B*Error_currents
    #return Error_X , A*Error_sep , B*Error_currents

    Error_currents = I_temp**2
    Error_X1 = ( psi_config(xpos[0,0]-delx,xpos[0,1],I_temp) - psi_config(xpos[0,0]+delx,xpos[0,1],I_temp))/delx
    Error_X2 = ( psi_config(xpos[0,0],xpos[0,1]-dely,I_temp) - psi_config(xpos[0,0],xpos[0,1]+dely,I_temp))/dely
    #Find actual x point
    #xpt = least_squares(derivfunc, x0=[xpos[0,0], xpos[0,1]], args = ([I_temp])).x
    #Error_X = np.r_[xpos[0,0]-xpt[0], xpos[0,1]-xpt[1]]

    Error_sep = psi_config(xpos[0,0],xpos[0,1],I_temp) - psi_config(conpos[:,0], conpos[:,1],I_temp)
    #Error_sep = psi_config(xpt[0],xpt[1],I_temp) - psi_config(conpos[:,0], conpos[:,1],I_temp)
    return np.r_[Error_X1,Error_X2,A*Error_sep,B*I_temp]#, C*I_temp[-2], D*I_temp[-1]]
    #return np.r_[Error_X,A*Error_sep,B*I_temp]

def func_check(I_in):
    I_in = np.r_[I_in[:-2],I_in]
    Error_X = (( psi_config(xpos[0,0]-delx,xpos[0,1],I_in) - psi_config(xpos[0,0]+delx,xpos[0,1],I_in))/delx)**2 \
            + (( psi_config(xpos[0,0],xpos[0,1]-dely,I_in) - psi_config(xpos[0,0],xpos[0,1]+dely,I_in))/dely)**2
    Error_sep = np.sum((psi_config(xpos[0,0],xpos[0,1],I_in) - psi_config(conpos[:,0], conpos[:,1],I_in))**2)
    Error_currents = np.sum(I_in**2)
    return Error_X + A*Error_sep + B*Error_currents, Error_X,A*Error_sep,B*Error_currents

#func to find the actual x point
#call with outderiv = least_squares(derivfunc,x0 = [xpos[0,0], xpos[0,1]])
def derivfunc(xvec,I_temp):
     Error_X1 = ( psi_config(xvec[0]-delx,xvec[1],I_temp) - psi_config(xvec[0]+delx,xvec[1],I_temp))/delx
     Error_X2 = ( psi_config(xvec[0],xvec[1]-dely,I_temp) - psi_config(xvec[0],xvec[1]+dely,I_temp))/dely
     return Error_X1,Error_X2





out = least_squares(func,x0=np.r_[Is[0:len(Is)//2],10*MA,50*MA] )
#out = least_squares(func,x0=np.zeros(len(Is)//2+2))



Rgrid = np.arange(0,8,0.05)
Zgrid = np.arange(-8.0,8.0,0.05)
#theta = np.arange(0,np.pi)
psiplot = np.zeros((len(Rgrid), len(Zgrid)))

for i,r in enumerate(Rgrid):
    for j,z in enumerate(Zgrid):
        #psiplot[i,j] = psi_config(r,z,Is)
        psiplot[i,j] = psi_config(r,z,np.r_[out.x[:-2],out.x])





levels = np.arange(0.0,2.3,0.01)

#choose levels based on psiSep
psisep = psi_config(xpos[:,0],xpos[:,1],np.r_[out.x[:-2],out.x])[0]
levels = np.arange(psisep-0.5,psisep+0.5,0.02)

X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T, levels = levels)
ax.clabel(CS, inline=True, fontsize=10, levels = levels)
fig.colorbar(CS)
ax.scatter(conpos[:,0],conpos[:,1], label = 'Control Points')
ax.scatter(xpos[:,0],xpos[:,1], label = 'Desired X point')

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


#Find actual x point
xpt = least_squares(derivfunc, x0=[xpos[0,0], xpos[0,1]], args = ([np.r_[out.x[:-2],out.x]])).x
psisep = psi_config(xpt[0],xpt[1],np.r_[out.x[:-2],out.x])
levels = [0,psisep]
X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T,[psisep])#] levels = levels)
#ax.clabel(CS, inline=True, fontsize=10)#, levels = levels)
#fig.colorbar(CS)
ax.scatter(conpos[:,0],conpos[:,1], label = 'Control Points')
ax.scatter(xpos[:,0],xpos[:,1], label = 'Desired X point')
#Calculate Geometry
R0 = a_mono
Ip = 16.5
xsep, ysep = xpt[0],xpt[1]
Rmax = conpos[11][0]
Rmin = conpos[0][0]
a = (Rmax-Rmin)/2
R = a + Rmin
kappa = ysep/a
delta = (R-xsep)/a
A=R/a
title_string = r"$A=%1.2f,\,\kappa = %1.2f,\, \delta = %1.2f$"%(A,kappa,delta)
plt.title(title_string)



#plot the coils
ax.scatter(coilpos[:,0],coilpos[:,1],marker='x', label = 'PF Coils')
labels = out.x[:-2]/MA
for i in range(len(coilpos)//2):
    yoffset = np.sign(coilpos[i][1] - 0.8)*0.1
    xoffset = 0.1
    ax.annotate('%1.1f'%labels[i], xy=(coilpos[i][0]+xoffset,coilpos[i][1]+yoffset))

#poles = np.array([[a_mono -delR,0], [a_mono + delR,0], [a_mono,delZ], [a_mono,-delZ], [a_mono,0]])
poles = np.array([[a_mono -delR,0], [a_mono + delR,0], [a_mono,delZ], [a_mono,-delZ]])
labels = [out.x[-2], -out.x[-2], out.x[-1], out.x[-1], -2*out.x[-1]]
ax.scatter(poles[:,0],poles[:,1],marker='x', label = '$I_{poles}$')
for i in range(len(poles)):
    xoffset = np.sign(poles[i][0]-a_mono)*0.2
    yoffset = np.sign(poles[i][1])*0.2
    ax.annotate('%d'%(labels[i]/MA), xy=(poles[i][0] + xoffset, poles[i][1] + yoffset))

#plot plate positions
ax.scatter(plate_positions[:,0],plate_positions[:,1], marker='s',s=80,color='k',label='Divertor Plate')

ax.legend(loc='lower right')


plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.show()


#Save the loop positions ans currents in a way that lua can read
#coils plus poles, plus central quad and I_mono
current_locations = np.row_stack((coilpos,poles,np.r_[a_mono,0], np.r_[a_mono,0]))
currents = np.r_[out.x[:-2],out.x[:-2], out.x[-2], -out.x[-2], out.x[-1], out.x[-1], -2*out.x[-1], I_mono ]
current_array = np.column_stack((current_locations,currents))

