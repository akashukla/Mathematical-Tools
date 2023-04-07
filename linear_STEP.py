import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
import splines_modular as interp
import numpy.linalg as lin

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


#Plasma Currents
#Monopole
I_mono = 16.5*MA
a_mono = 2.6
#Dipole
afac=1/1000
delR=a_mono*afac
#dipole_fac=1/16
#I_dipole = I_mono*a_mono/delR*dipole_fac
#Quadrapole
#delZ = 0.1*a_mono
delZ = delR


#Pf Coil Currents
#Read the coil positions
coilpos = np.genfromtxt('data/STEP_upper_coils_BSD.csv',delimiter=',',skip_header=6, usecols=(0,1))
#Add center coils
coilpos = np.row_stack((coilpos,[0.1,2]))
coilpos = np.row_stack((coilpos,[0.1,0]))
coilpos = np.row_stack((coilpos,[0.1,0.75]))
coilpos = np.row_stack((coilpos,[0.1,0.9]))
coilpos = np.row_stack((coilpos,[1.2,8]))
#Reflect about R axis to get lower coils too
#coilpos = np.r_[coilpos,np.c_[coilpos[:,0], -coilpos[:,1]]]

#Read the control positions
conpos = np.genfromtxt('data/STEP_upper_controlpoints_BSD3.csv',delimiter=',',skip_header=6, usecols=(0,1))
conpos_addendum_plate = np.genfromtxt('data/STEP_upper_controlpoints_BSD2_addendum_plate.csv',delimiter=',',skip_header=6, usecols=(0,1))
plate_positions =  np.genfromtxt('data/STEP_upper_controlpoints_BSD2_plate_positions.csv',delimiter=',',skip_header=6, usecols=(0,1))
conpos = np.row_stack((conpos,conpos_addendum_plate))
#Try relaxing outer midplane
#conpos[11][0]=4.35
#Reflect about R axis to get lower points too
conpos = np.r_[conpos,np.c_[conpos[:,0], -conpos[:,1]]]


#Read the x point posision
xpos = np.genfromtxt('data/STEP_upper_xpoint.csv',delimiter=',',skip_header=6, usecols=(0,1))
#xpostemp = np.zeros((2,2))
#xpostemp[0] = xpos
#xpostemp[1] = np.r_[xpos[0], -xpos[1]]
#xpos = xpostemp

#All current positions: coils and poles: di,di,quad,quad,quad
current_pos = np.row_stack((coilpos, [a_mono-delR,0], [a_mono+delR,0], [a_mono, -delZ],
                            [a_mono,delZ], [a_mono,0]))


#Define the currents
#BSD
Is = np.r_[4,-4,-8,-5,-5,-5,-5,-5,2]*MA/2
#Is = np.zeros(coilpos.shape[0]//2)
#duplicate for lower coils
Is = np.r_[Is,Is]

#Generate the matrix A : (#Control Points + 2 for derivs) x #Currents including dipole,quadrapole
#For 9 pairs of coils, 28 control points this is 30 x 11
A = np.zeros((conpos.shape[0]+2,coilpos.shape[0]+2))

#Leave last two rows for x point derivatives
delx = 0.01*1.55
dely=0.01*1.55
control_weight=5
x_weight=1
#First do the PF coils. This will require adding the reflected coils as well
for i in range(A.shape[0]-2):
    xi,yi = conpos[i,0], conpos[i,1]
    for j in range(A.shape[1]-2):
        #signature: psi_f(rho,rho0, z,z0, a, I)
        #original coils
        A[i,j] += control_weight*(psi_f(xi,0,yi,current_pos[j,1], current_pos[j,0],1) \
                - psi_f(xpos[0],0,xpos[1],current_pos[j,1], current_pos[j,0],1))
        #reflected coils
        A[i,j] += control_weight*(psi_f(xi,0,yi,-current_pos[j,1], current_pos[j,0],1) \
                - psi_f(xpos[0],0,xpos[1],-current_pos[j,1], current_pos[j,0],1))
# do the x point contibution from coils
for j in range(A.shape[1]-2):
    #original coils
    A[-2,j] += x_weight*( psi_f(xpos[0]-delx,0,xpos[1],current_pos[j,1], current_pos[j,0],1)
               - psi_f(xpos[0]+delx,0,xpos[1],current_pos[j,1], current_pos[j,0],1) )/delx
    A[-1,j] += x_weight*( psi_f(xpos[0],0,xpos[1]-dely,current_pos[j,1], current_pos[j,0],1)
               - psi_f(xpos[0],0,xpos[1]+dely,current_pos[j,1], current_pos[j,0],1) )/dely
    #reflected coils
    A[-2,j] += x_weight*( psi_f(xpos[0]-delx,0,xpos[1],-current_pos[j,1], current_pos[j,0],1)
               - psi_f(xpos[0]+delx,0,xpos[1],-current_pos[j,1], current_pos[j,0],1) )/delx
    A[-1,j] += x_weight*( psi_f(xpos[0],0,xpos[1]-dely,-current_pos[j,1], current_pos[j,0],1)
               - psi_f(xpos[0],0,xpos[1]+dely,-current_pos[j,1], current_pos[j,0],1) )/dely

#Now do the pole currents
for i in range(A.shape[0]-2):
    xi,yi = conpos[i,0], conpos[i,1]
    #dipoles
    A[i,-2] += control_weight*(psi_f(xi,0,yi,current_pos[-5,1], current_pos[-5,0],1) \
            - psi_f(xpos[0],0,xpos[1],current_pos[-5,1], current_pos[-5,0],1))
    A[i,-2] += control_weight*(-psi_f(xi,0,yi,current_pos[-4,1], current_pos[-4,0],1) \
            + psi_f(xpos[0],0,xpos[1],current_pos[-4,1], current_pos[-4,0],1))
    #quadrapoles
    A[i,-1] += control_weight*(psi_f(xi,0,yi,current_pos[-3,1], current_pos[-3,0],1)\
            - psi_f(xpos[0],0,xpos[1],current_pos[-3,1], current_pos[-3,0],1))
    A[i,-1] += control_weight*(psi_f(xi,0,yi,current_pos[-2,1], current_pos[-2,0],1)\
            - psi_f(xpos[0],0,xpos[1],current_pos[-2,1], current_pos[-2,0],1))
    A[i,-1] += control_weight*(-2*psi_f(xi,0,yi,current_pos[-1,1], current_pos[-1,0],1)\
            + 2*psi_f(xpos[0],0,xpos[1],current_pos[-1,1], current_pos[-1,0],1))
#Now do x points for poles
#dipoles
#x deriv
A[-2,-2] += x_weight*( psi_f(xpos[0]-delx,0,xpos[1],current_pos[-5,1], current_pos[-5,0],1)
            - psi_f(xpos[0]+delx,0,xpos[1],current_pos[-5,1], current_pos[-5,0],1) )/delx
A[-2,-2] += x_weight*( -psi_f(xpos[0]-delx,0,xpos[1],current_pos[-4,1], current_pos[-4,0],1)
            +  psi_f(xpos[0]+delx,0,xpos[1],current_pos[-4,1], current_pos[-4,0],1) )/delx
#y deriv
A[-1,-2] += x_weight*( psi_f(xpos[0],0,xpos[1]-dely,current_pos[-5,1], current_pos[-5,0],1)
            - psi_f(xpos[0],0,xpos[1]+dely,current_pos[-5,1], current_pos[-5,0],1) )/dely
A[-1,-2] += x_weight*( -psi_f(xpos[0],0,xpos[1]-dely,current_pos[-4,1], current_pos[-4,0],1)
            +  psi_f(xpos[0],0,xpos[1]+dely,current_pos[-4,1], current_pos[-4,0],1) )/dely
#quadrapoles
#x deriv
A[-2,-1] += x_weight*( psi_f(xpos[0]-delx,0,xpos[1],current_pos[-3,1], current_pos[-3,0],1)
            - psi_f(xpos[0]+delx,0,xpos[1],current_pos[-3,1], current_pos[-3,0],1) )/delx
A[-2,-1] += x_weight*( psi_f(xpos[0]-delx,0,xpos[1],current_pos[-2,1], current_pos[-2,0],1)
            - psi_f(xpos[0]+delx,0,xpos[1],current_pos[-2,1], current_pos[-2,0],1) )/delx
A[-2,-1] += x_weight*( -2*psi_f(xpos[0]-delx,0,xpos[1],current_pos[-1,1], current_pos[-1,0],1)
            +  2*psi_f(xpos[0]+delx,0,xpos[1],current_pos[-1,1], current_pos[-1,0],1) )/delx
#y deriv
A[-1,-1] += x_weight*( psi_f(xpos[0],0,xpos[1]-dely,current_pos[-3,1], current_pos[-3,0],1)
            - psi_f(xpos[0],0,xpos[1]+dely,current_pos[-3,1], current_pos[-3,0],1) )/dely
A[-1,-1] += x_weight*( psi_f(xpos[0],0,xpos[1]-dely,current_pos[-2,1], current_pos[-2,0],1)
            - psi_f(xpos[0],0,xpos[1]+dely,current_pos[-2,1], current_pos[-2,0],1) )/dely
A[-1,-1] += x_weight*( -2*psi_f(xpos[0],0,xpos[1]-dely,current_pos[-1,1], current_pos[-1,0],1)
            +  2*psi_f(xpos[0],0,xpos[1]+dely,current_pos[-1,1], current_pos[-1,0],1) )/dely


#Now construct b
# b will just be psi from plasma current
b = np.zeros(A.shape[0])
#control point
for i in range(b.shape[0]-2):
    xi,yi = conpos[i,0], conpos[i,1]
    b[i] = control_weight*-psi_f(xi, 0, yi, 0, a_mono, 16.5*MA)
#x point derivs
b[-1] = 0
b[-2] = 0

from scipy.optimize import lsq_linear
#Isol = lin.pinv(A).dot(b)
#print("Isol",Isol)
#Isol = lsq_linear(A,b).x
#print("Isol",Isol)
lowerbound = np.r_[0,   0, -10, -10, -120, -250, -150, -150, 0,  -np.inf , -np.inf]*MA
upperbound = np.r_[10, 10,   0,   0,    0,    0,    0,    0, 10, np.inf, np.inf]*MA
#lowerbound = np.r_[-np.inf,-np.inf,-np.inf, -np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf, -20, -50 ]*MA
#upperbound = -lowerbound
Isol = lsq_linear(A,b, bounds=(lowerbound,upperbound)).x
print("Isol",Isol)


def psi_config(r,z,Is):
    #List of currents should be: coil currents (1 sided so 9 of them), dipole current, quadrapol current
    psiret=0
    #Psi from the PF coils
    for l in range(len(Is)-2):
        #original coils
        psiret += psi_f(r,0,z,coilpos[l,1], coilpos[l,0],Is[l])
        #reflected coils
        psiret += psi_f(r,0,z,-coilpos[l,1], coilpos[l,0],Is[l])
    #Psi from the plasma current
    psiret+= psi_f(r,0,z,0,a_mono,I_mono)
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

### #func to find the actual x point
### #call with outderiv = least_squares(derivfunc,x0 = [xpos[0,0], xpos[0,1]])
def derivfunc(xvec,I_temp):
     Error_X1 = ( psi_config(xvec[0]-delx,xvec[1],I_temp) - psi_config(xvec[0]+delx,xvec[1],I_temp))/delx
     Error_X2 = ( psi_config(xvec[0],xvec[1]-dely,I_temp) - psi_config(xvec[0],xvec[1]+dely,I_temp))/dely
     return Error_X1,Error_X2


Rgrid = np.arange(0.06,8,0.05)
Zgrid = np.arange(-8.01,8.0,0.05)
psiplot = np.zeros((len(Rgrid), len(Zgrid)))

for i,r in enumerate(Rgrid):
    for j,z in enumerate(Zgrid):
        psiplot[i,j] = psi_config(r,z,Isol)

#choose levels based on psiSep
psisep = psi_config(xpos[0],xpos[1],Isol)
levels = np.arange(psisep-0.5,psisep+0.5,0.02)

X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T, levels = levels)
ax.clabel(CS, inline=True, fontsize=10, levels = levels)
fig.colorbar(CS)
ax.scatter(conpos[:,0],conpos[:,1], label = 'Control Points')
ax.scatter(xpos[0],xpos[1], label = 'Desired X point')
plt.legend()
plt.show()

### 
### # #Calculate Geometry
### # R0 = acs[-1]
### # Ip = Ics[-1]/MA
### # xsep,ysep=1.76,4.0
### # Rmin = 0.38
### # Rmax = 4.73
### # a = (Rmax-Rmin)/2
### # R = a + Rmin
### # kappa = ysep/a
### # delta = (R-xsep)/a
### # A=R/a
### # 
### # print("R0 = %1.2f m, a = %1.2f m, R0/a=%1.2f, kappa = %1.2f, delta = %1.2f "%(R0,a,A,kappa,delta))
### # 
### # 
### # 
### # title_string = r"$R = %1.2f,\, a = %1.2f ,\, R/a=%1.2f,$"%(R,a,A)+"\n" \
### #     +r"$\kappa = %1.2f,\, \delta = %1.2f,\, I_p = %1.2f MA$"%(kappa,delta,Ip)
### #plt.title(title_string)
### plt.xlabel('$R$')
### plt.ylabel('$Z$')
### plt.show()
### 
### 
#Find actual x point
xpt = least_squares(derivfunc, x0=[xpos[0], xpos[1]], args = ([np.r_[Isol]])).x
psisep = psi_config(xpt[0],xpt[1],Isol)
psisim = psisep-0.005
levels = [0,psisep]
X,Y = np.meshgrid(Rgrid,Zgrid)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T,[psisep])#] levels = levels)
CS = ax.contour(X, Y, (psiplot).T,[psisim])#] levels = levels)
#ax.clabel(CS, inline=True, fontsize=10)#, levels = levels)
#fig.colorbar(CS)
ax.scatter(conpos[:,0],conpos[:,1], label = 'Control Points')
ax.scatter(xpos[0],xpos[1], label = 'Desired X point')
#Calculate Geometry
R0 = a_mono
Ip = 16.5
xsep, ysep = xpt[0],xpt[1]
Rmax = conpos[30][0]
Rmin = conpos[0][0]
a = (Rmax-Rmin)/2
R = a + Rmin
kappa = ysep/a
delta = (R-xsep)/a
A=R/a
title_string = 'delR=delZ = a/%d, I_dipole = %d, I_quad =%d'%(1/afac,Isol[-2]/MA,Isol[-1]/MA) + "\n" + r"$A=%1.2f,\,\kappa = %1.2f,\, \delta = %1.2f$"%(A,kappa,delta)
plt.title(title_string)
plt.show()



#plot the coils
ax.scatter(coilpos[:,0],coilpos[:,1],marker='x', label = 'PF Coils')
labels = Isol[:-2]/MA
for i in range(len(coilpos)//2):
    yoffset = np.sign(coilpos[i][1] - 0.8)*0.1
    xoffset = 0.1
    ax.annotate('%1.1f'%labels[i], xy=(coilpos[i][0]+xoffset,coilpos[i][1]+yoffset))

#poles = np.array([[a_mono -delR,0], [a_mono + delR,0], [a_mono,delZ], [a_mono,-delZ], [a_mono,0]])
poles = np.array([[a_mono -delR,0], [a_mono + delR,0], [a_mono,delZ], [a_mono,-delZ]])
labels = [Isol[-2], -Isol[-2], Isol[-1], Isol[-1], -2*Isol[-1]]
ax.scatter(poles[:,0],poles[:,1],marker='x', label = '$I_{poles}$')
for i in range(len(poles)):
    xoffset = np.sign(poles[i][0]-a_mono)*0.2
    yoffset = np.sign(poles[i][1])*0.2
    ax.annotate('%d'%(labels[i]/MA), xy=(poles[i][0] + xoffset, poles[i][1] + yoffset))
### #plot plate positions
ax.scatter(plate_positions[:,0],plate_positions[:,1], marker='s',s=80,color='k',label='Divertor Plate')
ax.legend(loc='lower right')
plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.show()



### #Save the loop positions ans currents in a way that lua can read
### #coils plus poles, plus central quad and I_mono
### current_locations = np.row_stack((coilpos,poles,np.r_[a_mono,0], np.r_[a_mono,0]))
### currents = np.r_[out.x[:-2],out.x[:-2], out.x[-2], -out.x[-2], out.x[-1], out.x[-1], -2*out.x[-1], I_mono ]
### current_array = np.column_stack((current_locations,currents))
### 
### def psi_config_wrapped(R,Z):
###     return psi_config(R,Z,np.r_[out.x[:-2],out.x])
### 
### #Now test the spline
### #xtest,ytest = 0.85,0.75
### #print("actual value =", psi_config_wrapped(xtest,ytest))
### #I = interp.CubicSplineInterpolator(Rgrid,Zgrid,psi_config_wrapped)
### #print("Interp value = ",I.interpolate(xtest,ytest))
### 
### def funcy(y,xf,psif):
###     return psif - psi_config_wrapped(xf,y)
### #Find the cutoff point
### psif = psisim
### candidates = np.zeros((200,2))
### for i,xf in enumerate(np.linspace(4,5.5,num=200)):
###     #out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
###     outy = least_squares(funcy,x0=5.5, bounds = (5,8), args = (xf,psif))
###     candidates[i] = np.r_[xf,outy.x]
### maxind = np.argmax(candidates[:,1])
### candidates[maxind]
