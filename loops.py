import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
mu0 = 1.256e-6
#mu0=1.
a = 1
I=1e7/100

def Aphi_f(r,theta, a,I):
    #from Jackson in spherical coords
    ksq = 4*a*r*np.sin(theta)/(a**2+r**2 + 2*a*r*np.sin(theta))

    fac1 = mu0/(4*np.pi)

    num2 = 4*I*a
    denom2 = np.sqrt(a**2 + r**2 + 2*a*r*np.sin(theta))
    fac2 = num2/denom2

    num3 = (2-ksq)*ellipk(ksq) - 2*ellipe(ksq)
    denom3 = ksq
    fac3 = num3/denom3

    Aphi = fac1*fac2*fac3
    #psi = r*Aphi
    #return psi
    return Aphi


def psi_f(rho, z, a, I):
    rs = np.sqrt(rho**2 + z**2)
    theta = np.arccos(z/rs)
    Aphi = Aphi_f(rs,theta,a,I)
    psi = Aphi #* rho
    return psi
    

z_up = 1.0
z_down = -0.5
rshift0 = 0.13
rshift_down = 0.0
a0 = 0.5
a_up = 1.0
a_down = 0.5
I0 = I
I_down = I*3


R = np.arange(0.01,2,0.01)
Z = np.arange(-1.5,0.5,0.01)
#theta = np.arange(0,np.pi)
psi0 = np.zeros((len(R), len(Z)))
psi_up = np.zeros((len(R), len(Z)))
psi_down = np.zeros((len(R), len(Z)))

for i,r in enumerate(R):
    for j,z in enumerate(Z):
        #rs = np.sqrt(r**2 + z**2)
        #theta = np.arccos(z/rs)
        #psi0[i,j] =Aphi_f(rs,theta, 1)*r
        psi0[i,j] =psi_f(r-rshift0,z, a0,I0)*r

#for i,r in enumerate(R):
#    for j,z in enumerate(Z):
#        z = z-z_up
#        #rs = np.sqrt(r**2 + z**2)
#        #theta = np.arccos(z/rs)
#        #psi_up[i,j] =Aphi_f(rs,theta, 1)*r
#        psi_up[i,j] =psi_f(r,z, a_up)

for i,r in enumerate(R):
    for j,z in enumerate(Z):
        z = z-z_down
        #rs = np.sqrt(r**2 + z**2)
        #theta = np.arccos(z/rs)
        #psi_down[i,j] =Aphi_f(rs,theta,1)*r
        psi_down[i,j] =psi_f(r-rshift_down,z,a_down, I_down)*r




#psi0[psi0==np.inf] = 0.0
#psi_up[psi_up==np.inf] = 0.0
#psi_down[psi_down==np.inf] = 0.0

psiplot = psi0 + psi_down #+ psi_up
#levels = np.arange(0,1e-6,1e-8)
#levels = np.r_[0, 4.2e-7]
#levels = np.r_[0, 3.70]
#psiplot =psi0
levels = np.arange(0,10.0,0.1)/100
#levels = np.arange(-10,10,0.1)
X,Y = np.meshgrid(R,Z)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, (psiplot).T, levels = levels)
#CS = ax.contour(X, Y, (psiplot).T, levels = np.flip(10.0**(np.arange(-6,-10,-0.5))) )
ax.clabel(CS, inline=True, fontsize=10, levels = levels)
fig.colorbar(CS)
plt.show()


xgrid = np.arange(0.2,1.5,.005)
ygrid = np.arange(-0.5,0.5,0.005)
X,Y = np.meshgrid(xgrid,ygrid)

load = False

if load:
    psiplot = np.load('./psiplot.npy')
else:
    psiplot = np.zeros((xgrid.shape[0],ygrid.shape[0]))
    for i in range(0,xgrid.shape[0]):
       for j in range(0,ygrid.shape[0]):
           psiplot[i,j] = psi_f(xgrid[i]-rshift0, ygrid[j], a0, I0)*xgrid[i]
           psiplot[i,j] += psi_f(xgrid[i] - rshift_down, ygrid[j] - z_down, a_down, I_down)*xgrid[i]
    np.save('psiplot',psiplot)


plotlevel = 3.9/100
fig1, ax1 = plt.subplots()
CS = ax1.contour(X, Y, psiplot.T, levels =  np.r_[0,plotlevel])
ax.clabel(CS, inline=True, fontsize=10)
ax1.hlines(-0.35,0,1)
plt.show()


from scipy.optimize import least_squares
from scipy.optimize import fsolve
#Solve for y given x
def func(y,xf,psif,a0,a_down):
    #print(xf,psif)
    return psif - (psi_f(xf-rshift0, y, a0, I0) + psi_f(xf-rshift_down, y-z_down, a_down, I_down))*xf

# def jac(y,xf,psif):
#     #print('x and y', xf, y)
#     #print(psi_y(xf,y[0]))
#     return float(-psi_y(xf,y[0]))
# 
# #Solve for x given y
# def funcx(x,yf,psif):
#     #print(xf,psif)
#     return psif - float(psi(x[0],yf))
# 
# def jacx(x,yf,psif):
#     #print('x and y', xf, y)
#     #print(psi_y(xf,y[0]))
#     return float(-psi_x(x[0],yf))



psif = plotlevel
candidates = np.zeros((1000,2))
searchspace=np.linspace(0.4,0.8,num=1000)
for i,xf in enumerate(searchspace):
    #out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
    out = least_squares(func,x0=0.1, bounds = (-0.6,0.6), args = (xf,psif,a0,a_down))
    candidates[i] = np.r_[xf,out.x]
maxind = np.argmax(candidates[:,1])
candidates[maxind]
Rtop,Ztop = candidates[maxind]
print("Rtop,Ztop = ", Rtop,Ztop )
ax1.scatter(Rtop,Ztop)

plt.figure()
plt.plot(searchspace, candidates[:,1])
plt.show()

