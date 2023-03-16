import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sympy as sm
import numpy.linalg as lin


x = sm.symbols('x')
y = sm.symbols('y')
c1 = sm.symbols('c1')
c2 = sm.symbols('c2')
c3 = sm.symbols('c3')
c4 = sm.symbols('c4')
c5 = sm.symbols('c5')
c6 = sm.symbols('c6')
c7 = sm.symbols('c7')
c8 = sm.symbols('c8')
c9 = sm.symbols('c9')
c10 = sm.symbols('c10')
c11 = sm.symbols('c11')
c12 = sm.symbols('c12')

c = np.r_[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]

def psi(x,y):
    return x**4/8 + A*(.5*x**2*sm.log(x) - x**4/8) \
        + c[0]*psi1(x,y) + c[1]*psi2(x,y) + c[2]*psi3(x,y) \
        + c[3]*psi4(x,y) + c[4]*psi5(x,y) + c[5]*psi6(x,y) \
        + c[6]*psi7(x,y) + c[7]*psi8(x,y) + c[8]*psi9(x,y) \
        + c[9]*psi10(x,y) + c[10]*psi11(x,y) + c[11]*psi12(x,y)

def psi1(x,y):
    return 1

def psi2(x,y):
    return x**2


def psi3(x,y):
    return y**2 - x**2*sm.log(x)

def psi4(x,y):
    return x**4 - 4*x**2*y**2

def psi5(x,y):
    return  2*y**4 - 9*y**2*x**2 + 3*x**4*sm.log(x) - 12*x**2*y**2*sm.log(x)

def psi6(x,y):
    return x**6 - 12*x**4*y**2 + 8*x**2*y**4

def psi7(x,y):
    return 8*y**6 - 120*y**4*x**2 + 75*y**2*x**4 - 15*x**6*sm.log(x) + 180*x**4*y**2*sm.log(x) \
        -120*x**2*y**4*sm.log(x)

def psi8(x,y):
    return y

def psi9(x,y):
    return y*x**2

def psi10(x,y):
    return y**3 - 3*y*x**2*sm.log(x)

def psi11(x,y):
    return 3*y*x**4 - 4*y**3*x**2

def psi12(x,y):
    return 8*y**5 - 45*y*x**4 - 80*y**3*x**2*sm.log(x) + 60*y*x**4*sm.log(x)

def psi_x(x0,y0):
    deriv = sm.diff(psi(x,y),x)
    return deriv.subs(x,x0).subs(y,y0)

def psi_xx(x0,y0):
    deriv = sm.diff(sm.diff(psi(x,y),x),x)
    return deriv.subs(x,x0).subs(y,y0)

def psi_y(x0,y0):
    deriv = sm.diff(psi(x,y),y)
    return deriv.subs(x,x0).subs(y,y0)

def psi_yy(x0,y0):
    deriv = sm.diff(sm.diff(psi(x,y),y),y)
    return deriv.subs(x,x0).subs(y,y0)

#Iter Params
epsilon = 0.32
kappa = 1.7
delta = 0.33
alpha = np.arcsin(delta)
x_sep = 0.88
y_sep = -0.60
q_star = 1.57
A = -0.155 # choose A corresponding  to beta = 0.05



N1 = -(1+alpha)**2/(epsilon*kappa**2)
N2 = (1-alpha)**2/(epsilon*kappa**2)
N3 = -kappa/(epsilon*np.cos(alpha)**2)


def get_constant_coeff(expr):
    """Helper function for getting constant coefficient"""
    return expr.as_independent(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12)[0]


#Boundary Equations
expressions=np.zeros(12,dtype=object)
expressions[0] = psi(1+epsilon,0)
expressions[1] = psi(1-epsilon,0)
expressions[2] = psi(1-delta*epsilon,kappa*epsilon)
expressions[3] = psi(x_sep,y_sep)
expressions[4] = psi_y(1+epsilon,0)
expressions[5] = psi_y(1-epsilon,0)
expressions[6] = psi_x(1-delta*epsilon,kappa*epsilon)
expressions[7] = psi_x(x_sep,y_sep)
expressions[8] = psi_y(x_sep,y_sep)
expressions[9] = psi_yy(1+epsilon,0) + N1*psi_x(1+epsilon,0)
expressions[10] = psi_yy(1-epsilon,0) + N2*psi_x(1-epsilon,0)
expressions[11] = psi_xx(1-delta*epsilon,kappa*epsilon) + N3*psi_y(1-delta*epsilon,kappa*epsilon)

#Lefthand side Matrix to multiply column vector c
L = np.zeros((12,12))
for i in range(0,len(expressions)):
    for j in range(0,len(c)):
        L[i,j] = expressions[i].coeff(c[j])

#Righthand side values
R = np.zeros(12)
for i in range(0,len(expressions)):
    R[i] = - get_constant_coeff(expressions[i])

c = lin.inv(L).dot(R)

print(psi(x,y))

#psival = sm.symbols('psival')
#expr = 5 - psi(x,y)
#solution = sm.solve(expr,y)

#eqn = sm.Eq(psi(x,y),0.1)
#solutions = sm.solve(eqn, y, implicit=True,dict=True)
#sol = solutions[0][y]

from scipy.optimize import least_squares
from scipy.optimize import fsolve
#Solve for y given x
def func(y,xf,psif):
    #print(xf,psif)
    return psif - float(psi(xf,y[0]))

def jac(y,xf,psif):
    #print('x and y', xf, y)
    #print(psi_y(xf,y[0]))
    return float(-psi_y(xf,y[0]))

#Solve for x given y
def funcx(x,yf,psif):
    #print(xf,psif)
    return psif - float(psi(x[0],yf))

def jacx(x,yf,psif):
    #print('x and y', xf, y)
    #print(psi_y(xf,y[0]))
    return float(-psi_x(x[0],yf))


# psif = 0.002
# 
# postable = np.zeros((110,4))
# for i in range(0,110):
#     xf = 0.4+0.01*i
#     out = least_squares(func,x0=0.1,jac = jac, bounds = (0,np.inf), args = (xf,psif))
#     yf  = out.x[0]
#     postable[i,:] =np.r_[xf,yf, out.cost,out.status]
#     #print(i, out.cost, out.status)
# possoltable = postable[postable[:,2]<1e-8]
# xpossol = possoltable[:,0]
# ypossol = possoltable[:,1]
# 
# midtable = np.zeros((110,4))
# for i in range(0,110):
#     xf = 0.4+0.01*i
#     out = least_squares(func,x0=-0.2,jac = jac, bounds = (-0.6,0), args = (xf,psif))
#     yf  = out.x[0]
#     midtable[i,:] =np.r_[xf,yf, out.cost,out.status]
#     #print(i, out.cost, out.status)
# midsoltable = midtable[midtable[:,2]<1e-8]
# xmidsol = midsoltable[:,0]
# ymidsol = midsoltable[:,1]
# 
# 
# 
# negtable = np.zeros((110,4))
# for i in range(0,110):
#     xf = 0.4+0.01*i
#     out = least_squares(func,x0=-0.8,jac = jac, bounds = (-np.inf,-0.6), args = (xf,psif))
#     yf  = out.x[0]
#     negtable[i,:] =np.r_[xf,yf, out.cost,out.status]
#     #print(i, out.cost, out.status)
# negsoltable = negtable[negtable[:,2]<1e-8]
# xnegsol = negsoltable[:,0]
# ynegsol = negsoltable[:,1]
# 
# plt.scatter(xpossol,ypossol)
# plt.scatter(xnegsol,ynegsol)
# plt.scatter(xmidsol,ymidsol)
# plt.xlim(0,2.5)
# plt.ylim(-1.2,1.2)
# plt.show()

load = False
#R0 =  1.1
# Try karge R0 for iter
#R0 = 6.2
R0 = 1.0

xgrid = R0*np.arange(0,1.5,.005)
#ygrid = R0*np.arange(-1.2,1.2,0.005)
ygrid = R0*np.arange(-1.5,1.5,0.005)
X,Y = np.meshgrid(xgrid,ygrid)

if load:
    psiplot = np.load('./psiplot.npy')
else:
    psiplot = np.zeros((xgrid.shape[0],ygrid.shape[0]))
    for i in range(0,xgrid.shape[0]):
       for j in range(0,ygrid.shape[0]):
           psiplot[i,j] = float(psi(xgrid[i]/R0,ygrid[j]/R0))
    np.save('psiplot',psiplot)


fig, ax = plt.subplots()
CS = ax.contour(X, Y, psiplot.T, levels =  np.arange(-0.04,0.5,0.005))
ax.clabel(CS, inline=True, fontsize=10)
plt.show()
psif = 0.005
candidates = np.zeros((200,2))
for i,xf in enumerate(np.linspace(R0*0.75,R0*0.95,num=200)):
    #out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
    out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
    candidates[i] = np.r_[xf,out.x]
maxind = np.argmax(candidates[:,1])
candidates[maxind]
