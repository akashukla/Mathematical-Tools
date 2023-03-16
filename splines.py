import numpy as np

x = np.r_[0,1]
y = np.r_[0,1]
f = np.outer(np.cos(x), np.cos(y))
fx = np.outer(-np.sin(x),np.cos(y))
fy = np.outer(np.cos(x),-np.sin(y))
fxy = np.outer(np.sin(x),np.sin(y))

#Create A matrix it realy A inverse
A = np.zeros((16,16))
A[0,0] = 1
A[1,4] = 1
A[2,0:6] = np.r_[-3,3,0,0,-2,-1]
A[3,0:6] = np.r_[2,-2,0,0,1,1]
A[4,8] = 1
A[5,12] = 1
A[6,8:14] = np.r_[-3,3,0,0,-2,-1]
A[7,8:14] = np.r_[2,-2,0,0,1,1]
A[8,0:3], A[8,8:11] = np.r_[-3,0,3], np.r_[-2,0,-1]
A[9,:] = np.r_[0,0,0,0,-3,0,3,0,0,0,0,0,-2,0,-1,0 ] 
A[10,:] = np.r_[9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1]
A[11,:] = np.r_[-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1]
A[12,:] = np.r_[2,0,-2,0,0,0,0,0,1,0,1,0,0,0,0,0]
A[13,:] = np.r_[0,0,0,0,2,0,-2,0,0,0,0,0,1,0,1,0]
A[14,:] = np.r_[-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1]
A[15,:] = np.r_[4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1]

xvec = np.r_[ f[0,0], f[1,0], f[0,1], f[1,1], fx[0,0], fx[1,0], fx[0,1], fx[1,1],fy[0,0], fy[1,0], fy[0,1], fy[1,1],fxy[0,0], fxy[1,0], fxy[0,1], fxy[1,1] ]

alpha1 = A.dot(xvec)

alpha1 = np.reshape(alpha1,(4,4)).T

def p(x,y):
    xvec = np.r_[1,x,x**2,x**3]
    yvec = np.r_[1,y,y**2,y**3]
    return xvec.dot(alpha1.dot(yvec))

print("Test unit one")
print("p(0.5,0.5)",p(0.9,0.9))
print("f(0.5,0.5)",np.cos(0.9)*np.cos(0.9))


# Now do for a grid cell with lx,ly = 0.5,0.5 not unit square
#Use the same A
#xvec is different
x = np.r_[0,0.5]
y = np.r_[0,0.5]
dx = np.diff(x)
dy = np.diff(y)
f = np.outer(np.cos(x), np.cos(y))
fx = np.outer(-np.sin(x),np.cos(y))
fy = np.outer(np.cos(x),-np.sin(y))
fxy = np.outer(np.sin(x),np.sin(y))

xvec = np.r_[ f[0,0], f[1,0], f[0,1], f[1,1], dx*fx[0,0], dx*fx[1,0], dx*fx[0,1], dx*fx[1,1], dy*fy[0,0], dy*fy[1,0], dy*fy[0,1], dy*fy[1,1], dx*dy*fxy[0,0], dx*dy*fxy[1,0], dx*dy*fxy[0,1], dx*dy*fxy[1,1] ]

alpha = A.dot(xvec)

alpha = np.reshape(alpha,(4,4)).T

def p2(x,y):
    x = (x-0)/dx
    y = (y-0)/dy
    xvec = np.r_[1,x,x**2,x**3]
    yvec = np.r_[1,y,y**2,y**3]
    return xvec.dot(alpha.dot(yvec))


print("Test grid one")
print("p2(0.5,0.5)",p2(0.5,0.5))
print("f(0.5,0.5)",np.cos(0.5)*np.cos(0.5))


#Now try it on a grid with more cells. Let's do 4 cells.
dx,dy = 0.5,0.5
x = np.r_[0:2+dx:dx]
y = np.r_[0:2+dy:dy]

f = np.outer(np.cos(x), np.cos(y))
fx = np.outer(-np.sin(x),np.cos(y))
fy = np.outer(np.cos(x),-np.sin(y))
fxy = np.outer(np.sin(x),np.sin(y))

alpha_grid = np.zeros((len(x)-1, len(y)-1, 4,4))

for i in range(alpha_grid.shape[0]):
    for j in range(alpha_grid.shape[1]):
        xvec = np.r_[ f[i+0,j+0], f[i+1,j+0], f[i+0,j+1], f[i+1,j+1],
                     dx*fx[i+0,j+0], dx*fx[i+1,j+0], dx*fx[i+0,j+1], dx*fx[i+1,j+1],
                     dy*fy[i+0,j+0], dy*fy[i+1,j+0], dy*fy[i+0,j+1], dy*fy[i+1,j+1],
                     dx*dy*fxy[i+0,j+0], dx*dy*fxy[i+1,j+0], dx*dy*fxy[i+0,j+1], dx*dy*fxy[i+1,j+1] ]
        if i==1 and j==1:
            print("xvec",xvec)
        alpha_ij = A.dot(xvec)
        alpha_ij = np.reshape(alpha_ij,(4,4)).T
        alpha_grid[i,j] = alpha_ij


def p3(x_eval,y_eval):
    # Determine cell location
    i = np.where(x>x_eval)[0][0]-1
    j = np.where(y>y_eval)[0][0]-1
    x_eval = (x_eval-x[i])/dx
    y_eval = (y_eval-y[j])/dy
    xvec = np.r_[1,x_eval,x_eval**2,x_eval**3]
    yvec = np.r_[1,y_eval,y_eval**2,y_eval**3]
    return xvec.dot(alpha_grid[i,j].dot(yvec))
