import math
import numpy as np
import numpy.linalg as lin
#Inputs:
# tx: xgrid
# ty: ygrid
# f(x,y) the function to approximate

def dot(v1,v2):
    output = 0.0
    for i in range(len(v1)):
        output = output+v1[i]*v2[i]
    return output

def dot_matvec(A,v):
    output = []
    for i in range(len(A)):
        output.append(dot(A[i], v))
    return output

class Interpolator:
    def __init__(self, tx,ty,f,fx,fy,fxy):
        self.tx = tx
        self.ty = ty
        self.f = f
        self.fx = fx
        self.fy = fy
        self.fxy = fxy
        self.dx = tx[1]-tx[0]
        self.dy = ty[1]-ty[0]
        # Create f and derivatives on grid
        fgrid = []
        fgridx = []
        fgridy = []
        fgridxy = []
        print("tx,ty")
        print(tx)
        print(ty)
        for i in range(len(tx)):
            fgrid.append([])
            fgridx.append([])
            fgridy.append([])
            fgridxy.append([])
            for j in range(len(ty)):
                #if i==2 and j==1:
                    #print("i,j=2,1")
                    #print(tx[i], ty[j])
                fgrid[i].append( f(tx[i], ty[j]) )
                fgridx[i].append( fx(tx[i], ty[j]) )
                fgridy[i].append( fy(tx[i], ty[j]) )
                fgridxy[i].append( fxy(tx[i], ty[j]) )
        self.fgrid = fgrid
        self.fgridx = fgridx
        self.fgridy = fgridy
        self.fgridxy = fgridxy

    def find_cell(self,x_eval,y_eval):
        # Find the subcell i0, j0
        for i in range(len(self.tx)):
            if x_eval < self.tx[i+1]:
                i0 = i
                break
        for j in range(len(self.ty)):
            if y_eval < self.ty[j+1]:
                j0 = j
                break
        return i0,j0

class BiLinearInterpolator(Interpolator):
    def __init__(self, tx,ty,f,fx,fy,fxy):
        super().__init__(tx,ty,f,fx,fy,fxy)
    def interpolate(self,x_eval,y_eval):
        i,j = self.find_cell(x_eval,y_eval)
        print("i,j = ",i,j)
        norm = 1/(self.tx[i+1]-self.tx[i])/(self.ty[j+1]-self.ty[j])
        xvec = [self.tx[i+1] - x_eval, x_eval-self.tx[i]]
        yvec = [self.ty[j+1] - y_eval, y_eval-self.ty[j]]
        alpha = []
        for l in range(2):
            alpha.append([])
            for m in range(2):
                alpha[l].append(self.fgrid[i+l][j+m])
        print("alpha:",alpha)
        return norm*dot(xvec,dot_matvec(alpha,yvec))

class BiCubicInterpolator(Interpolator):
    def __init__(self, tx,ty,f,fx,fy,fxy):
        super().__init__(tx,ty,f,fx,fy,fxy)
        #Construct A. Depends on Nothing
        Ainv = []
        Ainv.append([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([-3.,  3.,  0.,  0., -2., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 2., -2.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -3.,  3.,  0.,  0., -2., -1.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2., -2.,  0.,  0.,  1.,  1.,  0.,  0.])
        Ainv.append([-3.,  0.,  3.,  0.,  0.,  0.,  0.,  0., -2.,  0., -1.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0., -3.,  0.,  3.,  0.,  0.,  0.,  0.,  0., -2.,  0., -1.,  0.])
        Ainv.append([ 9., -9., -9.,  9.,  6.,  3., -6., -3.,  6., -6.,  3., -3.,  4.,  2.,  2.,  1.])
        Ainv.append([-6.,  6.,  6., -6., -3., -3.,  3.,  3., -4.,  4., -2.,  2., -2., -2., -1., -1.])
        Ainv.append([ 2.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.])
        Ainv.append([ 0.,  0.,  0.,  0.,  2.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.])
        Ainv.append([-6.,  6.,  6., -6., -4., -2.,  4.,  2., -3.,  3., -3.,  3., -2., -1., -2., -1.])
        Ainv.append([ 4., -4., -4.,  4.,  2.,  2., -2., -2.,  2., -2.,  2., -2.,  1.,  1.,  1.,  1.])
        self.Ainv = Ainv

    def interpolate(self,x_eval,y_eval):
        i0,j0 = self.find_cell(x_eval,y_eval)
        print("i0,j0 = ",i0,j0)
        fvec = [ self.fgrid[i0+0][j0+0], self.fgrid[i0+1][j0+0], self.fgrid[i0+0][j0+1], self.fgrid[i0+1][j0+1],
            self.dx*self.fgridx[i0+0][j0+0], self.dx*self.fgridx[i0+1][j0+0], self.dx*self.fgridx[i0+0][j0+1], self.dx*self.fgridx[i0+1][j0+1],
            self.dy*self.fgridy[i0+0][j0+0], self.dy*self.fgridy[i0+1][j0+0], self.dy*self.fgridy[i0+0][j0+1], self.dy*self.fgridy[i0+1][j0+1],
            self.dx*self.dy*self.fgridxy[i0+0][j0+0], self.dx*self.dy*self.fgridxy[i0+1][j0+0], self.dx*self.dy*self.fgridxy[i0+0][j0+1], self.dx*self.dy*self.fgridxy[i0+1][j0+1] ]
        #print("fvec:", fvec)
        alpha = dot_matvec(self.Ainv, fvec)
        #print(alpha)
        alpha_mat = []
        for i in range(4):
            alpha_mat.append([])
            for j in range(4):
                alpha_mat[i].append(alpha[i+4*j])
        #print(alpha_mat)
        x_eval = (x_eval-self.tx[i0])/self.dx
        y_eval = (y_eval-self.ty[j0])/self.dy
        xvec = [1,x_eval,x_eval**2,x_eval**3]
        yvec = [1,y_eval,y_eval**2,y_eval**3]
        return dot(xvec,dot_matvec(alpha_mat,yvec))

class CubicSplineInterpolator(Interpolator):
    def __init__(self, tx,ty,f,fx,fy,fxy):
        super().__init__(tx,ty,f,fx,fy,fxy)
        self.fgrid2 = []
        #First should do spline interpolation in the x direction.
        #Then at can get f(x_eval,y_i) = (z_i,y_i)

        #First create L for x dir
        mu = np.zeros(len(self.tx))
        for i in range(len(self.tx)):
            mu[i] = 0.5
        #set mu_n = 0
        mu[-1] = 0
        lamb = np.zeros(len(self.tx))
        for i in range(len(self.tx)):
            lamb[i] = 0.5
        #set mu_n = 0
        lamb[0] = 0
        #Construct L for Lm=d
        L = np.zeros((len(self.tx),len(self.tx)))
        for i in range(len(self.tx)):
            L[i,i] = 2
            if i>0:
                L[i,i-1] = mu[i]
            if i<len(self.tx)-1:
                L[i,i+1] = lamb[i]
        self.Lx = L

        #Now create L for y dir
        mu = np.zeros(len(self.ty))
        for i in range(len(self.ty)):
            mu[i] = 0.5
        #set mu_n = 0
        mu[-1] = 0
        lamb = np.zeros(len(self.ty))
        for i in range(len(self.ty)):
            lamb[i] = 0.5
        #set mu_n = 0
        lamb[0] = 0
        #Construct L for Lm=d
        L = np.zeros((len(self.ty),len(self.ty)))
        for i in range(len(self.ty)):
            L[i,i] = 2
            if i>0:
                L[i,i-1] = mu[i]
            if i<len(self.ty)-1:
                L[i,i+1] = lamb[i]
        self.Ly = L

        #Create the M's for x interpolation at each y
        Mx = np.zeros((len(self.ty), len(self.tx)))
        for j in range(len(self.ty)):
            fgridj = np.array(self.fgrid)[:,j]
            d = np.zeros(len(self.tx))
            for i in range(1,len(self.tx)-1):
                part1 = (fgridj[i+1] -fgridj[i] )/(self.tx[i+1] - self.tx[i])/(self.tx[i+1]-self.tx[i-1])
                part2 = (fgridj[i] -fgridj[i-1] )/(self.tx[i] - self.tx[i-1])/(self.tx[i+1]-self.tx[i-1])
                d[i] = 6*(part1-part2)
            #d[0] = 2*-np.cos(1.5) #specific bc
            #d[-1] = 2*-np.cos(1.5)*np.cos(2) #specific bc
            d[0] = 0 #natural splin
            d[-1] = 0 #natural spline
            Mx[j] = lin.inv(self.Lx).dot(d)
        self.Mx=Mx


    def interpolate_x(self,x_eval,y_eval):
        i0,j0 = self.find_cell(x_eval,y_eval)
        for j in range(len(self.ty)):
            fgridj = np.array(self.fgrid)[:,j]
            #Use M to calculate the interpolated value at (x_eval,y_j)
            x=self.tx
            i = i0+1 # for consitency
            part1 = self.Mx[j][i-1]*(x[i]-x_eval)**3/6 + self.Mx[j][i]*(x_eval-x[i-1])**3/6
            part2 = (fgridj[i-1] - self.Mx[j][i-1]*self.dx**2/6)*(x[i]-x_eval) + (fgridj[i] - self.Mx[j][i]*self.dx**2/6)*(x_eval-x[i-1])
            self.fgrid2.append( (part1+part2)/self.dx)

    def interpolate(self,x_eval,y_eval):

        #First do interpolation in the x direction. This does a spline interpolation at each y_i
        self.interpolate_x(x_eval,y_eval)
        #At the end we will have fgrid2 which is an array of the interpolated values at x_eval : f(x_eval, y_i)


        #Now Find d and hence M for y direction
        fgrid2 = np.array(self.fgrid2)
        #print("fgrid2")
        #print(fgrid2)
        self.fgrid2=fgrid2
        d = np.zeros(len(self.ty))
        for i in range(1,len(self.ty)-1):
            part1 = (fgrid2[i+1] -fgrid2[i] )/(self.ty[i+1] - self.ty[i])/(self.ty[i+1]-self.ty[i-1])
            part2 = (fgrid2[i] -fgrid2[i-1] )/(self.ty[i] - self.ty[i-1])/(self.ty[i+1]-self.ty[i-1])
            d[i] = 6*(part1-part2)
        #d[0] = 2*-np.cos(1.5) #specific bc
        #d[-1] = 2*-np.cos(1.5)*np.cos(2.5) #specific bc
        d[0] = 0 #natural spline
        d[-1] = 0 #natural spline
        #self.d = d
        M = lin.inv(self.Ly).dot(d)
        self.M=M

        # Do the interpolation in y
        i0,j0 = self.find_cell(x_eval,y_eval)
        y=self.ty
        print("i,j = ",i0,j0)
        j = j0+1 # for consitency
        part1 = M[j-1]*(y[j]-y_eval)**3/6 + M[j]*(y_eval-y[j-1])**3/6
        part2 = (self.fgrid2[j-1] - M[j-1]*self.dy**2/6)*(y[j]-y_eval) + (self.fgrid2[j] - M[j]*self.dy**2/6)*(y_eval-y[j-1])
        return (part1+part2)/self.dy






#Now do a Test:

def f(x,y):
    return math.cos(x)*math.cos(y)
def fx(x,y):
    return -math.sin(x)*math.cos(y)
def fy(x,y):
    return math.cos(x)*-math.sin(y)
def fxy(x,y):
    return math.sin(x)*math.sin(y)


xgrid = [0,0.5,1,1.5,2]
ygrid = [0,0.5,1,1.5,2, 2.5]
xgrid=np.arange(0,2,0.1)
ygrid=np.arange(0,2.5,0.1)
xtest,ytest = 0.85,0.75
print("actual value =", f(xtest,ytest))
I = CubicSplineInterpolator(xgrid,ygrid,f,fx,fy,fxy)
print("Interp value = ",I.interpolate(xtest,ytest))
