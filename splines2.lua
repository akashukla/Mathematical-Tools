local Proto = require "Lib.Proto"
local math = require("sci.math").generic
local matrix = require "matrix2"
local diff      = require("sci.diff-recursive")

--print("Initial testing")
--m1 = matrix{{8,4,1},{6,8,3}, {2,8,2}}
--m2 = matrix{{-8,1,3},{5,2,1}, {1,4,5}}
--print(m1)
--
--m1inv = matrix.invert(m1)
--print(m1inv)
--print(matrix.mul(m1,m1inv))
--
--v1 = matrix{1,2,3}
--v2=matrix {4,5,6}
--print("vectors")
--print(v1)
--print(v2)
--print("dot")
--num = v1^'T' * v2
--print(num)
--
--print("ncols of m1")
--print(matrix.columns(m1))
--
--print(m1[2][1])
--
--print("col of m1")
--print(matrix.get_col(m1,3))
--
--print("test for loop")
--for i = 1,5 do
--   print(i)
--end
--print("Done with Initial testing")
----yes lua includes upper bound in loop unlike python


--Make interpolator class
local Interpolator =  Proto()

function Interpolator:init(tbl)
   self.tx = tbl.tx
   self.ty = tbl.ty
   self.f = tbl.f
   self.dx = self.tx[2]-self.tx[1]
   self.dy = self.ty[2]-self.ty[1]
   --boundary conditions default to 0 second derivative
   self.fxx_lower = tbl.fxx_lower or 0.
   self.fxx_upper= tbl.fxx_upper or 0.
   self.fyy_lower = tbl.fyy_lower or 0.
   self.fyy_upper = tbl.fyy_upper or 0.

   -- Create f and derivatives on grid
   fgrid = {}
   --print("tx,ty")
   --print(matrix{self.tx})
   --print(matrix{self.ty})
   for i = 1, #self.tx do
      --print("in init interpolator, i = ",i)
      fgrid[i] = {}
      for j =1, #self.ty do
         --print("in init interpolator, j = ",j)
         fgrid[i][j] = self.f(self.tx[i], self.ty[j])
      end
   end
   self.fgrid = fgrid

   --Now do the more complicated stuff
   self.fgrid2 = {}
   --First should do spline interpolation in the x direction.
   --Then at can get f(x_eval,y_i) = (z_i,y_i)

   --First create L for x dir
   --First make mu and lambda then construct L
   mu = {}
   for i =1, #self.tx -1 do
      mu[i] = 0.5
   end
   --set mu_n = 0
   mu[#self.tx] = 0

   lamb = {} --np.zeros(len(self.tx))
   for i =2, #self.tx do
       lamb[i] = 0.5
   end
   --set lambda_0 = 0
   lamb[1] = 0

   --Construct L for Lm=d
   --L = np.zeros((len(self.tx),len(self.tx)))
   Lx = {}          -- create the matrix
   for i=1,#self.tx do
      Lx[i] = {}     -- create a new row
      for j=1,#self.tx do
         Lx[i][j] = 0
      end
   end

   for i =1, #self.tx do
       Lx[i][i] = 2
       if diff.lt(1,i) then Lx[i][i-1] = mu[i] end
       if diff.lt(i,#self.tx) then Lx[i][i+1] = lamb[i] end
   end
   self.Lx = Lx

   --Now make L for y dir
   mu = {}
   for i =1, #self.ty -1 do
      mu[i] = 0.5
   end
   --set mu_n = 0
   mu[#self.ty] = 0

   lamb = {} --np.zeros(len(self.ty))
   for i =2, #self.ty do
       lamb[i] = 0.5
   end
   --set lambda_0 = 0
   lamb[1] = 0

   --Construct L for Lm=d
   --L = np.zeros((len(self.ty),len(self.ty)))
   L = {}          -- create the matrix
   for i=1,#self.ty do
      L[i] = {}     -- create a new row
      for j=1,#self.ty do
         L[i][j] = 0
      end
   end

   for i =1, #self.ty do
       L[i][i] = 2
       if diff.lt(1,i) then L[i][i-1] = mu[i] end
       if diff.lt(i,#self.ty) then L[i][i+1] = lamb[i] end
   end
   self.Ly = L

   --Create the M's for x interpolation at each y
   --Mx = np.zeros((len(self.ty), len(self.tx)))
   Mx = {}          -- create the matrix
   for i=1,#self.ty do
      Mx[i] = {}     -- create a new row
      for j=1,#self.tx do
         Mx[i][j] = 0
      end
   end
   --print("Lx")
   print("Lx rows", #self.Lx)
   print("Lx cols ", #self.Lx[1])
   --print(matrix{self.Lx})
   --Lxinv = matrix.invert(self.Lx)
   --self.Lxinv = Lxinv
   self.Lx = Matrix(self.Lx)
   print("Lx rows", self.Lx:rows())
   print("Lx cols ", self.Lx:cols())
   self.Lxinv = self.Lx:inv()
   print("Lxinv rows",  self.Lxinv:rows())
   print("Lxinv cols ", self.Lxinv:cols())
   self.Lxinv=self.Lxinv:totable()
   for j = 1, #self.ty do
       --get the column of fgrid, fgridj
       fgridj = {}
       for i = 1,#self.tx do
          fgridj[i] = self.fgrid[i][j]
       end
       --print("fgridj")
       --print(matrix{fgridj})
       d = {} --np.zeros(len(self.tx))
       for i = 2, #self.tx-1 do
           part1 = (fgridj[i+1] -fgridj[i] )/(self.tx[i+1] - self.tx[i])/(self.tx[i+1]-self.tx[i-1])
           part2 = (fgridj[i] -fgridj[i-1] )/(self.tx[i] - self.tx[i-1])/(self.tx[i+1]-self.tx[i-1])
           d[i] = 6*(part1-part2)
       end
       --d[1] = 2*-math.cos(1.5) --specific bc
       --d[#self.tx] = 2*-math.cos(1.5)*math.cos(2) --specific bc
       d[1] = self.fxx_lower
       d[#self.tx] = self.fxx_upper
       self.d = d
       for i = 1, #d do
         --print("i for lxinv")
         lvec = self.Lxinv[i]
         rvec = d
         sum = 0
         for k = 1, #d do
            sum = sum+lvec[k]*rvec[k]
         end
         Mx[j][i] = sum
       end
   end
   --print("Mx")
   --print(matrix{Mx})
   self.Mx=Mx
   --self.Lyinv = matrix.invert(self.Ly)
   print("Ly rows", #self.Ly)
   print("Ly cols ", #self.Ly[1])
   self.Ly = Matrix(self.Ly)
   print("Ly rows", self.Ly:rows())
   print("Ly cols ", self.Ly:cols())
   self.Lyinv = self.Ly:inv()
   print("Lxinv rows",  self.Lyinv:rows())
   print("Lxinv cols ", self.Lyinv:cols())
   self.Lyinv=self.Lyinv:totable()
end

function Interpolator:find_cell(x_eval,y_eval)
    -- Find the subcell i0, j0
    for i = 1,#self.tx do
        if diff.lt(x_eval,self.tx[i+1]) then
            i0 = i
            break
         end
    end
    for j = 1,#self.ty do
        if diff.lt(y_eval, self.ty[j+1]) then
            j0 = j
            break
         end
    end
    return i0,j0
end


function Interpolator:interpolate_x(x_eval,y_eval)
   i0,j0 = self:find_cell(x_eval,y_eval)
   --compute d and hence M for each j
   for j =1,#self.ty do
       --get the column of fgrid, fgridj
       fgridj = {}
       for i = 1,#self.tx do
          fgridj[i] = self.fgrid[i][j]
       end
       --Use M to calculate the interpolated value at (x_eval,y_j)
       x=self.tx
       i = i0+1 -- for consitency
       part1 = self.Mx[j][i-1]*(x[i]-x_eval)^3/6 + self.Mx[j][i]*(x_eval-x[i-1])^3/6
       part2 = (fgridj[i-1] - self.Mx[j][i-1]*self.dx^2/6)*(x[i]-x_eval) + (fgridj[i] - self.Mx[j][i]*self.dx^2/6)*(x_eval-x[i-1])
       self.fgrid2[j] = (part1+part2)/self.dx
   end
end

function Interpolator:interpolate(x_eval,y_eval)
    --First do interpolation in the x direction. This does a spline interpolation at each y_i
    self:interpolate_x(x_eval,y_eval)

    Lyinv = self.Lyinv
    --print("Lyinv")
    --print(matrix{Lyinv})
    --At the end we will have fgrid2 which is an array of the interpolated values at x_eval : f(x_eval, y_i)
    --Now Find d and hence M for y direction
    fgrid2 = self.fgrid2
    d = {}--np.zeros(len(self.ty))
    for i = 2, #self.ty-1 do -- in range(1,len(self.ty)-1):
        part1 = (fgrid2[i+1] -fgrid2[i] )/(self.ty[i+1] - self.ty[i])/(self.ty[i+1]-self.ty[i-1])
        part2 = (fgrid2[i] -fgrid2[i-1] )/(self.ty[i] - self.ty[i-1])/(self.ty[i+1]-self.ty[i-1])
        d[i] = 6*(part1-part2)
    end
    --d[1] = 2*-math.cos(1.5) --specific bc
    --d[#self.ty] = 2*-math.cos(1.5)*math.cos(2.5) --specific bc
    d[1] = self.fyy_lower
    d[#self.ty] = self.fyy_upper
    --self.d = d
    --M = lin.inv(self.Ly).dot(d)
    --self.M=M
    --print("d in y")
    --print(matrix{d})
    My = {}
    for i = 1, #d do
      lvec = Lyinv[i]
      rvec = d
      sum = 0
      for j = 1, #d do
         sum = sum+lvec[j]*rvec[j]
      end
      My[i] = sum
    end
    self.My=My

    -- Do the interpolation in y
    i0,j0 = self:find_cell(x_eval,y_eval)
    y=self.ty
    --print("i,j = ",i0,j0)
    j = j0+1 -- for consitency
    part1 = My[j-1]*(y[j]-y_eval)^3/6 + My[j]*(y_eval-y[j-1])^3/6
    part2 = (self.fgrid2[j-1] - My[j-1]*self.dy^2/6)*(y[j]-y_eval) + (self.fgrid2[j] - My[j]*self.dy^2/6)*(y_eval-y[j-1])
    return (part1+part2)/self.dy
end


return Interpolator

--I = Interpolator{
--   tx = {0,0.5,1.0,1.5,2},
--   ty = {0,0.5,1.0,1.5,2,2.5},
--   f = function(x,y) return math.cos(x)*math.cos(y) end,
--   --fxx_lower = 2*-math.cos(1.5)
--   --fxx_upper = 2*-math.cos(1.5)*math.cos()
--   --fyy_lower = 2*-math.cos(1.5)
--   --fyy_upper = 2*-math.cos(1.5)*math.cos(2.5)
--}
--
--a = I:interpolate(0.7,0.9)
--print("a")
--print(a)
