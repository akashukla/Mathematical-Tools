local Plasma    = (require "App.PlasmaOnCartGrid").Gyrokinetic()
local math      = require("sci.math").generic
local diff      = require("sci.diff-recursive")
local df        = diff.df
local Grid      = require("Grid")
local Basis = require "Basis"
local Time=require("Lib.Time")
local DataStruct = require("DataStruct")
local Updater = require("Updater")
local calc_gij = require("Updater.calc_gij3d")
local PopApp = require "App.Species.Population"
local Messenger = require "Comm.Messenger"





-- Setup configuration space grid.
local lower       = {0.5, -math.pi/2., 0.} -- Configuration space lower left.
local upper       = {1.0 , math.pi/2., math.pi/2.}    -- Configuration space upper right.
local cells       = {xcells , ycells, zcells}                 -- Configuration space cells.
local basisNm       = "serendipity"            -- One of "serendipity" or "maximal-order".
local polyOrder   = 1                        -- Polynomial order.
local decompCuts = {1, 1, 1}
--local periodicDirs = {2}     -- Periodic in y only.
local commManager = Messenger{
   cells = cells,   decompCutsConf = decompCuts,
}
local population = PopApp{ messenger = commManager }


print("running with %d cells", xcells)

local mapc2p = function(xc)
   local r, theta, phi = xc[1], xc[2], xc[3]
   local X = r*math.sin(theta)*math.cos(phi)
   local Y = r*math.sin(theta)*math.sin(phi)
   local Z = r*math.cos(theta)
   --print("r,theta,phi,X,Y,Z = ", r,theta,phi,X,Y,Z)
   return X, Y, Z
end

local gij = function(xc)
   local r, theta, phi = xc[1], xc[2], xc[3]
   return 1., 0., 0.,
          0., r^2, 0.,
          0., 0., r^2*math.sin(theta)^2
end


local GridConstructor = Grid.MappedCart
local confGrid = GridConstructor {
   lower = lower,
   decomposition = commManager:getConfDecomp(),
   upper = upper,
   cells = cells,
   mapc2p = mapc2p,
   periodicDirs = periodicDirs,
}

local function createBasis(nm, ndim, polyOrder)
   if nm == "serendipity" then
      return Basis.CartModalSerendipity { ndim = ndim, polyOrder = polyOrder }
   elseif nm == "maximal-order" then
      return Basis.CartModalMaxOrder { ndim = ndim, polyOrder = polyOrder }
   elseif nm == "tensor" then
      return Basis.CartModalTensor { ndim = ndim, polyOrder = polyOrder }
   end
end
local confBasis = createBasis(basisNm, #cells, polyOrder)

local function createField(grid, basis, vComp)
   vComp = vComp or 1
   local fld = DataStruct.Field {
      onGrid        = grid,
      numComponents = basis:numBasis()*vComp,
      ghost         = {1, 1},
      metaData      = {polyOrder = basis:polyOrder(),
                       basisType = basis:id(),},
   }
   return fld
end


--First lets calculate the approximate gs (Recovered)
local evMap = Updater.EvalOnNodes {
   onGrid = confGrid,
   evaluate = function (t,xn) return mapc2p(xn) end,
   basis  = confBasis,
   onGhosts = true,
}

local separateComponents = Updater.SeparateVectorComponents {
   onGrid = confGrid,
   basis = confBasis,
}
local metricCalc = calc_gij{
   grid = confGrid,
   basis = confBasis,
}
local g11 = createField(confGrid,confBasis, 1)
local g12 = createField(confGrid,confBasis, 1)
local g13 = createField(confGrid,confBasis, 1)
local g21 = createField(confGrid,confBasis, 1)
local g22 = createField(confGrid,confBasis, 1)
local g23 = createField(confGrid,confBasis, 1)
local g31 = createField(confGrid,confBasis, 1)
local g32 = createField(confGrid,confBasis, 1)
local g33 = createField(confGrid,confBasis, 1)

local mapField = createField(confGrid,confBasis, 3)
local XField = createField(confGrid,confBasis, 1)
local YField = createField(confGrid,confBasis, 1)
local ZField = createField(confGrid,confBasis, 1)
evMap:advance(0.0,{},{mapField})
separateComponents:advance(0, {mapField}, {XField,YField, ZField})
metricCalc:advance(XField, YField, ZField, g11, g12, g13, g21, g22, g23, g31, g31, g33)

local s = {}
for i =1,3 do
   s[i] = {}
   for j = 1,3 do
      s[i][j] = 4.0/(confGrid:dx(i)*confGrid:dx(j))
   end
end


g11:scale(s[1][1])
g12:scale(s[1][2])
g13:scale(s[1][3])
g21:scale(s[2][1])
g22:scale(s[2][2])
g23:scale(s[2][3])
g31:scale(s[3][1])
g32:scale(s[3][2])
g33:scale(s[3][3])

--Now the approximate fields from the original method (AD)
local geoField = Plasma.Geometry {
   bmag = function (t, xn)
      return 2.0 -- unimportant, not used
   end,
   -- Geometry is not time-dependent.
   evolve = false,
   tEnd=0.,
   nFrame=1,
}

local ioMethod="MPI"
geoField:fullInit(geoField.tbl) -- Complete initialization.
print("did full init")
geoField:setIoMethod(ioMethod)
geoField:setBasis(confBasis)
geoField:setGrid(confGrid)
geoField:alloc(1)--timeStepper.numFields)
geoField:createDiagnostics()
geoField:createSolver(population)
geoField:initField()

local g11ad = geoField.geo.g_xx
local g12ad = geoField.geo.g_xy
local g13ad = geoField.geo.g_xz
local g21ad = geoField.geo.g_xy
local g22ad = geoField.geo.g_yy
local g23ad = geoField.geo.g_yz
local g31ad = geoField.geo.g_xz
local g32ad = geoField.geo.g_yz
local g33ad = geoField.geo.g_zz




--Now the exact fields
local evMetric = Updater.EvalOnNodes {
   onGrid = confGrid,
   evaluate = function (t,xn) return gij(xn) end,
   basis  = confBasis,
   onGhosts = true,
}

local metricField = createField(confGrid,confBasis, 9)
local g11exact = createField(confGrid,confBasis, 1)
local g12exact = createField(confGrid,confBasis, 1)
local g13exact = createField(confGrid,confBasis, 1)
local g21exact = createField(confGrid,confBasis, 1)
local g22exact = createField(confGrid,confBasis, 1)
local g23exact = createField(confGrid,confBasis, 1)
local g31exact = createField(confGrid,confBasis, 1)
local g32exact = createField(confGrid,confBasis, 1)
local g33exact = createField(confGrid,confBasis, 1)
evMetric:advance(0.0,{},{metricField})
separateComponents:advance(0, {metricField}, {g11exact,g12exact,g13exact,g21exact,g22exact,g23exact,g31exact,g32exact,g33exact})






glist = {g11, g22,g33}
gadlist = {g11ad, g22ad,g33ad}
gexactlist = {g11exact, g22exact, g33exact}


function l2diff(f1, f2)
   local localRange = f1:localRange()
   local indexer = f1:genIndexer()
   local vol = confGrid:cellVolume()
   local dfact = 1/2^confGrid:ndim()

   local l2 = 0.0
   for idxs in localRange:colMajorIter() do
      local f1Itr = f1:get(indexer(idxs))
      local f2Itr = f2:get(indexer(idxs))
      for k = 1, f1:numComponents() do
	 l2 = l2 + (f1Itr[k]-f2Itr[k])^2
      end
   end

   return math.sqrt(l2*vol*dfact)
end

local dx = math.sqrt(confGrid:cellVolume())
print("dx = %g",dx)



--print(string.format("L2 Error for cells = %d  %g\n", cells[1], l2diff(g11, g11exact)))
local errors = {}
local errorsad = {}
for i =1,3 do
   errors[i] = l2diff(glist[i], gexactlist[i])
   errorsad[i] = l2diff(gadlist[i], gexactlist[i])
   print(string.format("L2 Error from recovery method for cells = %d  %g\n", cells[1], l2diff(glist[i], gexactlist[i])))
   print(string.format("L2 Error from AD method for cells = %d  %g\n", cells[1], l2diff(gadlist[i], gexactlist[i])))
end



--g11:write('g11.bp',0,0, false)
--g22:write('g22.bp',0,0, false)
--g33:write('g33.bp',0,0, false)
--g11ad:write('g11ad.bp',0,0, false)
--g22ad:write('g22ad.bp',0,0, false)
--g33ad:write('g33ad.bp',0,0, false)
--g11exact:write('g11exact.bp',0,0, false)
--g22exact:write('g22exact.bp',0,0, false)
--g33exact:write('g33exact.bp',0,0, false)

file = io.open(string.format("allcell.csv"),"a")
file:write(string.format("%d,",cells[1]))
for i =1,3 do
  file:write(string.format("%g,",errors[i]) )
end
file:write(string.format("\n"))

file = io.open(string.format("allcellad.csv"),"a")
file:write(string.format("%d,",cells[1]))
for i =1,3 do
  file:write(string.format("%g,",errorsad[i]) )
end
file:write(string.format("\n"))


