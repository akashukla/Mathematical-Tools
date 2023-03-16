import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
#data = pg.Data('perlmutter_sims/loops_sim_read2/loops_grid.bp')
#data = pg.Data('alpha_fix_strongfield/cerfon_grid.bp')
#data = pg.Data('loops_sim_spline/loops_grid.bp')
data = pg.Data('alpha_fix_iterlike/cerfon_grid.bp')
grid = data.getValues()
x = grid[1:-1,0]
y = grid[1:-1,1]
z = grid[1:-1,2]


#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(x[len(x)//2:],y[len(y)//2:],z[len(z)//2:])
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(x[0:len(x)//2],y[0:len(y)//2],z[0:len(z)//2])
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x,y,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


R = np.sqrt(x**2 + y**2)
Z = z
phi = np.arctan(y/x)
ax = plt.figure().add_subplot()
ax.plot(R,z)
ax.set_xlabel('R')
ax.set_ylabel('Z')


ax = plt.figure().add_subplot()
ax.scatter(np.arange(0,len(phi)),phi)
#ax.set_ylabel('$\phi$')
plt.show()
