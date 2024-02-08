import firedrake as fd
import math
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


# Time discretisation step
T = 2
dt = 0.05
dtc = fd.Constant(dt)
gamma = fd.Constant(0.5) # regulariser parameter #TODO: in the end, consider gamma -> 0 limit.
N = int(T/dt + 1)
#mesh = fd.UnitSquareMesh(40, 40)
#x, y = fd.SpatialCoordinate(mesh)
mesh = fd.UnitIntervalMesh(40)
x, = fd.SpatialCoordinate(mesh)
V = fd.VectorFunctionSpace(mesh, "CG", 1, dim=N-1)
W = V * V
U = fd.Function(W)
u, p = fd.split(U)
u_til = fd.Function(V)
#du_trial = fd.TrialFunction(V)
test = fd.TestFunction(W)
v, w = fd.split(test)

V0 = fd.FunctionSpace(mesh, 'CG', 1)

# TODO: need to set the IVP for a test case.
f = fd.Function(V) # wave equation
#f1 = fd.Function(V0)
g = fd.Function(V) # tracking trajectory
#g1 = fd.Function(V0)
#g.interpolate(fd.sin(2*fd.pi*(x)))
f_exp = []
g_exp = []
print(fd.sin(x))

for i in range(N-1):
       g_exp.append(2 * (2*fd.exp(2*i*dtc) - fd.exp(fd.Constant(T)+i*dtc)) * fd.sin(fd.pi*x)
                    + fd.pi ** 2 * fd.sin(fd.pi*x)*(fd.exp(i*dtc)-fd.exp(fd.Constant(T)))**2
                    + fd.sin(fd.pi * x) * fd.cos(fd.pi * i * dtc))
       f_exp.append(- 1/ gamma * fd.sin(fd.pi *x) * (fd.exp(i*dtc) - fd.exp(fd.Constant(T)))**2)

g.interpolate(fd.as_vector(g_exp))
f.interpolate(fd.as_vector(f_exp))


u_0 = fd.Function(V0).interpolate(fd.sin(fd.pi * x))
u_1 = fd.Function(V0).interpolate(fd.Constant(0.0)) # Boundary conditions



for i in range(N-1): # loop over 0 to N-2
       if i == 0:
              unm1 = u_0 - u_1 * dtc
              unm2 = u_0
       elif i == 1:
              unm1 = u[0]
              unm2 = u_0 - u_1 * dtc
       else:
              unm1 = u[i-1]
              unm2 = u[i-2]
       if i == N-2:
              pnp1 = fd.Constant(0, domain=mesh)
       else:
              pnp1 = p[i+1]
       un = u[i]
       pn = p[i]
       u_tiln =  pnp1 / gamma
       u_tilnm1 = pn / gamma

       fn = f[i]
       gn = g[i]

       L = dtc / 2 * (un - gn)**2 * fd.dx
       L += gamma / 2 * dtc * u_tiln**2 * fd.dx
       L += dtc * pn * ((un - 2 * unm1 + unm2) / dtc ** 2
                        - fn - u_tilnm1) * fd.dx
       L += dtc * fd.dot(fd.grad(pn), fd.grad((un + unm2)/2)) * fd.dx

       if i == 0:
              S = L
       else:
              S += L


zeros = fd.Function(V).interpolate(fd.as_vector([0 for i in range(N-1)]))
bcs = [fd.DirichletBC(W.sub(0), zeros, 'on_boundary'),
       fd.DirichletBC(W.sub(1), zeros, 'on_boundary')]
du = fd.Function(V)
L = fd.derivative(S, U)

params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
prob_u = fd.NonlinearVariationalProblem(L, U, bcs=bcs)
solv_u = fd.NonlinearVariationalSolver(prob_u, solver_parameters=params)

solv_u.solve()
sol_file = fd.File('sol.pvd')
ana_file = fd.File('ana.pvd')
u_sol, p_sol = U.subfunctions
u_out = fd.Function(V0)
p_out = fd.Function(V0)
# ANALYTIC SOLUTION
u_ana = fd.Function(V0)
p_ana = fd.Function(V0)
for i in range(N-1):
       u_ana.interpolate(fd.sin(fd.pi*x)*fd.cos(fd.pi*i*dtc))
       p_ana.interpolate(fd.sin(fd.pi*x)*(fd.exp(i*dtc)-fd.exp(T))**2)
       u_out.interpolate(u_sol[i])
       p_out.interpolate(p_sol[i])
       sol_file.write(u_out, p_out)
       ana_file.write(u_ana, p_ana)