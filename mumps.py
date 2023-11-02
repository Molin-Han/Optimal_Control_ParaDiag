import firedrake as fd
import math
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


# Time discretisation step
T = 1
dt = 0.01
dtc = fd.Constant(dt)
N = int(T/dt + 1)
mesh = fd.UnitSquareMesh(40, 40)
x, y = fd.SpatialCoordinate(mesh)
V = fd.VectorFunctionSpace(mesh, "CG", 1, dim=N-1)
W = V * V
U = fd.Function(W)
g = fd.Function(W)
u, p = fd.split(U)
#du_trial = fd.TrialFunction(V)
test = fd.TestFunction(W) # TODO: can we do it without using split ANs: no!
v, w = fd.split(test)

V0 = fd.FunctionSpace(mesh, 'CG', 1)
u_0 = fd.Function(V0).assign(fd.Constant(1.0))
u_1 = fd.Function(V0).assign(fd.Constant(1.0)) # Boundary conditions

p_0 = fd.Constant(1.0) # TODO: what is the value of this

unp1 = fd.Constant(1.0) # TODO:


gamma = 0.5 # regulariser parameter

f = 1

#rhs = fd.inner(f,v)*fd.dx
# for i in range(N-1):
#     unp1 = u[i]
#     if i == 0:
#         un = u_0
#         unm1 = u_0 - u_1*dtc
#     elif i ==1:
#         un = u[i-1]
#         unm1 = u_0
#     else:
#         un = u[i-1]
#         unm1 = u[i-2]
#     pbarn = p[i]
# un = fd.Function(V0)
# pn = fd.Function(V0)
# unp1 = fd.Function(V0)
# unp2 = fd.Function(V0)
# vnp1 = fd.Function(V0)
# pnp1 = fd.Function(V0)
# pnp2 = fd.Function(V0)
# pnm1 = fd.Function(V0)
# unm1 = fd.Function(V0)
# fnp1 = fd.Function(V0)
# gn = fd.Function(V0)
# wn = fd.Function(V0) # TODO clean way of doing this

# M = (unp2 - 2 * unp1 + un) * vnp1 * fd.dx
# M += - dtc ** 2 * (pnp2 + pn) / 2 * vnp1 / gamma * fd.dx
# M += dtc ** 2 * fd.grad((unp2 + un) / 2) * fd.grad(vnp1) * fd.dx # FIXME: not able to get the grad of it.
# M += - dtc ** 2 * fnp1 * vnp1 * fd.dx

# N = (pnp1 - 2 * pn + pnm1) * wn * fd.dx
# N += dtc ** 2 * (unp1 + unm1) / 2 * wn * fd.dx
# N += dtc ** 2 *fd.grad((pnp1 + pnm1) / 2) * fd.grad(wn) * fd.dx
# N += - dtc ** 2 * gn * wn * fd.dx


for i in range(N-1):
       if i == 0:
              un = u_0
              pnm1 = p_0
              unm1 = u_0 - u_1 * dtc
       else:
              un = u[i]
              pnm1 = p[i-1]
              unm1 = u[i-1]
       pn = p[i]
       unp1 = u[i+1]
       unp2 = u[i+2]
       vnp1 = v[i+1]
       pnp1 = p[i+1]
       pnp2 = p[i+2]
       fnp1 = f[i+1]
       wn = w[i]
       gn = g[i]
       M = (unp2 - 2 * unp1 + un) * vnp1 * fd.dx
       M += - dtc ** 2 * (pnp2 + pn) / 2 * vnp1 / gamma * fd.dx
       M += dtc ** 2 * fd.grad((unp2 + un) / 2) * fd.grad(vnp1) * fd.dx # FIXME: not able to get the grad of it.
       M += - dtc ** 2 * fnp1 * vnp1 * fd.dx

       N = (pnp1 - 2 * pn + pnm1) * wn * fd.dx
       N += dtc ** 2 * (unp1 + unm1) / 2 * wn * fd.dx
       N += dtc ** 2 *fd.grad((pnp1 + pnm1) / 2) * fd.grad(wn) * fd.dx
       N += - dtc ** 2 * gn * wn * fd.dx
       if i == 0:
              L_u = M
              L_p = N
       else:
              L_u += M
              L_p += N



bcs = [fd.DirichletBC(W.sub(0), fd.Constant(0), 'on_boundary'),
       fd.DirichletBC(W.sub(1), fd.Constant(0), 'on_boundary')]
du = fd.Function(V)



params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
prob_u = fd.NonlinearVariationalProblem(L_u + L_p, bcs=bcs)
solv_u = fd.NonlinearVariationalSolver(prob_u, solver_parameters=params)

solv_u.solve()