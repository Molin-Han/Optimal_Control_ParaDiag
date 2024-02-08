import firedrake as fd
import math
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def Optimal_Wave(T, dt, gamma, mesh, dim=1, test_u=False, test_p=False):
    #dtc = fd.Constant(dt)
    N = int(T/dt + 1)
    if dim == 2:
            x, y = fd.SpatialCoordinate(mesh)
    if dim == 1:
            x, = fd.SpatialCoordinate(mesh)
    V = fd.VectorFunctionSpace(mesh, "CG", 1, dim=N-1)
    R = fd.FunctionSpace(mesh, 'R', 0)
    dtc = fd.Function(R).assign(dt)
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
    g = fd.Function(V) # tracking trajectory
    f_exp = []
    g_exp = []

    if dim == 2:
            for i in range(N-1):
                    g_exp.append((fd.exp(i*dtc) + fd.Function(R).assign(2.0) 
                                   + 2*fd.pi**2*(i*dtc-T)**2) * fd.sin(fd.pi*x) * fd.sin(fd.pi*y))
                    f_exp.append((1+2*fd.pi**2) * fd.exp(i*dtc) * fd.sin(fd.pi*x) * fd.sin(fd.pi*y)
                            -1/gamma *(i*dtc - fd.Function(R).assign(T))**2 * fd.sin(fd.pi*x) * fd.sin(fd.pi*y))

            g.interpolate(fd.as_vector(g_exp))
            f.interpolate(fd.as_vector(f_exp))

            u_0 = fd.Function(V0).interpolate(fd.sin(fd.pi * x)*fd.sin(fd.pi * y))
            u_1 = fd.Function(V0).interpolate(fd.sin(fd.pi * x)*fd.sin(fd.pi * y)) # Boundary conditions

    # initial condition
    if dim == 1:
            for i in range(N-1):
                    i_i = i-1
                    g_exp.append(2 * (2*fd.exp(2*i_i*dtc) - fd.exp(fd.Function(R).assign(T)+i_i*dtc)) * fd.sin(fd.pi*x)
                            + fd.pi ** 2 * fd.sin(fd.pi*x)*(fd.exp(i_i*dtc)-fd.exp(fd.Function(R).assign(T)))**2
                            + fd.sin(fd.pi * x) * fd.cos(fd.pi * i_i * dtc))
                    f_exp.append(- 1/ gamma * fd.sin(fd.pi *x) * (fd.exp((i-1)*dtc) - fd.exp(fd.Function(R).assign(T)))**2)

            g.interpolate(fd.as_vector(g_exp))
            f.interpolate(fd.as_vector(f_exp))

            u_0 = fd.Function(V0).interpolate(fd.sin(fd.pi * x))
            u_1 = fd.Function(V0).interpolate(fd.Function(R).assign(0.0)) 



    for i in range(N-1): # loop over 0 to N-2
            if i == 0:
                    unm1 = u_0 + u_1 * dtc
                    unm2 = u_0
            elif i == 1:
                    unm1 = u[0]
                    unm2 = u_0 + u_1 * dtc
            else:
                    unm1 = u[i-1]
                    unm2 = u[i-2]
            if i == N-2:
                    pnp1 = fd.Function(R).assign(0.0)
            else:
                    pnp1 = p[i+1]
            un = u[i]
            pn = p[i]
            u_tiln =  pn / gamma
            #u_tilnm1 = pn / gamma
            u_bar = (un + unm2) / 2

            fn = f[i]
            gn = g[i]

            L = dtc / 2 * (un - gn)**2 * fd.dx
            L += gamma / 2 * dtc * u_tiln**2 * fd.dx
            L += dtc * pn * ((un - 2 * unm1 + unm2) / dtc ** 2
                            - fn - u_tiln) * fd.dx
            L += dtc * fd.dot(fd.grad(pn), fd.grad(u_bar)) * fd.dx

            if i == 0:
                    S = L
            else:
                    S += L


    zeros = fd.Function(V).interpolate(fd.as_vector([0 for i in range(N-1)]))
    bcs = [fd.DirichletBC(W.sub(0), zeros, 'on_boundary'),
            fd.DirichletBC(W.sub(1), zeros, 'on_boundary')]
    du = fd.Function(V)
    M = fd.derivative(S, U)
    #print(fd.assemble(M).dat.data[:]) # TODO: seems to be not the correct one. why?

    params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
    prob_u = fd.NonlinearVariationalProblem(M, U, bcs=bcs)
    solv_u = fd.NonlinearVariationalSolver(prob_u, solver_parameters=params)

    solv_u.solve()
    u_sol, p_sol = U.subfunctions

    del_M = fd.assemble(M, bcs=bcs).dat.data[:]
    print(del_M)

    # WRITE SOLUTION
    if dim == 1:
            sol_file = fd.File('sol_1d.pvd')
            ana_file = fd.File('ana_1d.pvd')
    elif dim == 2:
            sol_file = fd.File('sol_2d.pvd')
            ana_file = fd.File('ana_2d.pvd')
    u_out = fd.Function(V0, name='u_out')
    p_out = fd.Function(V0, name='p_out')
    # ANALYTIC SOLUTION
    u_ana = fd.Function(V0, name='u_ana')
    p_ana = fd.Function(V0, name='p_ana')
    g_out = fd.Function(V0, name='g_out')
    for i in range(N-1):
            if dim == 2:
                    u_ana.interpolate(fd.exp(i*dtc)*fd.sin(fd.pi*x)*fd.sin(fd.pi*y))
                    p_ana.interpolate((i*dtc - T)**2*fd.sin(fd.pi*x)*fd.sin(fd.pi*y))
            elif dim == 1:
                    u_ana.interpolate(fd.sin(fd.pi*x)*fd.cos(fd.pi*i*dtc))
                    p_ana.interpolate(fd.sin(fd.pi*x)*(fd.exp(i*dtc)-fd.exp(T))**2)
            u_out.interpolate(u_sol[i])
            p_out.interpolate(p_sol[i])
            g_out.interpolate(g[i])
            sol_file.write(u_out, p_out)
            ana_file.write(u_ana, p_ana, g_out)

    return u_sol, p_sol, del_M


# Using the defined function (workable)
# Time discretisation step
T = 2
dt = 0.1
dim = 1
if dim == 1:
    mesh = fd.UnitIntervalMesh(40)
if dim == 2:
    mesh = fd.UnitSquareMesh(40, 40)            
R = fd.FunctionSpace(mesh, 'R', 0)
gamma = fd.Function(R).assign(0.5) # regulariser parameter #TODO: in the end, consider gamma -> 0 limit.
V0 = fd.FunctionSpace(mesh, 'CG', 1)


u_sol, p_sol, del_M = Optimal_Wave(T, dt, gamma, mesh, dim=1, test_u=False, test_p=False)

#print(del_M[0][1])
for i in range(len(del_M[0])):
    for j in range(2):
        if np.linalg.norm(del_M[j][i]) > 1e-9:
                print(i)
                print(j)
                print(del_M[j][i])
