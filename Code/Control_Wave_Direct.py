import firedrake as fd
import math
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


class Optimal_Control_Wave_Equation:

       def __init__(self, N_x, T, N_t, gamma, dim=1):
              if dim == 1:
                     self.mesh = fd.UnitIntervalMesh(N_x)
              if dim == 2:
                     self.mesh = fd.UnitSquareMesh(N_x, N_x)
              self.N_x = N_x # spatial step
              self.T = T # total time interval
              self.R = fd.FunctionSpace(self.mesh, 'R', 0) # constant function space
              self.gamma = fd.Function(self.R).assign(gamma) # Regularising coefficient
              self.dtc = fd.Function(self.R).assign(T/N_t) # firedrake constant for time step
              self.N = N_t # number of discretisation # FIXME:N is changed
              self.dim = dim # dimension of the problem
              # Setup for the coordinates
              if dim == 1:
                     self.x, = fd.SpatialCoordinate(self.mesh)
              if dim == 2:
                     self.x, self.y = fd.SpatialCoordinate(self.mesh)
              self.FunctionSpace = fd.VectorFunctionSpace(self.mesh, "CG", 1, dim=self.N-1) # This is FunctionSpace V
              self.MixedSpace = self.FunctionSpace * self.FunctionSpace # this is FunctionSpace W = V*V
              self.U = fd.Function(self.MixedSpace) # this is Mixed Function U = [u, p]
              self.u, self.p = fd.split(self.U)
              self.u_til = fd.Function(self.FunctionSpace) # U_til
              #du_trial = fd.TrialFunction(V)
              self.TestFunction = fd.TestFunction(self.MixedSpace)
              self.v, self.w = fd.split(self.TestFunction) # test function
              self.CG1 = fd.FunctionSpace(self.mesh, 'CG', 2)


       def f_1d(self, x, i, gamma):
              return -1/gamma * fd.sin(fd.pi * x) * (fd.exp(i * self.dtc))

#TODO: make a transformation function to transfer from u-coord to t-coord?
       def Build_f(self):
              self.f = fd.Function(self.FunctionSpace)
              self.f_a = fd.Function(self.FunctionSpace)
              self.func = fd.Function(self.CG1)
              f_exp = []
              f_0 = []
              if self.dim == 2: # need correction
                     for i in range(self.N - 1):
                            f_exp.append((1+2*fd.pi**2) * fd.exp(i*self.dtc) * fd.sin(fd.pi*self.x) * fd.sin(fd.pi*self.y)
                                   -1/self.gamma *(i*self.dtc - fd.Function(self.R).assign(self.T))**2 * fd.sin(fd.pi*self.x) * fd.sin(fd.pi*self.y))
              if self.dim == 1:
                     for i in range(1, self.N):
                            #TODO: f=0 gives when gamma is really small, to avoid numerical instability in 1/gamma
                            f_exp.append(- 1 / self.gamma * fd.sin(fd.pi * self.x) * (fd.exp(i*self.dtc) - fd.exp(fd.Function(self.R).assign(self.T)))**2)
                            f_0.append(fd.Function(self.R).assign(0))
              self.f.interpolate(fd.as_vector(f_exp)) # Solve for f = 0 problem
              self.f_a.interpolate(fd.as_vector(f_exp))
              self.f_exp = f_exp
              # print(fd.Function(self.CG1).interpolate(self.f_exp[0]).dat.data)


       def Build_g(self):
              self.g = fd.Function(self.FunctionSpace)
              g_exp = []
              if self.dim == 2:
                     for i in range(self.N - 1): # need to be changed
                            g_exp.append((fd.exp(i*self.dtc) + fd.Function(self.R).assign(2.0) 
                                          + 2*fd.pi**2*(i*self.dtc-self.T)**2) * fd.sin(fd.pi*self.x) * fd.sin(fd.pi*self.y))
              if self.dim == 1:
                     for i in range(2, self.N + 1):
                            g_exp.append(2 * (2*fd.exp(2*i*self.dtc) - fd.exp(fd.Function(self.R).assign(self.T)+i*self.dtc)) * fd.sin(fd.pi*self.x)
                                   + fd.pi ** 2 * fd.sin(fd.pi*self.x)*(fd.exp(i*self.dtc)-fd.exp(fd.Function(self.R).assign(self.T)))**2
                                   + fd.sin(fd.pi * self.x) * fd.cos(fd.pi * i * self.dtc))
              self.g.interpolate(fd.as_vector(g_exp))


       def Build_Initial_Condition(self):
              if self.dim == 2:
                     self.u_0 = fd.Function(self.CG1).interpolate(fd.sin(fd.pi * self.x)*fd.sin(fd.pi * self.y))
                     self.u_1 = fd.Function(self.CG1).interpolate(fd.sin(fd.pi * self.x)*fd.sin(fd.pi * self.y))
              if self.dim == 1:
                     self.u_0 = fd.Function(self.CG1).interpolate(fd.sin(fd.pi * self.x))
                     self.u_1 = fd.Function(self.CG1).interpolate(fd.Function(self.R).assign(0.0))


       def Build_Action(self): # TODO : Check the indexing and coefficient! DONE at 18/1
              for i in range(self.N-1): # loop over 0 to N-2
                     if i == 0:
                            unm1 = self.u_0 * fd.cos(self.T / self.N * fd.pi) + self.u_1 * self.dtc #Changed here
                            unm2 = self.u_0
                     elif i == 1:
                            unm1 = self.u[0]
                            unm2 = self.u_0 * fd.cos(self.T / self.N * fd.pi) + self.u_1 * self.dtc
                     else:
                            unm1 = self.u[i-1]
                            unm2 = self.u[i-2]
                     if i == self.N-2:
                            pnp1 = fd.Function(self.R).assign(0.0)
                     else:
                            pnp1 = self.p[i+1]
                     un = self.u[i]
                     pn = self.p[i]
                     u_tiln =  pn / self.gamma
                     u_bar = (un + unm2) / 2

                     fn = self.f[i] # TODO:doubt?
                     gn = self.g[i]

                     L_g = 1 / 2 * (un - gn)**2 * fd.dx
                     L_u_til = self.gamma / 2 * u_tiln**2 * fd.dx
                     L_p = pn * ((un - 2 * unm1 + unm2) / (self.dtc ** 2) - fn - u_tiln) * fd.dx
                     L_p += fd.dot(fd.grad(pn), fd.grad(u_bar)) * fd.dx

                     # L_g = self.dtc / 2 * (un - gn)**2 * fd.dx
                     # L_u_til = self.gamma / 2 * self.dtc * u_tiln**2 * fd.dx
                     # L_p = self.dtc * pn * ((un - 2 * unm1 + unm2) / (self.dtc ** 2) - fn - u_tiln) * fd.dx
                     # L_p += self.dtc * fd.dot(fd.grad(pn), fd.grad(u_bar)) * fd.dx

                     L = L_g + L_u_til + L_p
                     if i == 0:
                            S_g = L_g
                            S_til = L_u_til
                            S_p = L_p
                            S = L
                     else:
                            S_g += L_g
                            S_til += L_u_til
                            S_p += L_p
                            S += L
              self.S = S
              self.S_g = S_g
              self.S_til = S_til
              self.S_p = S_p


       def solve(self):
              self.Build_f()
              self.Build_g()
              self.Build_Initial_Condition()
              self.Build_Action()
              zeros = fd.Function(self.FunctionSpace).interpolate(fd.as_vector([0 for i in range(self.N-1)]))
              bcs = [fd.DirichletBC(self.MixedSpace.sub(0), zeros, 'on_boundary'),
                     fd.DirichletBC(self.MixedSpace.sub(1), zeros, 'on_boundary')]
              du = fd.Function(self.FunctionSpace)
              M = fd.derivative(self.S, self.U)
              #print(fd.assemble(M).dat.data[:])

              params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
              prob_u = fd.NonlinearVariationalProblem(M, self.U, bcs=bcs)
              solv_u = fd.NonlinearVariationalSolver(prob_u, solver_parameters=params)

              solv_u.solve()
              u_sol, p_sol = self.U.subfunctions

              del_M = fd.assemble(M, bcs=bcs).dat.data[:]
              print("check if solver is working, norm of assembled del S:", np.linalg.norm(del_M))
              del_g = fd.assemble(self.S_g)
              del_S_til = fd.assemble(self.S_til)
              del_p = fd.assemble(self.S_p)
              print("check assembled action parts", del_g, del_S_til, del_p)
              #TODO: this del_p should be 0 and del_g should be near 0 if gamma is nearly 0 i.e. we can perfect control without caring cost
              return u_sol, p_sol


       def write(self, u_sol, p_sol): # TODO: check this: DONE at 28/1
              # WRITE SOLUTION
              if self.dim == 1:
                     sol_file = fd.File('sol_1d.pvd')
                     ana_file = fd.File('ana_1d.pvd')
              elif self.dim == 2:
                     sol_file = fd.File('sol_2d.pvd')
                     ana_file = fd.File('ana_2d.pvd')
              u_out = fd.Function(self.CG1, name='u_out')
              p_out = fd.Function(self.CG1, name='p_out')
              # ANALYTIC SOLUTION
              u_ana = fd.Function(self.CG1, name='u_ana')
              p_ana = fd.Function(self.CG1, name='p_ana')
              g_out = fd.Function(self.CG1, name='g_out')
              # Test
              fa = fd.Function(self.CG1)
              print('!!!',fd.norm(u_sol - self.g)) # if gamma is small, that should tend to 0
              
              for i in range(self.N+1): # loop over all time from t=0 to t=T
                     if i == 0:
                            p_out.interpolate(fd.Constant(0))
                            u_out.interpolate(self.u_0)
                            g_out.interpolate(fd.Constant(0))
                     elif i == 1:
                            p_out.interpolate(p_sol[0])
                            u_out.interpolate(fd.cos(self.T/self.N*fd.pi)*self.u_0+self.dtc*self.u_1) #TODO: problem here!!!!!
                            g_out.interpolate(fd.Constant(0))
                     elif i >= self.N:
                            p_out.interpolate(fd.Constant(0))
                            u_out.interpolate(u_sol[i-2])
                            g_out.interpolate(self.g[i-2])
                     else:
                            #print(i)
                            u_out.interpolate(u_sol[i-2])
                            p_out.interpolate(p_sol[i-1])
                            g_out.interpolate(self.g[i-2])
                     if self.dim == 2:
                            u_ana.interpolate(fd.exp(i*self.dtc)*fd.sin(fd.pi*self.x)*fd.sin(fd.pi*self.y))
                            p_ana.interpolate((i*self.dtc - self.T)**2*fd.sin(fd.pi*self.x)*fd.sin(fd.pi*self.y))
                     elif self.dim == 1:
                            u_ana.interpolate(fd.sin(fd.pi*self.x)*fd.cos(fd.pi*i*self.dtc))
                            p_ana.interpolate(fd.sin(fd.pi*self.x)*(fd.exp(i*self.dtc)-fd.exp(self.T))**2)
                     #print(fd.norm(u_out-g_out))
                     #print(i)
                     # TODO: test for f = 0
                     print("error for f=0", fd.norm(p_out - self.gamma * fd.Function(self.CG1).interpolate(self.f_exp[i]) + p_ana)/fd.norm(p_out))
                     # TODO: test for relative error for true problem.
                     print("Relative Error in u,p is", [fd.norm(u_out-u_ana) / fd.norm(u_ana), fd.norm(p_out-p_ana) / fd.norm(p_ana)])
                     if i < self.N - 1:
                            pass
                            # print(fd.cos(fd.pi * i *self.dtc))
                            #print('scaling', fd.interpolate(u_sol[i],self.CG1).dat.data[int(self.N_x/2)] / u_ana.dat.data[int(self.N_x/2)])
                            #print('ana_mid', u_ana.dat.data[int(self.N_x/2)])
                            #print('sol_mid', fd.interpolate(u_sol[i],self.CG1).dat.data[int(self.N_x/2)])
                     sol_file.write(u_out, p_out, g_out)
                     ana_file.write(u_ana, p_ana)


T = 2
N_t = 128
N_x = 128
dim = 1
gamma = 1e-1 # regulariser parameter #TODO: in the end, consider gamma -> 0 limit.

equ = Optimal_Control_Wave_Equation(N_x, T, N_t, gamma, dim=dim)
u_sol, p_sol = equ.solve()
equ.write(u_sol, p_sol)
