import firedrake as fd
import math
import numpy as np
import scipy as sp
import cmath
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

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
              self.dt = T / N_t
              self.dtc = fd.Function(self.R).assign(T/N_t) # firedrake constant for time step
              self.N = N_t # number of discretisation # FIXME: N is changed
              self.dim = dim # dimension of the problem
              # Setup for the coordinates
              if dim == 1:
                     self.x, = fd.SpatialCoordinate(self.mesh)
              if dim == 2:
                     self.x, self.y = fd.SpatialCoordinate(self.mesh)
              self.FunctionSpace = fd.VectorFunctionSpace(self.mesh, "CG", 1, dim=self.N) # This is FunctionSpace V
              self.MixedSpace = self.FunctionSpace * self.FunctionSpace # this is FunctionSpace W = V*V
              self.U = fd.Function(self.MixedSpace) # this is Mixed Function U = [u, p]
              self.u, self.p = fd.split(self.U)
              self.u_til = fd.Function(self.FunctionSpace) # U_til
              self.TrialFunction = fd.TrialFunction(self.MixedSpace)
              self.tv, self.tw = fd.split(self.TrialFunction)
              self.TestFunction = fd.TestFunction(self.MixedSpace)
              self.v, self.w = fd.split(self.TestFunction) # test function
              self.CG1 = fd.FunctionSpace(self.mesh, 'CG', 1)
              zeros = fd.Function(self.FunctionSpace).interpolate(fd.as_vector([0 for i in range(self.N)]))
              self.bcs = [fd.DirichletBC(self.MixedSpace.sub(0), zeros, 'on_boundary'),
                     fd.DirichletBC(self.MixedSpace.sub(1), zeros, 'on_boundary')]


       def Build_f(self):
              self.f = fd.Function(self.FunctionSpace)
              self.f_a = fd.Function(self.FunctionSpace)
              self.func = fd.Function(self.CG1)
              f_exp = []
              if self.dim == 1:
                     for i in range(self.N): #TODO: I did a extra dt**2 here! modified
                            f_exp.append(- 1 / self.gamma * fd.sin(fd.pi * self.x) * (fd.exp(i*self.dtc) - fd.exp(fd.Function(self.R).assign(self.T)))**2)
              if pc:
                     self.f.interpolate(fd.as_vector(f_exp) * fd.sqrt(self.gamma)) # scale factor root gamma
              else:
                     self.f.interpolate(fd.as_vector(f_exp))
              self.f_a.interpolate(fd.as_vector(f_exp))
              self.f_exp = f_exp
              # print(fd.Function(self.CG1).interpolate(self.f_exp[0]).dat.data)


       def Build_g(self):
              self.g = fd.Function(self.FunctionSpace)
              g_exp = []
              if self.dim == 1:
                     for i in range(1, self.N + 1):
                            g_exp.append((2 * (2*fd.exp(2*i*self.dtc) - fd.exp(fd.Function(self.R).assign(self.T)+i*self.dtc)) * fd.sin(fd.pi*self.x)
                                   + fd.pi ** 2 * fd.sin(fd.pi*self.x)*(fd.exp(i*self.dtc)-fd.exp(fd.Function(self.R).assign(self.T)))**2
                                   + fd.sin(fd.pi * self.x) * fd.cos(fd.pi * i * self.dtc)))
              self.g.interpolate(fd.as_vector(g_exp))


       def Build_Initial_Condition(self):
              if self.dim == 1:
                     if pc:
                            self.u_0 = fd.Function(self.CG1).interpolate(fd.sqrt(self.gamma) * fd.sin(fd.pi * self.x))
                            self.u_1 = fd.Function(self.CG1).interpolate(fd.sqrt(self.gamma) * fd.Function(self.R).assign(0.0))
                     else:
                            self.u_0 = fd.Function(self.CG1).interpolate(fd.sin(fd.pi * self.x))
                            self.u_1 = fd.Function(self.CG1).interpolate(fd.Function(self.R).assign(0.0))


       def Build_L(self):
              scale = fd.sqrt(self.gamma) # used for pc
              for i in range(self.N): # loop over 0 to N-1
                     un = self.u[i]
                     pn = self.p[i]
                     fn = self.f[i]
                     gn = self.g[i]
                     if i == 1: #TODO: i=0 case is separated
                            unm1 = self.u[0]
                            unm2 = self.u_0
                     elif i == 0:
                            unm1 = fd.Constant(math.nan)
                            unm2 = fd.Constant(math.nan)
                     else:
                            unm1 = self.u[i-1]
                            unm2 = self.u[i-2]
                     if i == self.N - 2: #TODO: i = N-1 case is separated
                            pnp2 = fd.Constant(0)
                            pnp1 = self.p[self.N - 1]
                     elif i == self.N - 1:
                            pnp2 = fd.Constant(math.nan)
                            pnp1 = fd.Constant(math.nan)
                     else:
                            pnp1 = self.p[i+1]
                            pnp2 = self.p[i+2]


                     if i == 0:
                            if pc:
                                   Lu = fd.inner(un, self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 / 2 * fd.grad(un), fd.grad(self.v[i])) * fd.dx
                                   Lu += fd.inner(- self.dtc**2 / 2 / scale * pn, self.v[i]) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * (1/2 * fn + self.u_1 /self.dtc + self.u_0 /self.dtc**2),self.v[i]) * fd.dx

                                   Lp = fd.inner(self.dtc**2 * un / scale, self.w[i]) * fd.dx
                                   Lp += fd.inner((pn-2*pnp1+pnp2), self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2 * fd.grad((pn+pnp2)/2), fd.grad(self.w[i])) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * gn, self.w[i]) * fd.dx
                            else:
                                   Lu = fd.inner(un, self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 / 2 * fd.grad(un), fd.grad(self.v[i])) * fd.dx
                                   Lu += fd.inner(- self.dtc**2 / 2 / self.gamma * pn, self.v[i]) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * (1/2 * fn + self.u_1 /self.dtc + self.u_0 /self.dtc**2),self.v[i]) * fd.dx
                                   
                                   Lp = fd.inner(self.dtc**2 * un, self.w[i]) * fd.dx
                                   Lp += fd.inner((pn-2*pnp1+pnp2), self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2 * fd.grad((pn+pnp2)/2), fd.grad(self.w[i])) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * gn, self.w[i]) * fd.dx

                     elif i == self.N - 1:
                            if pc:
                                   Lu = fd.inner(((un-2*unm1+unm2) - self.dtc**2 * pn / scale), self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 * fd.grad((un+unm2)/2) * scale, fd.grad(self.v[i])) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * fn, self.v[i]) * fd.dx
                                   
                                   Lp = fd.inner(pn, self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2 / 2 * fd.grad(pn), fd.grad(self.w[i])) * fd.dx
                                   Lp += fd.inner(self.dtc**2 / 2 * un / scale, self.w[i]) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * 1/2 * gn, self.w[i]) * fd.dx
                            else:
                                   Lu = fd.inner(((un-2*unm1+unm2) - self.dtc**2 * pn / self.gamma), self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 * fd.grad((un+unm2)/2), fd.grad(self.v[i])) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * fn, self.v[i]) * fd.dx

                                   Lp = fd.inner(pn, self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2/2 * fd.grad(pn), fd.grad(self.w[i])) * fd.dx
                                   Lp += fd.inner(self.dtc**2 / 2 * un, self.w[i]) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * 1/2 * gn, self.w[i]) * fd.dx

                     else:
                            if pc:
                                   Lu = fd.inner(((un-2*unm1+unm2) - self.dtc**2 * pn / scale), self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 * fd.grad((un+unm2)/2), fd.grad(self.v[i])) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * fn, self.v[i]) * fd.dx

                                   Lp = fd.inner(self.dtc**2 * un / scale, self.w[i]) * fd.dx
                                   Lp += fd.inner((pn-2*pnp1+pnp2), self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2 * fd.grad((pn+pnp2)/2), fd.grad(self.w[i])) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * gn, self.w[i]) * fd.dx
                            else:
                                   Lu = fd.inner(((un-2*unm1+unm2) - self.dtc**2 * pn / self.gamma), self.v[i]) * fd.dx
                                   Lu += fd.inner(self.dtc**2 * fd.grad((un+unm2)/2), fd.grad(self.v[i])) * fd.dx
                                   Lu -= fd.inner(self.dtc**2 * fn, self.v[i]) * fd.dx

                                   Lp = fd.inner(self.dtc**2 * un, self.w[i]) * fd.dx
                                   Lp += fd.inner((pn-2*pnp1+pnp2), self.w[i]) * fd.dx
                                   Lp += fd.inner(self.dtc**2 * fd.grad((pn+pnp2)/2), fd.grad(self.w[i])) * fd.dx
                                   Lp -= fd.inner(self.dtc**2 * gn, self.w[i]) * fd.dx
                     if i == 0:
                            LHS = Lu+Lp
                     else:
                            LHS += Lu + Lp

              self.L = LHS


       def solve(self, parameters=None, complex=False):
              if parameters: # set the solver parameter pc
                     params = parameters
              else:
                     params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

              if complex: # build the complete equation, do not use fd.derivative()
                     self.Build_f()
                     self.Build_g()
                     self.Build_Initial_Condition()
                     self.Build_L()

                     prob_up = fd.NonlinearVariationalProblem(self.L, self.U, bcs=self.bcs) # TODO: delete RHS
                     solv_up = fd.NonlinearVariationalSolver(prob_up, solver_parameters=params)
                     solv_up.solve()
                     u_sol, p_sol = self.U.subfunctions

                     v = fd.TestFunction(self.CG1)
                     w = fd.TestFunction(self.CG1)
                     dt = self.dtc
                     f = self.f
                     g = self.g
                     gamma = self.gamma
                     grad = fd.grad
                     bcs = fd.DirichletBC(self.CG1, fd.Constant(0), 'on_boundary')

                     u0 = fd.inner(u_sol[0], v)*fd.dx
                     u0 += dt**2 /2 * fd.inner(grad(u_sol[0]), grad(v)) * fd.dx
                     u0 -= dt**2 /2 / gamma * fd.inner(p_sol[0], v) * fd.dx
                     u0 -= fd.inner(self.u_0 + dt * self.u_1 + dt**2/2 * f[0], v) * fd.dx
                     print("~~~~~~~~~~~~~~~~~~~~~~", fd.norm(fd.assemble(u0, bcs=bcs).riesz_representation()))
                     


                     k = 2
                     un = u_sol[k]
                     unm1 = u_sol[k-1]
                     unm2 = u_sol[k-2]
                     #unm2 = self.u_0
                     fn = f[k]
                     gn = g[k]
                     pn = p_sol[k]
                     pnp1 = p_sol[k+1]
                     pnp2 = p_sol[k+2]
                     bcs = fd.DirichletBC(self.CG1, fd.Constant(0), 'on_boundary')

                     Lu = fd.inner(((un-2*unm1+unm2) - dt**2 * pn / gamma), v) * fd.dx
                     Lu += fd.inner(dt**2 * grad((un+unm2)/2), grad(v)) * fd.dx
                     Lu -= fd.inner(dt**2 * fn, v) * fd.dx

                     Lp = fd.inner(dt**2 * un, w) * fd.dx
                     Lp += fd.inner((pn-2*pnp1+pnp2), w) * fd.dx
                     Lp += fd.inner(dt**2 * grad((pn+pnp2)/2), grad(w)) * fd.dx
                     Lp -= fd.inner(dt**2 * gn, w) * fd.dx

                     print("!!!!!!!!!!!!!!!!!!!!!!!!", fd.norm(fd.assemble(Lu, bcs=bcs).riesz_representation()))
                     print("!!!!!!!!!!!!!!!!!!!!!!!!", fd.norm(fd.assemble(Lp, bcs=bcs).riesz_representation()))
                     # if fd.norm(fd.assemble(Lp, bcs=bcs).riesz_representation()) > 1e-6:
                     #        raise ValueError("The equation is not solved properly")
              return u_sol, p_sol


       def write(self, u_sol, p_sol): # TODO: check this: DONE at 28/1
              # WRITE SOLUTION
              if self.dim == 1:
                     sol_file = VTKFile('sol_1d.pvd')
                     ana_file = VTKFile('ana_1d.pvd')
              elif self.dim == 2:
                     sol_file = VTKFile('sol_2d.pvd')
                     ana_file = VTKFile('ana_2d.pvd')
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
                     if i < self.N-1:
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

# the control test problem
T = 2
N_t = 50
N_x = 10
dim = 1
gamma = 10 # regulariser parameter #TODO: in the end, consider gamma -> 0 limit.
# when gamma is too small the test problem f become ill conditioned and needs really small dt to 


equ = Optimal_Control_Wave_Equation(N_x, T, N_t, gamma, dim=dim)


# solver parameters for parallel method
parameters = {
       #'snes': snes_sparameters, #TODO: is this needed?
       'snes_type':'ksponly',
       'mat_type': 'matfree',
       'ksp_type': 'gmres',
       'ksp_gmres_modifiedgramschmidt':None,
       #'ksp_gmres_restart': 30,
       'ksp': {
       'monitor': None,
       'converged_reason': None,
       },
       'ksp_max_it':30,
       #'ksp_atol':1,
       #'ksp_rtol':0.5,
       'ksp_view':None,
       'pc_type': 'python',
       'pc_python_type': '__main__.DiagFFTPC', #TODO: needs to be put after the pc class.? not necessarily
}

# setup variables that we wish to use in the pc class.
V = equ.FunctionSpace
W = equ.MixedSpace # W = V*V
vu = equ.v # test function space for u
vp = equ.w # test function space for p
dt = equ.dt
bcs = equ.bcs

# Options
pc = True
complex = True


# here we define the preconditioner class, it is kept in the same file.
class DiagFFTPC(fd.PCBase):



       def initialize(self, pc): #FIXME: N_t variables now.
              self.xf = fd.Cofunction(W.dual()) # copy the code back.
              self.yf = fd.Function(W)
              self.w = fd.Function(W)
              self.f = fd.Function(W)

              # eigenvalues for Gamma 1&2 #TODO: should it be N_t-1?
              self.Lambda_1 = 1 - 2 * np.exp(2j*np.pi/ N_t * np.arange(N_t)) + np.exp(4j*np.pi/(N_t) * np.arange(N_t))
              self.Lambda_2 = 1 + np.exp(4j*np.pi/N_t * np.arange(N_t))

              self.S1 = np.sqrt(-np.conj(self.Lambda_2) / self.Lambda_2) #TODO: need to check this in ipython CHECKED
              self.S2 = -np.sqrt(-self.Lambda_2 / np.conj(self.Lambda_2))

              # self.Gamma = 1j * dt ** 2 / np.sqrt(gamma) * np.abs(1/self.Lambda_2)# TODO: CHECK?

              self.m1 = np.real(self.Lambda_1/self.Lambda_2)
              self.m2 = - dt ** 2 /np.conj(self.Lambda_2)/ np.sqrt(gamma)
              self.m3 = dt ** 2 /self.Lambda_2/ np.sqrt(gamma)

              self.Sigma_1 = self.m1 + self.m2 * self.S1
              self.Sigma_2 = self.m1 + self.m3 * self.S2

              # self.Sigma_1 = self.Lambda_1 / self.Lambda_2 + self.Gamma
              # self.Sigma_2 = self.Lambda_1 / self.Lambda_2 - self.Gamma


              tu, tp = fd.TrialFunctions(W)
              fu, fp = fd.split(self.f)


              # RHS
              L = fd.inner(1/2*(fu[0]+fd.conj(self.S1[0])*fp[0]), vu[0]) * fd.dx
              L += fd.inner(1/2*(fp[0]+ fd.conj(self.S2[0])*fu[0]), vp[0]) * fd.dx
              for i in range(1, N_t):
                     L += fd.inner(1/2*(fu[i]+ fd.conj(self.S1[i])*fp[i]), vu[i]) * fd.dx
                     L += fd.inner(1/2*(fp[i]+ fd.conj(self.S2[i])*fu[i]), vp[i]) * fd.dx


              # LHS
              D = fd.inner(self.Sigma_1[0]*tu[0], vu[0]) * fd.dx
              D += fd.inner(dt**2/2 * fd.grad(tu[0]),fd.grad(vu[0])) * fd.dx
              D += fd.inner(self.Sigma_2[0]*tp[0], vp[0]) * fd.dx
              D += fd.inner(dt**2/2 * fd.grad(tp[0]),fd.grad(vp[0])) * fd.dx
              for i in range(1, N_t): #TODO: dimension the same as unknowns
                     D += fd.inner(self.Sigma_1[i]*tu[i], vu[i]) * fd.dx
                     D += fd.inner(dt**2/2 * fd.grad(tu[i]),fd.grad(vu[i])) * fd.dx
                     D += fd.inner(self.Sigma_2[i]*tp[i], vp[i]) * fd.dx
                     D += fd.inner(dt**2/2 * fd.grad(tp[i]),fd.grad(vp[i])) * fd.dx


              # Simple solver for the test of the rest
              A = fd.inner(tu, vu) * fd.dx + fd.inner(tp, vp) * fd.dx
              B = fd.inner(fu, vu) * fd.dx + fd.inner(fp, vp) * fd.dx

              params_linear = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
              prob_w = fd.LinearVariationalProblem(D, L, self.w, bcs=bcs)
              #prob_w = fd.LinearVariationalProblem(A, B, self.w, bcs=bcs)
              self.solv_w = fd.LinearVariationalSolver(prob_w, solver_parameters=params_linear)


       def update(self, pc): # TODO: we don't need this?
              pass


       def apply(self, pc, x, y):
              PETSc.Sys.Print('applying')
              with self.xf.dat.vec_wo as v: # vector write only mode to ensure communication.
                     x.copy(v)
              
              #TODO: Why is it self.xf here? where is x
              #x_array = self.xf.dat.data # 2*N_x * N_t tensor
              u_array = self.xf.dat[0].data[:] # N_x * N_t array
              p_array = self.xf.dat[1].data[:]
              # scaling FFT step FFT of r
              vu1 = ifft(u_array, axis=0)
              vp1 = ifft(p_array, axis=0)
              self.xf.dat[0].data[:] = vu1
              self.xf.dat[1].data[:] = vp1

              # solve D w = 1/2 (S* X M) r = g
              f = self.xf.riesz_representation() # the function within the mixed space
              self.f.assign(f) # pass the copied value
              self.solv_w.solve()
              PETSc.Sys.Print('pc solver solved')

              # Apply S X M on w
              uf, pf = fd.split(self.w)
              uy, py = self.yf.subfunctions
              ufin = []
              pfin = []
              for i in range(N_t):
                     ufin.append(uf[i] + fd.Constant(self.S2[i]) * pf[i])
                     pfin.append(pf[i] + fd.Constant(self.S1[i]) * uf[i])
              uy.interpolate(fd.as_vector(ufin))
              py.interpolate(fd.as_vector(pfin))

              # apply lambda_2^-1 to the ffted vector #TODO: seems not to be consistent with the paper
              uf, pf = fd.split(self.yf)
              uy, py = self.yf.subfunctions
              ufin = []
              pfin = []
              for i in range(N_t):
                     ufin.append(uy[i]/fd.Constant(self.Lambda_2[i]))
                     pfin.append(py[i]/fd.Constant(self.Lambda_2[i].conj()))
              uy.interpolate(fd.as_vector(ufin))
              py.interpolate(fd.as_vector(pfin))

              yu_array = self.yf.dat[0].data[:]
              yp_array = self.yf.dat[1].data[:]

              # ifft to get s
              yu1 = fft(yu_array, axis=0)
              yp1 = fft(yp_array, axis=0)
              self.yf.dat[0].data[:] = yu1
              self.yf.dat[1].data[:] = yp1

              with self.yf.dat.vec_ro as v: # read only mode to ensure communication 
                     v.copy(y)

              PETSc.Sys.Print('applyed')

       def applyTranspose(self, pc, x, y):
              raise NotImplementedError



if pc:
       if complex:
              # pc version
              u_sol, p_sol = equ.solve(parameters=parameters, complex=True) #TODO: Build complex
              equ.write(u_sol, p_sol)
       else:
              print('Should use complex firedrake to implement the preconditioner.')
else:
       if complex:
              # direct version
              u_sol, p_sol = equ.solve(complex=True)
              equ.write(u_sol, p_sol)
       else:
              # direct version
              print("Use Complex mode")
