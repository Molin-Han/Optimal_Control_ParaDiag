import firedrake as fd
import math
import cmath
import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
from Optimal_Control_ParaDiag.Code.Control_Wave_PC import Optimal_Control_Wave_Equation



class DiagFFTPC(fd.PCBase):# TODO: Where to inherit from

    def __init__(self):
        self.initialized = False
        #self.control = control_problem # must be a Control_Wave_Direct problem
        self.xf = fd.Cofunction(V.dual()) # copy the code back.
        self.yf = fd.Function(V)

    def setUp(self, pc):#TODO: do we need this?
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)


    def initialize(self, pc):
        self.N_x = self.control.N_x
        self.N_t = self.control.N
        dt = self.control.dtc # TODO: change here
        self.gamma = self.control.gamma
        # eigenvalues for Gamma 1&2
        self.Lambda_1 = 1 - 2 * np.exp(2j*np.pi/self.N_t * np.arange(self.N_t)) + np.exp(4j*np.pi/self.N_t * np.arange(self.N_t))
        self.Lambda_2 = 1 + np.exp(4j*np.pi/self.N_t * np.arange(self.N_t))
        self.S1 = np.sqrt(- np.conj(self.Lambda_2) / self.Lambda_2) #TODO: need to check this in ipython CHECKED
        self.S2 = -np.sqrt(- self.Lambda_2 / np.conj(self.Lambda_2))
        self.Gamma = 1j * dt ** 2 / fd.sqrt(self.gamma) * fd.abs(1/self.Lambda_2)# TODO: CHECK?
        self.Sigma_1 = self.Lambda_1 / self.Lambda_2 + self.Gamma
        self.Sigma_2 = self.Lambda_1 / self.Lambda_2 - self.Gamma












    def update(self, pc): # TODO: we don't need this?
        pass

    


    def apply(self, pc, x, y):
        with self.xf.dat.vec_wo as v: #TODO: what should be this?
            x.copy(v)


        x_array = self.xf.dat.data
        u_array = self.xf.dat[0].data
        p_array = self.xf.dat[1].data
        # scaling 
        vu1 = fft(u_array, axis=0) # TODO: change the name
        vp1 = fft(p_array,axis=0)
        self.xf.dat[0].data[:] = vu1
        self.xf.dat[1].data[:] = vp1

        # don't need to form this
        form = v1[0,] + np.conj(self.S2[0])*v1[self.N_t, ]
        for i in range(2*self.N_t): #TODO:Check
            if 0 < i < self.N_t: #loop for u
                form += v1[i, ] + np.conj(self.S2[i])*v1[self.N_t+i, ]
            if self.N_t <= i <= 2*self.N_t: # loop for p
                form += np.conj(self.S1[i - self.N_t])*v1[i-self.N_t] + v1[i]

        g = fd.assemble(form)



        

        with self.yf.dat.vec_ro as v:
            v.copy(y)





    

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

