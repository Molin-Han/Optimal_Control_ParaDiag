import firedrake as fd
import math
import cmath
import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
from Control_Wave_Direct import Optimal_Control_Wave_Equation



class DiagFFTPC(fd.PCBase):# TODO: Where to inherit from

    def __init__(self, control_problem):
        self.initialized = False
        self.control = control_problem # must be a Control_Wave_Direct problem

    def setUp(self, pc):#TODO: do we need this?
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)


    def initialize(self, pc):
        N_x = self.control.N_x
        N_t = self.control.N
        # eigenvalues for Gamma 1&2
        Gamma_1 = 1 - 2 * np.exp(2j*np.pi/N_t * np.arange(N_t)) + np.exp(4j*np.pi/N_t * np.arange(N_t))
        Gamma_2 = 1 + np.exp(4j*np.pi/N_t * np.arange(N_t))
        S1 = np.sqrt(- np.conj(Gamma_2) / Gamma_2) #TODO: need to check this in ipython CHECKED
        S2 = -np.sqrt(- Gamma_2 / np.conj(Gamma_2))





    def update(self, pc): # TODO: we don't need this?
        pass

    


    def apply(self, pc, x, y):





    

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

