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
        self.mesh = fd.UnitIntervalMesh(N_x)
        self.N_x = N_x
        self.T = T
        self.gamma = fd.Constant(gamma)
        self.dt = T / N_t
        self.dtc = fd.Constant(self.dt)
        self.N_t = N_t
        self.dim = dim
        self.x, = fd.SpatialCoordinate(self.mesh)

        self.FunctionSpace = fd.VectorFunctionSpace(self.mesh, "CG", 1, dim=self.N_t)
        