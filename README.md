Molin Han's mastery project supervised by Prof. Colin Cotter at Imperial College London.

## Abstract
In this report, an efficient preconditioner is introduced for iteratively solving the large sparse
linear matrix-vector system, which arises from discretising the optimality system in the optimal
control of wave equations with an all-at-once second-order finite difference scheme in time and
a P1 finite element scheme in space. The preconditioner is designed for a time-parallelisable
implementation using a unitary diagonlisation decomposition with the ParaDiag technique. We
demonstrate the successful implementation of the algorithm using the automated finite element
package Firedrake. Moreover, the performance of the preconditioner and the solution accuracy
are illustrated through a one-dimensional test problem.

The report will be available upon request via molin.han20@imperial.ac.uk .

[1] .-L. Wu and J. Liu. A parallel-in-time block-circulant preconditioner for optimal control of wave equations. SIAM Journal on Scientific Computing, 42(3):A1510–A1540, 2020.
[2] . J. Gander, J. Liu, S.-L. Wu, X. Yue, and T. Zhou. ParaDiag: parallel-in-time algorithms based on the diagonalization technique, 2021.
