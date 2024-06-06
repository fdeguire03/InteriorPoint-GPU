# InteriorPoint-GPU
GPU-aware interior point solver for convex optimization problems

Please note that this package is still under development and contains a few bugs. Most notably, the solvers occasionally fail to converge on some problems, despite confirmed feasibility. We have noticed this issue most commonly on heavily constrained (hundreds of equality and inequality constraints) QPs and SOCPs. Likely caused by numerical errors in our backtracking search / linear solves or by suboptimal hyperparameter values (maximum number of iterations, initial t, mu, etc. are all set nominally instead of dynamically tuned to each problem instance).

See demo.ipynb for overview of usage. Init method of LPSolver also describes all of the input parameters.

Currently implemented for LPs, QPs, and SOCPs. Dedicated solver using ADMM for LASSO problems in parallel.

Implemented as part of the final project for EE364B Convex Optimization II Spring 2024 at Stanford University.
