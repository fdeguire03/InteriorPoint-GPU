import numpy as np
from SOCPSolver import SOCPSolver
from LPSolver import LPSolver
from QPSolver import QPSolver
import cProfile
import cvxpy as cp
from time import time

np.random.seed(1)
n = 200
m = 160
k = 40

A = np.random.uniform(low=-2, high=2, size=(m, n))

# Generate C
C = np.random.uniform(low=-2, high=2, size=(k, n))

# Generate x_feas and c
x_feas = np.random.uniform(low=-2, high=2, size=(n))
c = np.random.uniform(low=-2, high=2, size=(n))

# From this, calculate b and d
b = A @ x_feas
d = C @ x_feas

# Have upper and lower bounds
up_bnd = 3
lo_bnd = -3


ls_gpu = LPSolver(
    c=c,
    A=A,
    b=b,
    C=C,
    d=d,
    lower_bound=lo_bnd,
    upper_bound=up_bnd,
    use_gpu=True,
    suppress_print=True,
    check_cvxpy=False,
    epsilon=1e-4,
    mu=15,
    t0=1,
    max_inner_iters=20,
    beta=0.5,
    alpha=0.05
)
"""

# SOCP problem

m = 50
n = 750
p = 500
n_i = 1000
f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)
for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i] @ x0 + b[i], 2) - c[i].T @ x0)
F = np.random.randn(p, n)
g = F @ x0

tic = time()
ls_gpu = SOCPSolver(P=None, 
        q=f,
        A=A,
        b=b,
        c=c,
        d=d,
        F=F,
        g=g,
        use_gpu=True,
        upper_bound=None,
        lower_bound=None,
        suppress_print=True,
        update_slacks_every=0,
        check_cvxpy=True,
        alpha=0.02,)
toc = time()
print(f'CVXPY time: {toc-tic}')
"""


#x = cp.Variable(n)

# Objective
#obj = c @ x

# Constraints
#constr = [A @ x == b, C @ x <= d, lo_bnd <= x, x <= up_bnd]

# Create problem
#prob = cp.Problem(cp.Minimize(obj), constr)


pr = cProfile.Profile()

pr.enable()


"""PLACE CODE YOU WANT TO PROFILE HERE"""

ls_gpu.solve()

"""PLACE CODE YOU WANT TO PROFILE HERE"""


pr.disable()
pr.dump_stats("profiling_gpu_socp.prof")
print(ls_gpu.value - ls_gpu.cvxpy_val)
