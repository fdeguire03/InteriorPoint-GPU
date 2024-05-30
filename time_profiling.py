import numpy as np
from SOCPSolver import SOCPSolver
from LPSolver import LPSolver
from QPSolver import QPSolver
import cProfile

n = 100
m = 50
C = np.random.random((m, n))
C = np.where(
    np.random.binomial(1, 0.10, C.shape), C, 0
)  # <- apply a random mask with 10% density  (A will be 90% sparse)
d = np.random.randint(low=1, high=30, size=C.shape[0])

c = np.random.randint(low=1, high=n, size=C.shape[1]) - n / 2

# provide an x0 to force the solver to complete phase 1 solve

ls = LPSolver(
    c=c,
    A=None,
    b=None,
    C=C,
    d=d,
    use_gpu=False,
    upper_bound=None,
    lower_bound=0,
    suppress_print=False,
    epsilon=1e-12,
    x0=np.ones(len(c)),
)

pr = cProfile.Profile()

pr.enable()


"""PLACE CODE YOU WANT TO PROFILE HERE"""

ls.solve() - ls.cvxpy_val

"""PLACE CODE YOU WANT TO PROFILE HERE"""


pr.disable()
pr.dump_stats("profiling.prof")
