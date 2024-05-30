import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from NewtonSolver import *
from NewtonSolverInfeasibleStart import *
from PhaseOneSolver import PhaseOneSolver
from FunctionManager import *

try:
    import cupy as cp

    gpu_flag = True
except Exception:
    gpu_flag = False
    print("Not able to run with GPU")


class LPSolver:

    def __init__(
        self,
        c=None,
        A=None,
        b=None,
        C=None,
        d=None,
        lower_bound=0,
        upper_bound=None,
        t0=0.1,
        max_outer_iters=20,
        max_inner_iters=50,
        phase1_max_inner_iters=500,
        epsilon=1e-11,
        inner_epsilon=1e-5,
        check_cvxpy=True,
        linear_solve_method="cholesky",
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=15,
        suppress_print=False,
        use_gpu=False,
        try_diag=True,
        track_loss=False,
        get_dual_variables=False,
        phase1_tol=0,
        phase1_t0=0.01,
        x0=None,
        update_slacks_every=0,
    ):
        """Initialize LP problem of form:
        Minimize c^T x
        Subject to Ax == b
                   Cx <= d
                   lower_bound <= x <= upper_bound

        The remaining parameters:

            t0 (default 1): Starting point for interior point method. Consider using lower value (<0.1). Will take more outer iterations, but fewer inner iterations on Newton solver.

            max_outer_iters (default 50): Maximum number of iterations to run for interior point solver

            max_inner_iters (default 20): Maximum number of iterations of Newton's method to run at each step

            phase1_max_inner_iters (default 500): Maximum number of iterations of Newton's method to run while solving phase 1 method. Consider using larger value because if the solver
                                                    its max iterations, it may conclude that the problem is infeasible even though it is

            epsilon (default 1e-8): Desired optimality gap for entire problem

            inner_epsilon (default 1e-5): Minimum norm of residuals for each Newton solve

            phase1_tol (default 0): Required slack to consider phase 1 method complete. Set to above 0 if you want to find a point that is 'more' feasible.

            phase1_t0 (default 0.01): Starting point for interior point solver during phase 1. Usually performs better when even lower than the t0 for main solver because determining feasibility is paramount.

            check_cvxpy (default True): Performs an initial check on solving the problem using CVXPY to be able to compare optimal value and determine feasibility.
                        May take a long time if problem is large

            linear_solve_method (default 'cholesky'): Method to solve linear equations during Newton solves (options include 'np_solve' (call np.linalg.solve),
                'cholesky' (Solve using Cholesky-factorization and forward-backward substitution), 'cg' (conjugate gradient),
                'direct' (form matrix inverse), and 'np_lstsq' (call np.linalg.lstsq))

            max_cg_iters (default 50): maximum conjugate iterations to perform (used when linear solve method == "cg")

            alpha (default 0.2): parameter for backtrackign search

            beta (default 0.6): parameter for backtracking search

            mu (default 15): parameter for how much to increase t on each centering step

            suppress_print (default True): set to True if you want to suppress any potential warnings from being printed during solve method

            try_diag (default True): set to True if you want to try calcuating all Hessian matrices as diagonal matrices. Potential speedup if Hessian is diagonal,
                                    really no cost if its not, so recommended to keep as True

            track_loss (default True): set to True to be able to plot objective values after completion. Turn off for slight performance gains.

            get_dual_variables (default False): set to True if you want the solve function to calculate optimal dual variables in addition to optimal dual variables. Really no
                                                    performance cost to make the calculation, but it is not fully tested.

            x0 (default None): Leave as None to allow program to automatically generate an initial x. Pass an x with the correct dimensions if you want to supply the starting point.

        """

        # all attributes for LP
        self.A = A
        self.c = c
        self.C = C
        self.b = b
        self.d = d
        self.lb = lower_bound
        self.ub = upper_bound

        self.__check_inputs()

        self.equality_constrained = self.A is not None

        # initialize x
        # TODO: when using inequality constraints, will need to find a way to get a feasible point
        # (using Phase I methods?)
        if self.c is not None:
            self.n = len(self.c)
        elif self.A is not None:
            self.n = self.A.shape[1]
        elif self.C is not None:
            self.n = self.C.shape[1]

        self.x = x0

        self.bounded = self.lb is not None or self.ub is not None
        if self.x is None:
            if self.lb is not None and self.ub is not None:
                self.x = (
                    (np.maximum(self.lb, -1e2) + np.minimum(self.ub, 1e2))
                    / 2
                    * np.ones(self.n)
                )
            elif self.lb is not None:
                self.x = (np.maximum(self.lb, -1e2) + 1e-1) * np.ones(self.n)
            elif self.ub is not None:
                self.x = (np.minimum(self.ub, 1e2) - 1e-1) * np.ones(self.n)
            else:
                self.x = np.random.rand(self.n)

        # If specified, make sure that the problem is feasible usign CVXPY
        # the CVXPY solution can also be used later to verify the solution of the LinearSolver
        if check_cvxpy:
            if not suppress_print:
                print("Testing CVXPY")
            self.feasible, self.cvxpy_val, self.cvxpy_sol = self.__test_feasibility()
            if self.feasible == "infeasible":
                raise ValueError("Provided problem instance is infeasible!")
            elif self.feasible == "unbounded":
                raise ValueError("Provided problem instance is unbounded!")
        else:
            self.feasible, self.cvxpy_val, self.cvxpy_sol = None, None, None

        # transfer to GPU if specified to use GPU (and we are able to given available hardware)
        # TODO: Be able to break into smaller subproblems if not everything can fit onto GPU at the same time
        self.use_gpu = use_gpu and gpu_flag
        if self.use_gpu:
            self.x = cp.array(self.x)
            if self.A is not None:
                self.A = cp.array(self.A)
            if self.c is not None:
                self.c = cp.array(self.c)
            if self.C is not None:
                self.C = cp.array(self.C)
            if self.b is not None:
                self.b = cp.array(self.b)
            if self.d is not None:
                self.d = cp.array(self.d)
            if self.lb is not None:
                self.lb = cp.array(self.lb)
            if self.ub is not None:
                self.ub = cp.array(self.ub)

        # Count the number of inequality constraints (used to determine optimality gap during solve)
        self.num_constraints = 0
        if self.d is not None:
            self.num_constraints += len(self.d)
        if self.lb is not None:
            self.num_constraints += self.n
        if self.ub is not None:
            self.num_constraints += self.n

        # backtracking search parameters
        self.alpha = alpha
        self.beta = beta

        # initializations for interior point method
        self.t0 = t0
        self.mu = mu
        self.outer_iters = 0
        self.inner_iters = []
        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.epsilon = epsilon
        self.inner_epsilon = inner_epsilon
        self.max_cg_iters = max_cg_iters

        # other housekeeping
        # helper fields to denote when problem has been solved
        self.optimal = False
        self.value = None
        self.optimality_gap = None
        self.xstar = None
        self.lam_star = None
        self.vstar = None
        self.suppress_print = suppress_print
        self.try_diag = try_diag
        self.track_loss = track_loss
        self.linear_solve_method = linear_solve_method
        self.get_dual_variables = get_dual_variables
        self.phase1_t0 = phase1_t0
        self.phase1_tol = phase1_tol
        self.phase1_max_inner_iters = phase1_max_inner_iters
        self.update_slacks_every = update_slacks_every

        # initialize the newton solver for this problem
        if self.C is not None:
            self.phase1_solver = self.__make_phase1_solver(phase1_tol)
        self.fm = self.__get_function_manager()
        self.ns = self.__get_newton_solver(linear_solve_method)

    def __check_inputs(self):
        """Make sure that inputs have dimensions in agreement
        If A or b is specified, then both must be specified
        If C or d is specified then both must be specified
        """

        c_flag = self.c is not None
        if c_flag:
            if self.c.ndim != 1:
                raise ValueError("c must be 1-dimensional!")

        A_flag = self.A is not None
        if bool(A_flag) ^ bool(self.b is not None):
            raise ValueError("Both A and b must be defined, or neither!")
        if A_flag:
            if self.A.ndim != 2:
                raise ValueError("A must be 2-dimensional!")
            m, n_A = self.A.shape
            if self.b.ndim != 1:
                raise ValueError("b must be 1-dimensional!")
            if len(self.b) != m:
                raise ValueError("A and b must have agreeing dimensions!")
            if c_flag:
                if len(self.c) != n_A:
                    raise ValueError(
                        "c must have the same number of entries as A has columns!"
                    )

        C_flag = self.C is not None
        if bool(C_flag) ^ bool(self.d is not None):
            raise ValueError("Both C and d must be defined, or neither!")
        if C_flag:
            if self.C.ndim != 2:
                raise ValueError("C must be 2-dimensional!")
            m, n_C = self.C.shape
            if self.d.ndim != 1:
                raise ValueError("d must be 1-dimensional!")
            if len(self.d) != m:
                raise ValueError("C and d must have agreeing dimensions!")
            if c_flag:
                if len(self.c) != n_C:
                    raise ValueError(
                        "c must have the same number of entries as A has columns!"
                    )

        lb_flag = self.lb is not None
        ub_flag = self.ub is not None
        if lb_flag:
            try:
                self.lb = np.array(self.lb)
            except Exception:
                raise ValueError("Lower bound must be a scalar or list!")
            if self.lb.ndim > 0:
                try:
                    if c_flag:
                        assert len(self.lb) == len(self.c)
                    if C_flag:
                        assert len(self.lb) == n_C
                    if A_flag:
                        assert len(self.lb) == n_A
                except AssertionError:
                    raise ValueError(
                        "Lower bound must be a scalar or have the same number of dimensions as other parameters!"
                    )
        if ub_flag:
            try:
                self.ub = np.array(self.ub)
            except Exception:
                raise ValueError("Upper bound must be a scalar or list!")
            if self.ub.ndim > 0:
                try:
                    if c_flag:
                        assert len(self.ub) == len(self.c)
                    if C_flag:
                        assert len(self.ub) == n_C
                    if A_flag:
                        assert len(self.ub) == n_A
                except AssertionError:
                    raise ValueError(
                        "Upper bound must be a scalar or have the same number of dimensions as other parameters!"
                    )
        if ub_flag and lb_flag:
            diff = self.ub - self.lb
            if diff.ndim > 0:
                if (diff < 0).any():
                    raise ValueError("Lower bound must be lower than upper bound")
            else:
                if diff < 0:
                    raise ValueError("Lower bound must be lower than upper bound")

        if C_flag and A_flag:
            if n_C != n_A:
                raise ValueError("A and C must have the same number of columns!")

    def __make_phase1_solver(self, tol):

        phase1_solver = PhaseOneSolver(
            C=self.C,
            d=self.d,
            lower_bound=self.lb,
            upper_bound=self.ub,
            x0=self.x,
            max_outer_iters=self.max_outer_iters,
            max_inner_iters=self.phase1_max_inner_iters,
            epsilon=self.epsilon,
            inner_epsilon=self.inner_epsilon,
            linear_solve_method="cholesky",
            max_cg_iters=self.max_cg_iters,
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            suppress_print=self.suppress_print,
            use_gpu=self.use_gpu,
            track_loss=self.track_loss,
            n=self.n,
            tol=tol,
            t0=self.phase1_t0,
            update_slacks_every=self.update_slacks_every,
        )

        return phase1_solver

    def __get_function_manager(self):
        """Generate functions to use in solve method
        Be aware of where values were not passed for A,C,b,c,d matrices and vectors
        so that functions are as efficient as possible
        """

        fm = FunctionManagerLP(
            c=self.c,
            A=self.A,
            b=self.b,
            C=self.C,
            d=self.d,
            x0=self.x,
            lower_bound=self.lb,
            upper_bound=self.ub,
            t=1,
            use_gpu=self.use_gpu,
            n=self.n,
            try_diag=self.try_diag,
        )

        return fm

    def __get_newton_solver(self, linear_solve_method):
        """Initialize a NewtonSolver object with the specified linear solve method

        Pass all specified parameters from the user
        """

        if linear_solve_method == "np_lstsq":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverNPLstSqInfeasibleStart
                else:
                    NewtonClass = NewtonSolverNPLstSq
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverNPLstSqDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal

        elif linear_solve_method == "np_solve":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverNPSolveInfeasibleStart
                else:
                    NewtonClass = NewtonSolverNPSolve
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverNPSolveDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal
        elif linear_solve_method == "direct":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverDirectInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDirect
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverDirectDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal
        elif linear_solve_method == "cg":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverCGInfeasibleStart
                else:
                    NewtonClass = NewtonSolverCG
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverCGDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal

        elif linear_solve_method == "kkt":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverKKTNPSolveInfeasibleStart
                else:
                    raise ValueError(
                        "No KKT System non-equality-constrained problems! Please choose another solver"
                    )
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverKKTNPSolveDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal
        elif linear_solve_method == "cholesky":
            if self.C is not None or not self.try_diag:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverCholeskyInfeasibleStart
                else:
                    NewtonClass = NewtonSolverCholesky
            else:
                if self.equality_constrained:
                    NewtonClass = NewtonSolverCholeskyDiagonalInfeasibleStart
                else:
                    NewtonClass = NewtonSolverDiagonal
        else:
            raise ValueError("Please enter a valid linear solve method!")

        ns = NewtonClass(
            self.A,
            self.b,
            self.C,
            self.d,
            self.fm,
            max_iters=self.max_inner_iters,
            epsilon=self.inner_epsilon,
            suppress_print=self.suppress_print,
            max_cg_iters=self.max_cg_iters,
            lower_bound=self.lb,
            upper_bound=self.ub,
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            use_gpu=self.use_gpu,
            update_slacks_every=self.update_slacks_every,
        )

        return ns

    def __test_feasibility(self):
        """Use CVXPY to check whether the problem is feasible

        This function is included because determining feasibility of the problem is not the main
        goal of this project (our goal is to solve feasible problems faster)

        This can be suppressed with check_cvxpy = False. We may remove later if we implement better
        feasibility checking"""

        x = cvx.Variable(len(self.x))

        if self.c is not None:
            obj = cvx.Minimize(self.c.T @ x)
        else:
            obj = cvx.Minimize(cvx.sum(x))

        constr = []
        if self.A is not None:
            constr.append(self.A @ x == self.b)

        if self.C is not None:
            constr.append(self.C @ x <= self.d)

        if self.lb is not None:
            constr.append(x >= self.lb)
        if self.ub is not None:
            constr.append(self.ub >= x)

        prob = cvx.Problem(obj, constr)
        try:
            prob.solve(solver="CLARABEL")
        except Exception as e:
            print(e)

        return prob.status, prob.value, x.value

    def __str__(self):
        opt_val = "Not yet solved" if self.optimal is False else self.value
        return f"LinearSolver(Optimal Value: {opt_val})"

    def __repr__(self):
        return str(self)

    def solve(self, resolve=True, **kwargs):
        """Solve the Linear Program

        Parameters:
            resolve: set to True if you want to resolve the LP;
                    if set to False, the program will return the cached optimal
                    value if already solved
            t0: Override the t0 set during initialization (default 1)
            x0: Override random initialization of x
            track_loss: Override global setting to track loss during solve
        """
        if not resolve and self.optimal:
            return self.value

        # set initializations based on kwargs passed to function or defaults set when creating LinearSolver object
        t = kwargs.get("t0", self.t0)

        max_outer_iters = kwargs.get("max_outer_iters", self.max_outer_iters)
        self.track_loss = kwargs.get("track_loss", self.track_loss)

        # could add additional ability to override settings set during initialization of LinearSolver object
        # max_inner_iters = kwargs.get("max_inner_iters", self.max_inner_iters)
        # eps = kwargs.get("epsilon", self.epsilon)
        # inner_eps = kwargs.get("inner_epsilon", self.inner_epsilon)

        if "x0" in kwargs:
            x = kwargs["x0"]
            self.__check_x0(x)
            update_x = True
        else:
            x = self.x
            update_x = False
        if self.C is not None and self.phase1_solver.phase1_fm.s >= 1:
            if not self.suppress_print:
                print("running phase 1 solver")
            if update_x:
                x, s = self.phase1_solver.solve(x0=x)
            else:
                x, s = self.phase1_solver.solve()
            if s > -self.phase1_tol:
                # This mean infeasibility
                # TODO: Come up with what we do then
                raise ValueError(
                    "Phase 1 Solver did not successfully find a feasible point!"
                )
            if not self.suppress_print:
                print(f"found a feasible point with slack {s}")
        if not self.suppress_print:
            print("proceeding to solve method")

        # don't update self.x so that we don't make resolving easier
        # self.x = x

        self.outer_iters = 0

        objective_vals = []
        self.inner_iters = []

        # make sure everyone is on the same page at our starting point
        self.fm.update_x(x)
        self.fm.update_t(t)

        if self.equality_constrained:
            # intialize the dual variable
            if self.use_gpu:
                v = cp.zeros(self.A.shape[0])
            else:
                v = np.zeros(self.A.shape[0])
        else:
            v = None

        dual_gap = self.num_constraints
        best_x = x.copy()
        best_obj = np.inf

        for iter in range(max_outer_iters):

            x, v, numiters_t, _, success_flag = self.ns.solve(x, t, v0=v)

            self.outer_iters += 1
            self.inner_iters.append(numiters_t)

            if (
                self.A is None
                or (
                    self.use_gpu
                    and cp.linalg.norm(cp.matmul(self.A, x) - self.b) < 1e-3
                )
                or (np.linalg.norm(np.matmul(self.A, x) - self.b) < 1e-3)
            ):

                obj_val = self.fm.objective()
                if not self.suppress_print:
                    print(f"Objective value is now {obj_val}")
                if self.track_loss:
                    objective_vals.append(obj_val)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = x.copy()
                elif success_flag:
                    # if the last step ran until convergence and the objective still increased, we can return
                    # if success_flag is False, that means that the solver quit for some reason (maybe backtracking search got stuck, maybe something else)
                    break

            else:
                if not self.suppress_print:
                    print(f"Newton step at iteration {iter+1} did not converge")

            if not self.suppress_print and numiters_t >= self.max_inner_iters:
                print(
                    f"Reached max Newton steps during {iter+1}th centering step (t={t})"
                )

            dual_gap = self.num_constraints / t
            # alg_progress = np.hstack([alg_progress, np.array([num_iters_t, dual_gap]).reshape(2,1)])

            # quit if n/t < epsilon
            if dual_gap < self.epsilon:
                break

            # increment t for next outer iteration
            t = min(t * self.mu, (self.n + 1.0) / self.epsilon)
            self.fm.update_t(t)

        self.xstar = best_x
        if self.get_dual_variables:
            if self.C is not None or self.bounded:
                self.fm.update_x(best_x)
                self.lam_star = 1 / (t * self.fm.slacks)
            if self.A is not None:
                self.v_star = v / t

        self.optimal = True
        self.value = best_obj
        self.optimality_gap = dual_gap
        self.objective_vals = objective_vals

        return self.value

    def __check_x0(self, x):
        """Helper function to ensure initial x is in the domain of the problem

        We need a strictly feasible starting point"""

        if self.lb is not None and (x <= self.lb).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries greater than lower bound)"
            )
        elif self.ub is not None and (x >= self.ub).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries less than upper bound)"
            )

        if self.c is not None:
            if len(self.c) != len(x):
                raise ValueError("Initial x must be the same dimension as c!")

        if self.C is not None:
            # this error checking is solved by phase 1 solver
            # if (self.C @ x >= self.d).any():
            #    raise ValueError("Initial x must be in domain of problem (Cx <= d)")
            if self.C.shape[1] != len(x):
                raise ValueError("Initial x must have the same number of columns as C!")

        if self.A is not None:
            if self.A.shape[1] != len(x):
                raise ValueError("Initial x must have the same number of columns as A!")

    def plot(self, subtract_cvxpy=True):
        if not (self.optimal and self.track_loss):
            raise ValueError(
                "Need to solve problem with track_loss set to True to be able to plot convergence!"
            )

        if self.use_gpu:
            obj_vals = [val.get() for val in self.objective_vals]
        else:
            obj_vals = self.objective_vals

        ax = plt.subplot()
        ax.step(
            np.cumsum(self.inner_iters),
            obj_vals - self.cvxpy_val,
            where="post",
        )
        ax.set_xlabel("Cumulative Newton iterations")
        ax.set_ylabel("Optimality gap")
        ax.set_title("Convergence of LPSolver")
        ax.set_yscale("log")
        return ax
