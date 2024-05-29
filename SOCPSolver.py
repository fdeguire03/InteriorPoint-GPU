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


class SOCPSolver:

    def __init__(
        self,
        P=None,
        q=None,
        A=None,
        b=None,
        c=None,
        d=None,
        F=None,
        g=None,
        lower_bound=0,
        upper_bound=None,
        t0=0.1,
        phase1_t0=0.01,
        max_outer_iters=50,
        max_inner_iters=20,
        phase1_max_inner_iters=500,
        epsilon=1e-8,
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
        use_psd_condition=False,
        x0=None,
    ):
        """Initialize LP problem of form:
        Minimize 1/2 x^T P x + q^T x
        Subject to ||A_i x + b_i|| <= c_i^T x + d_i
                   Fx == g

        with f in R^n, A in R^(mxn), b in R^m, c in R^n, d in R, F in R^(pxn), g in R^p, and x in R^n

        Provide multiple instances of A, b, c, and d through a list

        The remaining parameters:
            sign (default 1): Used to set the sign of x (sign=1 to make x positive, sign=-1 to make x negative, sign=0 if unconstrained on sign)

            t0 (default 1): Starting point for interior point method

            max_outer_iters (default 50): Maximum number of iterations to run for interior point solver

            max_inner_iters (default 20): Maximum number of iterations of Newton's method to run at each step

            epsilon (default 1e-8): Desired optimality gap for entire problem

            inner_epsilon (default 1e-5): Minimum optimality gap for each Newton solve

            check_cvxpy (default True): Performs an initial check on solving the problem using CVXPY to be able to compare optimal value and determine feasibility.
                        May take a long time if problem is large

            linear_solve_method (default 'np_solve): Method to solve linear equations during Newton solves (options include 'np_solve' (call np.linalg.solve),
                'cholesky' (Solve using Cholesky-factorization and forward-backward substitution), 'cg' (conjugate gradient),
                'direct' (form matrix inverse), and 'np_lstsq' (call np.linalg.lstsq))

            max_cg_iters (default 50): maximum conjugate iterations to perform (used when linear solve method == "cg")

            alpha (default 0.2): parameter for backtrackign search

            beta (default 0.6): parameter for backtracking search

            mu (default 15): parameter for how much to increase t on each centering step

            suppress_print (default True): set to True if you want to suppress any potential warnings from being printed during solve method

            try_diag (default True): set to True if you want to try calcuating all Hessian matrices as diagonal matrices. Potential speedup if Hessian is diagonal,
                                    really no cost if its not, so recommended to keep as True
        """

        self.use_gpu = use_gpu and gpu_flag

        # all attributes for LP
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.c = c
        self.b = b
        self.d = d
        self.F = F
        self.g = g
        self.lb = lower_bound
        self.ub = upper_bound

        self.__check_inputs()

        self.equality_constrained = self.F is not None
        self.inequality_constrained = self.A is not None
        if not self.inequality_constrained:
            raise ValueError(
                "No cone contraints detected. Run with LPSolver or QPSolver for better performance."
            )

        # initialize x
        # TODO: when using inequality constraints, will need to find a way to get a feasible point
        # (using Phase I methods?)
        if self.q is not None:
            self.n = len(self.q)
        elif self.P is not None:
            self.n = self.P.shape[1]
        elif self.A is not None:
            self.n = self.A[0].shape[1]
        elif self.F is not None:
            self.n = self.F.shape[1]
        elif self.c is not None:
            self.n = len(self.c[0])

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
            self.feasible, self.cvxpy_val = None, None

        # transfer to GPU if specified to use GPU (and we are able to given available hardware)
        # TODO: Be able to break into smaller subproblems if not everything can fit onto GPU at the same time

        if self.use_gpu:
            self.x = cp.array(self.x)
            if self.A is not None:
                self.A = cp.array(self.A)
            if self.P is not None:
                self.P = cp.array(self.P)
            if self.q is not None:
                self.q = cp.array(self.q)
            if self.F is not None:
                self.F = cp.array(self.F)
            if self.b is not None:
                self.b = cp.array(self.b)
            if self.d is not None:
                self.d = cp.array(self.d)
            if self.c is not None:
                self.c = cp.array(self.c)
            if self.g is not None:
                self.g = cp.array(self.g)
            if self.lb is not None:
                self.lb = cp.array(self.lb)
            if self.ub is not None:
                self.ub = cp.array(self.ub)

        # Count the number of inequality constraints (used to determine optimality gap during solve)
        self.num_constraints = 0
        if self.A is not None:
            self.num_constraints += len(self.A)
        if self.lb is not None:
            self.num_constraints += self.n
        if self.ub is not None:
            self.num_constraints += self.n

        # backtracking search parameters
        self.alpha = alpha
        self.beta = beta

        # initializations for interior point method
        self.t0 = t0
        self.phase1_t0 = phase1_t0
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
        self.track_loss = track_loss
        self.linear_solve_method = linear_solve_method
        self.get_dual_variables = get_dual_variables
        self.use_psd_condition = use_psd_condition
        self.phase1_tol = phase1_tol
        self.phase1_max_inner_iters = phase1_max_inner_iters

        # initialize the newton solver for this problem
        if self.A is not None:
            self.phase1_solver = self.__make_phase1_solver(phase1_tol)
        self.fm = self.__get_function_manager()
        self.ns = self.__get_newton_solver(linear_solve_method)

    def __check_inputs(self):
        """Make sure that inputs have dimensions in agreement
        If A or b is specified, then both must be specified
        If C or d is specified then both must be specified
        """

        P_flag = self.P is not None
        if P_flag:
            if self.P.ndim != 2:
                raise ValueError("P must be 2-dimensional!")
            if self.P.shape[0] != self.P.shape[1]:
                raise ValueError("P must be a symmetric, square PSD matrix!")
        q_flag = self.q is not None
        if q_flag:
            if self.q.ndim != 1:
                raise ValueError("q must be q-dimensional!")
            if P_flag and self.P.shape[1] != len(self.q):
                raise ValueError("P and q must have the same dimension")

        A_flag = self.A is not None
        if A_flag:
            if not isinstance(self.A, list):
                self.A = [self.A]
            for i, A in enumerate(self.A):
                if A.ndim > 2:
                    raise ValueError("A must be 1- or 2-dimensional!")

                if A.ndim == 2:
                    m, n_A = A.shape

                    # check for diagonality to see if we can compress (will lead to performance gains in solve function)
                    diag_elem = np.diag(A).copy()
                    np.fill_diagonal(A, 0)
                    is_diag = (A == 0).all()
                    if is_diag:
                        self.A[i] = diag_elem
                    else:
                        np.fill_diagonal(a, diag_elem)

                else:
                    n_A = A.shape[0]
                    m = n_A
                if q_flag:
                    if len(self.q) != n_A:
                        raise ValueError(
                            "q must have the same number of entries as A has columns!"
                        )
            b_flag = self.b is not None
            if b_flag:
                if not isinstance(self.b, list):
                    self.b = [self.b]

                for b in self.b:
                    if b.ndim != 1:
                        raise ValueError("b must be 1-dimensional!")
                    if len(b) != m and not np.isscalar(b):
                        raise ValueError("A and b must have agreeing dimensions!")

                if len(self.b) == 1:
                    self.b = self.b * len(self.A)
                if len(self.A) != len(self.b):
                    raise ValueError("Must provide an equal number of A and b")

        F_flag = self.F is not None
        if F_flag:
            if self.F.ndim != 2:
                raise ValueError("F must be 2-dimensional!")
            m_F, n_F = self.F.shape
            if q_flag:
                if len(self.q) != n_F:
                    raise ValueError(
                        "q must have the same number of entries as F has columns!"
                    )
        g_flag = self.g is not None
        if g_flag:
            if self.g.ndim != 1:
                raise ValueError("g must be 1-dimensional!")
            if F_flag:
                if len(self.g) != m_F:
                    raise ValueError("F and g must have agreeing dimensions!")

        if F_flag and A_flag:
            if n_F != n_A:
                raise ValueError("A and F must have the same number of columns!")

        c_flag = self.c is not None
        if c_flag:
            if not isinstance(self.c, list):
                self.c = [self.c]
            for c in self.c:
                if c.ndim != 1:
                    raise ValueError("c must be 1-dimensional!")
                if q_flag:
                    if len(c) != len(self.q):
                        raise ValueError(
                            "q and c must have the same number of entries!"
                        )
                if F_flag:
                    if len(self.c[0]) != n_F:
                        raise ValueError(
                            "c must have the same number of entries as F has columns!"
                        )
                if A_flag:
                    if len(self.c[0]) != n_A:
                        raise ValueError(
                            "c must have the same number of entries as A has columns!"
                        )
        d_flag = self.d is not None
        if d_flag:
            if not isinstance(self.d, list):
                self.d = [self.d]
            for d in self.d:
                if not np.isscalar(d):
                    raise ValueError("d must be a scalar!")

        if d_flag and c_flag:
            if len(self.d) != len(self.c):
                raise ValueError("Must provide equal number of c and d")

        if d_flag and A_flag:
            if len(self.d) == 1:
                self.d = self.d * len(self.A)
            if len(self.d) != len(self.A):
                raise ValueError("Must provide equal number of A and d")

        if A_flag and c_flag:
            if len(self.A) != len(self.c):
                raise ValueError("Must provide equal number of c and A")

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
                    if F_flag:
                        assert len(self.lb) == n_F
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
                    if F_flag:
                        assert len(self.ub) == n_F
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

    def __make_phase1_solver(self, tol):

        phase1_solver = PhaseOneSolver(
            socp=True,
            socp_params=(self.A, self.b, self.c, self.d),
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
            use_psd_condition=self.use_psd_condition,
            t0=self.phase1_t0,
        )

        return phase1_solver

    def __get_function_manager(self):
        """Generate functions to use in solve method
        Be aware of where values were not passed for A,C,b,c,d matrices and vectors
        so that functions are as efficient as possible
        """

        # gradient and hessian of the objective

        fm = FunctionManagerSOCP(
            P=self.P,
            q=self.q,
            A=self.A,
            b=self.b,
            c=self.c,
            d=self.d,
            F=self.F,
            g=self.g,
            x0=self.x,
            # lower_bound=self.lb,
            # upper_bound=self.ub,
            t=1,
            use_gpu=self.use_gpu,
            n=self.n,
        )

        return fm

    def __get_newton_solver(self, linear_solve_method):
        """Initialize a NewtonSolver object with the specified linear solve method

        Pass all specified parameters from the user
        """

        if linear_solve_method == "np_lstsq":
            if self.equality_constrained:
                NewtonClass = NewtonSolverNPLstSqInfeasibleStart
            else:
                NewtonClass = NewtonSolverNPLstSq

        elif linear_solve_method == "np_solve":
            if self.equality_constrained:
                NewtonClass = NewtonSolverNPSolveInfeasibleStart
            else:
                NewtonClass = NewtonSolverNPSolve

        elif linear_solve_method == "direct":

            if self.equality_constrained:
                NewtonClass = NewtonSolverDirectInfeasibleStart
            else:
                NewtonClass = NewtonSolverDirect

        elif linear_solve_method == "cg":

            if self.equality_constrained:
                NewtonClass = NewtonSolverCGInfeasibleStart
            else:
                NewtonClass = NewtonSolverCG

        elif linear_solve_method == "kkt":

            if self.equality_constrained:
                NewtonClass = NewtonSolverKKTNPSolveInfeasibleStart
            else:
                raise ValueError(
                    "No KKT System non-equality-constrained problems! Please choose another solver"
                )

        elif linear_solve_method == "cholesky":

            if self.equality_constrained:
                NewtonClass = NewtonSolverCholeskyInfeasibleStart
            else:
                NewtonClass = NewtonSolverCholesky
        else:
            raise ValueError("Please enter a valid linear solve method!")

        ns = NewtonClass(
            self.F,
            self.g,
            C=None,
            d=None,
            function_manager=self.fm,
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
            use_psd_condition=self.use_psd_condition,
        )

        return ns

    def __test_feasibility(self):
        """Use CVXPY to check whether the problem is feasible

        This function is included because determining feasibility of the problem is not the main
        goal of this project (our goal is to solve feasible problems faster)

        This can be suppressed with check_cvxpy = False. We may remove later if we implement better
        feasibility checking"""

        x = cvx.Variable(len(self.x))

        obj = 0
        if self.P is not None:
            obj += 0.5 * cvx.quad_form(x, self.P)
        if self.q is not None:
            obj += self.q.T @ x
        obj = cvx.Minimize(obj)

        constr = []
        if self.inequality_constrained:

            for i, A in enumerate(self.A):
                if A.ndim > 1:
                    norm = A @ x
                else:
                    norm = cvx.multiply(A, x)
                if self.b is not None:
                    norm += self.b[i]
                norm = cvx.norm2(norm)
                rhs = 0
                if self.c is not None:
                    rhs += self.c[i].T @ x
                if self.d is not None:
                    rhs += self.d[i]

                constr.append(norm <= rhs)
        if self.F is not None:
            constr.append(self.F @ x == self.g)

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
        return f"SOCPSolver(Optimal Value: {opt_val})"

    def __repr__(self):
        return str(self)

    def solve(self, resolve=True, **kwargs):
        """Solve the SOCP

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
        if self.A is not None and self.phase1_solver.phase1_fm.s >= 1:
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

        if self.equality_constrained:
            # intialize the dual variable
            if self.use_gpu:
                v = cp.zeros(self.F.shape[0])
            else:
                v = np.zeros(self.F.shape[0])
        else:
            v = None

        dual_gap = self.num_constraints
        best_x = x.copy()
        best_obj = np.inf

        # make sure everyone is on the same page at our starting point
        self.fm.update_x(x)
        self.fm.update_t(t)

        for iter in range(max_outer_iters):

            x, v, numiters_t, _ = self.ns.solve(x, t, v0=v)

            self.outer_iters += 1
            self.inner_iters.append(numiters_t)

            if (
                self.F is None
                or (
                    self.use_gpu
                    and cp.linalg.norm(cp.matmul(self.F, x) - self.g) < 1e-3
                )
                or (np.linalg.norm(np.matmul(self.F, x) - self.g) < 1e-3)
            ):
                obj_val = self.fm.objective()
                if not self.suppress_print:
                    print(f"Objective value is now {obj_val}")
                if self.track_loss:
                    objective_vals.append(obj_val)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = x.copy()
                else:
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

        if self.q is not None:
            if len(self.q) != len(x):
                raise ValueError("Initial x must be the same dimension as q!")
        if self.P is not None:
            if len(self.P) != len(x):
                raise ValueError("Initial x must be the same dimension as P!")

        if self.A is not None:
            for A in self.A:
                if (
                    A.ndim < 2
                    and len(x) != len(A)
                    or A.ndim == 2
                    and A.shape[1] != len(x)
                ):
                    raise ValueError(
                        "Initial x must have the same number of columns as A!"
                    )

        if self.F is not None:
            if self.F.shape[1] != len(x):
                raise ValueError("Initial x must have the same number of columns as F!")

        if self.inequality_constrained:
            for i, A in enumerate(self.A):
                norm = A @ self.x
                if self.b is not None:
                    norm += self.b[i]
                if self.use_gpu:
                    norm = cp.linalg.norm(norm)
                else:
                    norm = np.linalg.norm(norm)
                rhs = 0
                if self.c is not None:
                    rhs += self.c[i].T @ self.x
                if self.d is not None:
                    rhs += self.d[i]
                if not (norm <= rhs):

                    raise ValueError(
                        f"Initial x must satisfy the cone constraint ||Ax + b ||_2 <= c^T x + d (failed constraint {i+1})"
                    )

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
        ax.set_title("Convergence of SOCPSolver")
        ax.set_yscale("log")
        return ax
