import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from NewtonSolver import *
from NewtonSolverInfeasibleStart import *
from PhaseOne import phase_one
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
        sign=1,
        t0=1,
        max_outer_iters=50,
        max_inner_iters=20,
        epsilon=1e-8,
        inner_epsilon=1e-5,
        check_cvxpy=True,
        linear_solve_method="np_solve",
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=15,
        suppress_print=False,
        use_gpu=False,
        try_diag=True,
        track_loss=False,
    ):
        """Initialize LP problem of form:
        Minimize c^T x
        Subject to Ax <= b
                   Cx == d
                   x >= 0

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

        # all attributes for LP
        self.A = A
        self.c = c
        self.C = C
        self.b = b
        self.d = d
        self.sign = sign

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
        self.x = np.random.rand(self.n)
        if self.sign:
            self.x *= self.sign

        # If specified, make sure that the problem is feasible usign CVXPY
        # the CVXPY solution can also be used later to verify the solution of the LinearSolver
        if check_cvxpy:
            print("Is testing CVXPY")
            self.feasible, self.cvxpy_val = self.__test_feasibility()
            if self.feasible == "infeasible":
                raise ValueError("Provided problem instance is infeasible!")
            elif self.feasible == "unbounded":
                raise ValueError("Provided problem instance is unbounded!")
        else:
            self.feasible, self.cvxpy_val = None, None

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

        # Count the number of inequality constraints (used to determine optimality gap during solve)
        self.num_constraints = 0
        if self.d is not None:
            self.num_constraints += len(self.d)
        if self.sign != 0:
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

        # initialize the newton solver for this problem
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

        if C_flag and A_flag:
            if n_C != n_A:
                raise ValueError("A and C must have the same number of columns!")

    def __get_function_manager(self):
        """Generate functions to use in solve method
        Be aware of where values were not passed for A,C,b,c,d matrices and vectors
        so that functions are as efficient as possible
        """

        # gradient and hessian of the objective
        if self.C is None and self.sign == 0:
            FunctionManagerClass = FunctionManagerUnconstrained
        elif self.C is None:
            FunctionManagerClass = FunctionManagerSigned
        elif self.sign == 0:
            FunctionManagerClass = FunctionManagerInequalityConstrained
        else:
            FunctionManagerClass = FunctionManagerConstrained

        fm = FunctionManagerClass(
            c=self.c,
            A=self.A,
            b=self.b,
            C=self.C,
            d=self.d,
            x0=self.x,
            sign=self.sign,
            t=1,
            use_gpu=self.use_gpu,
            n=self.n,
        )

        return fm

        # gradient and objective of the inequality constraints
        if self.sign == 0:
            signed = False
        else:
            signed = True
            signed_obj_grad = lambda x: -1 / x
            signed_hessian = lambda x: 1 / x**2
            signed_hessian_inverse = lambda x: x**2
            if self.sign > 0:
                if self.use_gpu:
                    signed_log_barrier = lambda x: -cp.log(x).sum()
                else:
                    signed_log_barrier = lambda x: -np.log(x).sum()
            else:
                if self.use_gpu:
                    signed_log_barrier = lambda x: -cp.log(-x).sum()
                else:
                    signed_log_barrier = lambda x: -np.log(-x).sum()

        if self.d is None:
            inequality = False
        else:
            inequality = True
            if self.use_gpu:
                inequality_log_barrier = lambda x: -cp.log(self.d - self.C @ x).sum()
            else:
                inequality_log_barrier = lambda x: -np.log(self.d - self.C @ x).sum()

            inequality_grad = lambda x: self.C.T @ (1 / (self.d - self.C @ x))
            inequality_hessian = lambda x: self.C.T @ (
                (1 / (self.d - self.C @ x) ** 2)[:, None] * self.C
            )

        if not (inequality or signed):
            barrier = False
        else:
            barrier = True
            if inequality and signed:
                log_barrier = lambda x: signed_log_barrier(x) + inequality_log_barrier(
                    x
                )
                barrier_grad = lambda x: signed_obj_grad(x) + inequality_grad(x)

                def barrier_hessian(x):
                    # get (the non-diagonal hessian of the inequality constraint),
                    # extract the diagonal, and add the diagonal elements of the signed
                    # hessian. Faster than changing the signed hessian to a full matrix
                    # and adding it (sqrt(n) fewer operations!)
                    # return False boolean to inform NewtonSolver that hessian is not diagonal
                    a = inequality_hessian(x)
                    d = np.einsum("ii->i", a)
                    d += signed_hessian(x)
                    return a

            elif inequality:
                log_barrier = inequality_log_barrier
                barrier_grad = inequality_grad
                barrier_hessian = inequality_hessian  # lambda x : inequality_hessian(x), False  # return False boolean to inform NewtonSolver that hessian is not diagonal
            elif signed:
                log_barrier = signed_log_barrier
                barrier_grad = signed_obj_grad
                barrier_hessian = signed_hessian  # lambda x : signed_hessian(x), True  # return True boolean to inform NewtonSolver that hessian is diagonal

        # define the objective, gradient, and hessian functions for the newton solver based on above
        if barrier:
            self.obj_newton = lambda x, t: t * self.obj(x) + log_barrier(x)
            self.grad_newton = lambda x, t: t * obj_grad + barrier_grad(x)
            self.hessian_newton = barrier_hessian
            if signed:
                self.inv_hessian_newton = lambda x: signed_hessian_inverse(x)
            else:

                def raiseException(x):
                    raise ValueError(
                        "Hessian is not diagonal, cannot use inv hessian function!"
                    )

                self.inv_hessian_newton = raiseException
        else:
            self.obj_newton = lambda x, t: t * self.obj(x)
            self.grad_newton = lambda x, t: t * obj_grad
            self.hessian_newton = 0
            self.inv_hessian_newton = 0

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
            sign=self.sign,
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            use_gpu=self.use_gpu,
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

        if self.sign:
            constr.append(self.sign * x >= 0)

        prob = cvx.Problem(obj, constr)
        try:
            prob.solve(solver="CLARABEL")
        except Exception as e:
            print(e)
            
        return prob.status, prob.value

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

        x = kwargs.get("x0", self.x)

        # Have an intial x
        if self.A is not None and self.b is not None:
          if (self.A @ x <= self.b).all():
            # Good point
            pass
          else:
            # Need to find a feasible point
            # Create needed matrices if sign is not 0
            if self.sign != 0:
              identity = np.eye(self.n)
              zeros_vec = np.zeros(self.n)
              
              G = np.vstack([self.A, - self.sign * identity])
              h = np.hstack([self.b, zeros_vec])
            else:
              G = self.A
              h = self.b
            
            # TODO: implement more solvers in phase on
            # Although this might not be needed since it usually runs very few iterations
            if self.linear_solve_method == 'cg':
              linear_solver = "cg"
            else:
              linear_solver = "solve"
            
            PhaseOne = phase_one(G, h, self.mu, x0 = x, eps = self.epsilon, 
                                max_iter_interior = self.max_outer_iters, 
                                max_iter_newton = self.max_inner_iters,
                                use_cupy = self.use_gpu, linear_solver = 'cg',
                                max_cg_iters = self.max_cg_iters)
            x, s, warn = PhaseOne.solve()
            
            if not warn and s > 0:
              # This mean infeasibility
              # TODO: Come up with what we do then
              pass

            del PhaseOne # Delete object since it is no longer needed

        self.__check_x0(x)

        self.outer_iters = 0

        objective_vals = []
        self.inner_iters = []

        if self.equality_constrained:
            # intialize the dual variable
            if self.use_gpu:
                v = cp.zeros(self.A.shape[0])
            else:
                v = np.zeros(self.A.shape[0])
        else:
            v = None

        dual_gap = self.num_constraints
        best_x = None
        best_obj = np.inf

        for iter in range(max_outer_iters):

            x, v, numiters_t, _ = self.ns.solve(x, t, v0=v)

            obj_val = self.fm.objective(x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = x
            if self.track_loss:
                objective_vals.append(obj_val)

            self.outer_iters += 1
            self.inner_iters.append(numiters_t)

            if not self.suppress_print and numiters_t <= self.max_inner_iters:
                print(
                    f"Reached max Newton steps during {iter}th centering step (t={t})"
                )

            dual_gap = self.n / t
            # alg_progress = np.hstack([alg_progress, np.array([num_iters_t, dual_gap]).reshape(2,1)])

            # quit if n/t < epsilon
            if dual_gap < self.epsilon:
                self.xstar = best_x
                if self.C is not None:
                    if self.sign != 0:
                        if self.use_gpu:
                            self.lam_star = cp.append(
                                1 / (t * (x)), 1 / (t * (self.d - self.C @ best_x))
                            )
                        else:
                            self.lam_star = np.append(
                                1 / (t * (x)), 1 / (t * (self.d - self.C @ best_x))
                            )
                    else:
                        self.lam_star = 1 / (t * (self.d - self.C @ best_x))
                elif self.sign != 0:
                    self.lam_star = 1 / (t * (best_x))
                if self.A is not None:
                    self.v_star = v / t

                self.optimal = True
                self.value = best_obj
                self.optimality_gap = dual_gap
                self.objective_vals = objective_vals

                return self.value

            # increment t for next outer iteration
            t = min(t * self.mu, (self.n + 1.0) / self.epsilon)
            self.fm.update_t(t)

    def __check_x0(self, x):
        """Helper function to ensure initial x is in the domain of the problem

        We need a strictly feasible starting point"""

        if self.sign > 0 and (x <= 0).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries positive)"
            )
        elif self.sign < 0 and (x >= 0).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries negative)"
            )

        if self.c is not None:
            if len(self.c) != len(x):
                raise ValueError("Initial x must be the same dimension as c!")

        if self.C is not None:
            if (self.C @ x >= self.d).any():
                raise ValueError("Initial x must be in domain of problem (Cx <= d)")
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
