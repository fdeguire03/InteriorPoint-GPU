import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from NewtonSolver import (
    NewtonSolverCG,
    NewtonSolverDirect,
    NewtonSolverNPLstSq,
    NewtonSolverNPSolve,
    NewtonSolverKKTNPSolve,
    NewtonSolverCholesky,
)

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
        num_variables=10,
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
    ):
        """Initialize LP problem of form:
        Minimize c^T x
        Subject to Ax <= b
                   Cx == d
                   x >= 0

        The remaining parameters:
            sign (default 1): Used to set the sign of x (sign=1 to make x positive, sign=-1 to make x negative, sign=0 if unconstrained on sign)

            num_variables (default 10): Length of vector x to solve for (Used only if no parameter is passed to c)

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
        """

        if C is not None or d is not None:
            raise NotImplementedError(
                "Inequality constraints beyond positive/negative x are not yet implemented!"
            )

        # all attributes for LP
        self.A = A
        self.c = c
        self.C = C
        self.b = b
        self.d = d
        self.sign = sign

        self.__check_inputs()

        # initialize x
        # TODO: when using inequality constraints, will need to find a way to get a feasible point
        # (using Phase I methods?)
        if self.c is None:
            self.n = num_variables
        else:
            self.n = len(self.c)
        self.x = np.random.rand(self.n)
        if self.sign:
            self.x *= self.sign

        # If specified, make sure that the problem is feasible usign CVXPY
        # the CVXPY solution can also be used later to verify the solution of the LinearSolver
        if check_cvxpy:
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
        self.inner_iters = 0
        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.epsilon = epsilon
        self.inner_epsilon = inner_epsilon

        # other housekeeping
        # helper fields to denote when problem has been solved
        self.optimal = False
        self.value = None
        self.optimality_gap = None
        self.xstar = None
        self.lam_star = None
        self.vstar = None
        self.suppress_print = suppress_print

        # initialize the newton solver for this problem
        self.__gen_functions()
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

    def __gen_functions(self):
        """Generate functions to use in solve method
        Be aware of where values were not passed for A,C,b,c,d matrices and vectors
        so that functions are as efficient as possible

        After adding inequality constraints, may need to add a flag for when hessian is diagonal
        or not to help newton solver be more efficient"""

        # gradient and hessian of the objective
        if self.c is None:
            self.obj = lambda x: x.sum()
            if self.use_gpu:
                ones_vector = cp.ones(len(self.x))
            else:
                ones_vector = np.ones(len(self.x))
            obj_grad = ones_vector
        else:
            if self.use_gpu:
                self.obj = lambda x: cp.dot(self.c, x)
            else:
                self.obj = lambda x: np.dot(self.c, x)
            obj_grad = self.c

        """I WROTE FUNCTIONS FOR GRADIENT AND HESSIAN OF INEQUALITY CONSTRAINTS Cx<=d BUT
        THESE FUNCTIONS ARE MOST LIKELY SLIGHTLY WRONG, WILL NEED TO REWRITE TO FUNCTION CORRECTLY!"""
        if self.C is not None or self.d is not None:
            raise NotImplementedError(
                "Inequality constraints beyond positive/negative x are not yet implemented (Need to rewrite gradient and hessian functions)!"
            )

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

        if self.d is None and self.C is None:
            inequality = False
        else:
            if self.d is None:
                if self.use_gpu:
                    inequality_log_barrier = lambda x: -cp.log(-self.C @ x).sum()
                else:
                    inequality_log_barrier = lambda x: -np.log(-self.C @ x).sum()
                C_sum = self.C.sum(axis=0)
                inequality_obj_grad = lambda x: C_sum / (-self.C @ x)
            elif self.C is None:
                if self.use_gpu:
                    inequality_log_barrier = lambda x: -cp.log(self.d - x).sum()
                else:
                    inequality_log_barrier = lambda x: -np.log(self.d - x).sum()

                inequality_obj_grad = lambda x: 1 / (self.d - x)
            else:
                if self.use_gpu:
                    inequality_log_barrier = lambda x: -cp.log(
                        self.d - self.C @ x
                    ).sum()
                else:
                    inequality_log_barrier = lambda x: -np.log(
                        self.d - self.C @ x
                    ).sum()

                C_sum = self.C.sum(axis=0)
                inequality_obj_grad = lambda x: C_sum / (self.d - self.C @ x)

        if not (inequality or signed):
            barrier = False
        else:
            barrier = True
            if inequality and signed:
                log_barrier = lambda x: signed_log_barrier(x) + inequality_log_barrier(
                    x
                )
                barrier_grad = lambda x: signed_obj_grad(x) + inequality_obj_grad(x)
            elif inequality:
                log_barrier = lambda x: inequality_log_barrier(x)
                barrier_grad = lambda x: inequality_obj_grad(x)
            elif signed:
                log_barrier = lambda x: signed_log_barrier(x)
                barrier_grad = lambda x: signed_obj_grad(x)

        # define the objective, gradient, and hessian functions for the newton solver based on above
        if barrier:
            self.obj_newton = lambda x, t: t * self.obj(x) + log_barrier(x)
            self.grad_newton = lambda x, t: t * obj_grad + barrier_grad(x)
            self.hessian_newton = lambda x: signed_hessian(x)
            self.inv_hessian_newton = lambda x: signed_hessian_inverse(x)
        else:
            self.obj_newton = lambda x, t: t * self.obj(x)
            self.grad_newton = lambda x, t: t * obj_grad
            self.hessian_newton = lambda x: 0
            self.inv_hessian_newton = lambda x: 0

    def __get_newton_solver(self, linear_solve_method):
        """Initialize a NewtonSolver object with the specified linear solve method

        Pass all specified parameters from the user
        """

        if linear_solve_method == "np_lstsq":
            ns = NewtonSolverNPLstSq(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
                max_iters=self.max_inner_iters,
                epsilon=self.inner_epsilon,
                suppress_print=self.suppress_print,
                sign=self.sign,
                alpha=self.alpha,
                beta=self.beta,
                mu=self.mu,
                use_gpu=self.use_gpu,
            )
        elif linear_solve_method == "np_solve":
            ns = NewtonSolverNPSolve(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
                max_iters=self.max_inner_iters,
                epsilon=self.inner_epsilon,
                suppress_print=self.suppress_print,
                sign=self.sign,
                alpha=self.alpha,
                beta=self.beta,
                mu=self.mu,
                use_gpu=self.use_gpu,
            )
        elif linear_solve_method == "direct":
            ns = NewtonSolverDirect(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
                max_iters=self.max_inner_iters,
                epsilon=self.inner_epsilon,
                suppress_print=self.suppress_print,
                sign=self.sign,
                alpha=self.alpha,
                beta=self.beta,
                mu=self.mu,
                use_gpu=self.use_gpu,
            )
        elif linear_solve_method == "cg":
            ns = NewtonSolverCG(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
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
        elif linear_solve_method == "kkt_system":
            ns = NewtonSolverKKTNPSolve(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
                max_iters=self.max_inner_iters,
                epsilon=self.inner_epsilon,
                suppress_print=self.suppress_print,
                sign=self.sign,
                alpha=self.alpha,
                beta=self.beta,
                mu=self.mu,
                use_gpu=self.use_gpu,
            )
        elif linear_solve_method == "cholesky":
            ns = NewtonSolverCholesky(
                self.A,
                self.b,
                self.C,
                self.d,
                self.obj_newton,
                self.grad_newton,
                self.hessian_newton,
                self.inv_hessian_newton,
                max_iters=self.max_inner_iters,
                epsilon=self.inner_epsilon,
                suppress_print=self.suppress_print,
                sign=self.sign,
                alpha=self.alpha,
                beta=self.beta,
                mu=self.mu,
                use_gpu=self.use_gpu,
            )
        else:
            raise ValueError("Please enter a valid linear solve method!")

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
            obj = cvx.Minimize(x)

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
        except Exception:
            pass
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
        """
        if not resolve and self.optimal:
            return self.value

        # set initializations based on kwargs passed to function or defaults set when creating LinearSolver object
        t = kwargs.get("t0", self.t0)
        max_outer_iters = kwargs.get("max_outer_iters", self.max_outer_iters)

        # could add additional ability to override settings set during initialization of LinearSolver object
        # max_inner_iters = kwargs.get("max_inner_iters", self.max_inner_iters)
        # eps = kwargs.get("epsilon", self.epsilon)
        # inner_eps = kwargs.get("inner_epsilon", self.inner_epsilon)

        x = kwargs.get("x0", self.x)
        self.__check_x0(x)

        # intialize the dual variable
        if self.use_gpu:
            v = cp.zeros(self.A.shape[0])
        else:
            v = np.zeros(self.A.shape[0])

        dual_gap = self.num_constraints

        for iter in range(max_outer_iters):

            x, v, numiters_t = self.ns.solve(x, t, v0=v)

            self.outer_iters += 1
            self.inner_iters += numiters_t

            if not self.suppress_print and numiters_t <= self.max_inner_iters:
                print(
                    f"Reached max Newton steps during {iter}th centering step (t={t})"
                )

            dual_gap /= t
            # alg_progress = np.hstack([alg_progress, np.array([num_iters_t, dual_gap]).reshape(2,1)])

            # quit if n/t < epsilon
            if dual_gap < self.epsilon:
                self.xstar = x
                if self.C is not None:
                    if self.sign != 0:
                        if self.use_gpu:
                            self.lam_star = cp.append(
                                1 / (t * (x)), 1 / (t * (self.d - self.C @ x))
                            )
                        else:
                            self.lam_star = np.append(
                                1 / (t * (x)), 1 / (t * (self.d - self.C @ x))
                            )
                    else:
                        self.lam_star = 1 / (t * (self.d - self.C @ x))
                elif self.sign != 0:
                    self.lam_star = 1 / (t * (x))
                if self.A is not None:
                    self.v_star = v / t

                self.optimal = True
                self.value = self.obj(self.xstar)
                self.optimality_gap = dual_gap

                return self.value

            # increment t for next outer iteration
            t *= self.mu

    def __check_x0(self, x):
        """Helper function to ensure initial x is in the domain of the problem"""

        if self.sign > 0 and (x < 0).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries positive)"
            )
        elif self.sign < 0 and (x > 0).any():
            raise ValueError(
                "Initial x must be in domain of problem (all entries negative)"
            )

        if self.C is not None:
            if (self.C @ x > self.d).any():
                raise ValueError("Initial x must be in domain of problem (Cx <= d)")
