import numpy as np
import scipy

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    gpu_flag = True
except Exception:
    gpu_flag = False


class NewtonSolverInfeasibleStart:

    def __init__(
        self,
        A,
        b,
        C,
        d,
        function_manager,
        sign=1,
        max_iters=50,
        epsilon=1e-5,
        suppress_print=True,
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=20,
        use_gpu=False,
        track_loss=False,
    ):
        """Solve convex optimization problem of the following form using infeasible start Newton's method:
        argmin_x  t * obj_fxn(x)

        Uses provided gradient and hessian functions to solve problem

        Currently, assumes that Hessian is diagonal, but this assumption will stop holding as solver gets
        more robust to more types of problems

        This class does not have a linear solve method implemented and will need to be overridden by a child class
        to be implemented on a problem
        """

        # problem specifications
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.sign = sign

        # problem functions for solve method
        self.fm = function_manager

        # other housekeeping
        self.max_iters = max_iters
        self.eps = epsilon
        self.suppress_print = suppress_print
        self.max_cg_iters = max_cg_iters
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.use_gpu = use_gpu and gpu_flag
        self.track_loss = track_loss

    def solve(self, x, t, v0=None):
        """Solve a convex optimization problem using Newton's method, using the provided initial values
        for x, t, and v0 (optional)
        """

        # initialize dual variable
        if v0 is None and self.A is not None:
            if self.use_gpu:
                v = cp.zeros(self.A.shape[0])
            else:
                v = np.zeros(self.A.shape[0])
        else:
            v = v0

        # residuals = (
        #    np.hstack([self.grad(x, t), self.A @ x - self.b])
        #    * np.ones((1, len(x) + self.A.shape[0]))
        # ).T

        # precompute gradient since it will be used in multiple locations
        gradf = self.fm.gradient(x)
        residual_norm = None

        # place everything in a try-except block so we can report if there was an error during solve
        try:

            for iter in range(self.max_iters):

                # invoke linear solve method -- needs to be implemented by a child class
                xstep, vstep = self.newton_linear_solve(x, v, gradf)

                # backtracking line search on norm of residual
                # also captures residual nad gradient calculations from backtracking search
                step_size, gradf, residual_norm = self.backtrack_search(
                    x, v, xstep, vstep, t, gradf, residual_norm
                )

                # update x and nu based on newton solve
                x += step_size * xstep
                v += step_size * vstep

                # check stopping criteria
                # residuals = np.hstack(
                #    [residuals, (residual * np.ones((1, len(x) + self.A.shape[0]))).T]
                # )

                # TODO: Implement more efficient syntax here (can we reuse steps from above that already
                # solve for the intermediate values like Av and Ax?)
                # Also, do we have to calculate the norm, or is there a more efficient calculation we can perform
                # like norm squared
                # gradf = self.grad(x, t)
                # r_dual = gradf + self.A.T @ v
                # r_primal = self.A @ x - self.b
                # if self.use_gpu:
                #    r = cp.append(r_dual, r_primal)
                #    residual_norm2 = cp.linalg.norm(r)
                # else:
                #    r = np.append(r_dual, r_primal)
                #    residual_norm2 = np.linalg.norm(r)

                # return if our equality constraint and problem are solved to satisfactory epsilon
                if residual_norm < self.eps:
                    return x, v, iter + 1, residual_norm

            # if we reach the maximum number of iterations, print warnings to the user unless specified not to
            if self.suppress_print:
                return x, v, iter + 1, residual_norm

            print(
                "REACHED MAX ITERATIONS: Problem likely infeasible or unbounded",
                end="",
            )

            # unbounded below if we have a feasible x
            if (self.A @ x - self.b < self.eps).all():
                if not self.suppress_print:
                    print(" (Likely unbounded)")
                return x, v, iter + 1, residual_norm

            # else we are not feasible
            else:
                if not self.suppress_print:
                    print(" (Likely infeasible)")
                return x, v, iter + 1, residual_norm

        except np.linalg.LinAlgError as e:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, v, iter + 1, residual_norm
        except cp.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, v, iter + 1, residual_norm

    def backtrack_search(self, x, v, xstep, vstep, t, gradf, residual_normp=None):
        """Backtracking search for Newton's method ensures that Newton step
        walks in a descent direction

        First, make sure that the next x is in the domain of the objective function (satisfies all log barriers)
        Then, make sure that we are going in a descent direction"""

        # default to step size of 1 -- can only get smaller
        step_size = 1
        next_x = x + step_size * xstep

        # make sure our next step is in the domain of f
        if self.sign > 0:
            while (next_x <= 0).any():
                step_size *= self.beta
                next_x = x + step_size * xstep
        elif self.sign < 0:
            while (next_x >= 0).any():
                step_size *= self.beta
                next_x = x + step_size * xstep
        if self.C is not None:
            while (self.d - self.C @ next_x <= 0).any():
                step_size *= self.beta
                next_x = x + step_size * xstep

        # capture results of some matrix multiplies so we don't do repeated calculations
        ATv_cache = self.A.T @ v
        ATvstep_cache = self.A.T @ vstep
        Axb_cache = self.A @ x - self.b
        Axstep_cache = self.A @ xstep

        # calculate residuals for current step (only if not provided with residual from last iteration)
        # TODO: Was getting some weird behavior trying to use the cached residual norm, can try to fix if we
        # think it will save time to not have to recalculate the residual norm here
        # if residual_norm is None:
        r_dual = gradf + ATv_cache
        r_primal = Axb_cache
        if self.use_gpu:
            r = cp.append(r_dual, r_primal)
            residual_norm = cp.linalg.norm(r)
        else:
            r = np.append(r_dual, r_primal)
            residual_norm = np.linalg.norm(r)

        # calculate residuals for proposed step
        next_grad = self.fm.gradient(x + step_size * xstep)
        rnext_dual = next_grad + ATv_cache + step_size * ATvstep_cache
        rnext_primal = Axb_cache + step_size * Axstep_cache
        if self.use_gpu:
            residual_step = cp.append(
                rnext_dual,
                rnext_primal,
            )
            next_residual_norm = cp.linalg.norm(residual_step)
        else:
            residual_step = np.append(
                rnext_dual,
                rnext_primal,
            )
            next_residual_norm = np.linalg.norm(residual_step)

        # make sure the residual is descending enough
        while next_residual_norm > (1 - self.alpha * step_size) * residual_norm:
            step_size *= self.beta
            next_grad = self.fm.gradient(x + step_size * xstep)
            rnext_dual = next_grad + ATv_cache + step_size * ATvstep_cache
            rnext_primal = Axb_cache + step_size * Axstep_cache
            if self.use_gpu:
                residual_step = cp.append(
                    rnext_dual,
                    rnext_primal,
                )
                next_residual_norm = cp.linalg.norm(residual_step)
            else:
                residual_step = np.append(
                    rnext_dual,
                    rnext_primal,
                )
                next_residual_norm = np.linalg.norm(residual_step)

        return step_size, next_grad, next_residual_norm

    def newton_linear_solve(self, x, v, gradient):
        raise NotImplementedError("Must be overridden by child class")


class NewtonSolverNPLstSqInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using the least squares method from numpy

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11 = self.fm.hessian(x)
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            A11_inv_AT = cp.linalg.lstsq(A11, self.A.T, rcond=None)[0]
            A11_inv_b1 = cp.linalg.lstsq(A11, b1, rcond=None)[0]
            w = cp.linalg.lstsq(
                self.A @ (A11_inv_AT),
                b2 - self.A @ (A11_inv_b1),
                rcond=None,
            )[0]
            xstep = -cp.linalg.lstsq(A11, (b1 + self.A.T @ w), rcond=None)[0]
        else:
            A11_inv_AT = np.linalg.lstsq(A11, self.A.T, rcond=None)[0]
            A11_inv_b1 = np.linalg.lstsq(A11, b1, rcond=None)[0]
            w = np.linalg.lstsq(
                self.A @ (A11_inv_AT),
                b2 - self.A @ (A11_inv_b1),
                rcond=None,
            )[0]
            xstep = -np.linalg.lstsq(A11, (b1 + self.A.T @ w), rcond=None)[0]
        vstep = w - v
        return xstep, vstep


class NewtonSolverNPSolveInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using the linalg.solve method from numpy

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11 = self.fm.hessian(x)
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            A11_inv_AT = cp.linalg.solve(A11, self.A.T)
            A11_inv_b1 = cp.linalg.solve(A11, b1)
            w = cp.linalg.solve(self.A @ (A11_inv_AT), b2 - self.A @ (A11_inv_b1))
            xstep = -cp.linalg.solve(A11, (b1 + self.A.T @ w))
        else:

            A11_inv_AT = np.linalg.solve(A11, self.A.T)
            A11_inv_b1 = np.linalg.solve(A11, b1)
            w = np.linalg.solve(self.A @ (A11_inv_AT), b2 - self.A @ (A11_inv_b1))
            xstep = -np.linalg.solve(A11, (b1 + self.A.T @ w))
        vstep = w - v

        return xstep, vstep


class NewtonSolverCholeskyInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using Cholesky factorization

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    On CPU, scipy has functions dedicated to solving Cholesky systems
    On GPU, must first calculate cholesky decomp (G = L L^T) and
    then solve two subsequent linear solves (x = L^-T L^-1 y)"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11 = self.fm.hessian(x)
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:

            L1 = cp.linalg.cholesky(A11)
            A11_inv_AT = solve_triangular(
                L1.T,
                solve_triangular(
                    L1,
                    self.A.T,
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )
            A11_inv_b1 = solve_triangular(
                L1.T,
                solve_triangular(
                    L1,
                    b1,
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )

            L = cp.linalg.cholesky(self.A @ (A11_inv_AT))
            w = solve_triangular(
                L.T,
                solve_triangular(
                    L,
                    b2 - self.A @ (A11_inv_b1),
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )
            xstep = -solve_triangular(
                L1.T,
                solve_triangular(
                    L1,
                    b1 + self.A.T @ w,
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )
        else:

            L1 = scipy.linalg.cho_factor(
                A11,
                overwrite_a=False,
                check_finite=False,
            )
            A11_inv_AT = scipy.linalg.cho_solve(
                L1,
                self.A.T,
                overwrite_b=False,
                check_finite=False,
            )
            A11_inv_b1 = scipy.linalg.cho_solve(
                L1,
                b1,
                overwrite_b=False,
                check_finite=False,
            )

            L, low_flag = scipy.linalg.cho_factor(
                self.A @ (A11_inv_AT),
                overwrite_a=False,
                check_finite=False,
            )
            w = scipy.linalg.cho_solve(
                (L, low_flag),
                b2 - self.A @ (A11_inv_b1),
                overwrite_b=False,
                check_finite=False,
            )

            xstep = -scipy.linalg.cho_solve(
                L1,
                b1 + self.A.T @ w,
                overwrite_b=False,
                check_finite=False,
            )

        vstep = w - v

        return xstep, vstep


class NewtonSolverDirectInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    by directly calculating the matrix inverse. Not recommended,
    inncluded only for timing and debugging purposes.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11 = self.fm.hessian(x)
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            A11_inv = cp.linalg.inv(A11)
            KKT_inv = cp.linalg.inv(self.A @ (A11_inv @ self.A.T))
        else:
            A11_inv = np.linalg.inv(A11)
            KKT_inv = np.linalg.inv(self.A @ (A11_inv @ self.A.T))
        w = KKT_inv @ (b2 - self.A @ (A11_inv @ b1))
        xstep = -A11_inv @ (b1 + self.A.T @ w)
        vstep = w - v
        return xstep, vstep


class NewtonSolverCGInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using conjugate gradient. Can see the maximum number of conjugate
    gradient steps using the max_cg_iters parameter.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving"""

    def __init__(
        self,
        A,
        b,
        C,
        d,
        obj_fxn,
        grad_fxn,
        hessian_fxn,
        inv_hessian_fxn,
        sign=1,
        max_iters=50,
        epsilon=1e-5,
        suppress_print=True,
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=20,
        use_gpu=False,
    ):

        raise NotImplementedError(
            "CONJUGATE GRADIENT GIVING UNSTABLE RESULTS, NEEDS TO BE DEBUGGED"
        )

        super().__init__(
            A,
            b,
            C,
            d,
            obj_fxn,
            grad_fxn,
            hessian_fxn,
            inv_hessian_fxn,
            sign=sign,
            max_iters=max_iters,
            epsilon=epsilon,
            suppress_print=suppress_print,
            max_cg_iters=max_cg_iters,
            alpha=alpha,
            beta=beta,
            mu=mu,
            use_gpu=False,
        )

        if self.use_gpu:
            self.last_w = cp.zeros(A.shape[0])
        else:
            self.last_w = np.zeros(A.shape[0])

    def newton_linear_solve(self, x, v, gradf):

        b1 = gradf
        b2 = self.A @ x - self.b
        A11 = self.fm.hessian(x)
        if A11.ndim < 2:
            A11 = np.diag(A11)

        # get initial x for conjugate gradient
        # TODO: WRITE A CHECK TO GET AN INITIAL W FOR CONJUGATE GRADIENT
        # THIS INITIAL W SHOULD BE BASED ON THE self.last_w PARAMETER THAT IS SAVED
        # descent_check = np.dot(x, gradf)
        # if descent_check < 0:
        #    x0 = -descent_check * x / np.dot(x, np.dot(self.fm.hessian(x), x))
        # else:
        #    x0 = np.zeros_like(v)

        # implement conjugate gradient for all of these intermediate solves?
        w = scipy.sparse.linalg.cg(
            self.A @ (A11_inv[:, None] * self.A.T),
            b2 - self.A @ (A11_inv * b1),
            # x0=x0,
            maxiter=self.max_cg_iters + 500,
        )[0]
        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        self.last_w = w
        return xstep, vstep


class NewtonSolverKKTNPSolveInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    by using numpy.linalg.solve. Differs from NewtonSolverNPSolve
    because this class does not use block elimination for faster solving.
    Not recommended, included only for timing and debugging purposes.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    """

    def newton_linear_solve(self, x, v, gradf):
        r_dual = gradf + self.A.T @ v
        r_primal = self.A @ x - self.b
        r = np.append(r_dual, r_primal)
        M = np.bmat(
            [
                [np.diag(self.fm.hessian(x)), self.A.T],
                [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))],
            ]
        )
        d = np.linalg.solve(M, -r)
        xstep = d[: self.A.shape[1]]
        vstep = d[self.A.shape[1] :]
        return xstep, vstep


class NewtonSolverNPLstSqDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using the least squares method from numpy

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11_inv = self.fm.inv_hessian(x)
        if self.use_gpu:
            w = cp.linalg.lstsq(
                self.A @ (A11_inv[:, None] * self.A.T),
                b2 - self.A @ (A11_inv * b1),
                rcond=None,
            )[0]
        else:
            w = np.linalg.lstsq(
                self.A @ (A11_inv[:, None] * self.A.T),
                b2 - self.A @ (A11_inv * b1),
                rcond=None,
            )[0]
        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        return xstep, vstep


class NewtonSolverNPSolveDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using the linalg.solve method from numpy

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11_inv = self.fm.inv_hessian(x)
        if self.use_gpu:
            w = cp.linalg.solve(
                self.A @ (A11_inv[:, None] * self.A.T), b2 - self.A @ (A11_inv * b1)
            )
        else:
            w = np.linalg.solve(
                self.A @ (A11_inv[:, None] * self.A.T), b2 - self.A @ (A11_inv * b1)
            )

        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        return xstep, vstep


class NewtonSolverCholeskyDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using Cholesky factorization

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    On CPU, scipy has functions dedicated to solving Cholesky systems
    On GPU, must first calculate cholesky decomp (G = L L^T) and
    then solve two subsequent linear solves (x = L^-T L^-1 y)

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11_inv = self.fm.inv_hessian(x)
        if self.use_gpu:

            L = cp.linalg.cholesky(self.A @ (A11_inv[:, None] * self.A.T))
            w = solve_triangular(
                L.T,
                solve_triangular(
                    L,
                    b2 - self.A @ (A11_inv * b1),
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )
        else:
            L, low_flag = scipy.linalg.cho_factor(
                self.A @ (A11_inv[:, None] * self.A.T),
                overwrite_a=False,
                check_finite=False,
            )
            w = scipy.linalg.cho_solve(
                (L, low_flag),
                b2 - self.A @ (A11_inv * b1),
                overwrite_b=False,
                check_finite=False,
            )

        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        return xstep, vstep


class NewtonSolverDirectDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    by directly calculating the matrix inverse. Not recommended,
    inncluded only for timing and debugging purposes.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        b2 = self.A @ x - self.b
        A11_inv = self.fm.inv_hessian(x)
        if self.use_gpu:
            KKT_inv = cp.linalg.inv(self.A @ (A11_inv[:, None] * self.A.T))
        else:
            KKT_inv = np.linalg.inv(self.A @ (A11_inv[:, None] * self.A.T))
        w = KKT_inv @ (b2 - self.A @ (A11_inv * b1))
        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        return xstep, vstep


class NewtonSolverCGDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    using conjugate gradient. Can see the maximum number of conjugate
    gradient steps using the max_cg_iters parameter.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Implements block elimination method for faster system solving

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def __init__(
        self,
        A,
        b,
        C,
        d,
        obj_fxn,
        grad_fxn,
        hessian_fxn,
        inv_hessian_fxn,
        sign=1,
        max_iters=50,
        epsilon=1e-5,
        suppress_print=True,
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=20,
    ):

        raise NotImplementedError(
            "CONJUGATE GRADIENT GIVING UNSTABLE RESULTS, NEEDS TO BE DEBUGGED"
        )

        super().__init__(
            A,
            b,
            C,
            d,
            obj_fxn,
            grad_fxn,
            hessian_fxn,
            inv_hessian_fxn,
            sign=sign,
            max_iters=max_iters,
            epsilon=epsilon,
            suppress_print=suppress_print,
            max_cg_iters=max_cg_iters,
            alpha=alpha,
            beta=beta,
            mu=mu,
            use_gpu=False,
        )

        if self.use_gpu:
            self.last_w = cp.zeros(A.shape[0])
        else:
            self.last_w = np.zeros(A.shape[0])

    def newton_linear_solve(self, x, v, gradf):

        b1 = gradf
        b2 = self.A @ x - self.b
        A11_inv = self.fm.inv_hessian(x)

        # get initial x for conjugate gradient
        # TODO: WRITE A CHECK TO GET AN INITIAL W FOR CONJUGATE GRADIENT
        # THIS INITIAL W SHOULD BE BASED ON THE self.last_w PARAMETER THAT IS SAVED
        # descent_check = np.dot(x, gradf)
        # if descent_check < 0:
        #    x0 = -descent_check * x / np.dot(x, np.dot(self.fm.hessian(x), x))
        # else:
        #    x0 = np.zeros_like(v)

        w = scipy.sparse.linalg.cg(
            self.A @ (A11_inv[:, None] * self.A.T),
            b2 - self.A @ (A11_inv * b1),
            # x0=x0,
            maxiter=self.max_cg_iters + 500,
        )[0]
        xstep = -A11_inv * (b1 + self.A.T @ w)
        vstep = w - v
        self.last_w = w
        return xstep, vstep


class NewtonSolverKKTNPSolveDiagonalInfeasibleStart(NewtonSolverInfeasibleStart):
    """Subclass of the NewtonSolver that solves linear equations
    by using numpy.linalg.solve. Differs from NewtonSolverNPSolve
    because this class does not use block elimination for faster solving.
    Not recommended, included only for timing and debugging purposes.

    Solves the system:

    [[H A^T]  [[xstep],     = -[[gradf(x)],
     [A 0]]   [v + vstep]]      [Ax - b]]

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, v, gradf):
        r_dual = gradf + self.A.T @ v
        r_primal = self.A @ x - self.b
        r = np.append(r_dual, r_primal)
        M = np.bmat(
            [
                [np.diag(self.fm.hessian(x)), self.A.T],
                [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))],
            ]
        )
        d = np.linalg.solve(M, -r)
        xstep = d[: self.A.shape[1]]
        vstep = d[self.A.shape[1] :]
        return xstep, vstep
