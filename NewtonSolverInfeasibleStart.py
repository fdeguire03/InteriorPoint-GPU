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
        lower_bound=None,
        upper_bound=None,
        max_iters=50,
        epsilon=1e-5,
        suppress_print=True,
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=20,
        use_gpu=False,
        track_loss=False,
        use_psd_condition=False,
        update_slacks_every=0,
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
        self.lb = lower_bound
        self.ub = upper_bound

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
        self.use_psd_condition = use_psd_condition
        self.update_slacks_every = update_slacks_every

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

        residual_norm = None

        # place everything in a try-except block so we can report if there was an error during solve
        try:

            for iter in range(self.max_iters):

                # precompute gradient since it will be used in multiple locations
                gradf = self.fm.gradient(x)

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
                self.fm.update_x(x)

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
                if step_size < 1e-13:
                    return x, v, iter + 1, residual_norm, False
                elif residual_norm < self.eps:
                    return x, v, iter + 1, residual_norm, True

            # if we reach the maximum number of iterations, print warnings to the user unless specified not to
            if self.suppress_print:
                return x, v, iter + 1, residual_norm, False

            print(
                "REACHED MAX ITERATIONS: Problem likely infeasible or unbounded",
                end="",
            )

            # unbounded below if we have a feasible x
            if (self.A @ x - self.b < self.eps).all():
                if not self.suppress_print:
                    print(" (Likely unbounded)")
                return x, v, iter + 1, residual_norm, False

            # else we are not feasible
            else:
                if not self.suppress_print:
                    print(" (Likely infeasible)")
                return x, v, iter + 1, residual_norm, False

        except np.linalg.LinAlgError as e:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, v, iter + 1, residual_norm, False
        except cp.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, v, iter + 1, residual_norm, False

    def backtrack_search(self, x, v, xstep, vstep, t, gradf, residual_normp=None):
        """Backtracking search for Newton's method ensures that Newton step
        walks in a descent direction

        First, make sure that the next x is in the domain of the objective function (satisfies all log barriers)
        Then, make sure that we are going in a descent direction"""

        # default to step size of 1 -- can only get smaller
        step_size = 1
        next_x = x + step_size * xstep

        # make sure our next step is in the domain of f
        self.fm.update_x(next_x)

        while ((self.fm.slacks) < 0).any():
            step_size *= self.beta
            if step_size < 1e-13:
                if not self.suppress_print:
                    print(
                        "Backtracking search got stuck, returning from Newton's method now..."
                    )
                return step_size, None, None
            next_x = x + step_size * xstep
            self.fm.update_x(next_x)

        # capture results of some matrix multiplies so we don't do repeated calculations
        if self.use_gpu:
            ATv_cache = cp.matmul(self.A.T, v)
            ATvstep_cache = cp.matmul(self.A.T, vstep)
            Axb_cache = cp.matmul(self.A, x) - self.b
            Axstep_cache = cp.matmul(self.A, xstep)
        else:
            ATv_cache = np.matmul(self.A.T, v)
            ATvstep_cache = np.matmul(self.A.T, vstep)
            Axb_cache = np.matmul(self.A, x) - self.b
            Axstep_cache = np.matmul(self.A, xstep)

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
        next_grad = self.fm.gradient()
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
        attempt = 0
        while next_residual_norm > (1 - self.alpha * step_size) * residual_norm:
            attempt += 1
            step_size *= self.beta
            if step_size < 1e-13:
                if not self.suppress_print:
                    print(
                        "Backtracking search got stuck, returning from Newton's method now..."
                    )
                break
            next_x = x + step_size * xstep
            if self.update_slacks_every > 0:
                update_slacks = attempt % self.update_slacks_every == self.update_slacks_every - 1
                self.fm.update_x(next_x, update_slacks=update_slacks)
            else:
                self.fm.update_x(next_x, update_slacks=False)

            next_grad = self.fm.gradient()
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

        self.fm.update_x(next_x)

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
        A11 = self.fm.hessian()
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            b2 = cp.matmul(self.A, x) - self.b
            A11_inv_AT = cp.linalg.lstsq(A11, self.A.T, rcond=None)[0]
            A11_inv_b1 = cp.linalg.lstsq(A11, b1, rcond=None)[0]
            w = cp.linalg.lstsq(
                cp.matmul(self.A, (A11_inv_AT)),
                b2 - cp.matmul(self.A, (A11_inv_b1)),
                rcond=None,
            )[0]
            xstep = -cp.linalg.lstsq(A11, (b1 + self.A.T @ w), rcond=None)[0]
        else:
            b2 = np.matmul(self.A, x) - self.b
            A11_inv_AT = np.linalg.lstsq(A11, self.A.T, rcond=None)[0]
            A11_inv_b1 = np.linalg.lstsq(A11, b1, rcond=None)[0]
            w = np.linalg.lstsq(
                np.matmul(self.A, A11_inv_AT),
                b2 - np.matmul(self.A, A11_inv_b1),
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
        A11 = self.fm.hessian()
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            b2 = cp.matmul(self.A, x) - self.b
            A11_inv_AT = cp.linalg.solve(A11, self.A.T)
            A11_inv_b1 = cp.linalg.solve(A11, b1)
            w = cp.linalg.solve(
                cp.matmul(self.A, (A11_inv_AT)), b2 - cp.matmul(self.A, A11_inv_b1)
            )
            xstep = -cp.linalg.solve(A11, (b1 + cp.matmul(self.A.T, w)))
        else:
            b2 = np.matmul(self.A, x) - self.b
            A11_inv_AT = np.linalg.solve(A11, self.A.T)
            A11_inv_b1 = np.linalg.solve(A11, b1)
            w = np.linalg.solve(
                np.matmul(self.A, (A11_inv_AT)), b2 - np.matmul(self.A, A11_inv_b1)
            )
            xstep = -np.linalg.solve(A11, (b1 + np.matmul(self.A.T, w)))
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

    def __init__(self, *args, **kwargs):
        #    use_psd_condition = kwargs.pop("use_psd_conditioning", False)
        #   self.use_psd_condition = use_psd_condition

        super().__init__(*args, **kwargs)
        self.use_backup = False

    def add_psd_conditioning(self, M):
        if self.use_gpu:
            diag = cp.einsum("ii->i", M)
        else:
            diag = np.einsum("ii->i", M)
        diag += 1e-9
        return M

    def newton_linear_solve(self, x, v, gradf):
        b1 = gradf
        A11 = self.fm.hessian()
        if A11.ndim < 2:
            A11 = np.diag(A11)

        if not self.use_backup:
            try:
                if self.use_psd_condition:
                    A11 = self.add_psd_conditioning(A11)
                if self.use_gpu:
                    b2 = cp.matmul(self.A, x) - self.b
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

                    L = cp.linalg.cholesky(cp.matmul(self.A, A11_inv_AT))
                    w = solve_triangular(
                        L.T,
                        solve_triangular(
                            L,
                            b2 - cp.matmul(self.A, A11_inv_b1),
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
                            b1 + cp.matmul(self.A.T, w),
                            lower=True,
                            overwrite_b=False,
                            check_finite=False,
                        ),
                        lower=False,
                        overwrite_b=False,
                        check_finite=False,
                    )
                else:
                    b2 = np.matmul(self.A, x) - self.b
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
                        np.matmul(self.A, A11_inv_AT),
                        overwrite_a=False,
                        check_finite=False,
                    )
                    w = scipy.linalg.cho_solve(
                        (L, low_flag),
                        b2 - np.matmul(self.A, A11_inv_b1),
                        overwrite_b=False,
                        check_finite=False,
                    )

                    xstep = -scipy.linalg.cho_solve(
                        L1,
                        b1 + np.matmul(self.A.T, w),
                        overwrite_b=False,
                        check_finite=False,
                    )
            except np.linalg.LinAlgError:
                if not self.suppress_print:
                    print(
                        "Cholesky solver failed due to numeric instability. Proceeding with Numpy solve..."
                    )
                self.use_backup = True
                xstep, w = self.backup_solve(x, v, gradf, A11=A11, b2=b2)
            except cp.linalg.LinAlgError:
                if not self.suppress_print:
                    print(
                        "Cholesky solver failed due to numeric instability. Proceeding with Numpy solve..."
                    )
                self.use_backup = True
                xstep, w = self.backup_solve(x, v, gradf, A11=A11, b2=b2)

        else:
            xstep, w = self.backup_solve(x, v, gradf, A11=A11)

        vstep = w - v

        return xstep, vstep

    def backup_solve(self, x, v, gradf, A11=None, b2=None):
        b1 = gradf
        if A11 is None:
            A11 = self.fm.hessian()
        if A11.ndim < 2:
            A11 = np.diag(A11)
        if self.use_gpu:
            if b2 is None:
                b2 = cp.matmul(self.A, x) - self.b
            A11_inv_AT = cp.linalg.solve(A11, self.A.T)
            A11_inv_b1 = cp.linalg.solve(A11, b1)
            w = cp.linalg.solve(
                cp.matmul(self.A, (A11_inv_AT)), b2 - cp.matmul(self.A, A11_inv_b1)
            )
            xstep = -cp.linalg.solve(A11, (b1 + cp.matmul(self.A.T, w)))
        else:
            if b2 is None:
                b2 = np.matmul(self.A, x) - self.b
            A11_inv_AT = np.linalg.solve(A11, self.A.T)
            A11_inv_b1 = np.linalg.solve(A11, b1)
            w = np.linalg.solve(
                np.matmul(self.A, (A11_inv_AT)), b2 - np.matmul(self.A, A11_inv_b1)
            )
            xstep = -np.linalg.solve(A11, (b1 + np.matmul(self.A.T, w)))

        return xstep, w


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
        A11 = self.fm.hessian()
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
        A11 = self.fm.hessian()
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
                [np.diag(self.fm.hessian()), self.A.T],
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
        A11_inv = self.fm.inv_hessian()
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
        A11_inv = self.fm.inv_hessian()
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
        A11_inv = self.fm.inv_hessian()
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
        A11_inv = self.fm.inv_hessian()
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
        A11_inv = self.fm.inv_hessian()

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
                [np.diag(self.fm.hessian()), self.A.T],
                [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))],
            ]
        )
        d = np.linalg.solve(M, -r)
        xstep = d[: self.A.shape[1]]
        vstep = d[self.A.shape[1] :]
        return xstep, vstep
