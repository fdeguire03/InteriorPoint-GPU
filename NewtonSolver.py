import numpy as np
import scipy

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from cupyx.scipy.sparse.linalg import cg

    gpu_flag = True
except Exception:
    gpu_flag = False


class NewtonSolver:

    def __init__(
        self,
        A=None,
        b=None,
        C=None,
        d=None,
        function_manager=None,
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
        phase1_flag=False,
        phase1_tol=0.1,
        use_psd_condition=False,
        update_slacks_every=0,
    ):
        """Solve convex optimization problem of the following form using Newton's method:
        argmin_x  t * obj_fxn(x)

        This class does not support infeaisble start Newton's method. Use NewtonSolverInfeasibleStart for equality-constrained problems.

        Uses provided gradient and hessian functions to solve problem

        Currently, assumes that Hessian is diagonal, but this assumption will stop holding as solver gets
        more robust to more types of problems

        This class does not have a linear solve method implemented and will need to be overridden by a child class
        to be implemented on a problem
        """

        # problem specifications
        # A and b and normed constraints included only for consistency with the Infeaisble Start Solver, they are not used
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
        self.phase1_flag = phase1_flag
        self.phase1_tol = phase1_tol
        self.use_psd_condition = use_psd_condition
        self.update_slacks_every = update_slacks_every

    def solve(self, x, t, v0=None):
        """Solve a convex optimization problem using Newton's method, using the provided initial values
        for x and t

        v0 parameter is included for compatibility with the NewtonSolverInfeasibleStart class, but it is not used
        """

        # place everything in a try-except block so we can report if there was an error during solve
        try:

            for iter in range(self.max_iters):

                # precompute gradient since it will be used in multiple locations
                gradf = self.fm.gradient(x)

                # invoke linear solve method -- needs to be implemented by a child class
                xstep = self.newton_linear_solve(x, gradf)

                # backtracking line search on norm of residual
                # also captures residual nad gradient calculations from backtracking search
                step_size = self.backtrack_search(x, xstep, t, gradf)

                # update x and nu based on newton solve
                x += step_size * xstep
                self.fm.update_x(x)
                if self.phase1_flag:
                    if x[-1] < -self.phase1_tol:
                        return x, None, iter + 1, None, True

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
                nd = -gradf.dot(xstep) / 2
                if step_size < 1e-13:
                    return x, None, iter + 1, nd, False
                elif nd < self.eps:
                    return x, None, iter + 1, nd, True

            # if we reach the maximum number of iterations, print warnings to the user unless specified not to
            if self.suppress_print:
                return x, None, iter + 1, nd, False

            print(
                "REACHED MAX ITERATIONS: Problem likely infeasible or unbounded",
                end="",
            )

            # else we are not feasible
            print(" (Likely infeasible)")
            return x, None, iter + 1, nd, False

        except np.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, None, iter + 1, nd, False
        except cp.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, None, iter + 1, nd, False

    def backtrack_search(self, x, xstep, t, gradf):
        """Backtracking search for Newton's method ensures that Newton step
        walks in a descent direction

        First, make sure that the next x is in the domain of the objective function (satisfies all log barriers)
        Then, make sure that we are going in a descent direction"""

        # default to step size of 1 -- can only get smaller
        step_size = 1
        fx = self.fm.newton_objective()
        next_x = x + step_size * xstep
        grad_check = gradf.dot(x)

        # make sure our next step is in the domain of f

        self.fm.update_x(next_x)

        while (self.fm.slacks < 0).any():
            step_size *= self.beta
            if step_size < 1e-13:
                if not self.suppress_print:
                    print(
                        "Backtracking search got stuck, returning from Newton's method now..."
                    )
                return step_size
            next_x = x + step_size * xstep
            self.fm.update_x(next_x)

        attempt = 0
        while self.fm.newton_objective() > fx + self.alpha * step_size * grad_check:
            attempt += 1
            next_x = x + step_size * xstep
            if step_size < 1e-13:
                if not self.suppress_print:
                    print(
                        "Backtracking search got stuck, returning from Newton's method now..."
                    )
                return step_size
            step_size *= self.beta
            if self.update_slacks_every > 0:
                update_slacks = (
                    attempt % self.update_slacks_every == self.update_slacks_every - 1
                )
                self.fm.update_x(next_x, update_slacks=update_slacks)
            else:
                self.fm.update_x(next_x, update_slacks=False)

        self.fm.update_x(next_x)

        return step_size

    def newton_linear_solve(self, x, gradient):
        raise NotImplementedError("Must be overridden by child class")


class NewtonSolverNPLstSq(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    using the least squares method from numpy

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):
        H = self.fm.hessian()
        if self.use_gpu:
            xstep = cp.linalg.lstsq(H, -gradf, rcond=None)[0]
        else:
            xstep = np.linalg.lstsq(H, -gradf, rcond=None)[0]
        return xstep


class NewtonSolverNPSolve(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    using the linalg.solve method from numpy

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):

        H = self.fm.hessian()

        if self.use_gpu:
            xstep = cp.linalg.solve(H, -gradf)
        else:
            xstep = np.linalg.solve(H, -gradf)
        return xstep


class NewtonSolverCholesky(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    using Cholesky factorization

    Solves the system:

    H * xstep = -grad(x)

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

    def newton_linear_solve(self, x, gradf):

        H = self.fm.hessian()

        if not self.use_backup:
            try:
                if self.use_psd_condition:
                    H = self.add_psd_conditioning(H)
                if self.use_gpu:
                    L = cp.linalg.cholesky(H)
                    xstep = solve_triangular(
                        L.T,
                        solve_triangular(
                            L,
                            -gradf,
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
                        H,
                        overwrite_a=True,
                        check_finite=False,
                    )
                    xstep = scipy.linalg.cho_solve(
                        (L, low_flag),
                        -gradf,
                        overwrite_b=True,
                        check_finite=False,
                    )
            except np.linalg.LinAlgError:
                if not self.suppress_print:
                    print(
                        "Cholesky solver failed due to numeric instability. Proceeding with Numpy solve..."
                    )
                self.use_backup = True
                xstep = self.backup_solve(x, gradf, H=H)
            except cp.linalg.LinAlgError:
                if not self.suppress_print:
                    print(
                        "Cholesky solver failed due to numeric instability. Proceeding with Numpy solve..."
                    )
                self.use_backup = True
                xstep = self.backup_solve(x, gradf, H=H)

        else:
            xstep = self.backup_solve(x, gradf, H=H)

        return xstep

    def backup_solve(self, x, gradf, H=None):
        if H is None:
            H = self.fm.hessian()
        if self.use_gpu:
            xstep = cp.linalg.lstsq(H, -gradf, rcond=None)[0]
        else:
            xstep = np.linalg.lstsq(H, -gradf, rcond=None)[0]
        return xstep


class NewtonSolverDirect(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    by directly calculating the matrix inverse. Not recommended,
    inncluded only for timing and debugging purposes.

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):
        H = self.fm.hessian()
        if self.use_gpu:
            H_inv = cp.linalg.inv(H)
            xstep = cp.matmul(H_inv, -gradf)
        else:
            H_inv = np.linalg.inv(H)
            xstep = np.matmul(H_inv, -gradf)
        return xstep


class NewtonSolverCG(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    using conjugate gradient. Can see the maximum number of conjugate
    gradient steps using the max_cg_iters parameter.

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):

        H = self.fm.hessian()

        descent_check = np.dot(x, gradf)
        if descent_check < 0:
            x0 = -descent_check * x / np.dot(x, np.dot(H, x))
        else:
            x0 = np.zeros_like(x)

        if self.use_gpu:
            xstep = cg(
                -H,
                gradf,
                x0=x0,
                maxiter=self.max_cg_iters,
            )[0]
        else:
            xstep = scipy.sparse.linalg.cg(
                -H,
                gradf,
                x0=x0,
                maxiter=self.max_cg_iters,
            )[0]

        return xstep


class NewtonSolverDiagonal(NewtonSolver):
    """Subclass of the NewtonSolver that solves by exploiting diagonal structure
    of Hessian matrix. We do not need to implement any solve steps (using numpy,
    cholesky, etc.) because the system is already solved when we get the matrix
    inverse of H.

    Solves the system:

    H * xstep = -grad(x)

    Assumes a diagonal hessian matrix that we can take the inverse of easily"""

    def newton_linear_solve(self, x, gradf):

        H_inv = self.fm.inv_hessian()
        xstep = -H_inv * gradf

        return xstep
