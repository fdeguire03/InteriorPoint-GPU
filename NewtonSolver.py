import numpy as np
import scipy

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from cupyx.scipy.sparse.linalg import cg

    gpu_flag = True
except Exception:
    gpu_flag = False
    print("Not able to run with GPU")


class NewtonSolver:

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
        track_loss=False,
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
        # A and b included only for consistency with the Infeaisble Start Solver, they are not used
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.sign = sign

        # problem functions for solve method
        self.obj = obj_fxn
        self.grad = grad_fxn
        self.hessian = hessian_fxn
        self.inv_hessian = inv_hessian_fxn

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
        for x and t

        v0 parameter is included for compatibility with the NewtonSolverInfeasibleStart class, but it is not used
        """

        # place everything in a try-except block so we can report if there was an error during solve
        try:

            for iter in range(self.max_iters):

                # precompute gradient since it will be used in multiple locations
                gradf = self.grad(x, t)

                # invoke linear solve method -- needs to be implemented by a child class
                xstep = self.newton_linear_solve(x, gradf)

                # backtracking line search on norm of residual
                # also captures residual nad gradient calculations from backtracking search
                step_size = self.backtrack_search(x, xstep, t, gradf)

                # update x and nu based on newton solve
                x += step_size * xstep

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
                nd = -gradf.T @ xstep / 2
                if step_size < 1e-15 or nd < self.eps:
                    return x, None, iter + 1, nd

            # if we reach the maximum number of iterations, print warnings to the user unless specified not to
            if self.suppress_print:
                return x, None, iter + 1, nd

            print(
                "REACHED MAX ITERATIONS: Problem likely infeasible or unbounded",
                end="",
            )

            # else we are not feasible
            print(" (Likely infeasible)")
            return x, None, iter + 1, nd

        except np.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, None, iter + 1, nd
        except cp.linalg.LinAlgError:
            if not self.suppress_print:
                print("OVERFLOW ERROR: Problem likely unbounded")
            return x, None, iter + 1, nd

    def backtrack_search(self, x, xstep, t, gradf):
        """Backtracking search for Newton's method ensures that Newton step
        walks in a descent direction

        First, make sure that the next x is in the domain of the objective function (satisfies all log barriers)
        Then, make sure that we are going in a descent direction"""

        # default to step size of 1 -- can only get smaller
        step_size = 1
        fx = self.obj(x, t)
        next_x = x + step_size * xstep
        grad_check = gradf.T @ x

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

        while self.obj(next_x, t) > fx + self.alpha * step_size * grad_check:
            next_x = x + step_size * xstep
            step_size *= self.beta

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
        g = gradf
        H = self.hessian(x)
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

        H = self.hessian(x)

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

    def newton_linear_solve(self, x, gradf):

        H = self.hessian(x)
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

        return xstep


class NewtonSolverDirect(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    by directly calculating the matrix inverse. Not recommended,
    inncluded only for timing and debugging purposes.

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):
        H = self.hessian(x)
        if self.use_gpu:
            H_inv = cp.linalg.inv(H)
            xstep = H_inv @ -gradf
        else:
            H_inv = np.linalg.inv(H)
            xstep = H_inv @ -gradf
        return xstep


class NewtonSolverCG(NewtonSolver):
    """Subclass of the NewtonSolver that solves linear equations
    using conjugate gradient. Can see the maximum number of conjugate
    gradient steps using the max_cg_iters parameter.

    Solves the system:

    H * xstep = -grad(x)
    """

    def newton_linear_solve(self, x, gradf):

        H = self.hessian(x)

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

        H_inv = self.inv_hessian(x)
        xstep = -H_inv * gradf

        return xstep
