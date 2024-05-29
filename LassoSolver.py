import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cvx

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    gpu_flag = True
except Exception:
    gpu_flag = False
    print("Not able to run with GPU")


class LassoSolver:

    def __init__(
        self,
        A,
        b,
        reg=1,
        rho=0.4,
        max_iters=1000,
        check_stop=10,
        add_bias=False,
        normalize_A=False,
        mask=None,
        positive=False,
        compute_loss=False,
        adaptive_rho=False,
        eps_abs=1e-4,
        eps_rel=3e-2,
        use_gpu=False,
        num_chunks=0,
        check_cvxpy=True
    ):
        """
        Solve problem of the form using ADMM:

        Minimize_x  1/(2*n) ||A x - b||_2^2 + λ||x||_1

        If a matrix is passed to b, the problem is solved for multiple instances simultaneously

        Also supports solving the problem with a masked x or all positive entries in x
        """

        self.use_gpu = use_gpu and gpu_flag
        if self.use_gpu:
            X_shape = (A.shape[1], b.shape[0])
            Xb_size = (
                np.prod(X_shape) * 3 + np.prod(b.shape)
            ) * 8  # need to be able to fit three copies the size of X on GPU
            if mask is not None:
                Xb_size += np.prod(mask.shape) * 8
                Xb_size -= (
                    (b.shape[1] + 3 * X_shape[0]) * ((mask.sum(axis=0) == 0).sum()) * 8
                )
            A_size = (
                np.prod(A.shape) * 8 + A.shape[1] ** 2 * 8
            )  # be able to fit A and A^TA in GPU
            # assume 1.5 GB can reside on GPU at any time (gives some slack to true amount),
            # especially since the above is only a rough estimate of memory being used
            avail_memory = 1.5 * 1000**3 - A_size
            # print(avail_memory, C_size, YA_size)
            self.num_chunks = max(int(Xb_size // avail_memory) + 1, num_chunks)
        else:
            self.num_chunks = max(1, num_chunks)

        self.A = A
        self.b = b
        if self.b.ndim < 2:
            self.b = self.b[:, None]
        self.reg = reg
        self.rho = rho
        self.mask = mask
        self.max_iters = max_iters
        self.check_stop = check_stop
        self.compute_loss = compute_loss
        self.EPS_ABS = eps_abs
        self.EPS_REL = eps_rel
        self.positive = positive
        
        self.num_samples = self.b.shape[1]

        if self.use_gpu:
            self.A = cp.array(A)
            self.gaps = cp.zeros((self.max_iters, self.num_samples))
        else:
            self.gaps = np.zeros((self.max_iters, self.num_samples))

        self.m, self.n = self.A.shape
        if normalize_A:
            self.A /= self.A.std(axis=0)
        self.add_bias = add_bias
        if self.add_bias:
            if self.use_gpu:
                self.A = cp.hstack((cp.ones((self.m, 1)), self.A))
            else:
                self.A = np.hstack((np.ones((self.m, 1)), self.A))
        self.n = self.A.shape[1]
        
        if check_cvxpy:
            print("Testing CVXPY")
            self.feasible, self.cvxpy_vals, self.cvxpy_sols = self.check_cvxpy()
            if self.feasible == "infeasible":
                raise ValueError("Provided problem instance is infeasible!")
            elif self.feasible == "unbounded":
                raise ValueError("Provided problem instance is unbounded!")
            #else:
            #    print(f'Optimal value of {round(self.cvxpy_val, 5)}')
        else:
            self.feasible, self.cvxpy_val = None, None

        self.AtA_cache = self.A.T @ self.A

        # not recommended to use this, just trying something out
        if adaptive_rho:
            # adaptively calculate rho based on condition number of matrix?
            e = cp.linalg.eigvalsh(self.AtA_cache)
            lambda_min = e[0]
            lambda_max = e[-1]
            # print('min: %d, max:%d\n', lambda_min, lambda_max)
            # Compute acceptable upper limit on condition number. This is based on
            # the expected condition number of a random gaussian square matrix,
            # which is theta(n) (n is # dimensions).
            # cond_limit_upper = XtX.shape[0] / 3
            # lambda_min = max(lambda_min, lambda_max / cond_limit_upper)
            rho = 10 / cp.sqrt(lambda_min)

        if self.use_gpu:
            I = cp.eye(self.n)
            L = cp.linalg.cholesky(
                cp.diag(cp.ones(self.n) * self.m * self.rho) + self.AtA_cache
            )
            Qinv_cache = solve_triangular(
                L.T,
                solve_triangular(
                    L,
                    I,
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                ),
                lower=False,
                overwrite_b=False,
                check_finite=False,
            )
        else:
            I = np.eye(self.n)
            L, low_flag = scipy.linalg.cho_factor(
                np.diag(np.ones(self.n) * self.m * self.rho) + self.AtA_cache,
                overwrite_a=False,
                check_finite=False,
            )
            Qinv_cache = scipy.linalg.cho_solve(
                (L, low_flag),
                I,
                overwrite_b=False,
                check_finite=False,
            )
        self.Qinv_cache = Qinv_cache
        self.X = np.zeros((self.n, self.b.shape[1]))

        # if only one chunk, we can precalcuate everything (and transfer everything to GPU if applicable)
        if self.num_chunks == 1:
            self.solve_func = self.__run_admm

            if self.use_gpu:
                self.b = cp.array(self.b)
                self.reg = cp.array(reg)
                if self.mask is not None:
                    self.mask = cp.array(self.mask)

                self.stop_multiplier = self.EPS_ABS * cp.sqrt(self.n * self.num_samples)

                # initialize the primal and dual parameters:
                self.x = cp.zeros((self.n, self.num_samples))
                self.alpha = cp.zeros((self.n, self.num_samples))
                self.u = cp.zeros((self.n, self.num_samples))
            else:
                self.b = np.array(self.b)
                self.reg = np.array(reg)
                if self.mask is not None:
                    self.mask = np.array(self.mask)

                self.stop_multiplier = self.EPS_ABS * np.sqrt(self.n * self.num_samples)

                # initialize the primal and dual parameters:
                self.x = np.zeros((self.n, self.num_samples))
                self.alpha = np.zeros((self.n, self.num_samples))
                self.u = np.zeros((self.n, self.num_samples))

            self.eta = self.reg / self.rho

            if self.compute_loss:
                self.Atb_cache = self.A.T @ self.b
                self.bA_cache = self.Qinv_cache @ (self.Atb_cache)
                self.bb_cache = self.b.T @ self.b
                if self.use_gpu:
                    self.Atb_cache = cp.array(self.Atb_cache)
                    self.bb_cache = cp.array(self.bb_cache)
            else:
                self.bA_cache = self.Qinv_cache @ (self.A.T @ self.b)
            self.Qinv_cache *= -self.m * self.rho
            if self.use_gpu:
                self.Qinv_cache = cp.array(self.Qinv_cache)
                self.bA_cache = cp.array(self.bA_cache)
        else:
            self.solve_func = self.__run_admm_chunks

    def solve(self):
        return self.solve_func()

    def __run_admm(self):

        # run ADMM
        for iteration in range(self.max_iters):
            # primal updates
            self.x = self.bA_cache + self.Qinv_cache @ (self.u - self.alpha)
            last_alpha = self.alpha
            self.alpha = self.prox(self.x + self.u, self.eta)
            if self.mask is not None:
                if self.add_bias:
                    self.alpha[1:] *= self.mask
                else:
                    self.alpha *= self.mask

            # dual update
            self.u += self.x - self.alpha

            if self.compute_loss:
                # save sub-optimality

                f = 1 / (2 * self.m) * ((self.A @ self.alpha - self.b) ** 2).sum(axis=0)
                if self.use_gpu and not self.positive:
                    x_abs = cp.abs(self.alpha)
                elif not self.positive:
                    x_abs = np.abs(self.alpha)
                else:
                    x_abs = self.alpha
                if self.add_bias:
                    norm1 = x_abs[1:].sum(axis=0)
                else:
                    norm1 = x_abs.sum(axis=0)
                f += self.reg * norm1

                self.gaps[iteration] = f

            if iteration % self.check_stop == self.check_stop - 1:
                r = self.x - self.alpha
                d = self.rho * (self.alpha - last_alpha)
                if self.use_gpu:
                    r_norm = cp.linalg.norm(r)
                    d_norm = cp.linalg.norm(d)
                    tol_primal = self.stop_multiplier + self.EPS_REL * cp.linalg.norm(
                        self.alpha
                    )
                    tol_dual = (
                        self.stop_multiplier
                        + self.EPS_REL * self.rho * cp.linalg.norm(self.u)
                    )
                else:
                    r_norm = np.linalg.norm(r)
                    d_norm = np.linalg.norm(d)
                    tol_primal = self.stop_multiplier + self.EPS_REL * np.linalg.norm(
                        self.alpha
                    )
                    tol_dual = (
                        self.stop_multiplier
                        + self.EPS_REL * self.rho * np.linalg.norm(self.u)
                    )
                if (r_norm < tol_primal) and (d_norm < tol_dual):
                    # print(f'Stopping solve at iteration {iteration}')
                    break
                # if adaptive_rho:
                # tau_param = cp.sqrt(r_norm / (z*d_norm))
                # if 1 <= tau_param < tau_max:
                #    tau = tau_param
                # elif 1/tau_max < tau_param < 1:
                #    tau = z * tau_param
                # else:
                #    tau = tau_max
                # if r_norm > d_norm * mu:
                #    rho *= tau
                # elif d_norm > r_norm * mu:
                #    rho /= tau
                # print(tau, rho)
                # rho *= 1.0001

        f = 1 / (2 * self.m) * ((self.A @ self.alpha - self.b) ** 2).sum(axis=0)
        if self.use_gpu and not self.positive:
            x_abs = cp.abs(self.alpha)
        elif not self.positive:
            x_abs = np.abs(self.alpha)
        else:
            x_abs = self.alpha
        if self.add_bias:
            norm1 = x_abs[1:].sum(axis=0)
        else:
            norm1 = x_abs.sum(axis=0)
        f += self.reg * norm1
        
        self.solutions = f
        if self.use_gpu:
            self.X = self.alpha.get()
            gaps = self.gaps.get()
        else:
            self.X = self.alpha
            gaps = self.gaps

        return self.X, self.solutions, self.gaps[: iteration + 1], iteration+1

    def __run_admm_chunks(self):

        num_iterations = []
        if self.use_gpu:
            self.solutions = cp.empty(self.num_samples)
        else:
            self.solutions = np.empty(self.num_samples)

        reg_is_array = isinstance(self.reg, np.ndarray)

        indices = np.array(range(self.b.shape[1]))
        for i in range(self.num_chunks):
            iter_indices = indices[i :: self.num_chunks]
            if self.use_gpu:
                b_iter = cp.array(self.b[..., iter_indices])
                num_samples = b_iter.shape[1]

                if reg_is_array:
                    iter_reg = cp.array(self.reg[iter_indices])
                else:
                    iter_reg = cp.array(self.reg)
                if self.mask is not None:
                    iter_mask = cp.array(self.mask)

                iter_stop_multiplier = self.EPS_ABS * cp.sqrt(self.n * num_samples)

                # initialize the primal and dual parameters:
                x = cp.zeros((self.n, num_samples))
                alpha = cp.zeros((self.n, num_samples))
                u = cp.zeros((self.n, num_samples))

            else:
                b_iter = np.array(self.b[..., iter_indices])
                num_samples = b_iter.shape[1]

                if reg_is_array:
                    iter_reg = np.array(self.reg[iter_indices])
                else:
                    iter_reg = np.array(self.reg)
                if self.mask is not None:
                    iter_mask = np.array(self.mask)

                iter_stop_multiplier = self.EPS_ABS * np.sqrt(self.n * num_samples)

                # initialize the primal and dual parameters:
                x = np.zeros((self.n, num_samples))
                alpha = np.zeros((self.n, num_samples))
                u = np.zeros((self.n, num_samples))

            eta = iter_reg / self.rho

            if self.compute_loss:
                Atb_cache = self.A.T @ b_iter
                bA_cache = self.Qinv_cache @ (Atb_cache)
                bb_cache = b_iter.T @ b_iter
            else:
                bA_cache = self.Qinv_cache @ (self.A.T @ b_iter)
            Qinv_cache_iter = self.Qinv_cache * -self.m * self.rho

            # run ADMM
            for iteration in range(self.max_iters):
                # primal updates
                x = bA_cache + Qinv_cache_iter @ (u - alpha)
                last_alpha = alpha
                alpha = self.prox(x + u, eta)
                if self.mask is not None:
                    if self.add_bias:
                        alpha[1:] *= iter_mask
                    else:
                        alpha *= iter_mask

                # dual update
                u = u + (x - alpha)

                if self.compute_loss:
                    # save sub-optimality
                    f = 1 / (2 * self.m) * ((self.A @ alpha - b_iter) ** 2).sum(axis=0)
                    if self.use_gpu and not self.positive:
                        x_abs = cp.abs(alpha)
                    elif not self.positive:
                        x_abs = np.abs(alpha)
                    else:
                        x_abs = alpha
                    if self.add_bias:
                        norm1 = x_abs[1:].sum(axis=0)
                    else:
                        norm1 = x_abs.sum(axis=0)
                    f += iter_reg * norm1

                    self.gaps[iteration, iter_indices] = f

                if iteration % self.check_stop == self.check_stop - 1:
                    r = x - alpha
                    d = self.rho * (alpha - last_alpha)
                    if self.use_gpu:
                        r_norm = cp.linalg.norm(r)
                        d_norm = cp.linalg.norm(d)
                        tol_primal = (
                            iter_stop_multiplier + self.EPS_REL * cp.linalg.norm(alpha)
                        )
                        tol_dual = (
                            iter_stop_multiplier
                            + self.EPS_REL * self.rho * cp.linalg.norm(u)
                        )
                    else:
                        r_norm = np.linalg.norm(r)
                        d_norm = np.linalg.norm(d)
                        tol_primal = (
                            iter_stop_multiplier + self.EPS_REL * np.linalg.norm(alpha)
                        )
                        tol_dual = (
                            iter_stop_multiplier
                            + self.EPS_REL * self.rho * np.linalg.norm(u)
                        )
                    if (r_norm < tol_primal) and (d_norm < tol_dual):
                        # print(f'Stopping solve at iteration {iteration}')
                        break
                    # if adaptive_rho:
                    # tau_param = cp.sqrt(r_norm / (z*d_norm))
                    # if 1 <= tau_param < tau_max:
                    #    tau = tau_param
                    # elif 1/tau_max < tau_param < 1:
                    #    tau = z * tau_param
                    # else:
                    #    tau = tau_max
                    # if r_norm > d_norm * mu:
                    #    rho *= tau
                    # elif d_norm > r_norm * mu:
                    #    rho /= tau
                    # print(tau, rho)
                    # rho *= 1.0001

            f = 1 / (2 * self.m) * ((self.A @ alpha - b_iter) ** 2).sum(axis=0)
            if self.use_gpu and not self.positive:
                x_abs = cp.abs(alpha)
            elif not self.positive:
                x_abs = np.abs(alpha)
            else:
                x_abs = alpha
            if self.add_bias:
                norm1 = x_abs[1:].sum(axis=0)
            else:
                norm1 = x_abs.sum(axis=0)
            f += iter_reg * norm1
            self.solutions[iter_indices] = f
            if self.use_gpu:
                self.X[:, iter_indices] = alpha = alpha.get()
            else:
                self.X[:, iter_indices] = alpha
            num_iterations.append(iteration)

        if self.use_gpu:
            gaps = self.gaps.get()
        else:
            gaps = self.gaps
        return self.X, self.solutions, self.gaps, num_iterations

    def objective(self):
        """Compute the objective.

        Note: We do not regularize the bias term.

        Params:
            A: the training data matrix.
            b: the training targets.
            x: the variables
            reg: the regularization strength.

        Returns:
            objective: the full training objective
        """

        f = 1 / (2 * self.m) * ((self.A @ self.alpha - self.b) ** 2).sum(axis=0)
        if self.use_gpu and not self.positive:
            x_abs = cp.abs(self.alpha)
        elif self.positive:
            x_abs = np.abs(self.alpha)
        else:
            x_abs = self.alpha
        if self.add_bias:
            norm1 = x_abs[1:].sum(axis=0)
        else:
            norm1 = x_abs.sum(axis=0)
        f += self.reg * norm1

        return f

    def prox(self, v, eta):
        """Evaluate the proximal operator

        Do not regularize the bias term.

        Params:
            v: the vector at which to evaluate the proximal operator.
            eta: the parameter controlling the strength of the proximal term.

        Returns:
            prox(v): the result of the proximal operator.
        """

        # group_norms = np.sum(v)
        if self.use_gpu:
            x = cp.maximum(v - eta, 0)
        else:
            x = np.maximum(v - eta, 0)
        if not self.positive:
            if self.use_gpu:
                x -= cp.maximum(-v - eta, 0)
            else:
                x -= np.maximum(-v - eta, 0)
        if self.add_bias:
            x[0] = v[0]  # don't penalize the bias term

        return x
    
    def check_cvxpy(self):
        if self.use_gpu:
            A = self.A.get()
        else:
            A = self.A
        cvxpy_sols = []
        cvxpy_vals = []
        for i in range(self.num_samples):
            print(f'CVXPY solving sample {i+1}...', end='')

            x = cvx.Variable(self.n)
            obj = cvx.Minimize(1/(2*self.m)*cvx.norm2(A@x - self.b[:,i])**2 + self.reg[i]*cvx.norm(x, 1))
            prob = cvx.Problem(obj, [])
            try:
                prob.solve(solver="CLARABEL")
                print(f'Optimal value of {round(prob.value, 4)}')
            except Exception as e:
                print('CVXPY SOLVER FAILED DUE TO THE FOLLOWING EXCEPTION:')
                print(e)
                
            cvxpy_vals.append(prob.value)
            cvxpy_sols.append(x.value)

        return prob.status, np.array(cvxpy_vals), cvxpy_sols
