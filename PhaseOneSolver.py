import numpy as np
from NewtonSolver import *
from FunctionManager import FunctionManagerPhase1, FunctionManagerSOCPPhase1


class PhaseOneSolver:

    def __init__(
        self,
        C=None,
        d=None,
        lower_bound=0,
        upper_bound=None,
        x0=None,
        max_outer_iters=50,
        max_inner_iters=20,
        epsilon=1e-8,
        inner_epsilon=1e-5,
        linear_solve_method="cholesky",
        max_cg_iters=50,
        alpha=0.2,
        beta=0.6,
        mu=15,
        t0=1,
        suppress_print=False,
        use_gpu=False,
        track_loss=False,
        n=None,
        tol=0.1,
        socp=False,
        socp_params=None,
        use_psd_condition=False,
        update_slacks_every=0
    ):

        # all attributes for LP
        self.C = C
        self.d = d

        self.lb = lower_bound
        self.ub = upper_bound
        self.x = x0
        self.n = n

        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.epsilon = epsilon
        self.inner_epsilon = inner_epsilon
        self.linear_solve_method = linear_solve_method

        self.max_cg_iters = max_cg_iters
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.suppress_print = suppress_print
        self.use_gpu = use_gpu
        self.track_loss = track_loss
        self.tol = tol
        self.t0 = t0
        self.update_slacks_every = update_slacks_every

        if not socp:
            self.phase1_fm = FunctionManagerPhase1(
                C=self.C,
                d=self.d,
                x0=self.x,
                lower_bound=self.lb,
                upper_bound=self.ub,
                t=self.t0,
                use_gpu=self.use_gpu,
                n=self.n,
                suppress_print=self.suppress_print,
            )
        else:
            self.phase1_fm = FunctionManagerSOCPPhase1(
                *socp_params,
                x0=self.x,
                lower_bound=self.lb,
                upper_bound=self.ub,
                t=self.t0,
                use_gpu=self.use_gpu,
                n=self.n,
                suppress_print=self.suppress_print,
            )

        if self.use_gpu:
            self.x = cp.append(self.x, self.phase1_fm.s)
        else:
            self.x = np.append(self.x, self.phase1_fm.s)

        self.phase1_ns = NewtonSolverCholesky(
            C=self.C,
            d=self.d,
            function_manager=self.phase1_fm,
            lower_bound=self.lb,
            upper_bound=self.ub,
            max_iters=self.max_inner_iters,
            epsilon=self.inner_epsilon,
            suppress_print=self.suppress_print,
            max_cg_iters=self.max_cg_iters,
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            use_gpu=self.use_gpu,
            track_loss=self.track_loss,
            phase1_flag=True,
            phase1_tol=self.tol,
            use_psd_condition=use_psd_condition,
            update_slacks_every=self.update_slacks_every
        )

    def solve(self, x0=None):

        if x0 is not None:
            self.phase1_fm.update_x(x0)

        t = self.t0

        self.outer_iters = 0
        self.inner_iters = []

        for iter in range(self.max_outer_iters):

            if not self.suppress_print:
                print(f"Current slack: {self.phase1_fm.s}")

            self.x, _, numiters_t, _, success_flag = self.phase1_ns.solve(self.x, t)

            self.outer_iters += 1
            self.inner_iters.append(numiters_t)

            obj_val = self.phase1_fm.objective(self.x)
            if obj_val < -self.tol:
                break

            # if self.track_loss:
            #    objective_vals.append(obj_val)
            # if obj_val < best_obj:
            #    best_obj = obj_val
            #    best_x = x.copy()
            # else:
            #    break

            if numiters_t >= self.max_inner_iters:
                if not self.suppress_print:
                    print(
                        f"Reached max Newton steps during {iter}th centering step (t={t}) of phase 1"
                    )

            # increment t for next outer iteration
            t = min(t * self.mu, (self.n + 1.0) / self.epsilon)
            self.phase1_fm.update_t(t)

        return self.x[:-1], obj_val
