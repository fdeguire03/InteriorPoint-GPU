import numpy as np

try:
    import cupy as cp

    gpu_flag = True
except Exception:
    gpu_flag = False


class FunctionManager:

    def __init__(
        self,
        c=None,
        A=None,
        b=None,
        C=None,
        d=None,
        x0=None,
        lower_bound=None,
        upper_bound=None,
        t=1,
        use_gpu=False,
        n=None,
        try_diag=True,
    ):

        # problem specifications
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.c = c
        self.lb = lower_bound
        self.ub = upper_bound

        self.x = x0
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True

        self.update_slacks = True
        self.update_inv_slacks = True
        self.is_bounded = self.ub is not None or self.lb is not None
        self.is_constrained = self.C is not None or self.is_bounded

        start_index = 0
        end_index = 0
        if self.C is not None:
            end_index = start_index + len(self.C)
            self.inequality_slack_indices = slice(start_index, len(self.C))
            start_index += len(self.C)
        if self.ub is not None:
            end_index = start_index + len(self.x)
            self.ub_slack_indices = slice(start_index, end_index)
            start_index = end_index
        if self.lb is not None:
            end_index = start_index + len(self.x)
            self.lb_slack_indices = slice(start_index, end_index)

        self.obj = None
        self.newton_obj = None
        self.grad = None
        self.hess = None
        self.inv_hess = None
        self.slacks = None
        self.inv_slacks = None

        self.use_gpu = use_gpu and gpu_flag
        self.try_diag = try_diag

        self.t = t
        if self.c is None:
            if self.x is not None:
                if self.use_gpu:
                    self.c = cp.ones(len(self.x))
                else:
                    self.c = np.ones(len(self.x))
            elif n is None:

                raise ValueError(
                    "If no x vector is provided, you need to pass a value to n (problem dimension)"
                )

            else:
                if self.use_gpu:
                    self.c = cp.ones(n)
                else:
                    self.c = np.ones(n)

    def update_x(self, x):

        self.x = x
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True
        self.update_slacks = True
        self.update_inv_slacks = True

    def update_t(self, t):

        if self.t != t:
            self.t = t
            self.update_obj = True
            self.update_newton_obj = True
            self.update_grad = True
            # self.update_slacks = True

        # t does not affect hessian or inverse hessian for linear problems
        # self.update_hessian = True
        # self.update_inv_hessian = True

    def update_slacks_fxn(self):

        initialized = False
        if self.d is not None:
            if self.use_gpu:
                self.slacks = self.d - cp.matmul(self.C, self.x)
            else:
                self.slacks = self.d - np.matmul(self.C, self.x)
            initialized = True
        if self.ub is not None:
            ub_slacks = self.ub - self.x
            if initialized:
                if self.use_gpu:
                    self.slacks = cp.append(self.slacks, ub_slacks)
                else:
                    self.slacks = np.append(self.slacks, ub_slacks)
            else:
                self.slacks = ub_slacks
                initialized = True
        if self.lb is not None:
            lb_slacks = self.x - self.lb
            if initialized:
                if self.use_gpu:
                    self.slacks = cp.append(self.slacks, lb_slacks)
                else:
                    self.slacks = np.append(self.slacks, lb_slacks)
            else:
                self.slacks = lb_slacks
        if self.slacks.ndim > 1:
            self.slacks = self.slacks.flatten()
        self.update_inv_slacks = True
        self.update_slacks = False

    def objective(self, x=None):

        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        self.obj = self.c.dot(self.x)
        self.update_obj = False

        return self.obj

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective()
        self.newton_obj = self.t * obj
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None):
        raise NotImplementedError(
            "Gradient method needs to be overridden by a child class!"
        )

    def hessian(self, x=None):
        raise NotImplementedError(
            "Hessian method needs to be overridden by a child class!"
        )

    def inv_hessian(self, x=None):
        raise NotImplementedError(
            "Hessian method needs to be overridden by a child class!"
        )


class FunctionManagerLP(FunctionManager):

    def update_x(self, x, update_slacks=True):
        super().update_x(x)
        if self.is_constrained:
            if update_slacks:
                self.update_slacks_fxn()
            else:
                self.update_slacks = True
                self.update_inv_slacks = True

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective(x)
        self.newton_obj = self.t * obj
        if self.is_constrained:
            if self.use_gpu:
                self.newton_obj -= cp.log(
                    self.slacks + 1e-15
                ).sum()  # small addition for numerical stability
            else:
                self.newton_obj -= np.log(self.slacks + 1e-15).sum()
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        self.grad = self.t * self.c
        if self.lb is not None:
            self.grad -= self.inv_slacks[self.lb_slack_indices]
        if self.ub is not None:
            self.grad += self.inv_slacks[self.ub_slack_indices]
        if self.C is not None:
            if self.use_gpu:
                self.grad += cp.matmul(
                    self.C.T, self.inv_slacks[self.inequality_slack_indices]
                )
            else:
                self.grad += np.matmul(
                    self.C.T, self.inv_slacks[self.inequality_slack_indices]
                )
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        if self.C is None:

            if self.try_diag and self.is_bounded:

                if self.lb is not None:
                    self.hess = self.inv_slacks[self.lb_slack_indices] ** 2
                    if self.ub is not None:
                        self.hess += self.inv_slacks[self.ub_slack_indices] ** 2
                elif self.ub is not None:
                    self.hess = self.inv_slacks[self.ub_slack_indices] ** 2

                return self.hess

            else:
                if self.use_gpu:
                    self.hess = cp.zeros((self.x.shape[0], self.x.shape[0]))
                else:
                    self.hess = np.zeros((self.x.shape[0], self.x.shape[0]))

        else:
            if self.use_gpu:
                self.hess = cp.matmul(
                    self.C.T,
                    (self.inv_slacks[self.inequality_slack_indices] ** 2)[:, None]
                    * self.C,
                )
            else:
                self.hess = np.matmul(
                    self.C.T,
                    (self.inv_slacks[self.inequality_slack_indices] ** 2)[:, None]
                    * self.C,
                )

        if self.is_bounded:
            if self.use_gpu:
                diag = cp.einsum("ii->i", self.hess)
            else:
                diag = np.einsum("ii->i", self.hess)
            if self.lb is not None:
                diag += 1 / (self.slacks[self.lb_slack_indices]) ** 2
            if self.ub is not None:
                diag += 1 / (self.slacks[self.ub_slack_indices]) ** 2

        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_inv_hessian:
            return self.inv_hess

        if self.C is None and self.try_diag:

            if self.is_bounded:

                hess = self.hessian()
                self.inv_hess = 1 / hess

            else:
                if self.use_gpu:
                    self.hess = cp.zeros((self.x.shape[0], self.x.shape[0]))
                else:
                    self.hess = np.zeros((self.x.shape[0], self.x.shape[0]))

        else:
            raise ValueError(
                "Hessian is not diagonal, cannot use inv hessian function!"
            )

        self.update_inv_hessian = False

        return self.inv_hess


class FunctionManagerPhase1(FunctionManager):

    def __init__(
        self,
        c=None,
        A=None,
        b=None,
        C=None,
        d=None,
        x0=None,
        lower_bound=None,
        upper_bound=None,
        t=1,
        use_gpu=False,
        n=None,
        try_diag=False,
        suppress_print=True,
    ):

        # problem specifications
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.c = c
        self.x = x0
        self.lb = lower_bound
        self.ub = upper_bound

        self.use_gpu = use_gpu

        self.s = 0
        self.update_slacks_fxn()
        self.s = -self.slacks.min() + 1
        self.update_slacks_fxn()
        if not suppress_print:
            print(f"Starting slack of {round(self.s, 4)}")

        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True

        self.update_slacks = False
        self.update_inv_slacks = True
        self.is_bounded = self.ub is not None or self.lb is not None
        self.is_constrained = self.C is not None or self.is_bounded
        self.inequality_slack_indices = slice(0, len(self.C))
        start_index = len(self.C)
        if self.ub is not None:
            end_index = start_index + len(self.x)
            self.ub_slack_indices = slice(start_index, end_index)
            start_index = end_index
        if self.lb is not None:
            end_index = start_index + len(self.x)
            self.lb_slack_indices = slice(start_index, end_index)

        self.obj = None
        self.newton_obj = None
        self.grad = None
        self.hess = None
        self.inv_hess = None
        self.inv_slacks = None

        self.t = t

    def update_slacks_fxn(self):

        if self.use_gpu:
            self.slacks = self.s + self.d - cp.matmul(self.C, self.x)
        else:
            self.slacks = self.s + self.d - np.matmul(self.C, self.x)

        if self.ub is not None:
            if self.use_gpu:
                self.slacks = cp.append(self.slacks, self.s + self.ub - self.x)
            else:
                self.slacks = np.append(self.slacks, self.s + self.ub - self.x)
        if self.lb is not None:
            if self.use_gpu:
                self.slacks = cp.append(self.slacks, self.s + self.x - self.lb)
            else:
                self.slacks = np.append(self.slacks, self.s + self.x - self.lb)
        if self.slacks.ndim > 1:
            self.slacks = self.slacks.flatten()
        self.update_inv_slacks = True
        self.update_slacks = False

    def update_x(self, x, update_slacks=True):

        if len(x) == len(self.x) + 1:
            self.x = x[:-1]
            self.s = x[-1]
        elif len(x) == len(self.x):
            self.x = x
        else:
            raise ValueError("Provided x does not have the right dimensions!")

        if update_slacks:
            self.update_slacks_fxn()
        else:
            self.update_slacks = True
            self.update_inv_slacks = True
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True

    def objective(self, x=None):
        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        self.obj = self.s
        self.update_obj = False

        return self.obj

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective()
        self.newton_obj = self.t * obj
        if self.use_gpu:
            self.newton_obj -= cp.log(
                self.slacks + 1e-15
            ).sum()  # small addition for numerical stability
        else:
            self.newton_obj -= np.log(
                self.slacks + 1e-15
            ).sum()  # small addition for numerical stability
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        if self.use_gpu:
            grad_x = cp.matmul(self.C.T, self.inv_slacks[self.inequality_slack_indices])
        else:
            grad_x = np.matmul(self.C.T, self.inv_slacks[self.inequality_slack_indices])

        if self.lb is not None:
            grad_x -= self.inv_slacks[self.lb_slack_indices]
        if self.ub is not None:
            grad_x += self.inv_slacks[self.ub_slack_indices]

        grad_s = self.t - self.inv_slacks.sum()

        if self.use_gpu:
            self.grad = cp.append(grad_x, grad_s)
        else:
            self.grad = np.append(grad_x, grad_s)

        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        inv_slacks_squared = self.inv_slacks**2

        if self.use_gpu:
            hess_xx = cp.matmul(
                self.C.T,
                (inv_slacks_squared[self.inequality_slack_indices])[:, None] * self.C,
            )
            hess_xs = -cp.matmul(
                self.C.T, inv_slacks_squared[self.inequality_slack_indices]
            )
            diag = cp.einsum("ii->i", hess_xx)
        else:
            hess_xx = np.matmul(
                self.C.T,
                (inv_slacks_squared[self.inequality_slack_indices])[:, None] * self.C,
            )
            hess_xs = -np.matmul(
                self.C.T, inv_slacks_squared[self.inequality_slack_indices]
            )
            diag = np.einsum("ii->i", hess_xx)

        if self.lb is not None:
            diag += inv_slacks_squared[self.lb_slack_indices]
            hess_xs += inv_slacks_squared[self.lb_slack_indices]
        if self.ub is not None:
            diag += inv_slacks_squared[self.ub_slack_indices]
            hess_xs -= inv_slacks_squared[self.ub_slack_indices]

        hess_ss = inv_slacks_squared.sum()

        if self.use_gpu:
            self.hess = cp.bmat(
                [
                    [hess_xx, hess_xs.reshape(-1, 1)],
                    [hess_xs.reshape(1, -1), cp.array(hess_ss).reshape(1, 1)],
                ],
            )
        else:
            self.hess = np.bmat(
                [
                    [hess_xx, hess_xs.reshape(-1, 1)],
                    [hess_xs.reshape(1, -1), np.array(hess_ss).reshape(1, 1)],
                ],
            )

        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):
        raise ValueError(
            "Hessian is not diagonal, so inverse hessian cannot be directly computed!!"
        )


class FunctionManagerQP(FunctionManager):

    def __init__(
        self,
        P=None,
        q=None,
        A=None,
        b=None,
        C=None,
        d=None,
        x0=None,
        lower_bound=None,
        upper_bound=None,
        t=1,
        use_gpu=False,
        n=None,
    ):

        # problem specifications
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.lb = lower_bound
        self.ub = upper_bound
        self.P = P
        self.q = q

        self.x = x0
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True

        self.slacks = None
        self.update_slacks = True
        self.update_inv_slacks = True
        self.is_bounded = self.ub is not None or self.lb is not None
        self.is_constrained = self.C is not None or self.is_bounded

        start_index = 0
        if self.C is not None:
            self.inequality_slack_indices = slice(start_index, len(self.C))
            start_index += len(self.C)
        if self.ub is not None:
            end_index = start_index + len(self.x)
            self.ub_slack_indices = slice(start_index, end_index)
            start_index = end_index
        if self.lb is not None:
            end_index = start_index + len(self.x)
            self.lb_slack_indices = slice(start_index, end_index)

        self.obj = None
        self.newton_obj = None
        self.grad = None
        self.hess = None
        self.inv_hess = None
        self.slacks = None
        self.inv_slacks = None

        self.use_gpu = use_gpu and gpu_flag

        self.t = t

    def objective(self, x=None):

        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        self.obj = 0
        if self.use_gpu:
            if self.P is not None:
                self.obj += 1 / 2 * self.x.dot(cp.matmul(self.P, self.x))
            if self.q is not None:
                self.obj += self.q.dot(self.x)
        else:
            if self.P is not None:
                self.obj += 1 / 2 * self.x.dot(np.matmul(self.P, self.x))
            if self.q is not None:
                self.obj += self.q.dot(self.x)
        self.update_obj = False

        return self.obj

    def update_x(self, x, update_slacks=True):
        super().update_x(x)
        if self.is_constrained:
            if update_slacks:
                self.update_slacks_fxn()
            else:
                self.update_slacks = True
                self.update_inv_slacks = True

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective(x)
        self.newton_obj = self.t * obj
        if self.is_constrained:
            if self.use_gpu:
                self.newton_obj -= cp.log(
                    self.slacks + 1e-15
                ).sum()  # small addition for numerical stability
            else:
                self.newton_obj -= np.log(
                    self.slacks + 1e-15
                ).sum()  # small addition for numerical stability
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        if self.use_gpu:
            self.grad = cp.matmul(self.P, self.x)
        else:
            self.grad = np.matmul(self.P, self.x)
        if self.q is not None:
            self.grad += self.q
        self.grad *= self.t

        if self.lb is not None:
            self.grad -= self.inv_slacks[self.lb_slack_indices]
        if self.ub is not None:
            self.grad += self.inv_slacks[self.ub_slack_indices]
        if self.C is not None:
            if self.use_gpu:
                self.grad += cp.matmul(
                    self.C.T, self.inv_slacks[self.inequality_slack_indices]
                )
            else:
                self.grad += np.matmul(
                    self.C.T, self.inv_slacks[self.inequality_slack_indices]
                )
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        self.hess = self.t * self.P

        if self.C is not None:
            if self.use_gpu:
                self.hess = cp.matmul(
                    self.C.T,
                    (self.inv_slacks[self.inequality_slack_indices] ** 2)[:, None]
                    * self.C,
                )
                if self.is_bounded:
                    diag = cp.einsum("ii->i", self.hess)
            else:
                self.hess = np.matmul(
                    self.C.T,
                    (self.inv_slacks[self.inequality_slack_indices] ** 2)[:, None]
                    * self.C,
                )

        if self.is_bounded:
            if self.use_gpu:
                diag = cp.einsum("ii->i", self.hess)
            else:
                diag = np.einsum("ii->i", self.hess)
            if self.lb is not None:
                diag += 1 / (self.slacks[self.lb_slack_indices]) ** 2
            if self.ub is not None:
                diag += 1 / (self.slacks[self.ub_slack_indices]) ** 2

        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        raise ValueError("Hessian is not diagonal, cannot use inv hessian function!")


class FunctionManagerSOCP(FunctionManager):

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
        lower_bound=None,
        upper_bound=None,
        x0=None,
        t=1,
        use_gpu=False,
        n=None,
    ):

        # problem specifications
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

        self.use_gpu = use_gpu

        self.AtA_cache = []
        self.Atb_cache = []
        for i, A in enumerate(self.A):
            if self.b is not None:
                b = self.b[i]
            if A.ndim > 1:
                if self.use_gpu:
                    self.AtA_cache.append(cp.matmul(A.T, A))
                    if self.b is not None:
                        self.Atb_cache.append(cp.matmul(A.T, b))
                else:
                    self.AtA_cache.append(np.matmul(A.T, A))
                    if self.b is not None:
                        self.Atb_cache.append(np.matmul(A.T, b))
            else:
                if self.use_gpu:
                    self.AtA_cache.append(cp.diag(A**2))
                else:
                    self.AtA_cache.append(np.diag(A**2))
                if self.b is not None:
                    self.Atb_cache.append(A * b)
        if self.c is not None:
            if self.use_gpu:
                self.cct_cache = [cp.outer(c, c) for c in self.c]
            else:
                self.cct_cache = [np.outer(c, c) for c in self.c]

        self.x = x0
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True

        self.slacks = None
        self.slack_lhs = None
        self.slack_rhs = None
        self.update_slacks = True
        self.update_inv_slacks = True
        self.is_bounded = self.ub is not None or self.lb is not None
        self.is_constrained = self.A is not None or self.is_bounded

        start_index = 0
        end_index = len(self.A)
        self.inequality_slack_indices = slice(start_index, end_index)
        start_index = end_index
        if self.ub is not None:
            end_index = start_index + len(self.x)
            self.ub_slack_indices = slice(start_index, end_index)
            start_index = end_index
        if self.lb is not None:
            end_index = start_index + len(self.x)
            self.lb_slack_indices = slice(start_index, end_index)
        self.constraint_indices = slice(0, end_index)  # we will add some additional
        # variables to the end of the slacks list to force them to be nonnegative.
        # Remember the indices for all constraints for gradient and hessian calculations

        self.obj = None
        self.newton_obj = None
        self.grad = None
        self.hess = None
        self.inv_slacks = None

        self.t = t

    def update_slacks_fxn(self):

        inside_norms = []
        for A in self.A:
            if A.ndim > 1:
                if self.use_gpu:
                    inside_norms.append(cp.matmul(A, self.x))
                else:
                    inside_norms.append(np.matmul(A, self.x))
            else:
                inside_norms.append(A * self.x)

        if self.b is not None:
            for i in range(len(inside_norms)):
                inside_norms[i] += self.b[i]

        rhss = None
        if self.c is not None:
            rhss = [c.dot(self.x) for c in self.c]
            if self.d is not None:
                for i in range(len(rhss)):
                    rhss[i] += self.d[i]
        elif self.d is not None:
            rhss = self.d
        else:
            rhss = 0

        self.slack_lhs = inside_norms
        self.slack_rhs = rhss
        if self.use_gpu:
            self.slacks = cp.array(
                [
                    rhs**2 - (inside_norm**2).sum()
                    for rhs, inside_norm in zip(self.slack_rhs, self.slack_lhs)
                ]
            )
        else:
            self.slacks = np.array(
                [
                    rhs**2 - (inside_norm**2).sum()
                    for rhs, inside_norm in zip(self.slack_rhs, self.slack_lhs)
                ]
            )

        if self.use_gpu:
            if self.ub is not None:
                self.slacks = cp.append(self.slacks, self.ub - self.x)
            if self.lb is not None:
                self.slacks = cp.append(self.slacks, self.x - self.lb)
            self.slacks = cp.append(self.slacks, self.slack_rhs)
        else:
            if self.ub is not None:
                self.slacks = np.append(self.slacks, self.ub - self.x)
            if self.lb is not None:
                self.slacks = np.append(self.slacks, self.x - self.lb)
            self.slacks = np.append(self.slacks, self.slack_rhs)

        if self.slacks.ndim > 1:
            self.slacks = self.slacks.flatten()

        self.update_slacks = False
        self.update_inv_slacks = True

    def objective(self, x=None):

        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        self.obj = 0

        if self.use_gpu:
            if self.P is not None:
                self.obj += 1 / 2 * self.x.dot(cp.matmul(self.P, self.x))
            if self.q is not None:
                self.obj += self.q.dot(self.x)
        else:
            if self.P is not None:
                self.obj += 1 / 2 * self.x.dot(np.matmul(self.P, self.x))
            if self.q is not None:
                self.obj += self.q.dot(self.x)
        self.update_obj = False

        return self.obj

    def update_x(self, x, update_slacks=True):
        super().update_x(x)
        if self.is_constrained:
            if update_slacks:
                self.update_slacks_fxn()
            else:
                self.update_slacks = True
                self.updTE_inv_slacks = True

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective(x)
        self.newton_obj = self.t * obj
        if self.is_constrained:
            if self.use_gpu:
                self.newton_obj -= cp.log(
                    self.slacks[self.constraint_indices] + 1e-15
                ).sum()  # small addition for numerical stability
            else:
                self.newton_obj -= np.log(
                    self.slacks[self.constraint_indices] + 1e-15
                ).sum()  # small addition for numerical stability
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        self.grad = 0
        if self.P is not None:
            if self.use_gpu:
                self.grad = cp.matmul(self.P, self.x)
            else:
                self.grad = np.matmul(self.P, self.x)
        if self.q is not None:
            self.grad += self.q
        self.grad *= self.t

        for i, (s, rhs, lhs) in enumerate(
            zip(
                self.slacks[self.inequality_slack_indices],
                self.slack_rhs,
                self.slack_lhs,
            )
        ):
            if self.c is not None:
                self.grad -= 2 * self.c[i] * rhs / (s + 1e-12)
            if self.A is not None:
                A = self.A[i]
                if A.ndim > 1:
                    if self.use_gpu:
                        self.grad += 2 * cp.matmul(A.T, lhs) / (s + 1e-12)
                    else:
                        self.grad += 2 * np.matmul(A.T, lhs) / (s + 1e-12)
                else:
                    self.grad += 2 * A * lhs / (s + 1e-12)

        if self.lb is not None:
            self.grad -= 1 / (self.slacks[self.lb_slack_indices] + 1e-15)
        if self.ub is not None:
            self.grad += 1 / (self.slacks[self.ub_slack_indices] + 1e-15)

        self.update_grad = False

        return self.grad

    def hessian(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_hessian:
            return self.hess

        self.hess = 0
        if self.P is not None:
            self.hess += self.t * self.P

        for i, (A, s) in enumerate(
            zip(self.A, self.slacks[self.inequality_slack_indices])
        ):

            s_hess = 0
            if A.ndim < 2:
                s_grad_term = A * self.slack_lhs[i]
            else:
                if self.use_gpu:
                    s_grad_term = cp.matmul(A.T, self.slack_lhs[i])
                else:
                    s_grad_term = np.matmul(A.T, self.slack_lhs[i])

            s_hess += self.AtA_cache[i]
            if self.c is not None:
                s_hess -= self.cct_cache[i]
                s_grad_term -= self.c[i] * self.slack_rhs[i]
            s_hess *= 2 / (s + 1e-12)
            s_grad_term *= 2 / (s + 1e-12)

            if self.use_gpu:
                s_hess += cp.outer(s_grad_term, s_grad_term)
            else:
                s_hess += np.outer(s_grad_term, s_grad_term)

            self.hess += s_hess

        if self.is_bounded:
            if self.use_gpu:
                diag = cp.einsum("ii->i", self.hess)
            else:
                diag = np.einsum("ii->i", self.hess)
            if self.lb is not None:
                diag += 1 / (self.slacks[self.lb_slack_indices] + 1e-12) ** 2
            if self.ub is not None:
                diag += 1 / (self.slacks[self.ub_slack_indices] + 1e-12) ** 2

        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        raise ValueError("Hessian is not diagonal, cannot use inv hessian function!")


class FunctionManagerSOCPPhase1(FunctionManagerSOCP):

    def __init__(
        self,
        A=None,
        b=None,
        c=None,
        d=None,
        x0=None,
        lower_bound=None,
        upper_bound=None,
        t=1,
        use_gpu=False,
        n=None,
        suppress_print=True,
    ):

        # problem specifications
        self.A = A
        self.b = b
        self.c = c
        self.d = d
        self.x = x0
        self.lb = lower_bound
        self.ub = upper_bound

        self.use_gpu = use_gpu

        self.AtA_cache = []
        self.Atb_cache = []
        for i, A in enumerate(self.A):
            if self.b is not None:
                b = self.b[i]
            if A.ndim > 1:
                if self.use_gpu:
                    self.AtA_cache.append(cp.matmul(A.T, A))
                    if self.b is not None:
                        self.Atb_cache.append(cp.matmul(A.T, b))
                else:
                    self.AtA_cache.append(np.matmul(A.T, A))
                    if self.b is not None:
                        self.Atb_cache.append(np.matmul(A.T, b))
            else:
                self.AtA_cache.append(np.diag(A**2))
                if self.b is not None:
                    self.Atb_cache.append(A * b)
        if self.c is not None:
            if self.use_gpu:
                self.cct_cache = [cp.outer(c, c) for c in self.c]
            else:
                self.cct_cache = [np.outer(c, c) for c in self.c]

        start_index = 0
        end_index = len(self.A)
        self.inequality_slack_indices = slice(start_index, end_index)
        start_index = end_index
        if self.ub is not None:
            end_index = start_index + len(self.x)
            self.ub_slack_indices = slice(start_index, end_index)
            start_index = end_index
        if self.lb is not None:
            end_index = start_index + len(self.x)
            self.lb_slack_indices = slice(start_index, end_index)
        self.constraint_indices = slice(0, end_index)  # we will add some additional
        # variables to the end of the slacks list to force them to be nonnegative.
        # Remember the indices for all constraints for gradient and hessian calculations

        self.s = 0
        self.update_slacks_fxn()
        self.s = -self.slacks.min() + 1
        self.update_slacks_fxn()
        if not suppress_print:
            print(f"Starting slack of {round(self.s, 4)}")

        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True

        self.update_slacks = True
        self.update_inv_slacks = True
        self.is_bounded = self.ub is not None or self.lb is not None
        self.is_constrained = True

        self.obj = None
        self.newton_obj = None
        self.grad = None
        self.hess = None

        self.inv_slacks = None

        self.t = t

    def update_slacks_fxn(self):

        super().update_slacks_fxn()
        self.slacks[self.constraint_indices] += self.s
        self.update_inv_slacks = True

    def update_x(self, x, update_slacks=True):

        if len(x) == len(self.x) + 1:
            self.x = x[:-1]
            self.s = x[-1]
        elif len(x) == len(self.x):
            self.x = x
        else:
            raise ValueError("Provided x does not have the right dimensions!")

        if update_slacks:
            self.update_slacks_fxn()
        else:
            self.update_slacks = True
            self.updTE_inv_slacks = True
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True

    def objective(self, x=None):

        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        self.obj = self.s
        self.update_obj = False

        return self.obj

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective()
        self.newton_obj = self.t * obj
        if self.use_gpu:
            self.newton_obj -= cp.log(
                self.slacks[self.constraint_indices] + 1e-15
            ).sum()  # small addition for numerical stability
        else:
            self.newton_obj -= np.log(
                self.slacks[self.constraint_indices] + 1e-15
            ).sum()  # small addition for numerical stability
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks[self.constraint_indices] + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        grad_x = 0
        for i, (invs, rhs, lhs) in enumerate(
            zip(
                self.inv_slacks[self.inequality_slack_indices],
                self.slack_rhs,
                self.slack_lhs,
            )
        ):
            if self.c is not None:
                grad_x -= 2 * self.c[i] * rhs * invs
            if self.A is not None:
                A = self.A[i]
                if A.ndim > 1:
                    if self.use_gpu:
                        grad_x += 2 * cp.matmul(A.T, lhs) * invs
                    else:
                        grad_x += 2 * np.matmul(A.T, lhs) * invs
                else:
                    grad_x += 2 * A * lhs * invs

        if self.lb is not None:
            grad_x -= self.inv_slacks[self.lb_slack_indices]
        if self.ub is not None:
            grad_x += self.inv_slacks[self.ub_slack_indices]

        grad_s = self.t - self.inv_slacks.sum()

        if self.use_gpu:
            self.grad = cp.append(grad_x, grad_s)
        else:
            self.grad = np.append(grad_x, grad_s)

        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        if not self.update_grad:
            self.gradient()  # need updated gradient for calculations here

        if self.is_constrained and self.update_inv_slacks:
            self.inv_slacks = 1 / (
                self.slacks[self.constraint_indices] + 1e-15
            )  # small addition for numerical stability
            self.update_inv_slacks = False

        inv_slacks_squared = self.inv_slacks**2
        hess_xx = 0
        hess_xs = 0
        for i, (A, invs) in enumerate(
            zip(self.A, self.inv_slacks[self.inequality_slack_indices])
        ):

            s_hess = 0
            if A.ndim < 2:
                s_grad_term = A * self.slack_lhs[i]
            else:
                if self.use_gpu:
                    s_grad_term = cp.matmul(A.T, self.slack_lhs[i])
                else:
                    s_grad_term = np.matmul(A.T, self.slack_lhs[i])
            s_hess += self.AtA_cache[i]
            if self.c is not None:
                s_hess -= self.cct_cache[i]
                s_grad_term -= self.c[i] * self.slack_rhs[i]
            s_hess *= 2 * invs
            s_grad_term *= 2 * invs

            hess_xs -= s_grad_term * invs

            if self.use_gpu:
                s_hess += cp.outer(s_grad_term, s_grad_term)
            else:
                s_hess += np.outer(s_grad_term, s_grad_term)

            hess_xx += s_hess

        if self.is_bounded:
            if self.use_gpu:
                diag = cp.einsum("ii->i", hess_xx)
            else:
                diag = np.einsum("ii->i", hess_xx)
            if self.lb is not None:
                diag += inv_slacks_squared[self.lb_slack_indices]
                hess_xs += inv_slacks_squared[self.lb_slack_indices]
            if self.ub is not None:
                diag += inv_slacks_squared[self.ub_slack_indices]
                hess_xs -= inv_slacks_squared[self.ub_slack_indices]

        hess_ss = inv_slacks_squared.sum()

        if self.use_gpu:
            self.hess = cp.bmat(
                [
                    [hess_xx, hess_xs.reshape(-1, 1)],
                    [hess_xs.reshape(1, -1), cp.array(hess_ss).reshape(1, 1)],
                ],
            )
        else:
            self.hess = np.bmat(
                [
                    [hess_xx, hess_xs.reshape(-1, 1)],
                    [hess_xs.reshape(1, -1), np.array(hess_ss).reshape(1, 1)],
                ],
            )

        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):
        raise ValueError(
            "Hessian is not diagonal, so inverse hessian cannot be directly computed!!"
        )
