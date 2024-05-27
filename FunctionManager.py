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
        sign=1,
        t=1,
        use_gpu=False,
        n=None,
    ):

        # problem specifications
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.c = c
        self.sign = sign

        self.x = x0
        self.update_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True

        self.slacks = None
        self.update_slacks = True

        self.obj = None
        self.grad = None
        self.hess = None
        self.inv_hess = None

        self.use_gpu = use_gpu and gpu_flag

        self.t = t
        if self.c is None:
            if n is None:
                raise ValueError(
                    "If no c vector is provided, you need to pass a value to n (problem dimension)"
                )
            if self.use_gpu:
                self.c = cp.ones(len(self.x))
            else:
                self.c = np.ones(len(self.x))

    def update_x(self, x):

        self.x = x
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True
        self.update_hessian = True
        self.update_inv_hessian = True
        self.update_slacks = True

    def update_t(self, t):

        self.t = t
        self.update_obj = True
        self.update_newton_obj = True
        self.update_grad = True

        # t does not affect hessian or inverse hessian for linear problems
        # self.update_hessian = True
        # self.update_inv_hessian = True

    def update_slacks_fxn(self):

        self.slacks = self.d - self.C @ self.x

    def objective(self, x=None):

        if x is not None:
            self.update_x(x)

        elif not self.update_obj:
            return self.obj

        if self.use_gpu:
            self.obj = cp.dot(self.c, self.x)
        else:
            self.obj = np.dot(self.c, self.x)
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


class FunctionManagerUnconstrained(FunctionManager):

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        self.grad = self.t * self.c
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        self.hess = 0
        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_inv_hessian:
            return self.inv_hess

        self.inv_hess = 0
        self.update_hessian = False

        return self.inv_hess

    def update_x(self, x):

        self.x = x
        self.update_obj = True
        self.update_newton_obj = True

        # don't need to update these for unconstrained problem
        # self.update_grad = True
        # self.update_hessian = True
        # self.update_inv_hessian = True


class FunctionManagerSigned(FunctionManager):

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
            if self.sign > 0:
                self.newton_obj -= cp.log(self.x).sum()
            else:
                self.newton_obj -= cp.log(-self.x).sum()
        else:
            if self.sign > 0:
                self.newton_obj -= np.log(self.x).sum()
            else:
                self.newton_obj -= np.log(-self.x).sum()
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        self.grad = self.t * self.c - 1 / self.x
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        self.hess = 1 / self.x**2
        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_inv_hessian:
            return self.inv_hess

        self.inv_hess = self.x**2
        self.update_hessian = False

        return self.inv_hess


class FunctionManagerInequalityConstrained(FunctionManager):

    def update_x(self, x):
        super().update_x(x)
        self.update_slacks_fxn()

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
            self.newton_obj -= cp.log(self.slacks).sum()
        else:
            self.newton_obj -= np.log(self.slacks).sum()
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        self.grad = self.t * self.c - self.C.T @ (1 / self.slacks)
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        self.hess = self.C.T @ (
            (1 / self.slacks**2)[:, None] * self.C
        )  # perform the max operation to add some conditioning to the hessian
        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        raise ValueError("Hessian is not diagonal, cannot use inv hessian function!")


class FunctionManagerConstrained(FunctionManager):

    def update_x(self, x):
        super().update_x(x)
        self.update_slacks_fxn()

    def newton_objective(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_newton_obj:
            return self.newton_obj

        obj = self.objective(x)
        self.newton_obj = self.t * obj
        if self.use_gpu:
            self.newton_obj -= cp.log(self.slacks).sum()
            if self.sign > 0:
                self.newton_obj -= cp.log(self.x).sum()
            else:
                self.newton_obj -= cp.log(-self.x).sum()
        else:
            self.newton_obj -= np.log(self.slacks).sum()
            if self.sign > 0:
                self.newton_obj -= np.log(self.x).sum()
            else:
                self.newton_obj -= np.log(-self.x).sum()
        self.update_newton_obj = False

        return self.newton_obj

    def gradient(self, x=None, t=None):

        if x is not None:
            self.update_x(x)

        if t is not None:
            self.update_t(t)

        if not self.update_grad:
            return self.grad

        self.grad = self.t * self.c - self.C.T @ (1 / self.slacks) - 1 / self.x
        self.update_grad = False

        return self.grad

    def hessian(self, x=None):

        if x is not None:
            self.update_x(x)

        if not self.update_hessian:
            return self.hess

        self.hess = self.C.T @ ((1 / self.slacks**2)[:, None] * self.C)
        diag = np.einsum("ii->i", self.hess)
        diag += 1 / self.x**2
        diag += 0.01  # add some conditioning
        self.update_hessian = False

        return self.hess

    def inv_hessian(self, x=None):

        raise ValueError("Hessian is not diagonal, cannot use inv hessian function!")
