import numpy as np
from numpy.linalg import solve as npsolve
from scipy.sparse.linalg import cg as spcg

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as cpcg
    from cupy.linalg import solve as cpsolve

    gpu_flag = True
except Exception:
    gpu_flag = False
    # Alias
    cp = np
    print("Not able to run tests with GPU")

class phase_one():
    """
    Class that executes phase one of an interior point method. Given a matrix G, 
    vector h and hyperparameter mu, 'phase_one' finds an point in the interior of 
    Gx <= h if such a point exists. The point is found using the interior point 
    method, solving the optimization problem

    minimize s over s and x
    subject to G @ x - h <= s
    """
    
    def __init__(
        self, 
        G,
        h,
        mu,
        x0 = None,
        eps = 1e-8,
        max_iter_interior = 200,
        max_iter_newton = 200,
        use_cupy = False,
        linear_solver = "solve",
        max_cg_iters = 50
        ):
        """
        Initialize the 'phase_one' object. 

        --inputs: G - numpy matrix used in G @ x <= h
                  h - numpy vector
                  mu - hyperparameter which increases barrier parameter t
                  max_iter_interior - sets how many times barrier paramter t may be increased
                  max_iter_newton - sets how many newtons steps may be taken when centering
                  use_cupy - whether to use cupy or nor
                  linear_solver - string that decides how the newton direction is calculated
                                  can be: 'solve' for np/cp.linalg.solve,
                                          'cg' for conjugate gradient

        """

        self.use_cupy = use_cupy

        # Check if cupy should be use but not available
        if self.use_cupy:
          if not gpu_flag:
            # Raise error
            raise RuntimeError("Tried using cupy without it being available")
      
        if self.use_cupy:
          # Store as cupy matrices
          self.G = cp.array(G)
          self.h = cp.array(h)
        else:
          # Store as numpy matrices
          self.G = G  
          self.h = h

        # Initialize x and s
        m, n = self.G.shape

        # x can be whatever
        # since x >= 0 usually is included, might as well initialize accordingly
        if self.use_cupy:
          if x0 is not None:
            self.x = cp.array(x0)
          else:
            self.x = cp.ones(n)

          # s must be feasible, but this can be chosen
          self.s = cp.max(self.G @ self.x - self.h) + 1
        else:
          # x
          if x0 is not None:
            self.x = x0
          else:
            self.x = np.ones(n)

          # s
          self.s = np.max(self.G @ self.x - self.h) + 1 

        # Left out variables
        self.mu = mu
        self.eps = eps
        self.max_iter_interior = max_iter_interior
        self.max_iter_newton = max_iter_newton
        self.solver = linear_solver
        if self.solver not in ['solve', 'cg']:
          raise RuntimeError("Invalid linear solver")
        self.max_iter_cg = max_cg_iters
        self.warn = False

    def phase_one_newtons_method(self, t):
        """
        Performs newtons method for phase one, calculating a newton direction and then
        using line search. 

        -- inputs: t - barrier parameter

        -- outputs: bool - boolean, True if maximum allowed steps was taken, False otherwise
        """

        for newton_iteration in range(self.max_iter_newton):
            m, n = self.G.shape

            # Calculate phase_one_hessian and phase_one_gradient
            if self.use_cupy:
              hess = self.phase_one_hessian() + 0.01 * cp.eye(n + 1) # some conditioning
            else:
              hess = self.phase_one_hessian() + 0.01 * np.eye(n + 1)
            
            grad = self.phase_one_gradient(t)

            # This can probably be done way quicker with Cupy
            if self.use_cupy:
              if self.solver == "solve":
                newton_direction = cpsolve(hess, -grad)
              else:
                newton_direction, _ = cpcg(hess, -grad, 
                    x0 = cp.hstack([self.x, self.s]), maxiter = self.max_iter_cg)
            else:
              if self.solver == "solve":
                newton_direction = npsolve(hess, -grad)
              else:
                newton_direction, _ = spcg(hess, -grad, 
                    x0 = np.hstack([self.x, self.s]), maxiter = self.max_iter_cg)

            lambda_sq = - grad @ newton_direction

            if lambda_sq / 2 <= self.eps:
              break

            # linesearch
            step = self.phase_one_linesearch(t, newton_direction, grad)

            # update
            self.x = self.x + step * newton_direction[:-1]
            self.s = self.s + step * newton_direction[-1]

            # is s < 0, can break already
            if self.s < 0:
              break

        return newton_iteration == self.max_iter_newton - 1

    def phase_one_objective(self, x, s, t):
        """
        Calculates the objective value for phase one

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                t - barrier parameter

        -- output: val - objective value
        """
        if self.use_cupy and gpu_flag:
          val = t * s - cp.sum(cp.log(s + self.h - self.G @ x))
        else:
          val = t * s - np.sum(np.log(s + self.h - self.G @ x))

        return val

    def phase_one_linesearch(self, t, direction, grad, alpha = 0.2, beta = 0.7):
        """
        Performs a linesearch in the given direction.

        -- inputs: t - barrier parameter
                direction - given direction to take a step in
                grad - gradient at the current point
                alpha - hyperparameter of linesearch
                beta - hyperparamter of linesearch

        -- output: step - how long the step in the given direction should be
        """

        step = 1

        # Check feasiblity
        while not self.phase_one_check_feasibility(self.x + step * direction[:-1], 
                          self.s + step * direction[-1]):
            step *= beta

        # Now line search for descent direction
        while self.phase_one_objective(self.x + step * direction[:-1], self.s + step * direction[-1], t) > \
              self.phase_one_objective(self.x, self.s, t) + alpha * step * grad @ direction:
            # Decrease step length
            step *= beta
            
        return step

    def phase_one_check_feasibility(self, x, s):
        """
        Checks if the combination of x and s is allowed. 
        Gx - h <= s must always hold

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized

        -- outputs: boolean that is False if infeasible and True if feasible

        """
        if self.use_cupy:
          bool = cp.max(self.G @ x - self.h) < s
        else:
          bool = np.max(self.G @ x - self.h) < s

        return bool

    def phase_one_gradient(self, t):
        """
        Calculates the gradient for the objective value with log-barrier
        of the phase one method

        -- inputs: t - barrier parameter

        -- outputs: grad - the resulting gradient
        """

        m, n = self.G.shape

        # from notes, g_i is a row of G
        factors = self.s + self.h - self.G @ self.x # factors.shape = (m)

        # To ensure correct broadcasting
        factors_matrix = np.tile(factors, (n, 1)).T

        scaled_G = self.G / factors_matrix

        if self.use_cupy:
          grad_x = cp.sum(scaled_G, axis = 0)
          grad_s = t - cp.sum(1 / factors)
          grad = cp.hstack([grad_x, grad_s])
        else:
          grad_x = np.sum(scaled_G, axis = 0)
          grad_s = t - np.sum(1 / factors)
          grad = np.hstack([grad_x, grad_s])

        return grad

    def phase_one_hessian(self):
        """
        Calculates the hessian for the objective value with log-barrier
        of the phase one method

        -- outputs: hess - the resulting hessian
        """

        m, n = self.G.shape

        # from notes, g_i is a row of G
        factors = self.s + self.h - self.G @ self.x # factors.shape = (m)
        
        if self.use_cupy:
          # To ensure correct broadcasting
          factors_matrix = cp.tile(factors, (n, 1)).T

          # Compute the sum of the outer products for hess_xx
          # Can be done a matrix multiplication
          hess_xx = (self.G.T / factors_matrix.T) @ ( self.G / factors_matrix)

          hess_xs = cp.reshape(cp.sum(- self.G / factors_matrix**2, axis = 0), newshape = (n, 1))

          hess_ss = cp.reshape(cp.array(cp.sum(1 / factors**2)), newshape = (1, 1))

          # Create phase_one_hessian
          hess_upper = cp.hstack([hess_xx, hess_xs])
          hess_lower = cp.hstack([hess_xs.T, hess_ss])
          hess = cp.vstack([hess_upper, hess_lower])
        else:
          # To ensure correct broadcasting
          factors_matrix = np.tile(factors, (n, 1)).T

          # Compute the sum of the outer products for hess_xx
          # Can be done a matrix multiplication
          hess_xx = (self.G.T / factors_matrix.T) @ ( self.G / factors_matrix)
          hess_xs = np.reshape(np.sum(- self.G / factors_matrix**2, axis = 0), newshape = (n, 1))
          hess_ss = np.sum(1 / factors**2)

          # Create phase_one_hessian
          hess = np.block([[hess_xx, hess_xs],
                          [hess_xs.T, hess_ss]])

        return hess

    def execute_phase_one(self):
        """
        Given a matrix G and a vector h, 'phase_one' conducts Newtons Interior Point Method to
        find a point in the interior of the polyhedra Gx <= h, if such a point exists.

        -- outputs: x - resulting point
                    s - scalar value. If s <= 0, x is feasible and if s > 0 then the set 
                        Gx <= h is either empty or maximum iterations were used
        """

        m, n = self.G.shape

        if np.max(self.G @ self.x - self.h) <= 0:
              # is feasible
              self.s = -1
              return

        # Initialize t
        t = 1

        for interior_iteration in range(self.max_iter_interior):

            # Execute centering
            warn = self.phase_one_newtons_method(t)

            if warn:
              print("Warning, Newtons method ran its maximum number of steps")
              self.warn = True

            # Check stopping criterion
            if m / t <= self.eps:
                break

            # If s < 0, then now it is feasible
            if self.s < 0:
                break

            # Increase t
            t *= self.mu
        
        # Check if ran out of steps
        if interior_iteration == self.max_iter_interior:
          print("Warning, the phase one interior method ran its maximum number of steps")
          self.warn = True

    def solve(self):
      """
      Calls upon 'execute_phase_one' to solve phase one

      -- outputs: x - numpy array for point x
                  s - scalar value, if s < 0 x is strictly feasible,
                                       s == 0, x is on the boundary,
                                       s > 0, the set is empty
                  warn - boolean, if True if either max_iter_newton or 
                         max_iter_interior was reached. Thus can the conclusions 
                         based on s not be sure if True.
      """

      self.execute_phase_one()

      if self.use_cupy:
        return self.x, self.s, self.warn
      else:
        return self.x, self.s, self.warn
