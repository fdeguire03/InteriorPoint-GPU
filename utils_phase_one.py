## EE 364B Project, utility functions for 

import numpy as np
from numpy.linalg import solve

class phase_one():
    
    def __init__(self, G, h, mu, eps = 1e-8, max_iter_interior = 200, max_iter_newton = 200):

        self.G = G
        self.h = h
        self.mu = mu
        self.eps = eps
        self.max_iter_interior = max_iter_interior
        self.max_iter_newton = max_iter_newton

    def phase_one_newtons_method(self, x, s, G, h, t, eps = 1e-8, max_iter = 200):
        """
        Performs newtons method for phase one, calculating a newton direction and then
        using line search. 

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above
                t - barrier parameter
                eps - tolerance for stopping
                max_iter - maximum number of newton steps allowed

        -- outputs: x - a new point x
                    s - a new point s
                    bool - boolean, True if maximum allowed steps was taken, False otherwise
        """

        for nm in range(max_iter):
            m, n = G.shape

            # Calculate phase_one_hessian and phase_one_gradient
            hess = self.phase_one_hessian(x, s, G, h) + 0.01 * np.eye(n + 1) # some conditioning
            grad = self.phase_one_gradient(x, s, G, h, t)

            # This can probably be done way quicker with Cupy
            newton_direction = solve(hess, -grad)

            lambda_sq = - grad @ newton_direction

            if lambda_sq / 2 <= eps:
                break

            # linesearch
            step = self.phase_one_linesearch(x, s, G, h, t, newton_direction, grad)
        
            # update
            x, s = x + step * newton_direction[:-1], s + step * newton_direction[-1]

        return x, s, nm == max_iter - 1

    def phase_one_objective_interior(self, x, s, G, h, t):
        """
        Calculates the objective value for phase one

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above
                t - barrier parameter

        -- output: val - objective value
        """
        
        val = t * s - np.sum(np.log(s + h - G @ x))

        return val

    def phase_one_linesearch(self, x, s, G, h, t, direction, grad, alpha = 0.2, beta = 0.7):
        """
        Performs a linesearch in the given direction.

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above
                t - barrier parameter
                direction - given direction to take a step in
                grad - gradient at the current point
                alpha - hyperparameter of linesearch
                beta - hyperparamter of linesearch

        -- output: step - how long the step in the given direction should be
        """

        step = 1

        # Check feasiblity
        while not self.phase_one_check_feasibility(x + step * direction[:-1], s + step * direction[-1], G, h):
            step *= beta

        # Now line search for descent direction
        while self.phase_one_objective_interior(x + step * direction[:-1], s + step * direction[-1], G, h, t) > \
            self.phase_one_objective_interior(x, s, G, h, t) + alpha * step * grad @ direction:
            # Decrease step length
            step *= beta
            
        return step

    def phase_one_check_feasibility(self, x, s, G, h):
        """
        Checks if the combination of x and s is allowed. 
        Gx - h <= s must always hold

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above

        -- outputs: boolean that is False if infeasible and True if feasible

        """

        return np.max(G @ x - h) < s

    def phase_one_gradient(self, x, s, G, h, t):
        """
        Calculates the gradient for the objective value with log-barrier
        of the phase one method

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above
                t - barrier parameter

        -- outputs: grad - the resulting gradient
        """

        m, n = G.shape

        # from notes, g_i is a row of G
        factors = s + h - G @ x # factors.shape = (m)

        # To ensure correct broadcasting
        factors_matrix = np.tile(factors, (n, 1)).T

        scaled_G = G / factors_matrix

        grad_x = np.sum(scaled_G, axis = 0)

        grad_s = t - np.sum(1 / factors)

        grad = np.hstack([grad_x, grad_s])

        return grad

    def phase_one_hessian(self, x, s, G, h):
        """
        Calculates the hessian for the objective value with log-barrier
        of the phase one method

        -- inputs: x - point x used for the inequality Gx <= h
                s - scalar being minimized
                G - matrix used in inequality above
                h - vector used in inequality above

        -- outputs: hess - the resulting hessian
        """

        m, n = G.shape

        # from notes, g_i is a row of G
        factors = s + h - G @ x # factors.shape = (m)

        # To ensure correct broadcasting
        factors_matrix = np.tile(factors, (n, 1)).T

        # Compute the sum of the outer products for hess_xx
        # Can be done a matrix multiplication
        hess_xx = (G.T / factors_matrix.T) @ ( G / factors_matrix)

        hess_xs = np.reshape(np.sum(- G / factors_matrix**2, axis = 0), newshape = (n, 1))

        hess_ss = np.sum(1 / factors**2)

        # Create phase_one_hessian
        hess = np.block([[hess_xx, hess_xs],
                        [hess_xs.T, hess_ss]])

        return hess

    def execute_phase_one(self, G, h, mu, eps = 1e-8, max_iter_interior = 200, max_iter_newton = 200):
        """
        Given a matrix G and a vector h, 'phase_one' conducts Newtons Interior Point Method to
        find a point in the interior of the polyhedra Gx <= h, if such a point exists.

        -- inputs: G  - a numpy array
                    h  - a numpy vector
                    mu - hyperparameter which increases barrier parameter t
                    max_iter_interior - sets how many times barrier paramter t may be increase
                    max_iter_newton - sets how many newtons steps may be taken when centering

        -- outputs: x - resulting point
                    s - scalar value. If s <= 0, x is feasible and if s > 0 then the set 
                        Gx <= h is either empty or maximum iterations were used
        """

        # Initialize x and s
        m, n = G.shape

        # x can be whatever
        # since x >= 0 usually is included, might as well initialize accordingly
        x = np.ones(n)

        if np.max(G @ x - h) <= 0:
            # is feasible
            return x, -1

        # s must be feasible, but this can be chosen
        s = np.max(G @ x - h) + 0.1

        # Initialize t
        t = 1

        for _ in range(max_iter_interior):

            # Execute centering
            x, s, _ = self.phase_one_newtons_method(x, s, G, h, t, eps, max_iter_newton)

            # Check stopping criterion
            if m / t <= eps:
                break

            # If s < 0, then now it is feasible
            if s < 0:
                break

            # Increase t
            t *= mu

        # if s < 0 then x is feasible, otherwise x is not feasible
        return x, s
    
    def solve(self):
        x, s = self.execute_phase_one(self.G, self.h, self.mu, self.eps, 
                         self.max_iter_interior, self.max_iter_newton)
        return x, s