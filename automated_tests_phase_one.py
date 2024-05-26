## EE 364B Project

# Automated tests

from utils_phase_one import *
import numpy as np
from numpy.linalg import norm

def test_phase_one_gradient(tol = 1e-8):
    """
    Tests the function 'phase_one_gradient' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """
    # Constants
    G = np.array([[1, 2, 3], 
                [4, 5, 6]])
    h = np.array([[2, 3]])
    x = np.ones(3)
    s = np.max(G @ x - h) + 1
    t = 1
    mu = 15

    true_grad_x = np.array([1, 2, 3]) / 9 + np.array([4, 5, 6])
    true_grad_s = np.array([- 1 / 9])
    true_grad = np.hstack([true_grad_x, true_grad_s])

    phaseOne = phase_one(G, h, mu)
    grad = phaseOne.phase_one_gradient(x, s, G, h, t)

    if norm(true_grad - grad) <= tol:
        # passed
        pass
    else:
        print("The function 'phase_one_gradient' did not pass the test")
        return 0
    
    G = np.array([[-1, -3],
                  [-1, 1],
                  [1, -2],
                  [1, 4]])
    h = np.array([-6, 2, -2, 12])
    x = np.array([1, 1])
    s = 3
    t = 1
    mu = 15

    true_grad_x = np.array([-1, -3]) + np.array([-1, 1]) / 5 + np.array([1, -2]) / 2 + np.array([1, 4]) / 10
    true_grad_s = - 1 / 5 - 1 / 2 - 1 / 10
    true_grad = np.hstack([true_grad_x, true_grad_s])

    phaseOne = phase_one(G, h, mu)
    grad = phaseOne.phase_one_gradient(x, s, G, h, t)

    if norm(true_grad - grad) <= tol:
        # passed
        return 1
    else:
        print("The function 'phase_one_gradient' did not pass the test")
        return 0

def test_phase_one_hessian(tol = 1e-8):
    """
    Tests the function 'phase_one_hessian' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """
    G = np.array([[1, 2, 3], 
                  [4, 5, 6]])
    h = np.array([[2, 3]])
    x = np.ones(3)
    s = np.max(G @ x - h) + 1
    mu = 15

    true_hess_xx = np.array([[1, 2, 3], 
                        [2, 4, 6], 
                        [3, 6, 9]]) / 81 + \
                np.array([[16, 20, 24], 
                        [20, 25, 30], 
                        [24, 30, 36]])
    true_hess_xs = np.reshape(- np.array([1, 2, 3]) / 81 - np.array([4, 5, 6]), newshape = (3, 1))
    true_hess_ss = np.array([1 + 1/81])

    true_hess = np.block([[true_hess_xx, true_hess_xs], 
                        [true_hess_xs.T, true_hess_ss]])

    phaseOne = phase_one(G, h, mu)
    hess = phaseOne.phase_one_hessian(x, s, G, h)

    if norm(true_hess - hess) <= tol:
        # passed
        pass
    else:
        print("The function 'phase_one_hessian' did not pass the test")
        return 0
    
    G = np.array([[-1, -3],
              [-1, 1],
              [1, -2],
              [1, 4]])
    h = np.array([-6, 2, -2, 12])
    x = np.array([1, 1])
    s = 3
    mu = 15

    true_hess_xx = np.array([[1, 3], [3, 9]]) + np.array([[1, -1], [-1, 1]]) / 25 + \
                np.array([[1, -2], [-2, 4]]) / 4 + np.array([[1, 4], [4, 16]]) / 100
    true_hess_xs = - np.array([-1, -3]) - np.array([-1, 1]) / 25 - np.array([1, -2]) / 4 - np.array([1, 4]) / 100
    true_hess_xs = np.reshape(true_hess_xs, (-1, 1))
    true_hess_ss = 1 + 1/25 + 1 / 4 + 1 / 100

    true_hess = np.block([[true_hess_xx, true_hess_xs], 
                        [true_hess_xs.T, true_hess_ss]])
    
    phaseOne = phase_one(G, h, mu)
    hess = phaseOne.phase_one_hessian(x, s, G, h)
    
    if norm(true_hess - hess) <= tol:
        # passed again
        return 1
    else:
        print("The function 'phase_one_hessian' did not pass the test")
        return 0

def test_phase_one_objective_interior(tol = 1e-8):
    """
    Tests the function 'phase_one_objective_interior' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """

    G = np.array([[1, 2, 3], 
                  [4, 5, 6]])
    h = np.array([[2, 3]])
    x = np.ones(3)
    s = np.max(G @ x - h) + 1
    t = 1
    mu = 15

    true_val = 13 - np.log(9)

    phaseOne = phase_one(G, h, mu)
    val = phaseOne.phase_one_objective_interior(x, s, G, h, t)

    if norm(true_val - val) <= tol:
        return 1
    else:
        print("The function 'phase_one_phase_one_objective_interior' did not pass the test")
        return 0

def test_phase_one():
    """
    Tests the function 'phase_one' with some numerical examples.

    -- inputs: None

    -- output: int - 1 if test was passed, 0 otherwise
    """

    # phase_one is initialized inside this set
    G = np.array([[1, 3],
                  [1, 1], 
                  [-1, 0], 
                  [0, -1]])
    h = np.array([9, 5, 0, 0])
    mu = 15

    phaseOne = phase_one(G, h, mu)
    x, s = phaseOne.solve()

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0
    
    # Set where phase_one is not initialized in
    G = np.array([[-1, -3],
                  [-1, 1],
                  [-1, 2],
                  [1, 4]])
    h = np.array([-6, 2, 2, 12])
    mu = 15

    phaseOne = phase_one(G, h, mu)
    x, s = phaseOne.solve()

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    # Unbounded set which phase_one is not initialied in
    G = np.array([[1, -2], 
                  [-3, 1]])
    h = np.array([-2, 0])
    mu = 15

    phaseOne = phase_one(G, h, mu)
    x, s = phaseOne.solve()

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0
    
    # Empty set
    G = np.array([[3, -1], 
                  [-1, 5], 
                  [-1, 0],
                  [0, -1]])
    h = np.array([-2, 1.5, 0, 0])
    mu = 15

    phaseOne = phase_one(G, h, mu)
    x, s = phaseOne.solve()

    if s > 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    return 1

def automated_tests():

    tests = [test_phase_one_objective_interior, test_phase_one_gradient, test_phase_one_hessian, test_phase_one]
    
    sum = 0
    max_score = len(tests)
    for test in tests:
        sum += test()

    print("Automated tests are done.")
    print("Passed {} / {} tests".format(sum, max_score))

automated_tests()