from PhaseOne import *
import numpy as np

try:
    import cupy as cp

    gpu_flag = True
except Exception:
    gpu_flag = False
    # Alias
    cp = np
    print("Not able to run tests with GPU")


def test_phase_one_gradient(use_cupy=False, tol=1e-8):
    """
    Tests the function 'phase_one_gradient' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """

    if use_cupy:
        print("Testing phase one gradient with cupy")
    else:
        print("Testing phase one gradient")

    # Constants
    G = np.array([[1, 2, 3], [4, 5, 6]])
    h = np.array([[2, 3]])
    x = np.ones(3)
    s = np.max(G @ x - h) + 1
    t = 1
    mu = 15

    true_grad_x = np.array([1, 2, 3]) / 9 + np.array([4, 5, 6])
    true_grad_s = np.array([-1 / 9])
    true_grad = np.hstack([true_grad_x, true_grad_s])

    if use_cupy:
        true_grad = cp.array(true_grad)

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy)
    grad = phaseOne.phase_one_gradient(t)

    if use_cupy:
        if cp.linalg.norm(true_grad - grad) <= tol:
            # passed
            pass
        else:
            print("The function 'phase_one_gradient' did not pass the test")
            return 0
    else:
        if np.linalg.norm(true_grad - grad) <= tol:
            # passed
            pass
        else:
            print("The function 'phase_one_gradient' did not pass the test")
            return 0

    G = np.array([[-1, -3], [-1, 1], [1, -2], [1, 4]])
    h = np.array([-6, 2, -2, 12])
    x = np.array([1, 1])
    s = 3
    t = 1
    mu = 15

    true_grad_x = (
        np.array([-1, -3])
        + np.array([-1, 1]) / 5
        + np.array([1, -2]) / 2
        + np.array([1, 4]) / 10
    )
    true_grad_s = -1 / 5 - 1 / 2 - 1 / 10
    true_grad = np.hstack([true_grad_x, true_grad_s])

    if use_cupy:
        true_grad = cp.array(true_grad)

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy)
    grad = phaseOne.phase_one_gradient(t)

    if use_cupy:
        if cp.linalg.norm(true_grad - grad) <= tol:
            # passed
            return 1
        else:
            print("The function 'phase_one_gradient' did not pass the test")
            return 0
    else:
        if np.linalg.norm(true_grad - grad) <= tol:
            # passed
            return 1
        else:
            print("The function 'phase_one_gradient' did not pass the test")
            return 0


def test_phase_one_hessian(use_cupy=False, tol=1e-8):
    """
    Tests the function 'phase_one_hessian' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """

    if use_cupy:
        print("Testing phase one hessian with cupy")
    else:
        print("Testing phase one hessian")

    G = np.array([[1, 2, 3], [4, 5, 6]])
    h = np.array([[2, 3]])
    x = np.ones(3)
    s = np.max(G @ x - h) + 1
    mu = 15

    true_hess_xx = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]) / 81 + np.array(
        [[16, 20, 24], [20, 25, 30], [24, 30, 36]]
    )
    true_hess_xs = np.reshape(
        -np.array([1, 2, 3]) / 81 - np.array([4, 5, 6]), newshape=(3, 1)
    )
    true_hess_ss = np.array([1 + 1 / 81])

    true_hess = np.block([[true_hess_xx, true_hess_xs], [true_hess_xs.T, true_hess_ss]])

    if use_cupy:
        true_hess = cp.array(true_hess)

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy)
    hess = phaseOne.phase_one_hessian()

    if use_cupy:
        if cp.linalg.norm(true_hess - hess) <= tol:
            # passed
            pass
        else:
            print("The function 'phase_one_hessian' did not pass the test")
            return 0
    else:
        if np.linalg.norm(true_hess - hess) <= tol:
            # passed
            pass
        else:
            print("The function 'phase_one_hessian' did not pass the test")
            return 0

    G = np.array([[-1, -3], [-1, 1], [1, -2], [1, 4]])
    h = np.array([-6, 2, -2, 12])
    x = np.array([1, 1])
    s = 3
    mu = 15

    true_hess_xx = (
        np.array([[1, 3], [3, 9]])
        + np.array([[1, -1], [-1, 1]]) / 25
        + np.array([[1, -2], [-2, 4]]) / 4
        + np.array([[1, 4], [4, 16]]) / 100
    )
    true_hess_xs = (
        -np.array([-1, -3])
        - np.array([-1, 1]) / 25
        - np.array([1, -2]) / 4
        - np.array([1, 4]) / 100
    )
    true_hess_xs = np.reshape(true_hess_xs, (-1, 1))
    true_hess_ss = 1 + 1 / 25 + 1 / 4 + 1 / 100

    true_hess = np.block([[true_hess_xx, true_hess_xs], [true_hess_xs.T, true_hess_ss]])

    if use_cupy:
        true_hess = cp.array(true_hess)

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy)
    hess = phaseOne.phase_one_hessian()

    if use_cupy:
        if cp.linalg.norm(true_hess - hess) <= tol:
            # passed again
            return 1
        else:
            print("The function 'phase_one_hessian' did not pass the test")
            return 0
    else:
        if np.linalg.norm(true_hess - hess) <= tol:
            # passed again
            return 1
        else:
            print("The function 'phase_one_hessian' did not pass the test")
            return 0


def test_phase_one_objective(use_cupy=False, tol=1e-8):
    """
    Tests the function 'phase_one_objective_interior' with some numerical examples.

    -- input: tol - tolerance for if test was passed or not

    -- output: int - 1 if test was passed, 0 otherwise
    """

    if use_cupy:
        print("Testing phase one objective with cupy")
    else:
        print("Testing phase one objective")

    G = np.array([[1, 2, 3], [4, 5, 6]])
    h = np.array([[2, 3]])
    if use_cupy:
        x = cp.ones(3)
    else:
        x = np.ones(3)
    s = 13
    t = 1
    mu = 15

    true_val = 13 - np.log(9)

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy)
    val = phaseOne.phase_one_objective(x, s, t)

    # Scalar, don't need to change norm
    if np.linalg.norm(true_val - val) <= tol:
        return 1
    else:
        print(
            "The function 'phase_one_phase_one_objective_interior' did not pass the test"
        )
        return 0


def test_phase_one(use_cupy=False, linear_solver="solve"):
    """
    Tests the function 'phase_one' with some numerical examples.

    -- inputs: use_cupy - boolean whether or not to use cupy
               linear_solver - which solver to use, 'solve' or 'cg'

    -- output: int - 1 if test was passed, 0 otherwise
    """

    if use_cupy:
        print("Testing phase one with cupy and solver '" + linear_solver + "'")
    else:
        print("Testing phase one without cupy but with solver '" + linear_solver + "'")

    print("Phase one is initialized inside the set")
    # phase_one is initialized inside this set
    G = np.array([[1, 3], [1, 1], [-1, 0], [0, -1]])
    h = np.array([9, 5, 0, 0])
    mu = 15

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy, linear_solver=linear_solver)
    x, s, _ = phaseOne.solve()

    if use_cupy:
        x = cp.asnumpy(x)

    if phaseOne.s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    print("Phase one is initialized outside the set")
    # Set which phase_one is not initialized in
    G = np.array([[-1, -3], [-1, 1], [-1, 2], [1, 4]])
    h = np.array([-6, 2, 2, 12])
    mu = 15

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy, linear_solver=linear_solver)
    x, s, _ = phaseOne.solve()
    if use_cupy:
        x = cp.asnumpy(x)

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    print("Phase one is initialized outside of an unbounded set")
    # Unbounded set which phase_one is not initialied in
    G = np.array([[1, -2], [-3, 1]])
    h = np.array([-2, 0])
    mu = 15

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy, linear_solver=linear_solver)
    x, s, _ = phaseOne.solve()
    if use_cupy:
        x = cp.asnumpy(x)

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    print("Phase one tries to find an empty set")
    # Empty set
    G = np.array([[3, -1], [-1, 5], [-1, 0], [0, -1]])
    h = np.array([-2, 1.5, 0, 0])
    mu = 15

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy, linear_solver=linear_solver)
    x, s, _ = phaseOne.solve()
    if use_cupy:
        x = cp.asnumpy(x)

    if s > 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    print("Phase one is given a problem with high dimension")
    # Try a problem with larger size
    np.random.seed(0)

    m, n = 200, 1000
    G = np.random.uniform(low=-10, high=10, size=(m, n))
    x = np.random.uniform(low=-5, high=5, size=(n))
    h = G @ x + 1
    mu = 15

    phaseOne = PhaseOneSolver(G, h, mu, use_cupy=use_cupy, linear_solver=linear_solver)
    x, s, _ = phaseOne.solve()
    if use_cupy:
        x = cp.asnumpy(x)

    if s < 0 and np.max(G @ x - h) < 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    return 1


def test_initialized_phase_one(use_cupy=False, linear_solver="solve"):
    """
    Tests the function 'phase_one' with an initialized point.

    -- inputs: use_cupy - boolean whether or not to use cupy
               linear_solver - which solver to use, 'solve' or 'cg'

    -- output: int - 1 if test was passed, 0 otherwise
    """
    if use_cupy:
        print(
            "Testing phase one given an intialization with cupy and solver '"
            + linear_solver
            + "'"
        )
    else:
        print(
            "Testing phase one given an initialization without cupy with solver '"
            + linear_solver
            + "'"
        )
    # Set which phase_one is not initialized in
    G = np.array([[-1, -3], [-1, 1], [-1, 2], [1, 4]])
    h = np.array([-6, 2, 2, 12])
    x0 = np.array([-2, -3])
    mu = 15

    phaseOne = PhaseOneSolver(
        G, h, mu, x0=x0, use_cupy=use_cupy, linear_solver=linear_solver
    )
    x, s, _ = phaseOne.solve()
    if use_cupy:
        x = cp.asnumpy(x)

    if s < 0 and np.max(G @ x - h) <= 0:
        # is ok
        pass
    else:
        print("The function 'phase_one' did not pass the test")
        return 0

    return 1


def automated_tests(use_cupy=True):

    tests = [
        test_phase_one_objective,
        test_phase_one_gradient,
        test_phase_one_hessian,
        test_initialized_phase_one,
        test_phase_one,
    ]

    passed = 0
    max_score = 0
    print("Start tests")
    for count, test in enumerate(tests):
        passed += test()
        max_score += 1

        if use_cupy:
          passed += test(use_cupy=True)
          max_score += 1

        if count >= len(tests) - 2:
            passed += test(linear_solver="cg")
            max_score += 1

            if use_cupy:
              passed += test(use_cupy=True, linear_solver="cg")
              max_score += 1

    print("Automated tests are done.")
    print("Passed {} / {} tests".format(passed, max_score))
