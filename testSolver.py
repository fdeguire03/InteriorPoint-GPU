## Test different solvers

import numpy as np
import matplotlib.pyplot as plt
from LPSolver import LPSolver
from QPSolver import QPSolver
from SOCPSolver import SOCPSolver
from LassoSolver import LassoSolver
import cvxpy as cp
import cupy
from time import time
import pandas as pd

def test_LP(n_values, verbose = False, N = 10, filename = None, use_jax=False):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """
    
    if use_jax:
        import jax.numpy as jnp
        from jaxopt import BoxOSQP

    if verbose:
      print("------ Starting LP-test ------")
      
    # Set seed
    np.random.seed(1)

    # Make sure n values are integers
    n_values = np.array(n_values, dtype = np.int32)

    # Calculate m and k values
    m_values = (0.8 * n_values).astype(int)
    k_values = (0.2 * n_values).astype(int)

    # How many tests
    num_tests = len(n_values)

    # Containers for storing times
    cvxpy_times = np.zeros((num_tests, N))  # Every row is for a different
    ls_gpu_times = np.zeros((num_tests, N)) # dimension and every
    ls_cpu_times = np.zeros((num_tests, N)) # column is a different test
    

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N))
    ls_gpu_values = np.zeros((num_tests, N))
    ls_cpu_values = np.zeros((num_tests, N))
    
    if use_jax:
        jax_values = np.zeros((num_tests, N))
        jax_times = np.zeros((num_tests, N))

    ### Run test on LP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
      
      if verbose:
          print("\n")
          print(f"n is {n}")
          print("Start testing")

      if n < 1000:
        num_iters = N 
      elif n < 2500:
        num_iters = int(N/2)
      else:
        num_iters = 3

      for i in range(num_iters):
        
        # Generate A
        A = np.random.uniform(low = -2, high = 2, size = (m, n))

        # Generate C
        C = np.random.uniform(low = -2, high = 2, size = (k, n))

        # Generate x_feas and c
        x_feas = np.random.uniform(low = -2, high = 2, size = (n))
        c = np.random.uniform(low = -2, high = 2, size = (n))

        # From this, calculate b and d
        b = A @ x_feas
        d = C @ x_feas

        # Have upper and lower bounds
        up_bnd = 3
        lo_bnd = -3

        # Create CVXPY problem

        # Solve in CVXPY
        x = cp.Variable(n)

        # Objective
        obj = c @ x

        # Constraints
        constr = [A @ x == b, C @ x <= d, lo_bnd <= x, x <= up_bnd]

        # Create problem
        prob = cp.Problem(cp.Minimize(obj), constr)

        # Solve
        tik = time()
        prob.solve(solver = cp.CLARABEL)
        tok = time()

        if verbose:
          print(f"CVXPY solved {i + 1} time(s). Time: {tok-tik}")

        if i == num_iters - 1 and verbose:
          print(f"CVXPY gets optimal value {obj.value}")
      
        # Store
        cvxpy_times[count, i] = tok - tik
        cvxpy_values[count, i] = obj.value

        # Delete
        del x
        del constr
        del prob

        # Move on to LP-solver on GPU

        # Create instance
        ls_gpu = LPSolver(
            c = c,
            A = A,
            b = b,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = True,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-4, 
            mu = 15, 
            t0 = 1,
            max_inner_iters = 20, 
            max_outer_iters=10,
            beta = 0.5,
            alpha = 0.05
        )

        # Time
        tik = time()
        ls_gpu.solve()
        tok = time()

        if verbose:
          print(f"LP-solver, GPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        ls_gpu_times[count, i] = tok - tik
        ls_gpu_values[count, i] = ls_gpu.value

        if i == num_iters - 1 and verbose:
          print(f"LP-solver, GPU, gets optimal value {ls_gpu.value}")

        # Delete, clear GPU memory
        del ls_gpu
        cupy._default_memory_pool.free_all_blocks()

        # LP-solver on CPU

        # Create instance
        ls_cpu = LPSolver(
            c = c,
            A = A,
            b = b,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = False,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-4, 
            mu = 15, 
            t0 = 1,
            max_inner_iters = 20, 
            max_outer_iters=10,
            beta = 0.5,
            alpha = 0.05
          )

        # Time
        tik = time()
        ls_cpu.solve()
        tok = time()

        if verbose:
          print(f"LP-solver, CPU solved {i + 1} time(s). Time: {tok-tik}")

        # Store
        ls_cpu_times[count, i] = tok - tik
        ls_cpu_values[count, i] = ls_cpu.value

        if i == num_iters - 1 and verbose:
          print(f"LP-solver, CPU, gets optimal value {ls_cpu.value}")

        # Delete
        del ls_cpu
        
        if use_jax:

            # JAX-opt
            # Identity
            identity = jnp.eye(n)
            #Q = None
            Q = jnp.zeros((n, n))
            c_jnp = jnp.array(c)
            A_jnp = jnp.array(A)
            b_jnp = jnp.array(b)
            C_jnp = jnp.array(C)
            d_jnp = jnp.array(d)
            lo_jnp = lo_bnd * jnp.ones((n))
            up_jnp = up_bnd * jnp.ones((n))
            inf_jnp = -jnp.inf * jnp.ones(d.shape[0])
            E = jnp.vstack([A_jnp, C_jnp, identity])
            l = jnp.hstack([b_jnp, inf_jnp, lo_jnp])
            u = jnp.hstack([b_jnp, d_jnp, up_jnp])

            qp = BoxOSQP()

            tik = time()
            sol = qp.run(params_obj=(Q, c_jnp), params_eq=E, params_ineq=(l, u))
            tok = time()

            if verbose:
              print(f"JAX-opt solved {i + 1} time(s). Time: {tok-tik}")

            x_jnp_opt = np.array(sol.params.primal[0])
            opt_val = c @ x_jnp_opt

            jax_times[count, i] = tok - tik
            jax_values[count, i] = opt_val

            if i == num_iters - 1 and verbose:
              print(f"JAX-opt gets optimal value {opt_val}")


    if filename is not None:
     
      # Data to save
      data = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "cvxpy_values" : cvxpy_values.reshape((-1)),
              "ls_gpu_times" : ls_gpu_times.reshape((-1)),
              "ls_gpu_values" : ls_gpu_values.reshape((-1)),
              "ls_cpu_times" : ls_cpu_times.reshape((-1)),
              "ls_cpu_values" : ls_cpu_values.reshape((-1))}
      if use_jax:
            data["jax_times"] = jax_times.reshape((-1))
            data["jax_values"] = jax_values.reshape(-1)

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("------ Ending LP-test ------")

    return

def test_QP(n_values, verbose = False, N = 10, filename = None, use_jax=False):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """
    
    if use_jax:
        import jax.numpy as jnp
        from jaxopt import BoxOSQP

    if verbose:
      print("------ Starting QP-test ------")
      
    # Set seed
    np.random.seed(1)

    # Make sure n values are integers
    n_values = np.array(n_values, dtype = np.int32)

    # Calculate m and k values
    m_values = np.array(0.8 * n_values, dtype = np.int32)
    k_values = np.array(0.2 * n_values, dtype = np.int32)

    # How many tests
    num_tests = n_values.shape[0]

    # Containers for storing times
    cvxpy_times = np.zeros((num_tests, N))  # Every row is for a different
    qp_gpu_times = np.zeros((num_tests, N)) # dimension and every
    qp_cpu_times = np.zeros((num_tests, N)) # column is a different test
    qp_gpu_times_allinequality = np.zeros((num_tests, N)) # dimension and every
    qp_cpu_times_allinequality = np.zeros((num_tests, N)) # column is a different test
    

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N))
    qp_gpu_values = np.zeros((num_tests, N))
    qp_cpu_values = np.zeros((num_tests, N))
    qp_gpu_values_allinequality = np.zeros((num_tests, N))
    qp_cpu_values_allinequality = np.zeros((num_tests, N))

    if use_jax:
        jax_values = np.zeros((num_tests, N))
        jax_times = np.zeros((num_tests, N))

    ### Run test on QP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
        
      k = 20
      
      if verbose:
        print("\n")
        print(f"n is {n}")
        print("Generate some data")

      if n < 1000:
        num_iters = N 
      elif n < 2500:
        num_iters = int(N/2)
      else:
        num_iters = 3

      for i in range(num_iters):

        # Generate P
        P_partial = np.random.uniform(low = -2, high = 2, size = (m, n))
        P = P_partial.T @ P_partial + np.eye(n)
        
        # Generate A
        A = np.random.uniform(low = -2, high = 2, size = (m, n))

        # Generate C
        C = np.random.uniform(low = -2, high = 2, size = (k, n))

        # Generate x_feas and q
        x_feas = np.random.uniform(low = -2, high = 2, size = (n))
        q = np.random.uniform(low = -2, high = 2, size = (n))

        # From this, calculate b and d
        b = A @ x_feas
        d = C @ x_feas

        # Have upper and lower bounds
        up_bnd = 3
        lo_bnd = -3

        # Create CVXPY problem
      
        x = cp.Variable(n)

        # Objective
        P_cp = cp.psd_wrap(P)
        obj = 0.5 * cp.quad_form(x, P_cp) + q @ x

        # Constraints
        constr = [A @ x == b, C @ x <= d, lo_bnd <= x, x <= up_bnd]

        # Create problem
        prob = cp.Problem(cp.Minimize(obj), constr)

        # Solve
        tik = time()
        prob.solve(solver = cp.CLARABEL)
        tok = time()

        if verbose:
          print(f"CVXPY solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        cvxpy_times[count, i] = tok - tik
        cvxpy_values[count, i] = obj.value

        if i == num_iters - 1 and verbose:
            print(f"Problem is {prob.status}")
            print(f"CVXPY gets optimal value of {obj.value}")

        # Delete objects just to be sure
        del x
        del obj
        del constr
        del prob

        # Move on to QP-solver on GPU

        # Create instance
        qp_gpu = QPSolver(
            P = P,
            q = q,
            A = A,
            b = b,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = True,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-8, 
            mu = 15, 
            t0 = 0.01,
            max_inner_iters = 100, 
            max_outer_iters=10,
            beta = 0.6,
            alpha = 0.4
        )

        # Time
        tik = time()
        qp_gpu.solve()
        tok = time()

        if verbose:
          print(f"QP-solver, GPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        qp_gpu_times[count, i] = tok - tik
        qp_gpu_values[count, i] = qp_gpu.value

        if i == num_iters - 1 and verbose:
          print(f"QP-solver, GPU, gets optimal value {qp_gpu.value}")

        # Delete
        del qp_gpu
        cupy._default_memory_pool.free_all_blocks()

        # Finally QP-solver on CPU

        # Create instance
        # Create instance
        qp_cpu = QPSolver(
            P = P,
            q = q,
            A = A,
            b = b,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = False,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-8, 
            mu = 15, 
            t0 = 0.01,
            max_inner_iters = 100, 
            max_outer_iters=10,
            beta = 0.6,
            alpha = 0.4
        )

        # Time
        tik = time()
        qp_cpu.solve()
        tok = time()

        if verbose:
          print(f"QP-solver, CPU solved {i + 1} time(s). Time: {tok-tik}")

        # Store
        qp_cpu_times[count, i] = tok - tik
        qp_cpu_values[count, i] = qp_cpu.value

        if i == num_iters - 1 and verbose:
          print(f"QP-solver, CPU, gets optimal value {qp_cpu.value}")

        # Delete
        del qp_cpu
        
        """
        # Move on to QP-solver on GPU
        
        C = np.vstack((C, A, -A))
        d = np.hstack((d, np.ones(len(A)*2)*1e-3))

        # Create instance
        qp_gpu_inequ = QPSolver(
            P = P,
            q = q,
            A = None,
            b = None,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = True,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-8, 
            mu = 15, 
            t0 = 0.01,
            phase1_t0 = 0.00001,
            max_inner_iters = 100, 
            max_outer_iters=10,
            beta = 0.6,
            alpha = 0.4
        )

        # Time
        tik = time()
        qp_gpu_inequ.solve()
        tok = time()

        if verbose:
          print(f"QP-solver, GPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        qp_gpu_times_allinequality[count, i] = tok - tik
        qp_gpu_values_allinequality[count, i] = qp_gpu_inequ.value

        if i == num_iters - 1 and verbose:
          print(f"QP-solver, GPU, gets optimal value {qp_gpu_inequ.value}")

        # Delete
        del qp_gpu_inequ
        cupy._default_memory_pool.free_all_blocks()

        # Finally QP-solver on CPU

        # Create instance
        # Create instance
        qp_cpu_inequ = QPSolver(
            P = P,
            q = q,
            A = None,
            b = None,
            C = C,
            d = d,
            lower_bound = lo_bnd,
            upper_bound = up_bnd,
            use_gpu = False,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-8, 
            mu = 15, 
            t0 = 0.01,
            phase1_t0 = 0.00001,
            max_inner_iters = 100, 
            max_outer_iters=10,
            beta = 0.6,
            alpha = 0.4
        )

        # Time
        tik = time()
        qp_cpu_inequ.solve()
        tok = time()

        if verbose:
          print(f"QP-solver, CPU solved {i + 1} time(s). Time: {tok-tik}")

        # Store
        qp_cpu_times_allinequality[count, i] = tok - tik
        qp_cpu_values_allinequality[count, i] = qp_cpu_inequ.value

        if i == num_iters - 1 and verbose:
          print(f"QP-solver, CPU, gets optimal value {qp_cpu_inequ.value}")

        # Delete
        del qp_cpu_inequ
        """

        if use_jax:

            # JAX-opt
            # Identity
            identity = jnp.eye(n)
            Q = jnp.array(P)
            c_jnp = jnp.array(q)
            A_jnp = jnp.array(A)
            b_jnp = jnp.array(b)
            C_jnp = jnp.array(C)
            d_jnp = jnp.array(d)
            lo_jnp = lo_bnd * jnp.ones((n))
            up_jnp = up_bnd * jnp.ones((n))
            inf_jnp = -jnp.inf * jnp.ones(d.shape[0])
            E = jnp.vstack([A_jnp, C_jnp, identity])
            l = jnp.hstack([b_jnp, inf_jnp, lo_jnp])
            u = jnp.hstack([b_jnp, d_jnp, up_jnp])

            qp = BoxOSQP()

            tik = time()
            sol = qp.run(params_obj=(Q, c_jnp), params_eq=E, params_ineq=(l, u))
            tok = time()

            if verbose:
              print(f"JAX-opt solved {i + 1} time(s). Time: {tok-tik}")

            x_jnp_opt = np.array(sol.params.primal[0])
            opt_val = 0.5 * x_jnp_opt.transpose() @ P @ x_jnp_opt + q @ x_jnp_opt

            jax_times[count, i] = tok - tik
            jax_values[count, i] = opt_val

            if i == num_iters - 1 and verbose:
              print(f"JAX-opt gets optimal value {opt_val}")

    if filename is not None:
     
      # Data to save
      data = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "cvxpy_values" : cvxpy_values.reshape((-1)),
              "qp_gpu_times" : qp_gpu_times.reshape((-1)),
              "qp_gpu_values" : qp_gpu_values.reshape((-1)),
              "qp_gpu_allinequality_times" : qp_gpu_times_allinequality.reshape((-1)),
              "qp_gpu_allinequality_values" : qp_gpu_values_allinequality.reshape((-1)),
              "qp_cpu_times" : qp_cpu_times.reshape((-1)),
              "qp_cpu_values" : qp_cpu_values.reshape((-1)),
              "qp_cpu_allinequality_times" : qp_cpu_times_allinequality.reshape((-1)),
              "qp_cpu_allinequality_values" : qp_cpu_values_allinequality.reshape((-1))}

      if use_jax:
            data["jax_times"] = jax_times.reshape((-1))
            data["jax_values"] = jax_values.reshape(-1)

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("------ Ending QP-test ------")

    return

def test_SOCP(n_values, verbose = False, N = 10, filename = None):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """

    if verbose:
      print("------ Starting SOCP-test ------")
      
    # Set seed
    np.random.seed(1)

    # Make sure n values are integers
    n_values = np.array(n_values, dtype = np.int32)

    # Calculate m and k values
    m_values = np.array(0.8 * n_values, dtype = np.int32)
    k_values = np.array(0.2 * n_values, dtype = np.int32)

    # How many tests
    num_tests = n_values.shape[0]

    # Containers for storing times
    cvxpy_times = np.zeros((num_tests, N))  # Every row is for a different
    socp_gpu_times = np.zeros((num_tests, N)) # dimension and every
    socp_cpu_times = np.zeros((num_tests, N)) # column is a different test

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N))
    socp_gpu_values = np.zeros((num_tests, N))
    socp_cpu_values = np.zeros((num_tests, N))

    ### Run test on QP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
        
      k = 50
      
      if verbose:
        print("\n")
        print(f"n is {n}")
        print("Generate some data")

      if n < 1000:
        num_iters = N 
      elif n < 2500:
        num_iters = int(N/2)
      else:
        num_iters = 3

      for i in range(num_iters):  

        # Generate P and q
        P_partial = np.random.uniform(low = -2, high = 2, size = (m, n))
        P = P_partial.T @ P_partial + np.eye(n)
        q = np.random.uniform(low = -2, high = 2, size = (n))
        
        # Generate a random feasible SOCP.
        num_con = 5
        A = []
        b = []
        c = []
        d = []
        x0 = np.random.randn(n)
        for j in range(num_con):
            A.append(np.random.randn(m, n))
            b.append(np.random.randn(m))
            c.append(np.random.randn(n))
            d.append(np.linalg.norm(A[j] @ x0 + b, 2) - c[j] @ x0)
        F = np.random.randn(k, n)
        g = F @ x0

        # Create CVXPY problem
        
        if n < 2000 and i == num_iters - 1:

            # Define and solve the CVXPY problem
            x = cp.Variable(n)

            # Constrains
            soc_constraints = [
                  cp.SOC(c[j].T @ x + d[j], A[j] @ x + b[j]) for j in range(num_con)
            ]

            # objective
            P_cp = cp.psd_wrap(P)
            obj = 0.5 * cp.quad_form(x, P_cp) + q @ x
            prob = cp.Problem(cp.Minimize(obj),
                              soc_constraints + [F @ x == g])

            # Solve
            tik = time()
            prob.solve(solver = cp.CLARABEL)
            tok = time()

            if verbose:
              print(f"CVXPY solved {i} time(s). Time: {tok-tik}")

            # Store
            cvxpy_times[count, i] = tok - tik
            cvxpy_values[count, i] = obj.value

            if i == num_iters - 1 and verbose:
                print(f"Problem is {prob.status}")
                print(f"CVXPY gets optimal value of {obj.value}")

            # Delete objects just to be sure
            del x
            del obj
            del soc_constraints
            del prob

        # Move on to SOCP-solver on GPU
        # Create instance
        socp_gpu = SOCPSolver(
            P = P,
            q = q,
            A = A,
            b = b,
            c = c,
            d = d,
            F = F, 
            g = g,
            lower_bound = None,
            upper_bound = None,
            use_gpu = True,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-4, 
            mu = 15, 
            t0 = 0.1,
            max_inner_iters = 500,
            max_outer_iters=10,
            beta = 0.5,
            alpha = 0.05
        )

        # Time
        tik = time()
        socp_gpu.solve()
        tok = time()

        if verbose:
          print(f"SOCP-solver, GPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        socp_gpu_times[count, i] = tok - tik
        socp_gpu_values[count, i] = socp_gpu.value

        if i == num_iters - 1 and verbose:
          print(f"SOCP-solver, GPU, gets optimal value {socp_gpu.value}")

        # Delete
        del socp_gpu
        cupy._default_memory_pool.free_all_blocks()

        # Move on to SOCP-solver on CPU
        # Create instance
        socp_cpu = SOCPSolver(
            P = P,
            q = q,
            A = A,
            b = b,
            c = c,
            d = d,
            F = F, 
            g = g,
            lower_bound = None,
            upper_bound = None,
            use_gpu = False,
            suppress_print = True,
            check_cvxpy = False,
            epsilon = 1e-4, 
            mu = 15, 
            t0 = 0.1,
            max_inner_iters = 500,
            max_outer_iters=10,
            beta = 0.5,
            alpha = 0.05
        )

        # Time
        tik = time()
        socp_cpu.solve()
        tok = time()

        if verbose:
          print(f"SOCP-solver, CPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        socp_cpu_times[count, i] = tok - tik
        socp_cpu_values[count, i] = socp_cpu.value

        if i == num_iters - 1 and verbose:
          print(f"SOCP-solver, CPU, gets optimal value {socp_cpu.value}")

        # Delete
        del socp_cpu

    if filename is not None:
     
      # Data to save
      data = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "cvxpy_values" : cvxpy_values.reshape((-1)),
              "socp_gpu_times" : socp_gpu_times.reshape((-1)),
              "socp_gpu_values" : socp_gpu_values.reshape((-1)),
              "socp_cpu_times" : socp_cpu_times.reshape((-1)),
              "socp_cpu_values" : socp_cpu_values.reshape((-1))}

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("------ Ending SOCP-test ------")

    return

def test_LASSO(n_values, verbose = False, N = 10, filename = None, use_jax=False):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """
    
    if use_jax:
        from jaxopt import ProximalGradient
        from jaxopt.prox import prox_lasso
        import jax.numpy as jnp

    if verbose:
      print("------ Starting LASSO-test ------")
      
    # Set seed
    np.random.seed(1)

    # Make sure n values are integers
    n_values = np.array(n_values, dtype = np.int32)

    # Calculate m and k values
    m_values = np.array(0.8 * n_values, dtype = np.int32)
    k_values = np.array(0.2 * n_values, dtype = np.int32)

    # How many tests
    num_tests = n_values.shape[0]

    # How many problems per test
    num_problems = 30

    # Containers for storing times
    cvxpy_times = np.zeros((num_tests, N))  # Every row is for a different
    lasso_gpu_times = np.zeros((num_tests, N)) # dimension and every
    lasso_cpu_times = np.zeros((num_tests, N)) # column is a different test
    lasso_jax_times = np.zeros((num_tests, N)) # column is a different test

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N, num_problems))
    lasso_gpu_values = np.zeros((num_tests, N, num_problems))
    lasso_cpu_values = np.zeros((num_tests, N, num_problems))
    lasso_jax_values = np.zeros((num_tests, N, num_problems))

    ### Run test on QP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
      
      if verbose:
        print("\n")
        print(f"n is {n}")
        print("Generate some data")

      if n < 1000:
        num_iters = N 
      elif n < 2500:
        num_iters = int(N/2)
      else:
        num_iters = 3

      for i in range(num_iters):
        
        # Generate data
        num_rows = m*3
        num_nonzero = int(n * num_problems / 4) # create a sparse x_true with this many nonzero entries
        A = np.random.rand(num_rows, n)
        x_true = np.zeros((n, num_problems))
        x_true[np.unravel_index(np.random.randint(0,n*num_problems, num_nonzero),
                        (n, num_problems))] = np.random.uniform(0, 50, num_nonzero)
        reg = 0.05 + 0.01*np.random.randn(num_problems) # give each subproblem a slightly different regularization
        b = A @ x_true + np.random.randn(num_rows, num_problems)
        A_prob = np.hstack((np.ones((num_rows,1)), A)) # add a bias term to A for CVXPY (LASSOSolver will automatically apply)

        if n < 600 and i == num_iters - 1: # CVXPY takes longer on LASSO problems, so run it only one time

            

            # Create CVXPY problem

            # time how long it takes to solve all the problems in CVXPY (we must solve sequentially)
            obj_vals = []

            tik = time()

            for j, r in enumerate(reg):
                x = cp.Variable(n+1)
                obj = cp.Minimize(1/(2*num_rows)*cp.norm2(A_prob  @ x - b[:,j])**2 + r*cp.norm(x[1:], 1))
                prob = cp.Problem(obj, [])
                prob.solve(solver = cp.CLARABEL)
                obj_vals.append(obj.value)

            tok = time()

            if verbose:
              print(f"CVXPY solved {i + 1} time(s). Time: {tok-tik}")

            # Store
            cvxpy_times[count, i] = tok - tik
            cvxpy_values[count, i, :] = np.array(obj_vals)

            if i == num_iters - 1 and verbose:
                print(f"Problem is {prob.status}")
                print(f"CVXPY 5th solution value was {obj_vals[4]}")
                

        # Move on to Lasso-solver on GPU

        # Create instance
        lasso_gpu = LassoSolver(
            A = A, 
            b = b,
            reg=reg,
            rho=0.4,
            max_iters=5000,
            check_stop=10,
            add_bias=True,
            normalize_A=False,
            positive=False,
            compute_loss=False,
            adaptive_rho=False,
            use_gpu=True,
            num_chunks=0,
            eps_rel=1e-6,
            eps_abs=1e-6,
            check_cvxpy=False
        )

        # Time
        tik = time()
        lasso_gpu.solve()
        tok = time()

        if verbose:
          print(f"LASSO-solver, GPU solved {i + 1} time(s). Time: {tok-tik}")

        # Store
        lasso_gpu_times[count, i] = tok - tik
        lasso_gpu_values[count, i, :] = lasso_gpu.solutions.get()
        
        if i == num_iters - 1 and verbose:
            print(f"GPU 5th solution value was {lasso_gpu_values[count, i, 4]}")

        # Delete
        del lasso_gpu
        cupy._default_memory_pool.free_all_blocks()

        # Move on to LASSO-solver on CPU

        # Create instance
        lasso_cpu = LassoSolver(
            A = A, 
            b = b,
            reg=reg,
            rho=0.4,
            max_iters=1000,
            check_stop=10,
            add_bias=True,
            normalize_A=False,
            positive=False,
            compute_loss=False,
            adaptive_rho=False,
            use_gpu=False,
            num_chunks=0,
            eps_rel=1e-6,
            eps_abs=1e-6,
            check_cvxpy=False
        )

        # Time
        tik = time()
        lasso_cpu.solve()
        tok = time()

        if verbose:
          print(f"LASSO-solver, CPU solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        lasso_cpu_times[count, i] = tok - tik
        lasso_cpu_values[count, i, :] = lasso_cpu.solutions
        
        if i == num_iters - 1 and verbose:
            print(f"CPU 5th solution value was {lasso_cpu_values[count, i, 4]}")
        
        

        # Delete
        del lasso_cpu
        
        if use_jax:

            def least_squares(w, data):
              X,y  = data
              residuals = jnp.dot(X, w) - y
              return jnp.mean(residuals ** 2)

            l1reg = 1.0
            pg = ProximalGradient(fun=least_squares, prox=prox_lasso)

            pg_sols = []

            tik = time()
            for j, r in enumerate(reg):
                pg_sol = pg.run(jnp.array(np.zeros(A_prob.shape[1])), hyperparams_prox=r, data=(A_prob, b[:,j])).params
                pg_sols.append(pg_sol)
            tok = time()

            if verbose:
              print(f"JAX-opt solved {i + 1} time(s). Time: {tok-tik}")

            opt_vals = [1/(2*A.shape[0]) * np.linalg.norm(A_prob@pg_sol - b[:,i])**2 + reg[i]*np.linalg.norm(pg_sol, 1) for i, pg_sol in enumerate(pg_sols)]

            # Store
            lasso_jax_times[count, i] = tok - tik
            lasso_jax_values[count, i, :] = opt_vals

            if i == num_iters - 1 and verbose:
              print(f"JAX-opt gets 5th optimal value {opt_vals[4]}")

    if filename is not None:
     
      # Data to save
      data_times = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "lasso_gpu_times" : lasso_gpu_times.reshape((-1)),
              "lasso_cpu_times" : lasso_cpu_times.reshape((-1)),}

      data_values = {"cvxpy_values" : cvxpy_values.reshape((-1)),
              "lasso_gpu_values" : lasso_gpu_values.reshape((-1)),
              "lasso_cpu_values" : lasso_cpu_values.reshape((-1))}
        
      if use_jax:
        data_values['lasso_jax_values'] = lasso_jax_values.reshape((-1))
        data_times['lasso_jax_times'] = lasso_jax_times.reshape((-1))

      df_times = pd.DataFrame(data_times)
      df_values = pd.DataFrame(data_values)

      filename_times = filename[:-4] + "Times" + ".csv"
      filename_values = filename[:-4] + "Values" + ".csv"

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df_times.to_csv(filename_times, index=False, na_rep='NA')
      df_values.to_csv(filename_values, index=False, na_rep='NA')

      # Add a comment to the CSV file with times for parsing
      with open(filename_times, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("------ Ending LASSO-test ------")

    return

def test_all_solvers(n_values, verbose = False, N = 10, filename = None, use_jax=False):
  """
  Test all solvers
  """
  #test_LP(n_values, verbose, N, filename + "LP.csv", use_jax)
  test_QP(n_values, verbose, N, filename + "QP.csv", use_jax)
  #test_SOCP(n_values, verbose, N, filename + "SOCP.csv")
  #test_LASSO(n_values, verbose, N, filename + "LASSO.csv", use_jax)

  return

def parse_csv(filename, origin):
  
  # Read the header (first line)
  header = pd.read_csv(filename, nrows=1, header=None)

  # Read the data, skipping the first line
  data = pd.read_csv(filename, skiprows=1)

  # Parse header
  num_tests = header[0].to_numpy()[0]
  N = header[1].to_numpy()[0]

  # Parse common data
  n_values_repeated = data["n_values"].to_numpy()
  n_values_matrix = np.reshape(n_values_repeated, (-1, N))
  n_values = n_values_matrix[:, 0]
  cvxpy_times = np.reshape(data["cvxpy_times"].to_numpy(), (num_tests, N))
  cvxpy_values = np.reshape(data["cvxpy_values"].to_numpy(), (num_tests, N))
  
  if origin == "LP":
    # Parse unique data for LP
    ls_gpu_times = np.reshape(data["ls_gpu_times"].to_numpy(), (num_tests, N))
    ls_gpu_values = np.reshape(data["ls_gpu_values"].to_numpy(), (num_tests, N))
    ls_cpu_times = np.reshape(data["ls_cpu_times"].to_numpy(), (num_tests, N))
    ls_cpu_values = np.reshape(data["ls_cpu_values"].to_numpy(), (num_tests, N))

    return N, num_tests, n_values, cvxpy_times, cvxpy_values, ls_gpu_times, \
          ls_gpu_values, ls_cpu_times, ls_cpu_values

  elif origin == "QP":
    # Parse unique data for QP
    qp_gpu_times = np.reshape(data["qp_gpu_times"].to_numpy(), (num_tests, N))
    qp_gpu_values = np.reshape(data["qp_gpu_values"].to_numpy(), (num_tests, N))
    qp_cpu_times = np.reshape(data["qp_cpu_times"].to_numpy(), (num_tests, N))
    qp_cpu_values = np.reshape(data["qp_cpu_values"].to_numpy(), (num_tests, N))

    return N, num_tests, n_values, cvxpy_times, cvxpy_values, qp_gpu_times, \
          qp_gpu_values, qp_cpu_times, qp_cpu_values

  elif origin == "SOCP":
    # Parse unique data for SOCP
    socp_gpu_times = np.reshape(data["socp_gpu_times"].to_numpy(), (num_tests, N))
    socp_gpu_values = np.reshape(data["socp_gpu_values"].to_numpy(), (num_tests, N))
    socp_cpu_times = np.reshape(data["socp_cpu_times"].to_numpy(), (num_tests, N))
    socp_cpu_values = np.reshape(data["socp_cpu_values"].to_numpy(), (num_tests, N))

    return N, num_tests, n_values, cvxpy_times, cvxpy_values, socp_gpu_times, \
          socp_gpu_values, socp_cpu_times, socp_cpu_values

  elif origin == "LASSO":
    raise NotImplementedError("Not implemented parsing of Lasso data")

  else:
    raise ValueError("Invalid origin passed to parse data")

def plot_results(filename, origin):
  
  # Parse data
  N, num_tests, n_values, cvxpy_times, cvxpy_values, gpu_times, \
  gpu_values, cpu_times, cpu_values = parse_csv(filename, origin)

  # Calculate averages
  cvxpy_time_average = cvxpy_times.sum(axis = 1) / N
  gpu_time_average = gpu_times.sum(axis = 1) / N
  cpu_time_average = cpu_times.sum(axis = 1) / N

  ### Plot results
  x_ticks = np.arange(len(n_values))
  x_labels = n_values
  plt.title("Graph of average solving times using T4 GPU in Google Colab")
  plt.plot(cvxpy_time_average, label = "CVXPY")
  plt.plot(gpu_time_average, label = origin + "-solver GPU")
  plt.plot(cpu_time_average, label = origin + "-solver CPU")
  plt.xticks(ticks = x_ticks, labels = x_labels)
  plt.xlabel("Dimension n")
  plt.ylabel("Average solving time [s]")
  plt.legend()
  plt.show()

  dims = n_values
  timeResults = {
      'CVXPY': cvxpy_time_average,
      'GPU': gpu_time_average,
      'CPU': cpu_time_average,
  }

  x = np.arange(len(dims))  # the label locations
  width = 0.25  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in timeResults.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attribute)
      ax.bar_label(rects, labels=[f'{val:.2f}' for val in measurement], padding=3)
      multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel("Average solving time [s]")
  ax.set_xlabel("Dimension n")
  ax.set_title("Bar plot of average solving times using T4 GPU in Google Colab")
  ax.set_xticks(x + width, dims)
  ax.legend()

  plt.show()

def main(use_jax=True):
  if use_jax:
        import jax.numpy as jnp
        from jaxopt import BoxOSQP
  n_values = np.array([100, 200, 300,400,500,600,700,800,900,1000,1200,1300,1500,1750,2000,2500,3000,4000,5000]) # Every problem dimension to try
  #n_values = np.array([100])
  verbose = True # Should probably be False
  N = 10 # Number of tests for each dimension
  filename = "timing_results_jax/testResultsTake2" # filename, files will have this as base and then stuff added

  test_all_solvers(n_values, verbose, N, filename, use_jax)

if __name__ == "__main__":
  main()
