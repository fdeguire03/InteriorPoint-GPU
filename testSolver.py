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
import jax.numpy as jnp
from jaxopt import BoxOSQP

def test_LP(n_values, verbose = False, N = 10, filename = None):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """

    if verbose:
      print("------ Starting LP-test ------\n")
      
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
    jax_times = np.zeros((num_tests, N))

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N))
    ls_gpu_values = np.zeros((num_tests, N))
    ls_cpu_values = np.zeros((num_tests, N))
    jax_values = np.zeros((num_tests, N))

    ### Run test on LP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
      
      if verbose:
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

        # JAX-opt
        # Identity
        identity = jnp.eye(n)
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
              "ls_cpu_values" : ls_cpu_values.reshape((-1)),
              "jax_times" : jax_times.reshape((-1)),
              "jax_values" : jax_values.reshape(-1)}

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("\n------ Ending LP-test ------")

    return

def test_LP_sparse(data_dir, data_files, verbose = False, N = 10, filename = None):
  """
  Takes in directory and list .npy files. Compares LP-solver to CVXPY on these files
  """

  cvxpy_times = np.zeros((len(data_files), N))
  ls_gpu_times = np.zeros((len(data_files), N))
  ls_cpu_times = np.zeros((len(data_files), N))

  for count, data_file in enumerate(data_files):

    if verbose:
      print(f"For file {data_file}")

    with open(data_dir + data_file, 'rb') as f:

      c = np.load(f)
      A = np.load(f)
      b = np.load(f)
      C = np.load(f)
      d = np.load(f)
      up_bnd = np.load(f)
      lo_bnd = np.load(f)
      n = c.shape[0]

      if verbose:
        print(f"Dimension is n = {n}")
        print(f"Number of equality constraints are m = {A.shape[0]}")

      # Solve in CVXPY N times
      for i in range(N):

        # Variable
        x = cp.Variable(n)
        
        # Objective
        obj = c @ x

        # Constraints
        constr = [A @ x == b, C @ x <= d, lo_bnd <= x, x <= up_bnd]

        # Problem
        prob = cp.Problem(cp.Minimize(obj), constr)

        # Solve
        tik = time()
        prob.solve()
        tok = time()

        if verbose:
          print(f"CVXPY solved {i + 1} time(s). Time: {tok-tik}")

        if i == N - 1 and verbose:
          print(f"CVXPY got objective value of {obj.value}")

        # Store time
        cvxpy_times[count, i] = tok - tik

      # Solve using LP-solver, GPU, N times
      for i in range(N):
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

        if i == N - 1 and verbose:
          print(f"LP-solver, GPU, gets optimal value {ls_gpu.value}")

        # Delete, clear GPU memory
        del ls_gpu
        cupy._default_memory_pool.free_all_blocks()

      # Solve using LP-solver, CPU, N times
      for i in range(N):
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

        if i == N - 1 and verbose:
          print(f"LP-solver, CPU, gets optimal value {ls_cpu.value}")

        # Delete, clear GPU memory
        del ls_cpu
        cupy._default_memory_pool.free_all_blocks()

  if filename is not None:
     
      # Data to save
      data = {"cvxpy_times" : cvxpy_times.reshape((-1)),
              "ls_gpu_times" : ls_gpu_times.reshape((-1)),
              "ls_cpu_times" : ls_cpu_times.reshape((-1))}

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)


  return 

def test_QP(n_values, verbose = False, N = 10, filename = None):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """

    if verbose:
      print("------ Starting QP-test ------\n")
      
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

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N))
    qp_gpu_values = np.zeros((num_tests, N))
    qp_cpu_values = np.zeros((num_tests, N))

    ### Run test on QP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
      
      if verbose:
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
            epsilon = 1e-4, 
            mu = 15, 
            t0 = 1,
            max_outer_iters = 20,
            max_inner_iters = 50, 
            beta = 0.5,
            alpha = 0.05
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

    if filename is not None:
     
      # Data to save
      data = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "cvxpy_values" : cvxpy_values.reshape((-1)),
              "qp_gpu_times" : qp_gpu_times.reshape((-1)),
              "qp_gpu_values" : qp_gpu_values.reshape((-1)),
              "qp_cpu_times" : qp_cpu_times.reshape((-1)),
              "qp_cpu_values" : qp_cpu_values.reshape((-1))}

      df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file, filling missing values with a specific value
      df.to_csv(filename, index=False, na_rep='NA')

      # Add a comment to the CSV file
      with open(filename, 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(f"{num_tests},{N}\n" + content)

    if verbose:
      print("\n------ Ending QP-test ------")

    return

def test_SOCP(n_values, verbose = False, N = 10, filename = None):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """

    if verbose:
      print("------ Starting SOCP-test ------\n")
      
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
      
      if verbose:
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
        num_con = 10
        A = []
        b = []
        c = []
        d = []
        x0 = np.random.randn(n)
        for i in range(num_con):
            A.append(np.random.randn(m, n))
            b.append(np.random.randn(m))
            c.append(np.random.randn(n))
            d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
        F = np.random.randn(k, n)
        g = F @ x0

        # Create CVXPY problem

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
            t0 = 1,
            max_inner_iters = 20, 
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
      print("\n------ Ending SOCP-test ------")

    return

def test_LASSO(n_values, verbose = False, N = 10, filename = None):
    """
    Takes in a numpy vector n_values. Every element is the
    number of dimensions for every test. If verbose if True, 
    the solver will print throughout the test. N is how many 
    times every problem should be solved.
    """

    if verbose:
      print("------ Starting LASSO-test ------\n")
      
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

    # Containers for storing optimal values
    cvxpy_values = np.zeros((num_tests, N, num_problems))
    lasso_gpu_values = np.zeros((num_tests, N, num_problems))
    lasso_cpu_values = np.zeros((num_tests, N, num_problems))

    ### Run test on QP
    for count, n, m, k in zip(np.arange(num_tests), n_values, m_values, k_values):
      
      if verbose:
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
        A = np.hstack((np.ones((num_rows,1)), A))

        # Create CVXPY problem

        # time how long it takes to solve all the problems in CVXPY (we must solve sequentially)
        obj_vals = []

        tik = time()

        for r in reg:
            x = cp.Variable(n+1)
            obj = cp.Minimize(1/(2*num_rows)*cp.norm2(A @ x - b[:,0])**2 + r*cp.norm(x[1:], 1))
            prob = cp.Problem(obj, [])
            prob.solve(solver = cp.CLARABEL)
            obj_vals.append(obj.value)

        tok = time()

        if verbose:
          print(f"CVXPY solved {i + 1} time(s). Time: {tok-tik}")
        
        # Store
        cvxpy_times[count, i] = tok - tik
        cvxpy_values[count, i, :] = np.array(obj_vals)

        if count == 0 and i == 0:
          print("CVXPY gets: ", obj_vals)

        if i == num_iters - 1 and verbose:
            print(f"Problem is {prob.status}")

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
        lasso_gpu_values[count, i, :] = lasso_gpu.objective().get()

        if count == 0 and i == 0:
          print("LASSO GPU gets: ", lasso_gpu.objective().get())

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
            eps_rel=1e-3,
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
        lasso_cpu_values[count, i, :] = lasso_cpu.objective()

        # Delete
        del lasso_cpu

    if filename is not None:
     
      # Data to save
      data_times = {"n_values" : np.repeat(n_values, N),
              "cvxpy_times" : cvxpy_times.reshape((-1)),
              "lasso_gpu_times" : lasso_gpu_times.reshape((-1)),
              "lasso_cpu_times" : lasso_cpu_times.reshape((-1)),}

      data_values = {"cvxpy_values" : cvxpy_values.reshape((-1)),
              "lasso_gpu_values" : lasso_gpu_values.reshape((-1)),
              "lasso_cpu_values" : lasso_cpu_values.reshape((-1))}

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
      print("\n------ Ending LASSO-test ------")

    return

def test_all_solvers(n_values, verbose = False, N = 10, filename = None):
  """
  Test all solvers
  """
  test_LP(n_values, verbose, N, filename + "LP.csv")
  test_QP(n_values, verbose, N, filename + "QP.csv")
  test_SOCP(n_values, verbose, N, filename + "SOCP.csv")
  test_LASSO(n_values, verbose, N, filename + "LASSO.csv")

  return

def parse_csv(filename, origin):
  
  if origin == "LASSO":
    # assuming things are stored in 'filenameTimes.csv' and 'filenameValues.csv' 
    filenameTimes = filename[:-4] + "Times.csv"
    filenameValues = filename[:-4] + "Values.csv"

    # Read the header (first line)
    header = pd.read_csv(filenameTimes, nrows=1, header=None)

    # Read the data, skipping the first line
    data_times = pd.read_csv(filenameTimes, skiprows=1)
    data_values = pd.read_csv(filenameValues, skiprows= 0)
    # Fix name error
    data_values = data_values.rename(columns = {"socp_gpu_values" : "lasso_gpu_values", 
                                                "socp_cpu_values" : "lasso_cpu_values"})
    
    # Parse header
    num_tests = header[0].to_numpy()[0]
    N = header[1].to_numpy()[0]

    # Parse data, time
    # n_values
    n_values_repeated = data_times["n_values"].to_numpy()
    n_values_matrix = np.reshape(n_values_repeated, (-1, N))
    n_values = n_values_matrix[:, 0]

    cvxpy_times = np.reshape(data_times["cvxpy_times"].to_numpy(), (num_tests, N))
    lasso_gpu_times = np.reshape(data_times["lasso_gpu_times"].to_numpy(), (num_tests, N))
    lasso_cpu_times = np.reshape(data_times["lasso_cpu_times"].to_numpy(), (num_tests, N))

    # Parse data, values
    num_problems = 30 # fixed from testing
    
    cvxpy_values = np.reshape(data_values["cvxpy_values"].to_numpy(), 
                              (num_tests, N, num_problems))
    lasso_gpu_values = np.reshape(data_values["lasso_gpu_values"].to_numpy(),
                              (num_tests, N, num_problems))
    lasso_cpu_values = np.reshape(data_values["lasso_cpu_values"].to_numpy(),
                              (num_tests, N, num_problems))

    return N, num_tests, n_values, cvxpy_times, cvxpy_values, lasso_gpu_times, \
            lasso_gpu_values, lasso_cpu_times, lasso_cpu_values

  else:  
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
    
    if origin == "LP":
      # Parse unique data for LP
      cvxpy_times = np.reshape(data["cvxpy_times"].to_numpy(), (num_tests, N))
      cvxpy_values = np.reshape(data["cvxpy_values"].to_numpy(), (num_tests, N))
      ls_gpu_times = np.reshape(data["ls_gpu_times"].to_numpy(), (num_tests, N))
      ls_gpu_values = np.reshape(data["ls_gpu_values"].to_numpy(), (num_tests, N))
      ls_cpu_times = np.reshape(data["ls_cpu_times"].to_numpy(), (num_tests, N))
      ls_cpu_values = np.reshape(data["ls_cpu_values"].to_numpy(), (num_tests, N))
      jax_times = np.reshape(data["jax_times"].to_numpy(), (num_tests, N))
      jax_values = np.reshape(data["jax_values"].to_numpy(), (num_tests, N))

      return N, num_tests, n_values, cvxpy_times, cvxpy_values, ls_gpu_times, \
            ls_gpu_values, ls_cpu_times, ls_cpu_values, jax_times, jax_values

    elif origin == "QP":
      # Parse unique data for QP
      cvxpy_times = np.reshape(data["cvxpy_times"].to_numpy(), (num_tests, N))
      cvxpy_values = np.reshape(data["cvxpy_values"].to_numpy(), (num_tests, N))
      qp_gpu_times = np.reshape(data["qp_gpu_times"].to_numpy(), (num_tests, N))
      qp_gpu_values = np.reshape(data["qp_gpu_values"].to_numpy(), (num_tests, N))
      qp_cpu_times = np.reshape(data["qp_cpu_times"].to_numpy(), (num_tests, N))
      qp_cpu_values = np.reshape(data["qp_cpu_values"].to_numpy(), (num_tests, N))

      return N, num_tests, n_values, cvxpy_times, cvxpy_values, qp_gpu_times, \
            qp_gpu_values, qp_cpu_times, qp_cpu_values

    elif origin == "SOCP":
      # Parse unique data for SOCP
      cvxpy_times = np.reshape(data["cvxpy_times"].to_numpy(), (num_tests, N))
      cvxpy_values = np.reshape(data["cvxpy_values"].to_numpy(), (num_tests, N))
      socp_gpu_times = np.reshape(data["socp_gpu_times"].to_numpy(), (num_tests, N))
      socp_gpu_values = np.reshape(data["socp_gpu_values"].to_numpy(), (num_tests, N))
      socp_cpu_times = np.reshape(data["socp_cpu_times"].to_numpy(), (num_tests, N))
      socp_cpu_values = np.reshape(data["socp_cpu_values"].to_numpy(), (num_tests, N))

      return N, num_tests, n_values, cvxpy_times, cvxpy_values, socp_gpu_times, \
            socp_gpu_values, socp_cpu_times, socp_cpu_values

    else:
      raise ValueError("Invalid origin passed to parse data")

def get_result(filename, origin):

  print("------ Getting results for " + filename + " ------ \n")
  
  # Parse data
  if origin == "LP":

    N, num_tests, n_values, cvxpy_times, cvxpy_values, gpu_times, \
    gpu_values, cpu_times, cpu_values , jax_times, jax_values = \
    parse_csv(filename, origin)

    # Calculate average relative error
    ls_gpu_av_err = calculate_average_relative_error(cvxpy_values, gpu_values)
    ls_cpu_av_err = calculate_average_relative_error(cvxpy_values, cpu_values)
    jax_av_err = calculate_average_relative_error(cvxpy_values, jax_values)

    # Print average realtive error
    print("Considering CVXPY as ground truth: ")
    print(f"For LP-solver using GPU, the average relative error is {ls_gpu_av_err}")
    print(f"For LP-solver using CPU, the average relative error is {ls_cpu_av_err}")
    print(f"For JAX, the average relative error is {jax_av_err}")

    # Calculate time averages
    cvxpy_time_average = cvxpy_times.sum(axis = 1) / N
    gpu_time_average = gpu_times.sum(axis = 1) / N
    cpu_time_average = cpu_times.sum(axis = 1) / N
    jax_time_average = jax_times.sum(axis = 1) / N

    ### Plot results
    x_ticks = np.arange(len(n_values))
    x_labels = n_values
    plt.title("Graph of average solving times for " + origin)
    plt.plot(cvxpy_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = "LP-solver GPU")
    plt.plot(cpu_time_average, label = "LP-solver CPU")
    plt.plot(jax_time_average, label = "JAXopt")
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
        'JAX': jax_time_average
    }

    x = np.arange(len(dims))  # the label locations
    width = 0.15  # the width of the bars
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
    ax.set_title("Bar plot of average solving times for " + origin)
    ax.set_xticks(x + width, dims)
    ax.legend()

    plt.show()

  else:
  
    N, num_tests, n_values, cvxpy_times, cvxpy_values, gpu_times, \
    gpu_values, cpu_times, cpu_values = parse_csv(filename, origin)
    
    # Calculate average relative error
    gpu_av_err = calculate_average_relative_error(cvxpy_values, gpu_values, origin == "LASSO")
    cpu_av_err = calculate_average_relative_error(cvxpy_values, cpu_values, origin == "LASSO")
    
    # Print average relative error
    print("Considering CVXPY as ground truth: ")
    print(f"For {origin}-solver using GPU, the average relative error is {gpu_av_err}")
    print(f"For {origin}-solver using CPU, the average relative error is {cpu_av_err}")

    # Calculate averages
    cvxpy_time_average = cvxpy_times.sum(axis = 1) / N
    gpu_time_average = gpu_times.sum(axis = 1) / N
    cpu_time_average = cpu_times.sum(axis = 1) / N

    ### Plot results
    x_ticks = np.arange(len(n_values))
    x_labels = n_values
    plt.title("Graph of average solving times for " + origin)
    plt.plot(cvxpy_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = origin + "-solver GPU")
    plt.plot(cpu_time_average, label = origin + "-solver CPU")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.show()

    # Bar plot
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
    ax.set_title("Bar plot of average solving times for " + origin)
    ax.set_xticks(x + width, dims)
    ax.legend()

    plt.show()

def calculate_average_relative_error(truth, test_results, lasso = False):
  if not lasso:
    num_tests, N = truth.shape

    # Container
    dim_wise_average_error = []

    # Do naively since need to handle inf values in nice way
    # Can probably be done vectorized
    for test in range(num_tests):
      test_truth = truth[test, :]
      test_result = test_results[test, :]

      # Mask out all real valued
      mask = test_result != np.inf

      print(f"Test {test + 1} had {N - mask.sum()} out of {N} iterations not converge")

      if mask.sum() == 0:
        # No tests converged, punish
        dim_wise_average_error.append(np.inf)
        continue

      rel_error_testwise = np.abs(test_truth[mask] - test_result[mask]) / \
                           test_truth[mask]

      rel_average_error = rel_error_testwise.sum() / mask.sum()

      dim_wise_average_error.append(rel_average_error)

    return sum(dim_wise_average_error) / num_tests

  else:
    # Know getting LASSO-values
    num_tests, N, num_problems = truth.shape

    # Container
    dim_wise_average_error = []

    for test in range(num_tests):
      test_truth = truth[test, :, :]
      test_result = test_results[test, :, :]

      # For every iteration:
      for n in range(N):
        lasso_truth = test_truth[n, :]
        lasso_result = test_result[n, :]

        print(lasso_truth)
        print(lasso_result)

        assert False
      
        # Mask out all real valued
        mask = lasso_result != np.inf

        print(f"Test {test + 1}. iteration {n + 1} had {num_problems - mask.sum()}" + \
        f" out of {num_problems} problems not converge")

        if mask.sum() == 0:
          # No tests converged, punish
          dim_wise_average_error.append(np.inf)
          continue
        
        rel_error_testwise = np.abs(lasso_truth[mask] - lasso_result[mask]) / \
                            lasso_truth[mask]

        rel_average_error = rel_error_testwise.sum() / mask.sum()

        dim_wise_average_error.append(rel_average_error)

    return sum(dim_wise_average_error) / len(dim_wise_average_error)

def summarize_results(filename, LP = True, QP = True, SOCP = True, LASSO = True):
  """
  Assumes file naming convetion as in 'test_all_solvers'.
  Summarizes all test results by calling 'get_result' for LP, QP, SOCP and LASSO
  if called upon
  """
  if LP:
    try:
      get_result(filename + "LP.csv", "LP")
    except Exception as e:
      print(e)
  if QP:
    try:
      get_result(filename + "QP.csv", "QP")
    except Exception as e:
      print(e)
  if SOCP:
    try:
      get_result(filename + "SOCP.csv", "SOCP")
    except Exception as e:
      print(e)
  if LASSO:
    try:
      get_result(filename + "LASSO.csv", "LASSO")
    except Exception as e:
      print(e)

  return

def main():
  """
  Runs test for all solvers and saves to file.
  To summarize results with plots and get average realitive
  error, use 'summarize_results'
  """

  print("script started", flush=True)
  n_values = np.array([100, 200, 300,400,500,600,700,800,900,1000,1200,1300,1500,1750,2000,2500,3000,4000,5000]) # Every problem dimension to try
  verbose = True # Should probably be False
  N = 10 # Number of tests for each dimension
  filename = "timing_results/testResults" # filename, files will have this as base and then stuff added

  test_all_solvers(n_values, verbose, N, filename)

  # test sparse as well, special function call
  test_LP_sparse("example_data/", ["aflow40b.npy", "30n20b8.npy"], 
                  verbose = True, N = 2, filename = "testResultsSparse.csv")
                                  # N should be rather low, LP-solver CPU
                                  # Takes 320 seconds on Colab. Of course, if 
                                  # ran on Sherlock can be higher

if __name__ == "__main__":
  main()
