## testin LP-solver

import numpy as np
import matplotlib.pyplot as plt
from LPSolver import LPSolver
from QPSolver import QPSolver
from SOCPSolver import SOCPSolver
import cvxpy as cp
from time import time

# Generate random, dense data and compare
np.random.seed(1)
n_values = [100, 500, 1000, 2000, 3000]
m_values = [80, 400, 800, 1600, 2400]
k_values = [20, 100, 200, 400, 600]

cvxpy_times_random = []
ls_gpu_times_random = []
ls_cpu_times_random = []


### Run test on LP

for n, m, k in zip(n_values[3:4], m_values[3:4], k_values[3:4]):
  print(f"n is {n}")
  print("Generate some data")
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
  print("Solve in CVXPY")
  total_cvxpy_time = 0
  for k in range(3):

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
    print(f"CVXPY solved {k} time(s)")
    tok = time()

    total_cvxpy_time += tok - tik

    if k == 2:
        print(f"Problem is {prob.status}")
        print(f"CVXPY gets optimal value of {obj.value}")

    del x
    del constr
    del prob



  cvxpy_time = total_cvxpy_time / 3
  cvxpy_times_random.append(cvxpy_time)

  print("LP-solver, GPU")
  total_ls_gpu_time = 0

  for k in range(3):
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
        beta = 0.5,
        alpha = 0.05
    )

    tik = time()
    ls_gpu.solve()
    print(f"LP-solver, GPU solved {k} time(s)")
    tok = time()
    total_ls_gpu_time += tok - tik

    if k == 2:
      print(f"LP-solver, GPU, gets optimal value {ls_gpu.value}")

    del ls_gpu

  ls_gpu_time = total_ls_gpu_time / 3
  ls_gpu_times_random.append(ls_gpu_time)

  print("LP-solver, CPU")
  total_ls_cpu_time = 0

  for k in range(3):
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
        beta = 0.5,
        alpha = 0.05
    )

    tik = time()
    ls_cpu.solve()
    print(f"LP-solver, CPU solved {k} time(s)")
    tok = time()
    total_ls_cpu_time += tok - tik

    if k == 2:
      print(f"LP-solver, CPU, gets optimal value {ls_cpu.value}")

    del ls_cpu

  ls_cpu_time = total_ls_cpu_time / 3
  ls_cpu_times_random.append(ls_cpu_time)

  print(f"Results for n = {n}")
  print(f"CVXPY average time: {cvxpy_time}")
  print(f"LS GPU average time: {ls_gpu_time}")
  print(f"LS CPU average time: {ls_cpu_time}")
  print("\n")

### Plot results

x_ticks = np.arange(len(n_values))
x_labels = n_values
plt.title("Graph of average solving times using T4 GPU in Google Colab")
plt.plot(cvxpy_times_random, label = "CVXPY")
plt.plot(ls_gpu_times_random, label = "LP-solver GPU")
plt.plot(ls_cpu_times_random, label = "LP-solver CPU")
plt.xticks(ticks = x_ticks, labels = x_labels)
plt.xlabel("Dimension n")
plt.ylabel("Average solving time [s]")
#plt.yscale("log")
plt.legend()
plt.show()

dims = n_values
timeResults = {
    'CVXPY': cvxpy_times_random,
    'LS_GPU': ls_gpu_times_random,
    'LS_CPU': ls_cpu_times_random,
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
#ax.set_yscale("log")
ax.legend()

plt.show()

### Test QP
# Generate random data
np.random.seed(1)
n_values = [100, 500, 1000, 2000, 3000]
m_values = [80, 400, 800, 1600, 2400]
k_values = [20, 100, 200, 400, 600]

cvxpy_times_random = []
qp_gpu_times_random = []
qp_cpu_times_random = []

for n, m, k in zip(n_values, m_values, k_values):
  print(f"n is {n}")
  print("Generate some data")
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
  print("Solve in CVXPY")
  total_cvxpy_time = 0
  for k in range(3):

    # Solve in CVXPY
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
    print(f"CVXPY solved {k} time(s)")
    tok = time()

    total_cvxpy_time += tok - tik

    if k == 2:
        print(f"Problem is {prob.status}")
        print(f"CVXPY gets optimal value of {obj.value}")

    del x
    del constr
    del prob

  cvxpy_time = total_cvxpy_time / 3
  cvxpy_times_random.append(cvxpy_time)

  print("QP-solver, GPU")
  total_qp_gpu_time = 0

  for k in range(3):
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
        max_outer_iters = 200,
        max_inner_iters = 200)

    tik = time()
    qp_gpu.solve()
    print(f"QP-solver, GPU solved {k} time(s)")
    tok = time()
    total_qp_gpu_time += tok - tik

    if k == 2:
      print(f"QP-solver, GPU, gets optimal value {qp_gpu.value}")

    del qp_gpu

  qp_gpu_time = total_qp_gpu_time / 3
  qp_gpu_times_random.append(qp_gpu_time)

  print("QP-solver, CPU")
  total_qp_cpu_time = 0

  for k in range(3):
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
        max_outer_iters = 200,
        max_inner_iters = 200
    )

    tik = time()
    qp_cpu.solve()
    print(f"QP-solver, CPU solved {k} time(s)")
    tok = time()
    total_qp_cpu_time += tok - tik

    if k == 2:
      print(f"QP-solver, CPU, gets optimal value {qp_cpu.value}")

    del qp_cpu

  qp_cpu_time = total_qp_cpu_time / 3
  qp_cpu_times_random.append(qp_cpu_time)

  print(f"Results for n = {n}")
  print(f"CVXPY average time: {cvxpy_time}")
  print(f"QP GPU average time: {qp_gpu_time}")
  print(f"QP CPU average time: {qp_cpu_time}")
  print("\n")