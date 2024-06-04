### File for parsing and plotting results

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
      # Parse unique data for LP
      cvxpy_times = np.reshape(data["cvxpy_times"].to_numpy(), (num_tests, N))
      cvxpy_values = np.reshape(data["cvxpy_values"].to_numpy(), (num_tests, N))
      qp_gpu_times = np.reshape(data["qp_gpu_times"].to_numpy(), (num_tests, N))
      qp_gpu_values = np.reshape(data["qp_gpu_values"].to_numpy(), (num_tests, N))
      qp_cpu_times = np.reshape(data["qp_cpu_times"].to_numpy(), (num_tests, N))
      qp_cpu_values = np.reshape(data["qp_cpu_values"].to_numpy(), (num_tests, N))
      jax_times = np.reshape(data["jax_times"].to_numpy(), (num_tests, N))
      jax_values = np.reshape(data["jax_values"].to_numpy(), (num_tests, N))

      return N, num_tests, n_values, cvxpy_times, cvxpy_values, qp_gpu_times, \
            qp_gpu_values, qp_cpu_times, qp_cpu_values, jax_times, jax_values
      
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
  if origin == "LP" or origin == "QP":

    N, num_tests, n_values, cvxpy_times, cvxpy_values, gpu_times, \
    gpu_values, cpu_times, cpu_values , jax_times, jax_values = \
    parse_csv(filename, origin)

    # Calculate average relative error
    ls_gpu_av_err = calculate_average_relative_error(cvxpy_values, gpu_values)
    ls_cpu_av_err = calculate_average_relative_error(cvxpy_values, cpu_values)
    jax_av_err = calculate_average_relative_error(cvxpy_values, jax_values)

    # Print average relative error
    print("Considering CVXPY as ground truth: ")
    print(f"For LP-solver using GPU, the average relative error is {ls_gpu_av_err}")
    print(f"For LP-solver using CPU, the average relative error is {ls_cpu_av_err}")
    print(f"For JAX, the average relative error is {jax_av_err}")

    # Calculate time averages
    cvxpy_time_average = cvxpy_times.sum(axis = 1) / N
    gpu_time_average = gpu_times.sum(axis = 1) / N
    cpu_time_average = cpu_times.sum(axis = 1) / N
    jax_time_average = jax_times.sum(axis = 1) / N

    # Calculate standard deviations
    cvxpy_std = np.std(cvxpy_times, axis = 1)
    gpu_std = np.std(gpu_times, axis = 1)
    cpu_std = np.std(cpu_times, axis = 1)
    jax_std = np.std(jax_times, axis = 1)

    ### Plot results
    x_ticks = np.arange(len(n_values))
    x_labels = n_values
    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin)
    plt.plot(cvxpy_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = origin + "-solver GPU")
    plt.plot(cpu_time_average, label = origin + "-solver CPU")
    plt.plot(jax_time_average, label = "JAXopt")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLinearLinePlot.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin)
    plt.plot(cvxpy_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = origin + "-solver GPU")
    plt.plot(cpu_time_average, label = origin + "-solver CPU")
    plt.plot(jax_time_average, label = "JAXopt")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.yscale("log")
    plt.savefig("testing/plots/" + origin + "averageLogarithmicLinePlot.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin)
    plt.errorbar(x_ticks, cvxpy_time_average, yerr = cvxpy_std, label = "CVXPY")
    plt.errorbar(x_ticks, gpu_time_average, yerr = gpu_std, label = origin + "-solver GPU")
    plt.errorbar(x_ticks, cpu_time_average, yerr = cpu_std, label = origin + "-solver CPU")
    plt.errorbar(x_ticks, jax_time_average, yerr = jax_std, label = "JAXopt")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "errorbarLinear.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin + " on a log-scale")
    plt.errorbar(x_ticks, cvxpy_time_average, yerr = cvxpy_std, label = "CVXPY")
    plt.errorbar(x_ticks, gpu_time_average, yerr = gpu_std, label = origin + "-solver GPU")
    plt.errorbar(x_ticks, cpu_time_average, yerr = cpu_std, label = origin + "LP-solver CPU")
    plt.errorbar(x_ticks, jax_time_average, yerr= jax_std, label = "JAXopt")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.yscale("log")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "errorbarLogarithmic.png")
    plt.clf()

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

    plt.figure(figsize=(10, 5))
    for attribute, measurement in timeResults.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding = 3)#, labels=[f'{val:.2f}' for val in measurement])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Average solving time [s]")
    plt.xlabel("Dimension n")
    plt.title("Bar plot of average solving times for " + origin)
    plt.xticks(x + width, dims)
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLinearBarPlot.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    for attribute, measurement in timeResults.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding = 3)#, labels=[f'{val:.2f}' for val in measurement])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Average solving time [s]")
    plt.xlabel("Dimension n")
    plt.title("Bar plot of average solving times for " + origin + " on a log-scale")
    plt.xticks(x + width, dims)
    plt.yscale("log")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLogarithmicBarPlot.png")
    plt.clf()

  else:

    N, num_tests, n_values, truth_times, truth_values, gpu_times, \
    gpu_values, cpu_times, cpu_values = parse_csv(filename, origin)
    
    # Calculate average relative error
    gpu_av_err = calculate_average_relative_error(truth_values, gpu_values, origin == "LASSO")
    cpu_av_err = calculate_average_relative_error(truth_values, cpu_values, origin == "LASSO")
    
    # Print average relative error
    print(f"For {origin}-solver using GPU, the average relative error is {gpu_av_err}")
    print(f"For {origin}-solver using CPU, the average relative error is {cpu_av_err}")

    # remove rows having all zeroes
    truth_times_cutoff = truth_times[~np.all(truth_times == 0, axis=1)]
    nonzero_rows = truth_times_cutoff.shape[0]

    # Calculate averages
    truth_time_average = truth_times_cutoff.sum(axis = 1) / N
    gpu_time_average = gpu_times.sum(axis = 1) / N
    cpu_time_average = cpu_times.sum(axis = 1) / N

    # Calculate standard deviations
    truth_std = np.std(truth_times_cutoff, axis = 1)
    gpu_std = np.std(gpu_times, axis = 1)
    cpu_std = np.std(cpu_times, axis = 1)

    ### Plot results
    x_ticks = np.arange(len(n_values))
    x_labels = n_values
    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin + " with errorbars")
    if origin == "LASSO":
      plt.errorbar(x_ticks[:nonzero_rows], truth_time_average, yerr = truth_std, label = "JAX")
    else:
      plt.errorbar(x_ticks[:nonzero_rows], truth_time_average, yerr = truth_std, label = "CVXPY")
    plt.errorbar(x_ticks, gpu_time_average, yerr = gpu_std, label = origin + "-solver GPU")
    plt.errorbar(x_ticks, cpu_time_average, yerr = cpu_std, label = origin + "-solver CPU")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "errorbarLinear.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin)
    if origin == "LASSO":
      plt.plot(truth_time_average, label = "JAX")
    else:
      plt.plot(truth_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = origin + "-solver GPU")
    plt.plot(cpu_time_average, label = origin + "-solver CPU")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLinearLinePlot.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin + " on a log-scale with errorbars")
    if origin == "LASSO":
      plt.errorbar(x_ticks[:nonzero_rows], truth_time_average, yerr = truth_std, label = "JAX")
    else:
      plt.errorbar(x_ticks[:nonzero_rows], truth_time_average, yerr = truth_std, label = "CVXPY")
    plt.errorbar(x_ticks, gpu_time_average, yerr = gpu_std, label = origin + "-solver GPU")
    plt.errorbar(x_ticks, cpu_time_average, yerr = cpu_std, label = origin + "-solver CPU")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.yscale("log")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "errorbarLogarithmic.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Graph of average solving times for " + origin + " on a log-scale")
    if origin == "LASSO":
      plt.plot(truth_time_average, label = "JAX")
    else:
      plt.plot(truth_time_average, label = "CVXPY")
    plt.plot(gpu_time_average, label = origin + "-solver GPU")
    plt.plot(cpu_time_average, label = origin + "-solver CPU")
    plt.xticks(ticks = x_ticks, labels = x_labels)
    plt.xlabel("Dimension n")
    plt.ylabel("Average solving time [s]")
    plt.yscale("log")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLogarithmicLinePlot.png")
    plt.clf()

    # Redefine cvxpy_time_average
    truth_time_average = truth_times.sum(axis = 1) / N

    # Bar plot
    dims = n_values
    if origin == "LASSO":
      timeResults = {
          'JAX': truth_time_average,
          'GPU': gpu_time_average,
          'CPU': cpu_time_average,
      }
    else:
      timeResults = {
          'CVXPY': truth_time_average,
          'GPU': gpu_time_average,
          'CPU': cpu_time_average,
      }

    x = np.arange(len(dims))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    plt.figure(figsize=(10, 5))

    for attribute, measurement in timeResults.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding = 3, labels = None)#, labels=[f'{val:.2f}' for val in measurement])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Average solving time [s]")
    plt.xlabel("Dimension n")
    plt.title("Bar plot of average solving times for " + origin)
    plt.xticks(x + width, dims)
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLinearBarPlot.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    for attribute, measurement in timeResults.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding = 3, labels = None)#, labels=[f'{val:.2f}' for val in measurement])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Average solving time [s]")
    plt.xlabel("Dimension n")
    plt.title("Bar plot of average solving times for " + origin + " on a log-scale")
    plt.xticks(x + width, dims)
    plt.yscale("log")
    plt.legend()
    plt.savefig("testing/plots/" + origin + "averageLogarithmicBarPlot.png")
    plt.clf()

def calculate_average_relative_error(truth, test_results, lasso = False, verbose = False):
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
        mask_real = test_result != np.inf
        # Mask out non-zeros
        mask_nonzeros = test_result != 0
        # Elementwise and
        mask = np.logical_and(mask_real, mask_nonzeros)

        if verbose:
            print(f"Test {test + 1} had {mask_nonzeros.sum() - mask.sum()} out of {mask.sum()} iterations not converge")

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

get_result("testing/testing_results/testResults_jax_included_LP.csv", "LP")
get_result("testing/testing_results/testResults_jax_included_QP.csv", "QP")
get_result("testing/testing_results/testResults_no_jax_NoCVXPYSOCP.csv", "SOCP")
get_result("testing/testing_results/testResults_jax_included_NoCVXPYLASSO.csv", "LASSO")