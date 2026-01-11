import time
import numpy as np
import pandas as pd
<<<<<<< HEAD
from Methods.NewtonMethod import NewtonMethod
=======
from Methods.ModifiedNewtonMethod import ModifiedNewtonMethod
>>>>>>> flag-addition
from Problems.Problem_fd import Problem_fd
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from scipy.optimize import minimize
from Methods.Finite_Differences import FiniteDifferences
from scipy import sparse
import scipy.sparse.linalg


def calculate_convergence_order(step_sizes):
    """
    Calculates the experimental order of convergence (p) from a list of step sizes.
    Input: step_sizes = [dist_1, dist_2, dist_3, ...]
    """
    # We need at least 3 steps to calculate one 'p' value
    if len(step_sizes) < 3:
        return np.nan

    # Calculate ratios of consecutive steps
    # ratio_k = s_{k+1} / s_k
    ratios = []
    for k in range(len(step_sizes) - 1):
        if step_sizes[k] == 0: # Avoid division by zero
            ratios.append(0)
        else:
            ratios.append(step_sizes[k+1] / step_sizes[k])

    # Calculate p values
    # p = ln(ratio_k) / ln(ratio_{k-1})
    p_values = []
    for k in range(1, len(ratios)):
        numerator = np.log(ratios[k]) if ratios[k] > 0 else 0
        denominator = np.log(ratios[k-1]) if ratios[k-1] > 0 else 0
        
        if denominator != 0 and numerator != 0:
            p = numerator / denominator
            p_values.append(p)
    
    # Return the last (most converged) p value, or nan if failed
    return p_values[-1] if p_values else np.nan


def final_1(x0, xRand, problem_main):

    all_results = {}
    tol = 1e-6

<<<<<<< HEAD
    modified_newt = NewtonMethod(tol, 1000, 0.6, 1e-4)
=======
    modified_newt = ModifiedNewtonMethod(tol, 1000, 0.6, 1e-4)
>>>>>>> flag-addition
    truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl', 0.6, 1e-4)

    x_initial_results = {}
    x_initial_results_tr = {}
    x_random_results = {}
    x_random_results_tr = {}


    for starting_point in x0:

        problem = type(problem_main)(starting_point.shape[0])

        #NM
        start_time = time.time()
<<<<<<< HEAD
        x, path, norm_gradient, converges, steps = modified_newt.minimize(
=======
        x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(
>>>>>>> flag-addition
            problem, starting_point
        )
        end_time = time.time() - start_time

        #TR
        start_time_tr = time.time()
<<<<<<< HEAD
        x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
=======
        x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
>>>>>>> flag-addition
        end_time_tr = time.time() - start_time_tr

        final_score_tr = problem.function(x_tr)
        final_score = problem.function(x)


        if starting_point.shape[0] == 2:
            path_to_save = path
            path_to_save_tr = path_tr
            steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
            steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
        else:
            path_to_save = []
            path_to_save_tr = []
            for i in range(1, len(path)):
                dist = np.linalg.norm(path[i] - path[i-1])
                path_to_save.append(dist)
            for i in range(1, len(path_tr)):
                dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                path_to_save_tr.append(dist)
            steps_for_calc = path_to_save
            steps_for_calc_tr = path_to_save_tr

        estimated_p = calculate_convergence_order(steps_for_calc)
        estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

        x_initial_results[starting_point.shape[0]] = {
            "norm_gradient": norm_gradient,
            "time": end_time,
            "final_score": final_score,
            "converges": converges,
            "iterations": steps,
            "path": path_to_save,
<<<<<<< HEAD
            "conv.": estimated_p
=======
            "conv.": estimated_p,
            "failure_reason": failure_reason
>>>>>>> flag-addition
        }

        x_initial_results_tr[starting_point.shape[0]] = {
            "norm_gradient": norm_gradient_tr,
            "time": end_time_tr,
            "final_score": final_score_tr,
            "converges": converges_tr,
            "iterations": steps_tr,
            "path": path_to_save_tr,
<<<<<<< HEAD
            "conv.":estimated_p_tr
=======
            "conv.":estimated_p_tr,
            "failure_reason": failure_reason_tr
>>>>>>> flag-addition
        }

    # random initialization (5 runs per size)
    for starting_size in xRand:

        n_dim = starting_size.shape[1]
        problem = type(problem_main)(n_dim)

        #NM
        path_history = []
        norm_grads = []
        times = []
        final_scores = []
        iterations = []
        converges_list = []
        conv_list = []
<<<<<<< HEAD
=======
        failure_reasons = []
>>>>>>> flag-addition

        #TR
        norm_grads_tr = []
        times_tr = []
        final_scores_tr = []
        iterations_tr = []
        converges_list_tr = []
        path_history_tr = []
        conv_list_tr = []
<<<<<<< HEAD
=======
        failure_reasons_tr = []
>>>>>>> flag-addition

        for starting_point in starting_size:

            start_time = time.time()
<<<<<<< HEAD
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
=======
            x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(
>>>>>>> flag-addition
                problem, starting_point
            )
            end_time = time.time() - start_time

            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
                steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
            else:
                path_to_save = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
                steps_for_calc = path_to_save

            estimated_p = calculate_convergence_order(steps_for_calc)

            path_history.append(path_to_save)
            norm_grads.append(norm_gradient)
            times.append(end_time)
            final_scores.append(final_score)
            iterations.append(steps)
            converges_list.append(converges)
            conv_list.append(estimated_p)
<<<<<<< HEAD
=======
            failure_reasons.append(failure_reason)
>>>>>>> flag-addition

            

            start_time_tr = time.time()
<<<<<<< HEAD
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
=======
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
>>>>>>> flag-addition
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)

            if starting_point.shape[0] == 2:
                path_to_save_tr = path_tr
                steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
            else:
                path_to_save_tr = []
                for i in range(1, len(path_tr)):
                    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                    path_to_save_tr.append(dist)
                steps_for_calc_tr = path_to_save_tr

            estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

            norm_grads_tr.append(norm_gradient_tr)
            times_tr.append(end_time_tr)
            final_scores_tr.append(final_score_tr)
            iterations_tr.append(steps_tr)
            converges_list_tr.append(converges_tr)
            path_history_tr.append(path_to_save_tr)
            conv_list_tr.append(estimated_p_tr)
<<<<<<< HEAD
=======
            failure_reasons_tr.append(failure_reason_tr)
>>>>>>> flag-addition


        for i, item in enumerate(norm_grads_tr):
            print(f"Item {i}: Type={type(item)} | Value={item}")
        print("-----------------------------\n")

        x_random_results[n_dim] = {
            "norm_gradient": np.mean(norm_grads),
            "time": np.mean(times),
            "final_score": np.mean(final_scores),
            "iterations": np.mean(iterations),
            "converges": np.all(converges_list),
            "paths": path_history,   # length = 5
<<<<<<< HEAD
            "conv.":np.mean(conv_list)
=======
            "conv.":np.mean(conv_list),
            "failure_reasons": failure_reasons
>>>>>>> flag-addition
        }
                   
        x_random_results_tr[n_dim] = {
            "norm_gradient": np.mean(norm_grads_tr),
            "time": np.mean(times_tr),
            "final_score": np.mean(final_scores_tr),
            "iterations": np.mean(iterations_tr),
            "converges": np.all(converges_list_tr),
            "paths" : path_history_tr,
<<<<<<< HEAD
            "conv.":np.mean(conv_list_tr)
=======
            "conv.":np.mean(conv_list_tr),
            "failure_reasons": failure_reasons_tr
>>>>>>> flag-addition
        }

        print(f"n = {n_dim}, stored paths = {len(path_history)}")

    
    # Save results for current tol  
    all_results[tol] = {
        "x_initial": x_initial_results,
        "x_initial_tr": x_initial_results_tr,
        "x_random": x_random_results,       
        "x_random_tr": x_random_results_tr
    }


    # random results to dataFrame
    table_nm_initial = []
    table_tr_initial = []
    table_nm_rand = []
    table_tr_rand = []
    
    for tol, res in all_results.items():
        for n_dim, metrics in res["x_initial"].items():
            table_nm_initial.append({
                "method":"Newton Initial Points",
                "tol": tol,
                "n": n_dim,
                "time": metrics["time"],
                "iterations": metrics["iterations"],
                "converges": metrics["converges"],
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"],
                "conv" : metrics["conv."],
<<<<<<< HEAD
                "path" : metrics["path"]
=======
                "path" : metrics["path"],
                "failure_reason": metrics["failure_reason"]
>>>>>>> flag-addition
                
            })

    df_nm_init = pd.DataFrame(table_nm_initial)

    for tol, res in all_results.items():
        #if res["x_initial_tr"]:
            for n_dim, metrics in res["x_initial_tr"].items():
                table_tr_initial.append({
                    "method": "Truncated Newton Initial Points",
                    "tol": tol,
                    "n": n_dim,
                    "time": metrics["time"],
                    "iterations": metrics["iterations"],
                    "converges": metrics["converges"],
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
                    "conv" : metrics["conv."],
<<<<<<< HEAD
                    "path" : metrics["path"]
=======
                    "path" : metrics["path"],
                    "failure_reason": metrics["failure_reason"]
>>>>>>> flag-addition
                    
                })
    df_tr_init = pd.DataFrame(table_tr_initial)



    for tol, res in all_results.items():
        for n_dim, metrics in res["x_random"].items():
            table_nm_rand.append({
                "method":"Newton Random Points",
                "tol": tol,
                "n": n_dim,
                "time": metrics["time"],
                "iterations": metrics["iterations"],
                "converges": metrics["converges"],
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"],
                "conv" : metrics["conv."],
<<<<<<< HEAD
                "paths" : metrics["paths"]
=======
                "paths" : metrics["paths"],
                "failure_reasons": metrics["failure_reasons"]
>>>>>>> flag-addition
                
            })

    df_nm_rand = pd.DataFrame(table_nm_rand)

    for tol, res in all_results.items():
        if res["x_random_tr"]:
            for n_dim, metrics in res["x_random_tr"].items():
                table_tr_rand.append({
                    "method": "Truncated Newton Random Points",
                    "tol": tol,
                    "n": n_dim,
                    "time": metrics["time"],
                    "iterations": metrics["iterations"],
                    "converges": metrics["converges"],
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
                    "conv" : metrics["conv."],
<<<<<<< HEAD
                    "paths" : metrics["paths"]
=======
                    "paths" : metrics["paths"],
                    "failure_reasons": metrics["failure_reasons"]
>>>>>>> flag-addition
                    
                })
    df_tr_rand = pd.DataFrame(table_tr_rand)
    
    return df_nm_init,df_tr_init,df_nm_rand,df_tr_rand

def final_2(x0, xRand, problem_main):
    k_values = [4, 8, 12]

<<<<<<< HEAD
    modified_newt = NewtonMethod(1e-6, 1000, 0.6, 1e-4)
=======
    modified_newt = ModifiedNewtonMethod(1e-6, 1000, 0.6, 1e-4)
>>>>>>> flag-addition
    truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl', 0.6, 1e-4)
    x_initial_fd = []
    x_initial_fd_tr = []

    for starting_point in x0:
        
        problem = type(problem_main)(starting_point.shape[0])
        fd_solver = FiniteDifferences(problem)
        

        for k in k_values:

            grad = problem.gradient
            hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2)

            problem_fd = Problem_fd(problem, grad, hess)

            start_time = time.time()
<<<<<<< HEAD
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
=======
            x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(
>>>>>>> flag-addition
                problem_fd, starting_point
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
<<<<<<< HEAD
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point)
=======
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point)
>>>>>>> flag-addition
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)
            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
                path_to_save_tr = path_tr
                steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
            else:
                path_to_save = []
                path_to_save_tr = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
                for i in range(1, len(path_tr)):
                    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                    path_to_save_tr.append(dist)
                steps_for_calc = path_to_save
                steps_for_calc_tr = path_to_save_tr

            estimated_p = calculate_convergence_order(steps_for_calc)
            estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

            x_initial_fd.append({
                #"strategy": strategy,
                "n":starting_point.shape[0],         
                "k": k,
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "conv": estimated_p,
<<<<<<< HEAD
                "path": path_to_save
=======
                "path": path_to_save,
                "failure_reason": failure_reason
>>>>>>> flag-addition
                
            })

            x_initial_fd_tr.append({
                #"strategy": strategy, 
                "n":starting_point.shape[0],         
                "k": k,
                "norm_gradient": norm_gradient_tr,
                "time": end_time_tr,
                "final_score": final_score_tr,
                "converges": converges_tr,
                "iterations": steps_tr,
                "conv" : estimated_p_tr,
<<<<<<< HEAD
                "path": path_to_save_tr
=======
                "path": path_to_save_tr,
                "failure_reason": failure_reason_tr
>>>>>>> flag-addition
                
            })


    x_random_fd = []
    x_random_fd_tr = []
    
    # 1. Loop Dimensions FIRST (e.g. 2, 1000, 10000...)
    for x_dataset in xRand:
        n_dim = x_dataset.shape[1]
        problem = type(problem_main)(n_dim)
        fd_solver = FiniteDifferences(problem)

        # 2. Loop K values (4, 8, 12)
        for k in k_values:
            
            # Reset metrics for this specific (N, k) batch
            path_history = []
            norm_grads = []
            times = []
            final_scores = []
            iterations = []
            converges_list = []
            conv_list = []
<<<<<<< HEAD
=======
            failure_reasons = []
>>>>>>> flag-addition

            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            path_history_tr = []
            conv_list_tr = []
<<<<<<< HEAD
=======
            failure_reasons_tr = []
>>>>>>> flag-addition
        
            # 3. Run the 5 random points
            for starting_point in x_dataset:
                
                grad = problem.gradient
                hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2)
                problem_fd = Problem_fd(problem, grad, hess)

                # --- NM ---
                start_time = time.time()
<<<<<<< HEAD
                x, path, norm_gradient, converges, steps = modified_newt.minimize(problem_fd, starting_point)
=======
                x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(problem_fd, starting_point)
>>>>>>> flag-addition
                end_time = time.time() - start_time
                final_score = problem.function(x)
                
                # --- TR ---
                start_time_tr = time.time()
<<<<<<< HEAD
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(
=======
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(
>>>>>>> flag-addition
                    problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
                )
                end_time_tr = time.time() - start_time_tr
                final_score_tr = problem.function(x_tr)

                # --- PROCESS PATHS ---
                if n_dim == 2:
                    path_to_save = path
                    path_to_save_tr = path_tr
                    steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                    steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
                else:
                    path_to_save = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                    path_to_save_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
                    steps_for_calc = path_to_save
                    steps_for_calc_tr = path_to_save_tr

                estimated_p = calculate_convergence_order(steps_for_calc)
                estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

                # Append NM Data
                path_history.append(path_to_save)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)
                conv_list.append(estimated_p)
<<<<<<< HEAD
=======
                failure_reasons.append(failure_reason)
>>>>>>> flag-addition

                # Append TR Data
                norm_grads_tr.append(norm_gradient_tr)
                times_tr.append(end_time_tr)
                final_scores_tr.append(final_score_tr)
                iterations_tr.append(steps_tr)
                converges_list_tr.append(converges_tr)
                path_history_tr.append(path_to_save_tr)
                conv_list_tr.append(estimated_p_tr)
<<<<<<< HEAD
=======
                failure_reasons_tr.append(failure_reason_tr)
>>>>>>> flag-addition

            print(f"Finished n={n_dim} k={k}, Avg Iter: {np.mean(iterations):.2f}")

            # Save Averaged Results
            x_random_fd.append({
                "n": n_dim, "k": k,
                "norm_gradient": np.mean(norm_grads),
                "time": np.mean(times),
                "final_score": np.mean(final_scores),
                "iterations": np.mean(iterations),
                "converges": np.all(converges_list),
                "conv" : np.nanmean(conv_list), 
<<<<<<< HEAD
                "paths": path_history   
=======
                "paths": path_history,
                "failure_reasons": failure_reasons
>>>>>>> flag-addition
            })
                        
            x_random_fd_tr.append({
                "n": n_dim, "k": k,
                "norm_gradient": np.mean(norm_grads_tr),
                "time": np.mean(times_tr),
                "final_score": np.mean(final_scores_tr),
                "iterations": np.mean(iterations_tr),
                "converges": np.all(converges_list_tr),
                "conv" : np.nanmean(conv_list_tr), 
<<<<<<< HEAD
                "paths" : path_history_tr
=======
                "paths" : path_history_tr,
                "failure_reasons": failure_reasons_tr
>>>>>>> flag-addition
            })

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)
            
    return x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df

def final_3(x0, xRand, problem_main):
    k_values = [4, 8, 12]

    # Solver Setup
<<<<<<< HEAD
    modified_newt = NewtonMethod(1e-6, 1000, 0.6, 1e-4)
=======
    modified_newt = ModifiedNewtonMethod(1e-6, 1000, 0.6, 1e-4)
>>>>>>> flag-addition
    truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl', 0.6, 1e-4)
    
    x_initial_fd = []
    x_initial_fd_tr = []

    # ==========================================
    # PART 1: FIXED POINTS (x0)
    # ==========================================
    for starting_point in x0:
        problem = type(problem_main)(starting_point.shape[0])   
        fd_solver = FiniteDifferences(problem)

        for k in k_values:
            # Setup Mixed FD
            grad = lambda x: fd_solver.approximate_gradient(x, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2) 
            hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2)
            problem_fd = Problem_fd(problem, grad, hess)

            # --- NM ---
            start_time = time.time()
<<<<<<< HEAD
            x, path, norm_gradient, converges, steps = modified_newt.minimize(problem_fd, starting_point)
=======
            x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(problem_fd, starting_point)
>>>>>>> flag-addition
            end_time = time.time() - start_time
            final_score = problem.function(x)

            # --- TR ---
            start_time_tr = time.time()
<<<<<<< HEAD
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(
=======
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(
>>>>>>> flag-addition
                problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
            )
            end_time_tr = time.time() - start_time_tr
            final_score_tr = problem.function(x_tr)

            # --- Process Paths ---
            if starting_point.shape[0] == 2:
                # N=2: Save full vectors, calculate steps separately
                path_to_save = path
                path_to_save_tr = path_tr
                steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
            else:
                # N>2: Save steps only, use same list for p-calc
                path_to_save = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                path_to_save_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
                steps_for_calc = path_to_save
                steps_for_calc_tr = path_to_save_tr

            estimated_p = calculate_convergence_order(steps_for_calc)
            estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

            # Save Results
            x_initial_fd.append({
                "n": starting_point.shape[0], "k": k,
                "norm_gradient": norm_gradient, "time": end_time,
                "final_score": final_score, "converges": converges,
<<<<<<< HEAD
                "iterations": steps, "conv": estimated_p, "path": path_to_save
=======
                "iterations": steps, "conv": estimated_p, "path": path_to_save,
                "failure_reason": failure_reason
>>>>>>> flag-addition
            })

            x_initial_fd_tr.append({
                "n": starting_point.shape[0], "k": k,
                "norm_gradient": norm_gradient_tr, "time": end_time_tr,
                "final_score": final_score_tr, "converges": converges_tr,
<<<<<<< HEAD
                "iterations": steps_tr, "conv": estimated_p_tr, "path": path_to_save_tr
=======
                "iterations": steps_tr, "conv": estimated_p_tr, "path": path_to_save_tr,
                "failure_reason": failure_reason_tr
>>>>>>> flag-addition
            })

    # ==========================================
    # PART 2: RANDOM POINTS (xRand)
    # ==========================================
    x_random_fd = []
    x_random_fd_tr = []
    
    # 1. Loop Dimensions FIRST (e.g. 2, 1000, 10000...)
    for x_dataset in xRand:
        n_dim = x_dataset.shape[1]
        problem = type(problem_main)(n_dim)
        fd_solver = FiniteDifferences(problem)

        # 2. Loop K values (4, 8, 12)
        for k in k_values:
            
            # Reset metrics for this specific (N, k) batch
            path_history = []
            norm_grads = []
            times = []
            final_scores = []
            iterations = []
            converges_list = []
            conv_list = []
<<<<<<< HEAD
=======
            failure_reasons = []
>>>>>>> flag-addition

            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            path_history_tr = []
            conv_list_tr = []
<<<<<<< HEAD
=======
            failure_reasons_tr = []
>>>>>>> flag-addition
        
            # 3. Run the 5 random points
            for starting_point in x_dataset:
                
                grad = lambda x: fd_solver.approximate_gradient(x, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2) 
                hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2)
                problem_fd = Problem_fd(problem, grad, hess)

                # --- NM ---
                start_time = time.time()
<<<<<<< HEAD
                x, path, norm_gradient, converges, steps = modified_newt.minimize(problem_fd, starting_point)
=======
                x, norm_gradient, converges, steps, path, failure_reason = modified_newt.modified_newton(problem_fd, starting_point)
>>>>>>> flag-addition
                end_time = time.time() - start_time
                final_score = problem.function(x)
                
                # --- TR ---
                start_time_tr = time.time()
<<<<<<< HEAD
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(
=======
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, failure_reason_tr = truncated_newt.truncated_newton(
>>>>>>> flag-addition
                    problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
                )
                end_time_tr = time.time() - start_time_tr
                final_score_tr = problem.function(x_tr)

                # --- PROCESS PATHS ---
                if n_dim == 2:
                    path_to_save = path
                    path_to_save_tr = path_tr
                    steps_for_calc = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                    steps_for_calc_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
                else:
                    path_to_save = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
                    path_to_save_tr = [np.linalg.norm(path_tr[i] - path_tr[i-1]) for i in range(1, len(path_tr))]
                    steps_for_calc = path_to_save
                    steps_for_calc_tr = path_to_save_tr

                estimated_p = calculate_convergence_order(steps_for_calc)
                estimated_p_tr = calculate_convergence_order(steps_for_calc_tr)

                # Append NM Data
                path_history.append(path_to_save)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)
                conv_list.append(estimated_p)
<<<<<<< HEAD
=======
                failure_reasons.append(failure_reason)
>>>>>>> flag-addition

                # Append TR Data
                norm_grads_tr.append(norm_gradient_tr)
                times_tr.append(end_time_tr)
                final_scores_tr.append(final_score_tr)
                iterations_tr.append(steps_tr)
                converges_list_tr.append(converges_tr)
                path_history_tr.append(path_to_save_tr)
                conv_list_tr.append(estimated_p_tr)
<<<<<<< HEAD
=======
                failure_reasons_tr.append(failure_reason_tr)
>>>>>>> flag-addition

            print(f"Finished n={n_dim} k={k}, Avg Iter: {np.mean(iterations):.2f}")

            # Save Averaged Results
            x_random_fd.append({
                "n": n_dim, "k": k,
                "norm_gradient": np.mean(norm_grads),
                "time": np.mean(times),
                "final_score": np.mean(final_scores),
                "iterations": np.mean(iterations),
                "converges": np.all(converges_list),
                "conv" : np.nanmean(conv_list), 
<<<<<<< HEAD
                "paths": path_history   
=======
                "paths": path_history,
                "failure_reasons": failure_reasons   
>>>>>>> flag-addition
            })
                        
            x_random_fd_tr.append({
                "n": n_dim, "k": k,
                "norm_gradient": np.mean(norm_grads_tr),
                "time": np.mean(times_tr),
                "final_score": np.mean(final_scores_tr),
                "iterations": np.mean(iterations_tr),
                "converges": np.all(converges_list_tr),
                "conv" : np.nanmean(conv_list_tr), 
<<<<<<< HEAD
                "paths" : path_history_tr
=======
                "paths" : path_history_tr,
                "failure_reasons": failure_reasons_tr
>>>>>>> flag-addition
            })

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)
            
    return x_initial_fd_df, x_initial_fd_tr_df, x_random_fd_df, x_random_fd_tr_df