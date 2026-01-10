import time
import numpy as np
import pandas as pd
from Methods.ModifiedNewtonMethod import ModifiedNewtonMethod
from Problems.Problem_fd import Problem_fd
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from scipy.optimize import minimize
from Methods.Finite_Differences import FiniteDifferences
from scipy import sparse
import scipy.sparse.linalg

def iterate_fd(x0, xRand,problem_main):
    
    k_values = [4, 8, 12]
    modes = ["scalar","adaptive"]

    modified_newt = ModifiedNewtonMethod(1e-6, 1000, 1e-4, 0.5)
    truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl')
    x_initial_fd = []
    x_initial_fd_tr = []

    for starting_point in x0:
        

        problem = type(problem_main)(starting_point.shape[0])
        fd_solver = FiniteDifferences(problem)
        

        for mode in modes:
            for k in k_values:

                grad = lambda x: fd_solver.approximate_gradient(x, k, mode=mode)
                hess = lambda x: fd_solver.finite_differences_H(x, grad, k)

                problem_fd = Problem_fd(problem, grad, hess)

                start_time = time.time()
                x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                    problem_fd, starting_point
                )
                end_time = time.time() - start_time

                #TR
                start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point)
                end_time_tr = time.time() - start_time_tr

                final_score_tr = problem.function(x_tr)
                final_score = problem.function(x)

                x_initial_fd.append({
                    #"strategy": strategy,
                    "n":starting_point.shape[0],
                    "mode": mode,         
                    "k": k,
                    "norm_gradient": norm_gradient,
                    "time": end_time,
                    "final_score": final_score,
                    "converges": converges,
                    "failure_reason": failure_reason,
                    "iterations": steps,
                    "path": path,
                })

                x_initial_fd_tr.append({
                    #"strategy": strategy, 
                    "n":starting_point.shape[0],
                    "mode": mode,         
                    "k": k,
                    "norm_gradient": norm_gradient_tr,
                    "time": end_time_tr,
                    "final_score": final_score_tr,
                    "converges": converges_tr,
                    "failure_reason": failure_reason_tr,
                    "iterations": steps_tr,
                    "path": path_tr,
                })


    n_dim = xRand[3].shape[1]
    problem = type(problem_main)(n_dim)
    fd_solver = FiniteDifferences(problem)
    x_random_fd = []
    x_random_fd_tr = []
    for mode in modes:
        for k in k_values:
            # NM
            path_history = []
            norm_grads = []
            times = []
            final_scores = []
            iterations = []
            converges_list = []
            failure_reason_list = []

            # TR
            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            failure_reason_list_tr = []
            path_history_tr = []

        
            for starting_point in xRand[3]:
                
                grad = lambda x: fd_solver.approximate_gradient(x, k_step=k, mode=mode)
                hess = lambda x: fd_solver.finite_differences_H(x, grad, k_step=k)
                problem_fd = Problem_fd(problem, grad, hess)

                # NM
                start_time = time.time()
                x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                    problem_fd, starting_point
                )
                end_time = time.time() - start_time
                final_score = problem.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)
                failure_reason_list.append(failure_reason)

                # TR
                start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = TR.truncated_newton(
                    problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
                )
                end_time_tr = time.time() - start_time_tr
                final_score_tr = problem.function(x_tr)

                norm_grads_tr.append(norm_gradient_tr)
                times_tr.append(end_time_tr)
                final_scores_tr.append(final_score_tr)
                iterations_tr.append(steps_tr)
                converges_list_tr.append(converges_tr)
                failure_reason_list_tr.append(failure_reason_tr)
                path_history_tr.append(path_tr)

        
            print(f"Finished {mode} k={k}, Avg Iter: {np.mean(iterations)}")

            
            x_random_fd.append({
                "n": n_dim,
                "mode": mode,         
                "k": k,
                "norm_gradient": np.mean(norm_grads),
                "time": np.mean(times),
                "final_score": np.mean(final_scores),
                "iterations": np.mean(iterations),
                "converges": np.all(converges_list),
                "failure_reason": ", ".join(failure_reason_list),
                "paths": path_history,   
            })
                        
            x_random_fd_tr.append({
                "n": n_dim,
                "mode": mode,         
                "k": k,
                "norm_gradient": np.mean(norm_grads_tr),
                "time": np.mean(times_tr),
                "final_score": np.mean(final_scores_tr),
                "iterations": np.mean(iterations_tr),
                "converges": np.all(converges_list_tr),
                "failure_reason": ", ".join(failure_reason_list_tr),
                "paths" : path_history_tr
            })

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)
            
    return x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df

def iterate_tol(x0, xRand, problem_main):
    tolerances = [1e-4, 1e-6, 1e-8]
    all_results = {}

    for tol in tolerances:

        print(f"\n=== Running experiments with tol = {tol:.1e} ===")

        modified_newt = ModifiedNewtonMethod(tol, 1000, 1e-4, 0.5)
        truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl', 1e-4, 0.5)

        x_initial_results = {}
        x_initial_results_tr = {}
        x_random_results = {}
        x_random_results_tr = {}

        '''problem_64 = Problem_64(50, 10)
        my_x, _, _, _, _ = modified_newt.modified_newton(problem_64, x0_50, mode="exact")

        # 2. Solve with SCIPY
        # (Assuming problem.function and problem.gradient are defined)
        scipy_res = modified_newton(fun=problem_64.function, 
                            x0=x0_50, 
                            jac=problem_64.gradient, 
                            method='BFGS', 
                            tol=tol)

        # 3. Compare
        diff = np.linalg.norm(my_x - scipy_res.x)
        print(f"Difference between My Newton and Scipy w tol {tol}: {diff:.2e}")'''


        
        print("--- using exact derivatives ---")

        # one loop for each dimension of the problem to be chaecked
        # n = [2, 10**3, 10**4, 10**5]
        for starting_point in x0:

            # get the current dimension of the problem
            n = starting_point.shape[0]
            # instatiate a new problem of dimension n
            problem = type(problem_main)(n)

            #NM -> run the method and store time required
            start_time = time.time()
            x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(problem, starting_point)
            end_time = time.time() - start_time

            #TR -> run the method and store time required
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)
            final_score = problem.function(x)

            x_initial_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "failure_reason": failure_reason,
                "iterations": steps,
                "path": path,
            }

            x_initial_results_tr[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient_tr,
                "time": end_time_tr,
                "final_score": final_score_tr,
                "converges": converges_tr,
                "failure_reason": failure_reason_tr,
                "iterations": steps_tr,
                "path": path_tr,
            }

        # (Optional)
        '''for starting_point in x_ground:

            problem_64 = Problem_64(starting_point.shape[0], 10)

            start_time = time.time()
            x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                problem_64, starting_point, mode="exact"
            )
            end_time = time.time() - start_time

            final_score = problem_64.function(x)

            x_ground_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "failure_reason": failure_reason,
                "iterations": steps,
                "path": path,
            }'''

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
            failure_reason_list = []
            #TR
            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            failure_reason_list_tr = []
            path_history_tr = []


            for starting_point in starting_size:

                start_time = time.time()
                x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                    problem, starting_point
                )
                end_time = time.time() - start_time

                final_score = problem.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)
                failure_reason_list.append(failure_reason)

                if tol == 1e-6:

                    start_time_tr = time.time()
                    x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
                    end_time_tr = time.time() - start_time_tr

                    final_score_tr = problem.function(x_tr)

                    norm_grads_tr.append(norm_gradient_tr)
                    times_tr.append(end_time_tr)
                    final_scores_tr.append(final_score_tr)
                    iterations_tr.append(steps_tr)
                    converges_list_tr.append(converges_tr)
                    failure_reason_list_tr.append(failure_reason_tr)
                    path_history_tr.append(path_tr)

            print("\n--- DEBUGGING NORM LIST ---")
            for i, item in enumerate(norm_grads_tr):
                print(f"Item {i}: Type={type(item)} | Value={item}")
            print("-----------------------------\n")

            x_random_results[n_dim] = {
                "norm_gradient": np.mean(norm_grads),
                "time": np.mean(times),
                "final_score": np.mean(final_scores),
                "iterations": np.mean(iterations),
                "converges": np.all(converges_list),
                "failure_reason": ", ".join(failure_reason_list),
                "paths": path_history,   # length = 5
            }
            if tol == 1e-6:            
                x_random_results_tr[n_dim] = {
                    "norm_gradient": np.mean(norm_grads_tr),
                    "time": np.mean(times_tr),
                    "final_score": np.mean(final_scores_tr),
                    "iterations": np.mean(iterations_tr),
                    "converges": np.all(converges_list_tr),
                    "failure_reason": ", ".join(failure_reason_list_tr),
                    "paths" : path_history_tr
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
                "failure_reason" : metrics["failure_reason"],
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"]
            })

    df_nm_init = pd.DataFrame(table_nm_initial)

    for tol, res in all_results.items():
        #if res["x_initial_tr"]:
            for n_dim, metrics in res["x_initial_tr"].items():
                table_tr_rand.append({
                    "method": "Truncated Newton Initial Points",
                    "tol": tol,
                    "n": n_dim,
                    "time": metrics["time"],
                    "iterations": metrics["iterations"],
                    "converges": metrics["converges"],
                    "failure_reason" : metrics["failure_reason"],
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
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
                "failure_reason" : metrics["failure_reason"],
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"]
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
                    "failure_reason" : metrics["failure_reason"],
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
                })
    df_tr_rand = pd.DataFrame(table_tr_rand)
    
    return df_nm_init,df_tr_init,df_nm_rand,df_tr_rand

def iterate_bcktrk(x0,xRand, problem_main):
    bck_trk_C1 = [0.2, 0.4, 0.6, 0.8]
    bck_trk_rho = [0.2, 0.4, 0.6, 0.8]
    x_initial_bck = []
    x_initial_tr_bck = []

    for rho in bck_trk_rho:
        for c1 in bck_trk_C1:

            modified_newt = ModifiedNewtonMethod(1e-6, 1000, rho, c1)
            truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl', rho, c1)
            

            for starting_point in x0:
                

                problem = type(problem_main)(starting_point.shape[0]) 
                
                start_time = time.time()
                x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                    problem, starting_point
                )
                end_time = time.time() - start_time

                #TR
                start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem.function, problem.gradient, problem.hessian, starting_point)
                end_time_tr = time.time() - start_time_tr

                final_score_tr = problem.function(x_tr)
                final_score = problem.function(x)

                x_initial_bck.append({
                    #"strategy": strategy,
                    "n":starting_point.shape[0],
                    "rho": rho,
                    "C1": c1,         
                    "norm_gradient": norm_gradient,
                    "time": end_time,
                    "final_score": final_score,
                    "converges": converges,
                    "failure_reason": failure_reason,
                    "iterations": steps,
                    "path": path,
                })

                x_initial_tr_bck.append({
                    #"strategy": strategy, 
                    "n":starting_point.shape[0],
                    "C1": c1,         
                    "rho": rho,
                    "norm_gradient": norm_gradient_tr,
                    "time": end_time_tr,
                    "final_score": final_score_tr,
                    "converges": converges_tr,
                    "failure_reason": failure_reason_tr,
                    "iterations": steps_tr,
                    "path": path_tr,
                })


    n_dim = xRand[3].shape[1]
    problem = type(problem_main)(n_dim)
    x_random_bck = []
    x_random_tr_bck = []
    for rho in bck_trk_rho:
        for c1 in bck_trk_C1:
            # NM
            path_history = []
            norm_grads = []
            times = []
            final_scores = []
            iterations = []
            converges_list = []
            failure_reason_list = []

            # TR
            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            failure_reason_list_tr = []
            path_history_tr = []


        
            for starting_point in xRand[3]:
                

                # NM
                start_time = time.time()
                x, norm_gradient, converges, failure_reason, steps, path = modified_newt.modified_newton(
                    problem, starting_point
                )
                end_time = time.time() - start_time
                final_score = problem.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)
                failure_reason_list.append(failure_reason)

                # TR
                start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, failure_reason_tr, steps_tr, path_tr = truncated_newt.truncated_newton(
                    problem.function, problem.gradient, problem.hessian, starting_point
                )
                end_time_tr = time.time() - start_time_tr
                final_score_tr = problem.function(x_tr)

                norm_grads_tr.append(norm_gradient_tr)
                times_tr.append(end_time_tr)
                final_scores_tr.append(final_score_tr)
                iterations_tr.append(steps_tr)
                converges_list_tr.append(converges_tr)
                failure_reason_list_tr.append(failure_reason_tr)                
                path_history_tr.append(path_tr)

        
            print(f"Rho= {rho} c1={c1}, Avg Iter: {np.mean(iterations)}")

            
            x_random_bck.append({
                "n": n_dim,
                "rho": rho,       
                "C1": c1, 
                "norm_gradient": np.mean(norm_grads),
                "time": np.mean(times),
                "final_score": np.mean(final_scores),
                "iterations": np.mean(iterations),
                "converges": np.all(converges_list),
                "failure_reason": ", ".join(failure_reason_list),
                "paths": path_history,   
            })
                        
            x_random_tr_bck.append({
                "n": n_dim,
                "rho": rho,       
                "C1": c1,
                "norm_gradient": np.mean(norm_grads_tr),
                "time": np.mean(times_tr),
                "final_score": np.mean(final_scores_tr),
                "iterations": np.mean(iterations_tr),
                "converges": np.all(converges_list_tr),
                "failure_reason": ", ".join(failure_reason_list_tr),
                "paths" : path_history_tr
            })
    x_initial_bck_df = pd.DataFrame(x_initial_bck)
    x_initial_tr_bck_df = pd.DataFrame(x_initial_tr_bck)
    x_random_bck_df = pd.DataFrame(x_random_bck)
    x_random_tr_bck_df = pd.DataFrame(x_random_tr_bck)
            
    return x_initial_bck_df,x_initial_tr_bck_df, x_random_bck_df,x_random_tr_bck_df