import time
import numpy as np
import pandas as pd
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_fd import Problem_fd
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from scipy.optimize import minimize
from Methods.Finite_Differences import FiniteDifferences
from scipy import sparse
import scipy.sparse.linalg


def final_1(x0, xRand, problem_main):

    all_results = {}
    tol = 1e-6

    modified_newt = NewtonMethod(tol, 1000, 0.6, 1e-4)
    truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl', 0.6, 1e-4)

    x_initial_results = {}
    x_initial_results_tr = {}
    x_random_results = {}
    x_random_results_tr = {}


    for starting_point in x0:

        problem = type(problem_main)(starting_point.shape[0])

        #NM
        start_time = time.time()
        x, path, norm_gradient, converges, steps = modified_newt.minimize(
            problem, starting_point
        )
        end_time = time.time() - start_time

        #TR
        start_time_tr = time.time()
        x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, flag_tr = truncated_newt.truncated_newton(
            problem.function, problem.gradient, problem.hessian, starting_point,
            return_flag=True
        )

        end_time_tr = time.time() - start_time_tr

        final_score_tr = problem.function(x_tr)
        final_score = problem.function(x)


        if starting_point.shape[0] == 2:
            path_to_save = path
            path_to_save_tr = path_tr
        else:
            path_to_save = []
            path_to_save_tr = []
            for i in range(1, len(path)):
                dist = np.linalg.norm(path[i] - path[i-1])
                path_to_save.append(dist)
            for i in range(1, len(path_tr)):
                dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                path_to_save_tr.append(dist)


        x_initial_results[starting_point.shape[0]] = {
            "norm_gradient": norm_gradient,
            "time": end_time,
            "final_score": final_score,
            "converges": converges,
            "iterations": steps,
            "path": path_to_save,
        }

        x_initial_results_tr[starting_point.shape[0]] = {
            "norm_gradient": norm_gradient_tr,
            "time": end_time_tr,
            "final_score": final_score_tr,
            "converges": converges_tr,
            "iterations": steps_tr,
            "flag": flag_tr,
            "path": path_to_save_tr
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

        #TR
        norm_grads_tr = []
        times_tr = []
        final_scores_tr = []
        iterations_tr = []
        converges_list_tr = []
        flags_tr = []
        path_history_tr = []

        for starting_point in starting_size:

            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem, starting_point
            )
            end_time = time.time() - start_time

            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
            else:
                path_to_save = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)

            path_history.append(path_to_save)
            norm_grads.append(norm_gradient)
            times.append(end_time)
            final_scores.append(final_score)
            iterations.append(steps)
            converges_list.append(converges)

            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, flag_tr = truncated_newt.truncated_newton(
                problem.function, problem.gradient, problem.hessian, starting_point,
                return_flag=True
            )

            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)

            if starting_point.shape[0] == 2:
                path_to_save_tr = path_tr
            else:
                path_to_save_tr = []
                for i in range(1, len(path_tr)):
                    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                    path_to_save_tr.append(dist)

            norm_grads_tr.append(norm_gradient_tr)
            times_tr.append(end_time_tr)
            final_scores_tr.append(final_score_tr)
            iterations_tr.append(steps_tr)
            converges_list_tr.append(converges_tr)
            flags_tr.append(flag_tr)
            path_history_tr.append(path_to_save_tr)


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
        }
                   
        x_random_results_tr[n_dim] = {
            "norm_gradient": np.mean(norm_grads_tr),
            "time": np.mean(times_tr),
            "final_score": np.mean(final_scores_tr),
            "iterations": np.mean(iterations_tr),
            "converges": np.all(converges_list_tr),
            "flags": flags_tr,
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
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"],
                "path" : metrics["path"]
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
                    "flag": metrics.get("flag", ""),
                    "path" : metrics["path"]
                })
    df_tr_init = pd.DataFrame(table_tr_initial)
    if "flag" not in df_tr_init.columns:
        print("WARNING: missing 'flag' in df_tr_init columns:", df_tr_init.columns)



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
                "paths" : metrics["paths"]
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
                    "flags": metrics.get("flags", []),
                    "paths" : metrics["paths"]
                })
    df_tr_rand = pd.DataFrame(table_tr_rand)
    if "flag" not in df_tr_rand.columns:
        print("WARNING: missing 'flag' in df_tr_rand columns:", df_tr_rand.columns)
    
    return df_nm_init,df_tr_init,df_nm_rand,df_tr_rand

def final_2(x0, xRand, problem_main):
    k_values = [4, 8, 12]

    modified_newt = NewtonMethod(1e-6, 1000, 0.6, 1e-4)
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
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_fd, starting_point
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, flag_tr = truncated_newt.truncated_newton(
                problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point,
                return_flag=True
            )

            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)
            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
                path_to_save_tr = path_tr
            else:
                path_to_save = []
                path_to_save_tr = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
                for i in range(1, len(path_tr)):
                    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                    path_to_save_tr.append(dist)

            x_initial_fd.append({
                #"strategy": strategy,
                "n":starting_point.shape[0],         
                "k": k,
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path_to_save,
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
                "flag": flag_tr,
                "path": path_to_save_tr,
            })


    n_dim = xRand[3].shape[1]
    problem = type(problem_main)(n_dim)
    fd_solver = FiniteDifferences(problem)
    x_random_fd = []
    x_random_fd_tr = []
    
    for k in k_values:
        # NM
        path_history = []
        norm_grads = []
        times = []
        final_scores = []
        iterations = []
        converges_list = []

        # TR
        norm_grads_tr = []
        times_tr = []
        final_scores_tr = []
        iterations_tr = []
        converges_list_tr = []
        path_history_tr = []

    
        for starting_point in xRand[3]:
            
            grad = problem.gradient
            hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2)
            problem_fd = Problem_fd(problem, grad, hess)

            # NM
            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_fd, starting_point
            )
            end_time = time.time() - start_time
            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
            else:
                path_to_save = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
               

            path_history.append(path_to_save)
            norm_grads.append(norm_gradient)
            times.append(end_time)
            final_scores.append(final_score)
            iterations.append(steps)
            converges_list.append(converges)

            # TR
            '''start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = TR.truncated_newton(
                problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
            )
            end_time_tr = time.time() - start_time_tr
            final_score_tr = problem.function(x_tr)

            norm_grads_tr.append(norm_gradient_tr)
            times_tr.append(end_time_tr)
            final_scores_tr.append(final_score_tr)
            iterations_tr.append(steps_tr)
            converges_list_tr.append(converges_tr)
            path_history_tr.append(path_tr)'''

    
        print(f"Finished k={k}, Avg Iter: {np.mean(iterations)}")

        
        x_random_fd.append({
            "n": n_dim,       
            "k": k,
            "norm_gradient": np.mean(norm_grads),
            "time": np.mean(times),
            "final_score": np.mean(final_scores),
            "iterations": np.mean(iterations),
            "converges": np.all(converges_list),
            "paths": path_history,   
        })
                    
        '''x_random_fd_tr.append({
            "n": n_dim,       
            "k": k,
            "norm_gradient": np.mean(norm_grads_tr),
            "time": np.mean(times_tr),
            "final_score": np.mean(final_scores_tr),
            "iterations": np.mean(iterations_tr),
            "converges": np.all(converges_list_tr),
            #"paths" : path_history_tr
        })'''

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)
            
    return x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df

def final_3(x0, xRand, problem_main):
    k_values = [4, 8, 12]

    modified_newt = NewtonMethod(1e-6, 1000, 0.6, 1e-4)
    truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl',0.6, 1e-4)
    x_initial_fd = []
    x_initial_fd_tr = []

    for starting_point in x0:
        

        problem = type(problem_main)(starting_point.shape[0])
        fd_solver = FiniteDifferences(problem)
        

        for k in k_values:

            grad = lambda x: fd_solver.approximate_gradient(x, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2)
            hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2)

            problem_fd = Problem_fd(problem, grad, hess)

            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_fd, starting_point
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr, flag_tr = truncated_newt.truncated_newton(
                problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point,
                return_flag=True
            )

            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem.function(x_tr)
            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
                path_to_save_tr = path_tr
            else:
                path_to_save = []
                path_to_save_tr = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
                for i in range(1, len(path_tr)):
                    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                    path_to_save_tr.append(dist)

            x_initial_fd.append({
                #"strategy": strategy,
                "n":starting_point.shape[0],         
                "k": k,
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path_to_save,
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
                "flag": flag_tr,
                "path": path_to_save_tr,
            })


    n_dim = xRand[3].shape[1]
    problem = type(problem_main)(n_dim)
    fd_solver = FiniteDifferences(problem)
    x_random_fd = []
    x_random_fd_tr = []
    
    for k in k_values:
        # NM
        path_history = []
        norm_grads = []
        times = []
        final_scores = []
        iterations = []
        converges_list = []

        # TR
        norm_grads_tr = []
        times_tr = []
        final_scores_tr = []
        iterations_tr = []
        converges_list_tr = []
        path_history_tr = []

    
        for starting_point in xRand[3]:
            
            grad = problem.gradient
            hess = lambda x: fd_solver.approximate_hessian_pentadiag(x, grad, k, step_mode="adaptive", x_ref=x,zero_floor=1e-2)
            problem_fd = Problem_fd(problem, grad, hess)

            # NM
            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_fd, starting_point
            )
            end_time = time.time() - start_time
            final_score = problem.function(x)

            if starting_point.shape[0] == 2:
                path_to_save = path
                #path_to_save_tr = path_tr
            else:
                path_to_save = []
                #path_to_save_tr = []
                for i in range(1, len(path)):
                    dist = np.linalg.norm(path[i] - path[i-1])
                    path_to_save.append(dist)
                #for i in range(1, len(path_tr)):
                #    dist = np.linalg.norm(path_tr[i] - path_tr[i-1])
                #    path_to_save_tr.append(dist)

            path_history.append(path_to_save)
            norm_grads.append(norm_gradient)
            times.append(end_time)
            final_scores.append(final_score)
            iterations.append(steps)
            converges_list.append(converges)

            # TR
            '''start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = TR.truncated_newton(
                problem_fd.function, problem_fd.gradient, problem_fd.hessian, starting_point
            )
            end_time_tr = time.time() - start_time_tr
            final_score_tr = problem.function(x_tr)

            norm_grads_tr.append(norm_gradient_tr)
            times_tr.append(end_time_tr)
            final_scores_tr.append(final_score_tr)
            iterations_tr.append(steps_tr)
            converges_list_tr.append(converges_tr)
            path_history_tr.append(path_tr)'''

    
        print(f"Finished k={k}, Avg Iter: {np.mean(iterations)}")

        
        x_random_fd.append({
            "n": n_dim,       
            "k": k,
            "norm_gradient": np.mean(norm_grads),
            "time": np.mean(times),
            "final_score": np.mean(final_scores),
            "iterations": np.mean(iterations),
            "converges": np.all(converges_list),
            "paths": path_history,   
        })
                    
        '''x_random_fd_tr.append({
            "n": n_dim,       
            "k": k,
            "norm_gradient": np.mean(norm_grads_tr),
            "time": np.mean(times_tr),
            "final_score": np.mean(final_scores_tr),
            "iterations": np.mean(iterations_tr),
            "converges": np.all(converges_list_tr),
            #"paths" : path_history_tr
        })'''

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)
            
    return x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df