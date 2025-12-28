import time
import numpy as np
import pandas as pd
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_31 import Problem_31
from Problems.Problem_31_fd import Problem_31_fd
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from scipy.optimize import minimize
from Methods.Finite_Differences import FiniteDifferences
from scipy import sparse
import scipy.sparse.linalg

def choose_h(problem):
    k_list = [4,8,12]
    results_grad = {}
    results_hess = {}
    fd = FiniteDifferences(problem)
    test_point = np.random.random(problem.n)

    exact_grad = problem.gradient(test_point)
    exact_hessian = problem.hessian(test_point)
    
    for k in k_list:
        approx_grad = fd.approximate_gradient(test_point, k_step=k)
        approx_hessian = fd.finite_differences_H(test_point, problem.gradient, k)
        #relative error
        error = np.linalg.norm(exact_grad - approx_grad) / np.linalg.norm(exact_grad)
        error_h = sparse.linalg.norm(exact_hessian - approx_hessian) / sparse.linalg.norm(exact_hessian)
        results_grad[k] = error
        results_hess[k] = error_h

        best_k_grad = min(results_grad, key=results_grad.get)
        best_k_hess = min(results_hess, key=results_hess.get)

    return max(best_k_grad,best_k_hess)

def iterate_fd(NM, TR, x0, xRand):
    
    k_values = [4, 8, 12]
    modes = ["scalar","adaptive"]

    modified_newt = NewtonMethod(1e-6, 1000, 1e-4, 0.5)
    truncated_newt = TruncatedNewtonMethod(1e-6, 1000, 500, 'sl')
    x_initial_fd = []
    x_initial_fd_tr = []

    for starting_point in x0:
        

        problem_31 = Problem_31(starting_point.shape[0])
        problem_52 = Problem_52(starting_point.shape[0])
        
        fd_solver_31 = FiniteDifferences(problem_31)
        fd_solver_52 = FiniteDifferences(problem_52)

        

        for mode in modes:
            for k in k_values:

                grad = lambda x: fd_solver_31.approximate_gradient(x, k, mode=mode)
                hess = lambda x: fd_solver_31.finite_differences_H(x, grad, k)

                problem_31_fd = Problem_31_fd(problem_31, grad, hess)

                start_time = time.time()
                x, path, norm_gradient, converges, steps = modified_newt.minimize(
                    problem_31_fd, starting_point
                )
                end_time = time.time() - start_time

                #TR
                '''start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_31_fd.function, problem_31_fd.gradient, problem_31_fd.hessian, starting_point)
                end_time_tr = time.time() - start_time_tr

                final_score_tr = problem_31.function(x_tr)'''
                final_score = problem_31.function(x)

                x_initial_fd.append({
                    #"strategy": strategy,
                    "n":starting_point.shape[0],
                    "mode": mode,         
                    "k": k,
                    "norm_gradient": norm_gradient,
                    "time": end_time,
                    "final_score": final_score,
                    "converges": converges,
                    "iterations": steps,
                    "path": path,
                })

                '''x_initial_fd_tr.append({
                    #"strategy": strategy, 
                    "n":starting_point.shape[0],
                    "mode": mode,         
                    "k": k,
                    "norm_gradient": norm_gradient_tr,
                    "time": end_time_tr,
                    "final_score": final_score_tr,
                    "converges": converges_tr,
                    "iterations": steps_tr,
                    "path": path_tr,
                })'''


    n_dim = xRand[3].shape[1]
    problem_31 = Problem_31(n_dim)
    fd_solver_31 = FiniteDifferences(problem_31)
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

            # TR
            norm_grads_tr = []
            times_tr = []
            final_scores_tr = []
            iterations_tr = []
            converges_list_tr = []
            path_history_tr = []

        
            for starting_point in xRand[3]:
                
                grad = lambda x: fd_solver_31.approximate_gradient(x, k_step=k, mode=mode)
                hess = lambda x: fd_solver_31.finite_differences_H(x, grad, k_step=k)
                problem_31_fd = Problem_31_fd(problem_31, grad, hess)

                # NM
                start_time = time.time()
                x, path, norm_gradient, converges, steps = modified_newt.minimize(
                    problem_31_fd, starting_point
                )
                end_time = time.time() - start_time
                final_score = problem_31.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)

                # TR
                '''start_time_tr = time.time()
                x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = TR.truncated_newton(
                    problem_31_fd.function, problem_31_fd.gradient, problem_31_fd.hessian, starting_point
                )
                end_time_tr = time.time() - start_time_tr
                final_score_tr = problem_31.function(x_tr)

                norm_grads_tr.append(norm_gradient_tr)
                times_tr.append(end_time_tr)
                final_scores_tr.append(final_score_tr)
                iterations_tr.append(steps_tr)
                converges_list_tr.append(converges_tr)
                path_history_tr.append(path_tr)'''

        
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
                "pathd" : path_history_tr
            })
            
    return x_initial_fd,x_initial_fd_tr, x_random_fd,x_random_fd_tr

def iterate_tol(NM,TR, x0, xRand):
    tolerances = [1e-4, 1e-5, 1e-6, 1e-8]
    all_results = {}

    for tol in tolerances:

        print(f"\n=== Running experiments with tol = {tol:.1e} ===")

        modified_newt = NewtonMethod(tol, 1000, 1e-4, 0.5)
        truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl', 1e-4, 0.5)

        x_initial_results = {}
        x_initial_results_tr = {}
        x_random_results = {}
        x_random_results_tr = {}

        '''problem_64 = Problem_64(50, 10)
        my_x, _, _, _, _ = modified_newt.minimize(problem_64, x0_50, mode="exact")

        # 2. Solve with SCIPY
        # (Assuming problem.function and problem.gradient are defined)
        scipy_res = minimize(fun=problem_64.function, 
                            x0=x0_50, 
                            jac=problem_64.gradient, 
                            method='BFGS', 
                            tol=tol)

        # 3. Compare
        diff = np.linalg.norm(my_x - scipy_res.x)
        print(f"Difference between My Newton and Scipy w tol {tol}: {diff:.2e}")'''


        
        print("--- using exact derivatives ---")

        for starting_point in x0:

            problem_31 = Problem_31(starting_point.shape[0])

            #NM
            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_31, starting_point
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_31.function, problem_31.gradient, problem_31.hessian, starting_point)
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem_31.function(x_tr)
            final_score = problem_31.function(x)

            x_initial_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path,
            }

            x_initial_results_tr[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient_tr,
                "time": end_time_tr,
                "final_score": final_score_tr,
                "converges": converges_tr,
                "iterations": steps_tr,
                "path": path_tr,
            }

        # (Optional)
        '''for starting_point in x_ground:

            problem_64 = Problem_64(starting_point.shape[0], 10)

            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_64, starting_point, mode="exact"
            )
            end_time = time.time() - start_time

            final_score = problem_64.function(x)

            x_ground_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path,
            }'''

        # random initialization (5 runs per size)
        for starting_size in xRand:

            n_dim = starting_size.shape[1]
            problem_31 = Problem_31(n_dim)

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
            path_history_tr = []

            for starting_point in starting_size:

                start_time = time.time()
                x, path, norm_gradient, converges, steps = modified_newt.minimize(
                    problem_31, starting_point
                )
                end_time = time.time() - start_time

                final_score = problem_31.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)

                if tol == 1e-6:

                    start_time_tr = time.time()
                    x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_31.function, problem_31.gradient, problem_31.hessian, starting_point)
                    end_time_tr = time.time() - start_time_tr

                    final_score_tr = problem_31.function(x_tr)

                    norm_grads_tr.append(norm_gradient_tr)
                    times_tr.append(end_time_tr)
                    final_scores_tr.append(final_score_tr)
                    iterations_tr.append(steps_tr)
                    converges_list_tr.append(converges_tr)
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
                "paths": path_history,   # length = 5
            }
            if tol == 1e-6:            
                x_random_results_tr[n_dim] = {
                    "norm_gradient": np.mean(norm_grads_tr),
                    "time": np.mean(times_tr),
                    "final_score": np.mean(final_scores_tr),
                    "iterations": np.mean(iterations_tr),
                    "converges": np.all(converges_list_tr),
                    "pathd" : path_history_tr
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
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
                })
    df_tr_rand = pd.DataFrame(table_tr_rand)
    
    return df_nm_init,df_tr_init,df_nm_rand,df_tr_rand

    
    

def main():


    n = [2, 10**3, 10**4, 10**5]
    tolerances = [1e-4, 1e-5, 1e-6, 1e-8]
    bck_trk_C1 = [0.2, 0.4, 0.6, 0.8]
    bck_trk_phi = [0.2, 0.4, 0.6, 0.8]



    np.random.seed(352283)

    x0 = [-1*np.ones(x) for x in n]
    x0_50 = np.ones(50)
    x_ground = [np.zeros(x) for x in n]
    xRand = [np.random.uniform(low=x-1, high=x+1, size=(5, x.shape[0])) for x in x0]

    all_results = {}   # tol -> results

    x_initial_fd,x_initial_fd_tr, x_random_fd,x_random_fd_tr = iterate_fd(0,0,x0,xRand)

    x_initial_fd_df = pd.DataFrame(x_initial_fd)
    x_initial_fd_tr_df = pd.DataFrame(x_initial_fd_tr)
    x_random_fd_df = pd.DataFrame(x_random_fd)
    x_random_fd_tr_df = pd.DataFrame(x_random_fd_tr)

    x_initial_fd_df.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\initial_results_nm_fd.csv",sep=',')
    x_initial_fd_tr_df.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\initial_results_tr_fd.csv",sep=',')
    x_random_fd_df.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\random_results_nm_fd.csv",sep=',')
    x_random_fd_tr_df.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\random_results_tr_fd.csv",sep=',')

    print(x_initial_fd_df)
    print()
    print(x_initial_fd_tr_df)
    print()
    print(x_random_fd_df)
    print()
    print(x_random_fd_tr_df)

    print("FD CONCLUDED")
    #exit()

    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = iterate_tol(0,0,x0,xRand)
    x_initial_nm.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\x_random_tr.csv",sep=',')
    
    print("TOL CONCLUDED")
    exit()


    '''for tol in tolerances:

        print(f"\n=== Running experiments with tol = {tol:.1e} ===")

        modified_newt = NewtonMethod(tol, 1000, 1e-4, 0.5)
        truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl')

        x_initial_results = {}
        x_initial_results_tr = {}
        x_ground_results = {}
        x_random_results = {}
        x_random_results_tr = {}

        
        problem_64 = Problem_64(50, 10)
        my_x, _, _, _, _ = modified_newt.minimize(problem_64, x0_50, mode="exact")

        # 2. Solve with SCIPY
        # (Assuming problem.function and problem.gradient are defined)
        scipy_res = minimize(fun=problem_64.function, 
                            x0=x0_50, 
                            jac=problem_64.gradient, 
                            method='BFGS', 
                            tol=tol)

        # 3. Compare
        diff = np.linalg.norm(my_x - scipy_res.x)
        print(f"Difference between My Newton and Scipy w tol {tol}: {diff:.2e}")


        
        print("--- using exact derivatives ---")

        for starting_point in x0:

            problem_31 = Problem_31(starting_point.shape[0])
            k = choose_h(problem_31)
            print("best h : 10^-",k)

            #NM
            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_31, starting_point
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_31.function, problem_31.gradient, problem_31.hessian, starting_point)
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem_31.function(x_tr)
            final_score = problem_31.function(x)

            x_initial_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path,
            }

            x_initial_results_tr[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient_tr,
                "time": end_time_tr,
                "final_score": final_score_tr,
                "converges": converges_tr,
                "iterations": steps_tr,
                "path": path_tr,
            }

        # (Optional)
        for starting_point in x_ground:

            problem_64 = Problem_64(starting_point.shape[0], 10)

            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_64, starting_point, mode="exact"
            )
            end_time = time.time() - start_time

            final_score = problem_64.function(x)

            x_ground_results[starting_point.shape[0]] = {
                "norm_gradient": norm_gradient,
                "time": end_time,
                "final_score": final_score,
                "converges": converges,
                "iterations": steps,
                "path": path,
            }

        # random initialization (5 runs per size)
        for starting_size in xRand:

            n_dim = starting_size.shape[1]
            problem_31 = Problem_31(n_dim)

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
            path_history_tr = []

            for starting_point in starting_size:

                start_time = time.time()
                x, path, norm_gradient, converges, steps = modified_newt.minimize(
                    problem_31, starting_point
                )
                end_time = time.time() - start_time

                final_score = problem_31.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)

                if tol == 1e-4:

                    start_time_tr = time.time()
                    x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_31.function, problem_31.gradient, problem_31.hessian, starting_point)
                    end_time_tr = time.time() - start_time_tr

                    final_score_tr = problem_31.function(x_tr)

                    norm_grads_tr.append(norm_gradient_tr)
                    times_tr.append(end_time_tr)
                    final_scores_tr.append(final_score_tr)
                    iterations_tr.append(steps_tr)
                    converges_list_tr.append(converges_tr)
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
                "paths": path_history,   # length = 5
            }
            if tol == 1e-4:            
                x_random_results_tr[n_dim] = {
                    "norm_gradient": np.mean(norm_grads_tr),
                    "time": np.mean(times_tr),
                    "final_score": np.mean(final_scores_tr),
                    "iterations": np.mean(iterations_tr),
                    "converges": np.all(converges_list_tr),
                    "pathd" : path_history_tr
                }

            print(f"n = {n_dim}, stored paths = {len(path_history)}")

        
        # Save results for current tol  
        all_results[tol] = {
            "x_initial": x_initial_results,
            "x_ground": x_ground_results,
            "x_initial_tr": x_initial_results_tr,
            "x_random": x_random_results,       
            "x_random_tr": x_random_results_tr
        }


    # random results to dataFrame
    table_nm = []
    table_tr = []
    


    for tol, res in all_results.items():
        for n_dim, metrics in res["x_random"].items():
            table_nm.append({
                "method":"Newton",
                "tol": tol,
                "n": n_dim,
                "time": metrics["time"],
                "iterations": metrics["iterations"],
                "converges": metrics["converges"],
                "final_score": metrics["final_score"],
                "norm_gradient": metrics["norm_gradient"]
            })

    df_nm = pd.DataFrame(table_nm)
    df_nm.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\random_results.csv",sep=',')
    print("\nsummary table for Newton Method:")
    print(df_nm)

    for tol, res in all_results.items():
        if res["x_random_tr"]:
            for n_dim, metrics in res["x_random_tr"].items():
                table_tr.append({
                    "method": "Truncated Newton",
                    "tol": tol,
                    "n": n_dim,
                    "time": metrics["time"],
                    "iterations": metrics["iterations"],
                    "converges": metrics["converges"],
                    "final_score": metrics["final_score"],
                    "norm_gradient": metrics["norm_gradient"],
                })

    df_tr = pd.DataFrame(table_tr)
    df_tr.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\random_results_tr.csv",sep=',')
    print("\nsummary table for Truncated Newton Method:")
    print(df_tr)'''


    return all_results, df_tr,df_nm

if __name__ == "__main__":
    main()