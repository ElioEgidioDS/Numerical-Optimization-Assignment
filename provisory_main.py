import time
import numpy as np
import pandas as pd
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_64 import Problem_64
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from scipy.optimize import minimize

def main():

    import numpy as np
    import pandas as pd
    import time

    n = [2, 10**3, 10**4, 10**5]
    tolerances = [1e-4, 1e-5, 1e-6, 1e-8]
    bck_trk_C1 = [0.2, 0.4, 0.6, 0.8]
    bck_trk_phi = [0.2, 0.4, 0.6, 0.8]


    np.random.seed(352283)

    x0 = [np.ones(x) for x in n]
    x0_50 = np.ones(50)
    x_ground = [np.zeros(x) for x in n]
    xRand = [np.random.uniform(low=x-1, high=x+1, size=(5, x.shape[0])) for x in x0]

    all_results = {}   # tol -> results

    for tol in tolerances:

        print(f"\n=== Running experiments with tol = {tol:.1e} ===")

        modified_newt = NewtonMethod(tol, 1000, 0.5, 0.5)
        truncated_newt = TruncatedNewtonMethod(tol, 1000, 1000, 'sl')

        x_initial_results = {}
        x_initial_results_tr = {}
        x_ground_results = {}
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

            problem_64 = Problem_64(starting_point.shape[0], 10)

            #NM
            start_time = time.time()
            x, path, norm_gradient, converges, steps = modified_newt.minimize(
                problem_64, starting_point, mode="exact"
            )
            end_time = time.time() - start_time

            #TR
            start_time_tr = time.time()
            x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_64.function, problem_64.gradient, problem_64.hessian, starting_point)
            end_time_tr = time.time() - start_time_tr

            final_score_tr = problem_64.function(x_tr)
            final_score = problem_64.function(x)

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
            problem_64 = Problem_64(n_dim, 10)

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
                    problem_64, starting_point, mode="exact"
                )
                end_time = time.time() - start_time

                final_score = problem_64.function(x)

                path_history.append(path)
                norm_grads.append(norm_gradient)
                times.append(end_time)
                final_scores.append(final_score)
                iterations.append(steps)
                converges_list.append(converges)

                if tol == 1e-4:

                    start_time_tr = time.time()
                    x_tr, norm_gradient_tr, converges_tr, steps_tr, path_tr = truncated_newt.truncated_newton(problem_64.function, problem_64.gradient, problem_64.hessian, starting_point,tol,1000,1000,'sl')
                    end_time_tr = time.time() - start_time_tr

                    final_score_tr = problem_64.function(x_tr)

                    norm_grads_tr.append(norm_gradient_tr)
                    times_tr.append(end_time_tr)
                    final_scores_tr.append(final_score_tr)
                    iterations_tr.append(steps_tr)
                    converges_list_tr.append(converges_tr)
                    path_history_tr.append(path_tr)


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
    print(df_tr)


    return all_results, df_tr,df_nm

if __name__ == "__main__":
    main()