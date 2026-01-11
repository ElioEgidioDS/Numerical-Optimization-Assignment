import time
import numpy as np
import pandas as pd
from Methods.ModifiedNewtonMethod import ModifiedNewtonMethod
from Problems.Problem_31 import Problem_31
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy import sparse
import scipy.sparse.linalg
from parameters import iterate_bcktrk,iterate_fd,iterate_tol
import os
from final import final_1, final_2, final_3

def main():
    #===============================
    #EXAMPLE OF USAGE OF ExperimentalConvergence.py
    #===============================

    print("RATE OF CONVERGES PLOT EXAMPLE GENERATION")
    from parameters import iterate_tol, iterate_fd
    from Methods.ExperimentalConvergence import (
        runs_from_iterate_tol_results,
        runs_from_iterate_fd_tables,
        generate_convergence_rate_figures,
    )
    from Problems.Problem_31 import Problem_31
    import numpy as np

    n_list = [2, 10**3]
    np.random.seed(352283)

    x0 = [-1*np.ones(n) for n in n_list]
    xRand = [np.random.uniform(low=-2, high=2, size=(5, n)) for n in n_list]

    problem = Problem_31(n_list[0])

    # EXACT
    df_nm_init, df_tr_init, df_nm_rand, df_tr_rand, all_results = iterate_tol(
        x0, xRand, problem, return_full=True #return full importante
    )
    runs_exact = runs_from_iterate_tol_results(all_results, problem="p31", tol=1e-6)

    # FD (Newton)
    x_init_fd_df, x_init_fd_tr_df, x_rand_fd_df, x_rand_fd_tr_df, raw_fd = iterate_fd(
        x0, xRand, problem, return_full=True  #return full importante
    )
    runs_fd_nm = runs_from_iterate_fd_tables(x_init_fd_df, x_rand_fd_df, problem="p31", method="nm")

    # Genera tutte le figure richieste (exact + fd mode/k separati)
    generate_convergence_rate_figures(runs_exact + runs_fd_nm, out_dir="./figures/convergence_rates_example")
    print("END OF EXAMPLE GENERATION")
    print("-"*40)
    exit()
    
    #=========================================================================
    n = [2, 10**3, 10**4, 10**5]                                                                                                                                                                            
    np.random.seed(352283)

    x0 = [-1*np.ones(x) for x in n]
    x0_50 = np.ones(50)
    xRand = [np.random.uniform(low=x-1, high=x+1, size=(5, x.shape[0])) for x in x0]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ==========================================
    # PROBLEM 31
    problem_31 = Problem_31(n[0])
    problem_52 = Problem_52(n[0])
    r'''print("FD STARTED")
    x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df = iterate_fd(x0,xRand, problem_31)

    x_initial_fd_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\initial_results_nm_fd.csv",sep=',')
    x_initial_fd_tr_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\initial_results_tr_fd.csv",sep=',')
    x_random_fd_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\random_results_nm_fd.csv",sep=',')
    x_random_fd_tr_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\random_results_tr_fd.csv",sep=',')

    print(x_initial_fd_df)
    print()
    print(x_initial_fd_tr_df)
    print()
    print(x_random_fd_df)
    print()
    print(x_random_fd_tr_df)

    print("FD CONCLUDED")
    exit()

    print("TOL STARTED")
    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = iterate_tol(x0,xRand, problem_31)
    x_initial_nm.to_csv(os.path.join(path_p31, "x_initial_nm.csv"), sep=',')
    x_initial_tr.to_csv(os.path.join(path_p31, "x_initial_tr.csv"), sep=',')
    x_random_nm.to_csv(os.path.join(path_p31, "x_random_nm.csv"), sep=',')
    x_random_tr.to_csv(os.path.join(path_p31, "x_random_tr.csv"), sep=',')
    
    print("TOL CONCLUDED")
    #exit()

    print("BCK STARTED")
    x_initial_bck_df,x_initial_tr_bck_df, x_random_bck_df,x_random_tr_bck_df = iterate_bcktrk(x0,xRand,problem_31)
    x_initial_bck_df.to_csv(os.path.join(path_p31, "x_initial_nm_bck.csv"), sep=',')
    x_initial_tr_bck_df.to_csv(os.path.join(path_p31, "x_initial_tr_bck.csv"), sep=',')
    x_random_bck_df.to_csv(os.path.join(path_p31, "x_random_nm_bck.csv"), sep=',')
    x_random_tr_bck_df.to_csv(os.path.join(path_p31, "x_random_tr_bck.csv"), sep=',')
    
    print("BCK concluded")
    exit()

    # ==========================================
    # PROBLEM 52

    problem_52 = Problem_52(n[0])
    path_p52 = os.path.join(base_dir, "csv", "p52")
    os.makedirs(path_p52, exist_ok=True)

    x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df = iterate_fd(x0,xRand, problem_52)

    x_initial_fd_df.to_csv(os.path.join(path_p52, "initial_results_nm_fd.csv"), sep=',')
    x_initial_fd_tr_df.to_csv(os.path.join(path_p52, "initial_results_tr_fd.csv"), sep=',')
    x_random_fd_df.to_csv(os.path.join(path_p52, "random_results_nm_fd.csv"), sep=',')
    x_random_fd_tr_df.to_csv(os.path.join(path_p52, "random_results_tr_fd.csv"), sep=',')

    print(x_initial_fd_df)
    print()
    print(x_initial_fd_tr_df)
    print()
    print(x_random_fd_df)
    print()
    print(x_random_fd_tr_df)

    print("FD CONCLUDED")
    #exit()

    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = iterate_tol(x0,xRand, problem_52)
    x_initial_nm.to_csv(os.path.join(path_p52, "x_initial_nm.csv"), sep=',')
    x_initial_tr.to_csv(os.path.join(path_p52, "x_initial_tr.csv"), sep=',')
    x_random_nm.to_csv(os.path.join(path_p52, "x_random_nm.csv"), sep=',')
    x_random_tr.to_csv(os.path.join(path_p52, "x_random_tr.csv"), sep=',')
    
    print("TOL CONCLUDED")
    #exit()

    x_initial_bck_df,x_initial_tr_bck_df, x_random_bck_df,x_random_tr_bck_df = iterate_bcktrk(x0,xRand,problem_52)
    x_initial_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_nm_bck.csv",sep=',')
    x_initial_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_tr_bck.csv",sep=',')
    x_random_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_nm_bck.csv",sep=',')
    x_random_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_tr_bck.csv",sep=',')'''

    #------FINAL------------
    print("START FINAL - problem 31")
    r'''x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_1(x0,xRand,problem_31)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_1\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_1\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_1\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_1\x_random_tr.csv",sep=',')

    r'''x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_2(x0,xRand,problem_31)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_2\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_2\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_2\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_2\x_random_tr.csv",sep=',')'''

    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_3(x0,xRand,problem_31)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_3\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_3\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_3\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}_final\final_3\x_random_tr.csv",sep=',')

    print("START FINAL - problem 52")
    r'''x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_1(x0,xRand,problem_52)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_1\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_1\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_1\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_1\x_random_tr.csv",sep=',')

    r'''x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_2(x0,xRand,problem_52)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_2\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_2\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_2\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_2\x_random_tr.csv",sep=',')'''

    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = final_3(x0,xRand,problem_52)
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_3\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_3\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_3\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}_final\final_3\x_random_tr.csv",sep=',')'''

if __name__ == "__main__":
    main()