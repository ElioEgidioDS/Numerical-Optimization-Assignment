import time
import numpy as np
import pandas as pd
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_31 import Problem_31
from Problems.Problem_52 import Problem_52
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy import sparse
import scipy.sparse.linalg
from parameters import iterate_bcktrk,iterate_fd,iterate_tol

def main():
    n = [2, 10**3, 10**4, 10**5]                                                                                                                                                                            
    np.random.seed(352283)

    x0 = [-1*np.ones(x) for x in n]
    x0_50 = np.ones(50)
    xRand = [np.random.uniform(low=x-1, high=x+1, size=(5, x.shape[0])) for x in x0]



    problem_31 = Problem_31(n[0])
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
    #exit()

    x_initial_nm,x_initial_tr, x_random_nm,x_random_tr = iterate_tol(0,0,x0,xRand)
    x_initial_nm.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(r"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_random_tr.csv",sep=',')
    
    print("TOL CONCLUDED")
    #exit()

    x_initial_bck_df,x_initial_tr_bck_df, x_random_bck_df,x_random_tr_bck_df = iterate_bcktrk(x0,xRand)
    x_initial_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_initial_nm_bck.csv",sep=',')
    x_initial_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_initial_tr_bck.csv",sep=',')
    x_random_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_random_nm_bck.csv",sep=',')
    x_random_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_31.name}\x_random_tr_bck.csv",sep=',')
    
    print("BCK concluded")
    #exit()



    problem_52 = Problem_52(n[0])
    x_initial_fd_df,x_initial_fd_tr_df, x_random_fd_df,x_random_fd_tr_df = iterate_fd(x0,xRand, problem_52)

    x_initial_fd_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\initial_results_nm_fd.csv",sep=',')
    x_initial_fd_tr_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\initial_results_tr_fd.csv",sep=',')
    x_random_fd_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\random_results_nm_fd.csv",sep=',')
    x_random_fd_tr_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\random_results_tr_fd.csv",sep=',')

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
    x_initial_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_nm.csv",sep=',')
    x_initial_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_tr.csv",sep=',')
    x_random_nm.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_nm.csv",sep=',')
    x_random_tr.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_tr.csv",sep=',')
    
    print("TOL CONCLUDED")
    #exit()

    x_initial_bck_df,x_initial_tr_bck_df, x_random_bck_df,x_random_tr_bck_df = iterate_bcktrk(x0,xRand)
    x_initial_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_nm_bck.csv",sep=',')
    x_initial_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_initial_tr_bck.csv",sep=',')
    x_random_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_nm_bck.csv",sep=',')
    x_random_tr_bck_df.to_csv(fr"C:\Users\UTENTE\Desktop\NUMERICAL HOMEWORK\Nuova cartella\Numerical-Optimization-Assignment\csv\{problem_52.name}\x_random_tr_bck.csv",sep=',')