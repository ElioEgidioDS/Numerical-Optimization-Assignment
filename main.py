import time
import numpy as np
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_64 import Problem_64
import matplotlib.pyplot as plt

#Entry point 
def main():

    
    n = [2, 10**3, 10**4, 10**5]
    tol = 1e-6
    x0 = [np.ones(x) for x in n] #default inital starting point
    np.random.seed(359806)
    xRand = [np.random.uniform(low=x-1,high=x+1,size=(5,x.shape[0])) for x in x0]

    starting_point = x0[3] #xRand[3][0]
    print(xRand[3].shape)
    #print(xRand[1][0])


    modified_newt = NewtonMethod(tol, n[3])
    function_31 = None
    problem_64 = Problem_64(n[3],10)

    for exact_derivatives in [True, False]:
        if exact_derivatives:
            print("--- using exact derivatives ---")
            start_time = time.time()
            x = modified_newt.minimize(problem_64,starting_point,mode='exact')
            end_time = time.time() - start_time


            final_score = problem_64.function(x)
            print(f"Final Score (should be ~0): {final_score:.5e}")
            # Also check the individual errors (residuals)
            errors = problem_64.function_k(x)
            print(f"Max Error in any link: {np.max(np.abs(errors)):.5e}")
            print(f"elaped time of newton method {end_time}")

        else:
            print("--- Running with FINITE DIFFERENCE ---")
            #modified_newt.minimize(problem_64, mode='fd', k=8)
    

    t_grid = np.linspace(0, 1, problem_64.n + 2) 
    # 2. Add the boundary conditions (0 on left, 1 on right)
    y_values = np.concatenate(([0], x, [1]))
    # 3. Plot
    plt.plot(t_grid, y_values, linewidth=2)
    plt.title(f"Troesch Solution (rho=10, N={problem_64.n})")
    plt.xlabel("Position t")
    plt.ylabel("Height x")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()