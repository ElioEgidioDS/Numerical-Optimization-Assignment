import numpy as np

class Backtracking:
    def __init__(self, C1, rho, max_k):

        # check for valid values of c1
        if not (0 < C1 < 1):
            raise ValueError(f"C1 must be between 0 and 1, got {C1}")
       
        # check for valid values of rho
        if not (0 < rho < 1):
            raise ValueError(f"rho must be between 0 and 1, got {rho}")
        
        self.C1 = C1
        self.rho = rho
        self.max_k = max_k
    
    def backtrack(self, p_desc, x_curr, function, alpha, grad_curr):
        k = 1
        f_curr = function(x_curr)
        slope = np.dot(grad_curr, p_desc)

        if slope > 0:
            print("slope was positive")
            return 0.0

        # two stopping criterion to avoid alpha getting too small
        while k < self.max_k and alpha > 1e-12:
            try: 
                # function evaluation at current alpha in exam
                f_next_step = function(x_curr + alpha*p_desc)
                armijo_condition = f_curr + alpha*self.C1*slope
                
                # check if new value is good
                if f_next_step <= armijo_condition:
                    return alpha
            except (OverflowError, ValueError, RuntimeWarning):
                pass
            
            # look for a new value of alpha
            alpha *= self.rho
            k += 1     
            
        # sapfeguard: max number of iterations
        print(f"no alpha was found in {k} steps" )
        return 0.0