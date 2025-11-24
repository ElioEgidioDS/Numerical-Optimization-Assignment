import numpy as np

class Backtracking:
    def __init__(self, C1, rho, max_k):
        #remember to check if >0 and < 1
        self.C1 = C1
        self.rho = rho
        self.max_k = max_k
    
    def backtrack(self, p_desc, x_curr, function, alpha, grad_curr):
    
        f_curr = function(x_curr)
        slope = np.dot(grad_curr, p_desc)

        if slope > 0:
            print("slope was positive")
            return 1e-4

        for k in range(1,self.max_k):
            f_next_step = function(x_curr + alpha*p_desc)
            armijo_condition = f_curr + alpha*self.C1*slope
            if f_next_step <= armijo_condition:
                return alpha
            else:
                alpha = self.rho*alpha
        
        print("no alpha was found")