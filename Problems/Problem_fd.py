class Problem_fd:
    def __init__(self, problem, gradient, hessian):
        self.function = problem.function
        self.gradient = gradient                 
        self.hessian = hessian                  
        self.n = problem.n