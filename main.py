import time
import numpy as np
from Methods.NewtonMethod import NewtonMethod
from Problems.Problem_64 import Problem_64
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_interactive_evolution(problem, path):
    """
    Opens a window with a slider to scroll through optimization iterations.
    """
    n_iterations = len(path)
    t_grid = np.linspace(0, 1, problem.n + 2)
    
    # 1. Setup the Figure and Axis
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25) # Make room for slider at the bottom
    
    # 2. Plot the initial state (Iteration 0)
    # Add boundaries [0] and [1] for visualization
    initial_y = np.concatenate(([0], path[0], [1]))
    
    # Create the line object (we will update this line's data later)
    line, = ax.plot(t_grid, initial_y, lw=2, color='blue', label='Current Iteration')
    
    # Set fixed limits so the graph doesn't jump around
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.2) # A bit of padding
    ax.set_title(f"Troesch Optimization Evolution (N={problem.n})")
    ax.set_xlabel("Position t")
    ax.set_ylabel("Height x")
    ax.grid(True)
    
    # Add a text label to show current iteration number
    iter_text = ax.text(0.02, 0.95, f'Iteration: 0', transform=ax.transAxes, fontsize=12)

    # 3. Create the Slider Widget
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Iteration',
        valmin=0,
        valmax=n_iterations - 1,
        valinit=0,
        valstep=1  # Force integer steps
    )

    # 4. Define the Update Function
    def update(val):
        iteration_idx = int(slider.val)
        
        # Get the shape at this specific iteration
        current_x = path[iteration_idx]
        new_y_values = np.concatenate(([0], current_x, [1]))
        
        # Update the plot data
        line.set_ydata(new_y_values)
        iter_text.set_text(f'Iteration: {iteration_idx}')
        
        # Redraw the figure
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)
    
    plt.show()

#Entry point 
def main():

    
    n = [2, 10**3, 10**4, 10**5]
    tol = 1e-12
    x0 = [np.ones(x) for x in n] #default inital starting point
    x_ground = [np.zeros(x) for x in n]
    np.random.seed(359806)
    xRand = [np.random.uniform(low=x-1,high=x+1,size=(5,x.shape[0])) for x in x0]

    starting_point = x_ground[1]
    print(xRand[3].shape)
    #print(xRand[1][0])


    modified_newt = NewtonMethod(tol, 1000)
    function_31 = None
    problem_64 = Problem_64(n[1],10)

    for exact_derivatives in [True, False]:
        if exact_derivatives:
            print("--- using exact derivatives ---")
            start_time = time.time()
            x, path, norm_gradient = modified_newt.minimize(problem_64,starting_point,mode='exact')
            print(norm_gradient)
        
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

    '''grid_min, grid_max = -1.2, 1.2
    x1_vals = np.linspace(grid_min, grid_max, 100)
    x2_vals = np.linspace(grid_min, grid_max, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            # Create a test vector [x1, x2]
            sample_x = np.array([X1[i,j], X2[i,j]])
            # Ask the problem class for the energy (objective function)
            Z[i,j] = problem_64.function(sample_x)
    
    plt.figure(figsize=(10, 8))
    
    # Draw the "Mountain" (Contours)
    # Using Log scale for contours helps see the valley floor better
    cp = plt.contour(X1, X2, Z, levels=np.logspace(-3, 2, 20), cmap='viridis')
    plt.colorbar(cp, label='Objective Function Value (Log Scale)')
    
    # Draw the "Path" (The sequence of x)
    # path[:, 0] are all the x1 coords, path[:, 1] are x2 coords
    plt.plot(path[:, 0], path[:, 1], 'r-o', linewidth=2, label='Newton Path', markersize=5)
    
    # Mark Start and End
    plt.plot(path[0, 0], path[0, 1], 'ko', label='Start') # Black dot start
    plt.plot(path[-1, 0], path[-1, 1], 'w*', markersize=15, label='Minimum') # White star end

    plt.title(f'Newton Method Trajectory (N={n})')
    plt.xlabel('Variable x1')
    plt.ylabel('Variable x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()'''

    '''errors = problem_64.function_k(x)

    plt.figure(figsize=(10, 4))
    plt.plot(errors, '.') # Plot as dots to see randomness
    plt.title("Residual Errors per Link (Should look like random noise)")
    plt.xlabel("Link Index")
    plt.ylabel("Error magnitude")
    plt.axhline(0, color='black', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.show()'''

    #----
    t_grid = np.linspace(0, 1, problem_64.n + 2)
    
    plt.figure(figsize=(10, 6))
    
    # 2. Pick specific iterations to plot (e.g., Start, Middle, End)
    # We use a logarithmic scale to pick indices because most changes happen early
    n_steps = len(path)
    print(n_steps)
    indices_to_plot = np.unique(np.geomspace(1, n_steps, num=10, dtype=int)) - 1
    
    # 3. Plot each selected iteration
    for idx in indices_to_plot:
        current_x = path[idx]
        # Add boundaries [0] and [1]
        y_values = np.concatenate(([0], current_x, [1]))
        
        # Color fades from Red (Start) to Blue (End)
        alpha_val = (idx + 1) / n_steps
        plt.plot(t_grid, y_values, color=plt.cm.jet(alpha_val), 
                 label=f'Iter {idx}', linewidth=1.5)

    plt.title(f"Evolution of the Solution (N={problem_64.n})")
    plt.xlabel("Position t")
    plt.ylabel("Height x")
    plt.grid(True)
    plt.legend()
    plt.show()
    plot_interactive_evolution(problem_64, path)


if __name__ == "__main__":
    main()