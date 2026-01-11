import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_error_plots(csv_file_path, output_dir="./figures/error_plots"):
    """
    reads the error CSV and generates one plot per configuration
    each plot compares different starting points
    """

    df = pd.read_csv(csv_file_path)

    df['h_mode'] = df['h_mode'].fillna('none')
    df['k_fd'] = df['k_fd'].fillna('0').astype(int) # 0 implies Exact/Not Applicable
    df['Case'] = df['Case'].fillna('Unknown')
    
    
    os.makedirs(output_dir, exist_ok=True)

    # define Grouping Columns cause we want one separate graph for every combination of these:
    group_cols = ['Problem', 'Size', 'Method', 'Case', 'h_mode', 'k_fd']
    grouped = df.groupby(group_cols)

    print(f"found {len(grouped)} unique configurations. Generating plots...")

   
    for name, group_df in grouped:
        # Unpack the tuple 'name' into variables
        prob, size, method, case, h_mode, k_val = name

       
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        # plot each run within this configuration
        runs = group_df.groupby(['start_type', 'run_id'])
        
        for (start_type, run_id), run_data in runs:
            run_data = run_data.sort_values('k')
            
            x = run_data['k']
            y = run_data['step_norm']
            
            if start_type == 'initial':
                label = "Fixed Point (x0)"
                color = 'black'
                linewidth = 2.5
                linestyle = '-'
                alpha = 1.0
            else:
                label = f"Random {run_id}"
                color = None # Let matplotlib pick colors
                linewidth = 1.2
                linestyle = '--'
                alpha = 0.7

            ax.semilogy(x, y, label=label, 
                        color=color, linewidth=linewidth, 
                        linestyle=linestyle, alpha=alpha)

        
        if case == "Exact":
            title_str = f"{prob} (N={size}) | {method.upper()} | Exact Derivatives"
            filename = f"{prob}_n{size}_{method}_Exact.png"
        else:
            title_str = f"{prob} (N={size}) | {method.upper()} | {case} | k={k_val} ({h_mode})"
            filename = f"{prob}_n{size}_{method}_{case}_{h_mode}_k{k_val}.png"

        ax.set_title(title_str, fontsize=14)
        ax.set_xlabel("Iteration (k)", fontsize=12)
        ax.set_ylabel("Step Size ||x_k - x_{k-1}|| (log scale)", fontsize=12)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        save_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Saved: {filename}")

if __name__ == "__main__":
        
    print("Generating plots from test data...")
    generate_error_plots("./csv/norms/convergence_errors.csv")
    print("Done! Check ./figures/error_plots")