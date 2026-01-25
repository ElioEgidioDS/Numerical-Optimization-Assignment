import pandas as pd
import os
import numpy as np

# ================= CONFIGURATION =================
CSV_FILE = "csv/final/final_results.csv"

def generate_grouped_tables(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Pre-formatting
    prob_map = {'p31': 'Problem 31 (Broyden)', 'p52': 'Problem 52 (Trig-Exp)'}
    method_map = {'nm': 'Mod. Newton', 'tr': 'Trunc. Newton'}
    case_map = {'Exact': 'Exact Derivatives', 'Mixed FD': 'Mixed Finite Diff.', 'Full FD': 'Full Finite Diff.'}

    problems = ['p31', 'p52']
    cases = ['Exact', 'Mixed FD', 'Full FD']
    


    for prob in problems:
        for case in cases:
            # 1. Filter Data (Main Tables)
            if case == 'Exact':
                subset = df[(df['Problem'] == prob) & (df['Case'] == case)].copy()
            else:
                subset = df[(df['Problem'] == prob) & (df['Case'] == case) & (df['h_mode'] == 'adaptive')].copy()
            
            if subset.empty:
                continue

            print(f"\\begin{{table}}[H]")
            print(f"\\centering")
            print(f"\\caption{{Results for {prob_map[prob]} using {case_map[case]} (Adaptive)}}")
            print(f"\\label{{tab:{prob}_{case.replace(' ', '').lower()}}}")
            print(f"\\begin{{tabular}}{{|c|l|c|c|c|}}")
            print(f"\\hline")
            print(f"\\textbf{{Dim ($n$)}} & \\textbf{{Method}} & \\textbf{{Avg Iters}} & \\textbf{{Avg Time (s)}} & \\textbf{{Success}} \\\\ \\hline")

            dimensions = sorted(subset['n'].unique())
            
            for n in dimensions:
                dim_subset = subset[subset['n'] == n]
                for method in ['nm', 'tr']:
                    m_data = dim_subset[dim_subset['Method'] == method]
                    if m_data.empty: continue
                    
                    total_runs = len(m_data)
                    success_runs = m_data[m_data['Success'] == True]
                    n_success = len(success_runs)
                    
                    if n_success > 0:
                        iter_str = f"{success_runs['Iterations'].mean():.1f}"
                        time_str = f"{success_runs['TimeSeconds'].mean():.4f}"
                    else:
                        iter_str = "-"
                        time_str = "-"
                    
                    success_str = f"{n_success}/{total_runs}"
                    print(f"{n} & {method_map[method]} & {iter_str} & {time_str} & {success_str} \\\\")
                print(f"\\hline") 
            print(f"\\end{{tabular}}")
            print(f"\\end{{table}}\n")
            print("% --------------------------------------------------\n")

def generate_step_size_tables(csv_path):
    df = pd.read_csv(csv_path)
    prob_map = {'p31': 'Problem 31', 'p52': 'Problem 52'}


    # 1. Check for 'k_fd' column
    if 'k_fd' not in df.columns:
        print("Error: Column 'k_fd' not found in CSV.")
        return

    # 2. Filter Strategy
    # Select: Mod. Newton + Full FD + n=100000
    # We look for rows where 'k_fd' exists
    target_subset = df[
        (df['Method'] == 'nm') & 
        (df['Case'] == 'Full FD') & 
        (df['n'] == 100000)
    ].copy()

    if target_subset.empty:
        print("% No data found matching (NM, Full FD, n=100000).")
        return

    problems = ['p31', 'p52']

    for prob in problems:
        prob_data = target_subset[target_subset['Problem'] == prob]
        
        # 3. Check for variance in k_fd
        # We need distinct values (e.g., 4, 8, 12) to make a valid comparison
        k_values = sorted(prob_data['k_fd'].unique())
        
        if len(k_values) < 2:
            print(f"% Skipping {prob}: Only one step size found (k={k_values}). Need multiple 'k_fd' values for analysis.")
            continue

        print(f"\\begin{{table}}[H]")
        print(f"\\centering")
        print(f"\\caption{{Impact of Step Size ($k$) on Stability ({prob_map.get(prob, prob)}, Mod. Newton, n=100000)}}")
        print(f"\\label{{tab:{prob}_step_analysis}}")
        print(f"\\begin{{tabular}}{{|c|c|c|c|c|}}")
        print(f"\\hline")
        print(f"\\textbf{{Exp (k)}} & \\textbf{{Step Size ($h$)}} & \\textbf{{Avg Iters}} & \\textbf{{Avg Time (s)}} & \\textbf{{Success Rate}} \\\\ \\hline")
        
        for k in k_values:
            k_data = prob_data[prob_data['k_fd'] == k]
            
            total_runs = len(k_data)
            success_runs = k_data[k_data['Success'] == True]
            n_success = len(success_runs)
            
            # Formatting h column (10^-k)
            h_str = f"$10^{{-{int(k)}}}$"
            
            if n_success > 0:
                iter_str = f"{success_runs['Iterations'].mean():.0f}"
                time_str = f"{success_runs['TimeSeconds'].mean():.2f}"
            else:
                iter_str = "-"
                time_str = "-"
            
            success_str = f"{n_success}/{total_runs}"
            print(f"{int(k)} & {h_str} & {iter_str} & {time_str} & {success_str} \\\\ \\hline")

        print(f"\\end{{tabular}}")
        print(f"\\end{{table}}\n")

if __name__ == "__main__":
    generate_grouped_tables(CSV_FILE)
    generate_step_size_tables(CSV_FILE)