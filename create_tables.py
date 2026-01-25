import pandas as pd
import numpy as np
import os
from scipy.stats import gmean


CSV_FILE = "csv/final/final_results.csv"
OUTPUT_FILE = "all_results_sequenced.tex"


PROBLEM_MAP = {'p31': 'Problem 31', 'p52': 'Problem 52'}
METHOD_MAP = {'nm': 'Modified Newton', 'tr': 'Truncated Newton'}
METHOD_SHORT_MAP = {'nm': 'MN', 'tr': 'TN'}
CASE_MAP = {'Exact': 'Exact derivatives', 'Mixed FD': 'Mixed FD', 'Full FD': 'Full FD'}

PROBLEMS = ['p31', 'p52']
DIMENSIONS = [2, 1000, 10000, 100000]
METHODS = ['nm', 'tr']
CASES_ORDER = ['Exact', 'Mixed FD', 'Full FD']


# Loads the CSV results file, handles boolean types, fills missing values, and formats LaTeX strings.
def load_data(csv_path):
    if not os.path.exists(csv_path):
        exit()
    df = pd.read_csv(csv_path)
    df['Success'] = df['Success'].astype(bool)
    
    if 'MaxIterations' not in df.columns:
        df['MaxIterations'] = 1000
    df['MaxIterations'] = df['MaxIterations'].fillna(1000).astype(int)

    # latex normalization
    df['point_id'] = df['point_id'].replace({
        'xbar': r'$\bar{x}$', 
        'rand1': '$x_1$', 'rand2': '$x_2$', 
        'rand3': '$x_3$', 'rand4': '$x_4$', 'rand5': '$x_5$'
    })
    return df

# salculates averages metrics (geometric for grad norm)
def calculate_averages(df_subset):
    #calculates averages for tables
    success_df = df_subset[df_subset['Success'] == True]
    if success_df.empty:
        return {
            'GradNorm': np.nan, 
            'Iterations': np.nan, 
            'MaxIterations': df_subset['MaxIterations'].iloc[0] if not df_subset.empty else 1000,
            'TimeSeconds': np.nan, 
            'ConvergenceRate': np.nan, 
            'Success': f"0/{len(df_subset)}"
        }

    grad_norm_avg = gmean(success_df['GradNorm'] + 1e-30)
    
    return {
        'GradNorm': grad_norm_avg,
        'Iterations': success_df['Iterations'].mean(),
        'MaxIterations': success_df['MaxIterations'].iloc[0], 
        'TimeSeconds': success_df['TimeSeconds'].mean(),
        'ConvergenceRate': success_df['ConvergenceRate'].mean(),
        'Success': f"{len(success_df)}/{len(df_subset)}"
    }

# generates the latex ready fiel for results tables
def generate_table_string(df, problem_key, method_key, n, case_key):

    prob_disp = PROBLEM_MAP.get(problem_key, problem_key)
    meth_disp = METHOD_MAP.get(method_key, method_key)
    case_disp = CASE_MAP.get(case_key, case_key)

    
    mask = (df['Problem'] == problem_key) & (df['Method'] == method_key) & (df['n'] == n) & (df['Case'] == case_key)
    df_data = df[mask].copy()
    
    is_fd = (case_key != "Exact")
    
    # formatting
    def fmt_sci(val): return f"{val:.2e}" if not pd.isna(val) else "-"
    def fmt_float(val): return f"{val:.2f}" if not pd.isna(val) else "-"
    def fmt_iter(val, max_val): 
        if pd.isna(val): return "-"
        return f"{int(val)}/{int(max_val)}"
    def fmt_time(val):
        if pd.isna(val): return "-"
        return f"{val:.2f}s"

    # columns and headers definitions
    if is_fd:
        col_def = "|l|l|c|c|c|c|c|c|c|"
        header_row = r"\textbf{starting point} & \textbf{pert. mode} & \textbf{k} & \textbf{grad. norm} & \textbf{iters / max} & \textbf{success} & \textbf{flag} & \textbf{conv. rate} & \textbf{time}"
        num_cols = 9
    else:
        col_def = "|l|c|c|c|c|c|c|"
        header_row = r"\textbf{starting point} & \textbf{grad. norm} & \textbf{iters / max} & \textbf{success} & \textbf{flag} & \textbf{conv. rate} & \textbf{time}"
        num_cols = 7

    
    latex_rows = []
    
    if df_data.empty:
        latex_rows.append(f"\\multicolumn{{{num_cols}}}{{|c|}}{{No data available}} \\\\ \\hline")
    else:
        if is_fd:
            # --- fd have different tables structure
            unique_k = sorted(df_data['k_fd'].unique())
            for k_idx, k in enumerate(unique_k):
                group_k = df_data[df_data['k_fd'] == k]
                modes = ['adaptive', 'scalar']
                
                for m_idx, mode in enumerate(modes):
                    group_mode = group_k[group_k['h_mode'] == mode].copy()
                    if group_mode.empty: continue

                    if m_idx > 0:
                        latex_rows.append("\\hline")

                    mode_display = "fixed" if mode == "scalar" else "adaptive"
                    group_mode['sort_order'] = group_mode['start_type'].apply(lambda x: 0 if x == 'initial' else 1)
                    group_mode = group_mode.sort_values(by=['sort_order', 'point_id'])
                    
                    
                    for _, row in group_mode.iterrows():
                        succ = "yes" if row['Success'] else "no"
                        latex_rows.append(
                            f"{row['point_id']} & {mode_display} & {int(k)} & "
                            f"{fmt_sci(row['GradNorm'])} & {fmt_iter(row['Iterations'], row['MaxIterations'])} & "
                            f"{succ} & {row['flag']} & {fmt_float(row['ConvergenceRate'])} & "
                            f"{fmt_time(row['TimeSeconds'])} \\\\"
                        )
                        latex_rows.append("\\hline") 
                    
                    # average Row
                    randoms = group_mode[group_mode['start_type'] == 'random']
                    if not randoms.empty:
                        avgs = calculate_averages(randoms)
                        latex_rows.append("\\hline\\hline") 
                        latex_rows.append(
                            f"\\textbf{{Avg}} & \\textbf{{{mode_display}}} & \\textbf{{{int(k)}}} & "
                            f"\\textbf{{{fmt_sci(avgs['GradNorm'])}}} & \\textbf{{{fmt_iter(avgs['Iterations'], avgs['MaxIterations'])}}} & "
                            f"\\textbf{{{avgs['Success']}}} & - & \\textbf{{{fmt_float(avgs['ConvergenceRate'])}}} & "
                            f"\\textbf{{{fmt_time(avgs['TimeSeconds'])}}} \\\\"
                        )
                        latex_rows.append("\\hline\\hline") 

        else:
            # --- exact tables structure
            df_data['sort_order'] = df_data['start_type'].apply(lambda x: 0 if x == 'initial' else 1)
            df_data = df_data.sort_values(by=['sort_order', 'point_id'])
            
            for _, row in df_data.iterrows():
                succ = "yes" if row['Success'] else "no"
                latex_rows.append(
                    f"{row['point_id']} & {fmt_sci(row['GradNorm'])} & {fmt_iter(row['Iterations'], row['MaxIterations'])} & "
                    f"{succ} & {row['flag']} & {fmt_float(row['ConvergenceRate'])} & "
                    f"{fmt_time(row['TimeSeconds'])} \\\\"
                )
                latex_rows.append("\\hline") 
            
            randoms = df_data[df_data['start_type'] == 'random']
            if not randoms.empty:
                avgs = calculate_averages(randoms)
                latex_rows.append("\\hline\\hline")
                latex_rows.append(
                    f"\\textbf{{Avg}} & \\textbf{{{fmt_sci(avgs['GradNorm'])}}} & \\textbf{{{fmt_iter(avgs['Iterations'], avgs['MaxIterations'])}}} & "
                    f"\\textbf{{{avgs['Success']}}} & - & \\textbf{{{fmt_float(avgs['ConvergenceRate'])}}} & "
                    f"\\textbf{{{fmt_time(avgs['TimeSeconds'])}}} \\\\"
                )
                latex_rows.append("\\hline\\hline")

    content = "\n".join(latex_rows)
    
    latex_code = f"""
\\begin{{longtable}}{{{col_def}}}
\\caption{{Results: {prob_disp}, n={n}, {meth_disp}, {case_disp}}} \\\\
\\hline
{header_row} \\\\
\\hline
\\endfirsthead
\\endhead
\\endfoot
\\hline
\\endlastfoot
{content}
\\end{{longtable}}
"""
    return latex_code


if __name__ == "__main__":
    df = load_data(CSV_FILE)
    
    
    with open(OUTPUT_FILE, "w") as f:
        
        # detailed results
        for p in PROBLEMS:
            f.write(f"\n\\section{{Results for {PROBLEM_MAP[p]}}}\n")
            
            for n in DIMENSIONS:
                f.write(f"\n\\subsection{{Dimension n = {n}}}\n")
                
                for m in METHODS:
                    f.write(f"\n\\subsubsection{{{METHOD_MAP[m]}}}\n")
                    
                    for case in CASES_ORDER:
                        table_tex = generate_table_string(df, p, m, n, case)
                        f.write(table_tex)
                        f.write("\n")
        
        
