import pandas as pd
import numpy as np
import os
from scipy.stats import gmean

# ================= CONFIGURATION =================
CSV_FILE = "csv/final/final_results.csv"
OUTPUT_FILE = "all_results_sequenced.tex"

# Mappings
PROBLEM_MAP = {'p31': 'Problem 31', 'p52': 'Problem 52'}
METHOD_MAP = {'nm': 'Modified Newton', 'tr': 'Truncated Newton'}
METHOD_SHORT_MAP = {'nm': 'MN', 'tr': 'TN'} # Short names for summary tables
CASE_MAP = {'Exact': 'Exact derivatives', 'Mixed FD': 'Mixed FD', 'Full FD': 'Full FD'}

PROBLEMS = ['p31', 'p52']
DIMENSIONS = [2, 1000, 10000, 100000]
METHODS = ['nm', 'tr']
CASES_ORDER = ['Exact', 'Mixed FD', 'Full FD']

# ================= HELPER FUNCTIONS =================

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_path} not found.")
        exit()
    df = pd.read_csv(csv_path)
    df['Success'] = df['Success'].astype(bool)
    
    if 'MaxIterations' not in df.columns:
        df['MaxIterations'] = 1000
    df['MaxIterations'] = df['MaxIterations'].fillna(1000).astype(int)

    # Normalize point names for LaTeX
    df['point_id'] = df['point_id'].replace({
        'xbar': r'$\bar{x}$', 
        'rand1': '$x_1$', 'rand2': '$x_2$', 
        'rand3': '$x_3$', 'rand4': '$x_4$', 'rand5': '$x_5$'
    })
    return df

def calculate_averages(df_subset):
    """Calculates averages for the detailed longtables."""
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

def generate_table_string(df, problem_key, method_key, n, case_key):
    """Generates the detailed longtable LaTeX string."""
    # Map keys
    prob_disp = PROBLEM_MAP.get(problem_key, problem_key)
    meth_disp = METHOD_MAP.get(method_key, method_key)
    case_disp = CASE_MAP.get(case_key, case_key)

    # 1. Filter Data
    mask = (df['Problem'] == problem_key) & (df['Method'] == method_key) & (df['n'] == n) & (df['Case'] == case_key)
    df_data = df[mask].copy()
    
    is_fd = (case_key != "Exact")
    
    # 2. Formatter Helpers
    def fmt_sci(val): return f"{val:.2e}" if not pd.isna(val) else "-"
    def fmt_float(val): return f"{val:.2f}" if not pd.isna(val) else "-"
    def fmt_iter(val, max_val): 
        if pd.isna(val): return "-"
        return f"{int(val)}/{int(max_val)}"
    def fmt_time(val):
        if pd.isna(val): return "-"
        return f"{val:.2f}s"

    # Define Columns and Headers
    if is_fd:
        col_def = "|l|l|c|c|c|c|c|c|c|"
        header_row = r"\textbf{starting point} & \textbf{pert. mode} & \textbf{k} & \textbf{grad. norm} & \textbf{iters / max} & \textbf{success} & \textbf{flag} & \textbf{conv. rate} & \textbf{time}"
        num_cols = 9
    else:
        col_def = "|l|c|c|c|c|c|c|"
        header_row = r"\textbf{starting point} & \textbf{grad. norm} & \textbf{iters / max} & \textbf{success} & \textbf{flag} & \textbf{conv. rate} & \textbf{time}"
        num_cols = 7

    # 3. Generate Content Rows
    latex_rows = []
    
    if df_data.empty:
        latex_rows.append(f"\\multicolumn{{{num_cols}}}{{|c|}}{{No data available}} \\\\ \\hline")
    else:
        if is_fd:
            # --- FINITE DIFFERENCE LOGIC ---
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
                    
                    # Single Rows
                    for _, row in group_mode.iterrows():
                        succ = "yes" if row['Success'] else "no"
                        latex_rows.append(
                            f"{row['point_id']} & {mode_display} & {int(k)} & "
                            f"{fmt_sci(row['GradNorm'])} & {fmt_iter(row['Iterations'], row['MaxIterations'])} & "
                            f"{succ} & {row['flag']} & {fmt_float(row['ConvergenceRate'])} & "
                            f"{fmt_time(row['TimeSeconds'])} \\\\"
                        )
                        latex_rows.append("\\hline") 
                    
                    # Average Row
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
            # --- EXACT LOGIC ---
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

# ================= SUMMARY TABLES GENERATION =================

def generate_summary_tables(df):
    """Generates the 4 specific summary tables for analysis."""
    
    summary_latex = []
    summary_latex.append("\n\\clearpage")
    summary_latex.append("\\section{Summary and Discussion}")

    # ---------------------------------------------------------
    # TABLE 1: Method Comparison (Exact Derivatives)
    # ---------------------------------------------------------
    summary_latex.append("\n\\subsection{Comparison by Method (Exact Derivatives)}")
    summary_latex.append("Comparison of Modified Newton (MN) and Truncated Newton (TN) using exact derivatives. "
                         "TN is generally faster for larger dimensions.")
    
    header1 = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + \
              r"\caption{Summary: Modified Newton vs. Truncated Newton (Exact Derivatives, n=1000)}" + "\n" + \
              r"\begin{tabular}{|c|c|c|c|c|c|}" + "\n" + \
              r"\hline" + "\n" + \
              r"\textbf{Problem} & \textbf{Dim} & \textbf{Method} & \textbf{Avg Iters} & \textbf{Avg Time (s)} & \textbf{Success} \\ \hline"

    rows1 = []
    for prob in ['p31', 'p52']:
        for meth in ['nm', 'tr']:
            mask = (df['Problem'] == prob) & (df['n'] == 1000) & (df['Method'] == meth) & (df['Case'] == 'Exact')
            sub = df[mask]
            if not sub.empty:
                avgs = calculate_averages(sub)
                m_disp = METHOD_SHORT_MAP.get(meth, meth.upper())
                p_disp = "P31" if prob == 'p31' else "P52"
                rows1.append(f"{p_disp} & 1000 & {m_disp} & {avgs['Iterations']:.0f} & {avgs['TimeSeconds']:.2f} & {avgs['Success']} \\\\")
                rows1.append(r"\hline")
    
    table1 = header1 + "\n" + "\n".join(rows1) + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    summary_latex.append(table1)

    # ---------------------------------------------------------
    # TABLE 2: Derivative Robustness (Problem 31, n=1000)
    # ---------------------------------------------------------
    summary_latex.append("\n\\subsection{Comparison by Derivative Type (Robustness)}")
    summary_latex.append("Impact of derivative approximation on Problem 31 (n=1000). Modified Newton fails with Full FD at high dimensions.")

    header2 = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + \
              r"\caption{Summary: Impact of Derivative Mode on Problem 31 ($n=1000$)}" + "\n" + \
              r"\begin{tabular}{|l|l|c|c|c|}" + "\n" + \
              r"\hline" + "\n" + \
              r"\textbf{Method} & \textbf{Derivative Mode} & \textbf{Avg Iters} & \textbf{Avg Time (s)} & \textbf{Success} \\ \hline"
    
    rows2 = []
    # Logic: For Full FD, we specifically look at k=12 as the "failure" case or high noise case
    target_modes = [('Exact', None), ('Mixed FD', None), ('Full FD', 12)]
    
    for meth in ['nm', 'tr']:
        m_disp = METHOD_MAP[meth]
        for (case, k_val) in target_modes:
            mask = (df['Problem'] == 'p31') & (df['n'] == 1000) & (df['Method'] == meth) & (df['Case'] == case)
            if k_val:
                mask = mask & (df['k_fd'] == k_val)
            
            sub = df[mask]
            
            # Special handling if data is missing or fully failed (0 success)
            if sub.empty:
                 row_str = f"{m_disp} & {case} (k={k_val}) & - & - & No Data \\\\"
            else:
                avgs = calculate_averages(sub)
                # If calculate_averages returned NaNs for Iters because of 0 success:
                iters = f"{avgs['Iterations']:.0f}" if not pd.isna(avgs['Iterations']) else "-"
                time = f"{avgs['TimeSeconds']:.2f}" if not pd.isna(avgs['TimeSeconds']) else "-"
                
                # Cleanup display for Mixed/Full
                case_print = case
                if k_val: case_print += f" (k={k_val})"
                
                row_str = f"{m_disp} & {case_print} & {iters} & {time} & {avgs['Success']} \\\\"
            
            rows2.append(row_str)
            rows2.append(r"\hline")

    table2 = header2 + "\n" + "\n".join(rows2) + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    summary_latex.append(table2)

    # ---------------------------------------------------------
    # TABLE 3: Dimension Scalability (Problem 31, Exact)
    # ---------------------------------------------------------
    summary_latex.append("\n\\subsection{Scalability Analysis}")
    summary_latex.append("Performance scaling from n=2 to n=1000 for Problem 31 (Exact Derivatives).")

    header3 = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + \
              r"\caption{Summary: Scalability from n=2 to n=1000 (Problem 31, Exact)}" + "\n" + \
              r"\begin{tabular}{|c|c|c|c|c|}" + "\n" + \
              r"\hline" + "\n" + \
              r"\textbf{Method} & \textbf{Dim} & \textbf{Avg Iters} & \textbf{Avg Time (s)} & \textbf{Increase} \\ \hline"
    
    rows3 = []
    for meth in ['nm', 'tr']:
        m_disp = METHOD_SHORT_MAP[meth]
        
        # Get n=2 data
        mask2 = (df['Problem'] == 'p31') & (df['n'] == 2) & (df['Method'] == meth) & (df['Case'] == 'Exact')
        sub2 = df[mask2]
        avg2 = calculate_averages(sub2)
        
        # Get n=1000 data
        mask1k = (df['Problem'] == 'p31') & (df['n'] == 1000) & (df['Method'] == meth) & (df['Case'] == 'Exact')
        sub1k = df[mask1k]
        avg1k = calculate_averages(sub1k)
        
        # Rows
        if not sub2.empty:
            rows3.append(f"{m_disp} & 2 & {avg2['Iterations']:.0f} & {avg2['TimeSeconds']:.3f} & - \\\\")
            rows3.append(r"\hline")
        if not sub1k.empty:
            factor = "-"
            if avg2['TimeSeconds'] > 0 and not pd.isna(avg1k['TimeSeconds']):
                factor = f"{avg1k['TimeSeconds']/avg2['TimeSeconds']:.1f}x"
            rows3.append(f"{m_disp} & 1000 & {avg1k['Iterations']:.0f} & {avg1k['TimeSeconds']:.3f} & {factor} \\\\")
            rows3.append(r"\hline")

    table3 = header3 + "\n" + "\n".join(rows3) + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    summary_latex.append(table3)

    # ---------------------------------------------------------
    # TABLE 4: Step Size Impact (Cancellation Error)
    # ---------------------------------------------------------
    summary_latex.append("\n\\subsection{Step Size Analysis (Numerical Stability)}")
    summary_latex.append("Impact of finite difference step size $h \approx 10^{-k}$ on Modified Newton (Problem 31, n=1000, Full FD). "
                         "Smaller step sizes ($k=12$) lead to cancellation errors.")

    header4 = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + \
              r"\caption{Summary: Impact of Step Size ($k$) on Stability (MN, P31, n=1000)}" + "\n" + \
              r"\begin{tabular}{|c|c|c|c|c|}" + "\n" + \
              r"\hline" + "\n" + \
              r"\textbf{Exp (k)} & \textbf{Step Size ($h$)} & \textbf{Avg Iters} & \textbf{Avg Time (s)} & \textbf{Success Rate} \\ \hline"
    
    rows4 = []
    # Filter for MN, P31, n=1000, Full FD
    mask_stab = (df['Problem'] == 'p31') & (df['n'] == 1000) & (df['Method'] == 'nm') & (df['Case'] == 'Full FD')
    sub_stab = df[mask_stab]
    
    if not sub_stab.empty:
        unique_k = sorted(sub_stab['k_fd'].unique())
        for k in unique_k:
            k_sub = sub_stab[sub_stab['k_fd'] == k]
            avgs = calculate_averages(k_sub)
            
            iters = f"{avgs['Iterations']:.0f}" if not pd.isna(avgs['Iterations']) else "-"
            time = f"{avgs['TimeSeconds']:.2f}" if not pd.isna(avgs['TimeSeconds']) else "-"
            
            rows4.append(f"{int(k)} & $10^{{-{int(k)}}}$ & {iters} & {time} & {avgs['Success']} \\\\")
            rows4.append(r"\hline")
            
    table4 = header4 + "\n" + "\n".join(rows4) + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    summary_latex.append(table4)

    return "\n".join(summary_latex)

# ================= MAIN LOOP =================

if __name__ == "__main__":
    df = load_data(CSV_FILE)
    
    print(f"Generating LaTeX output: {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("% =========================================\n")
        f.write("% AUTOMATICALLY GENERATED TABLES\n")
        f.write("% Includes Detailed Longtables and Executive Summary\n")
        f.write("% =========================================\n\n")
        
        # 1. Generate Detailed Results
        for p in PROBLEMS:
            f.write(f"\n\\section{{Results for {PROBLEM_MAP[p]}}}\n")
            
            for n in DIMENSIONS:
                f.write(f"\n\\subsection{{Dimension n = {n}}}\n")
                
                for m in METHODS:
                    f.write(f"\n\\subsubsection{{{METHOD_MAP[m]}}}\n")
                    
                    for case in CASES_ORDER:
                        # Generate LONGTABLE string
                        table_tex = generate_table_string(df, p, m, n, case)
                        f.write(table_tex)
                        f.write("\n")
        
        # 2. Generate Summary Tables
        summary_section = generate_summary_tables(df)
        f.write(summary_section)

    print(f"Done! File generated successfully at {OUTPUT_FILE}.")