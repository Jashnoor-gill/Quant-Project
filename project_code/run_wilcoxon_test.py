"""
Wilcoxon Test Runner (n=30)
===========================
Runs DTLZ benchmark suite with Wilcoxon rank-sum significance testing.
Number of runs per algorithm: 30
Prints progress after each problem/algorithm combination.
"""

import sys
import time
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

# Import from Quant project
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "QuantProjectResults"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
PICKLE_DIR = OUTPUT_DIR / "pickles"
REPORT_DIR = OUTPUT_DIR / "reports"

# Load Quant project module
quant_project_path = BASE_DIR / "Quant project.py"
spec = importlib.util.spec_from_file_location("quant_project", quant_project_path)
quant_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant_project)

# Use functions from quant_project
run_dtlz_suite = quant_project.run_dtlz_suite
ensure_dirs = quant_project.ensure_dirs
ranksum_summary = quant_project.ranksum_summary
save_pickle = quant_project.save_pickle
save_table = quant_project.save_table
print_dataframe = quant_project.print_dataframe
print_section = quant_project.print_section
plot_pvalue_heatmap = quant_project.plot_pvalue_heatmap

def run_wilcoxon_test_n30(max_fes=10000, NP=70):
    """
    Run Wilcoxon test with n=30 runs per algorithm.
    Prints progress after each problem.
    """
    print_section("WILCOXON TEST: n=30 runs per algorithm")
    print(f"Configuration: n_runs=30, max_fes={max_fes}, NP={NP}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    ensure_dirs()
    start_time = time.time()
    
    # Run DTLZ suite with n=30
    print("Starting DTLZ suite execution...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    dtlz_results, dtlz_df, dtlz_pvals = run_dtlz_suite(
        n_runs=30,
        max_fes=max_fes,
        NP=NP,
        include_3obj=True
    )
    
    elapsed = time.time() - start_time
    
    print_section(f"DTLZ Results Summary (n=30)")
    print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Problems tested: {len(dtlz_results)}")
    print(f"Total results rows: {len(dtlz_df)}\n")
    
    # Save results
    print_section("Saving Results")
    saved_files = []
    
    # Save DTLZ summary table
    dtlz_summary_path = save_table(dtlz_df, "dtlz_summary_n30.csv")
    saved_files.append(dtlz_summary_path)
    print(f"✓ Saved DTLZ summary: {dtlz_summary_path}")
    
    # Save Wilcoxon p-values and plots
    for pname, pval_df in dtlz_pvals.items():
        # Save CSV
        pval_path = save_table(pval_df, f"wilcoxon_pvals_{pname}_n30.csv")
        saved_files.append(pval_path)
        print(f"✓ Saved Wilcoxon p-values CSV: {pval_path}")
        
        # Save visualization
        plot_path = plot_pvalue_heatmap(pval_df, f"Wilcoxon p-values (n=30) - {pname}", f"{pname}_wilcoxon_n30.png")
        saved_files.append(plot_path)
        print(f"✓ Saved Wilcoxon p-values plot: {plot_path}")
    
    # Save pickle of all results
    pickle_path = save_pickle({
        "dtlz_results": dtlz_results,
        "dtlz_df": dtlz_df.to_dict(orient="records"),
        "dtlz_pvals": {k: v.to_dict() for k, v in dtlz_pvals.items()},
        "execution_time_seconds": elapsed,
        "n_runs": 30
    }, "wilcoxon_test_n30.pkl")
    saved_files.append(pickle_path)
    print(f"✓ Saved pickle bundle: {pickle_path}\n")
    
    # Print detailed results
    print_dataframe("DTLZ Summary (n=30 runs)", dtlz_df, sort_cols=["problem", "igd_mean"])
    
    for pname, pval_df in dtlz_pvals.items():
        print_dataframe(f"Wilcoxon P-Values: {pname} (n=30)", 
                       pval_df.reset_index().rename(columns={"index": "algorithm"}))
    
    # Final summary
    print_section("Execution Complete")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"Files saved: {len(saved_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return {
        "dtlz_results": dtlz_results,
        "dtlz_df": dtlz_df,
        "dtlz_pvals": dtlz_pvals,
        "execution_time_seconds": elapsed,
        "saved_files": saved_files
    }


if __name__ == "__main__":
    result = run_wilcoxon_test_n30()
    sys.exit(0)
