# Quick Start Guide - Wilcoxon Test (n=30)

## Step 1: Install Required Packages

```bash
pip install numpy pandas scipy matplotlib pymoo scikit-learn openpyxl
```

If you're using a virtual environment:
```bash
source venv/Scripts/activate  # Windows PowerShell
# OR
source venv/bin/activate      # Linux/Mac

pip install numpy pandas scipy matplotlib pymoo scikit-learn openpyxl
```

## Step 2: Navigate to Project

```bash
cd "c:\Users\Jashnoor\Desktop\sem 4\IQF\CFM4_final\project\project_code"
```

## Step 3: Run Wilcoxon Test

```bash
python run_wilcoxon_test.py
```

## What This Script Does

- **Runs:** 30 iterations per algorithm (n=30)
- **Problems:** DTLZ1, DTLZ2, DTLZ7 (both 2D and 3D variants)
- **Algorithms:** Standard DE, PCA-DE, PCA-Escape DE, PMODE, HECO-PDE
- **Test:** Wilcoxon rank-sum significance test
- **Metric:** IGD (Inverted Generational Distance)
- **Output:** 
  - CSV tables with p-values
  - PNG heatmap visualizations
  - Pickle files with detailed results

## Expected Output

**Console Output:**
- Progress messages after each problem
- Execution timing (usually 20-60 minutes)
- Summary statistics
- File paths of saved results

**Saved Files:**
```
project/results/QuantProjectResults/
├── plots/
│   ├── DTLZ1_2obj_wilcoxon_n30.png
│   ├── DTLZ1_3obj_wilcoxon_n30.png
│   ├── DTLZ2_2obj_wilcoxon_n30.png
│   └── ... (more plots)
├── tables/
│   ├── dtlz_summary_n30.csv
│   ├── wilcoxon_pvals_DTLZ1_2obj_n30.csv
│   └── ... (more tables)
└── pickles/
    └── wilcoxon_test_n30.pkl
```

## Interpreting Results

**P-value Heatmaps:**
- Values < 0.05 (darker colors) indicate **statistically significant** difference
- Values > 0.05 (lighter colors) indicate **no significant** difference
- Compare algorithms pairwise for each DTLZ problem variant

**CSV Tables:**
- One row per algorithm pair
- Columns show p-values for significance testing
- Values < 0.05 suggest one algorithm significantly outperforms the other

## Timing Reference

| Configuration | Estimated Time |
|---|---|
| n=30, quick (2K FEs) | 10-20 minutes |
| n=30, normal (10K FEs) | 30-60 minutes |
| n=30, full (100K FEs) | 2-4 hours |

## Troubleshooting

**ImportError: No module named 'pymoo'**
```bash
pip install pymoo
```

**FileNotFoundError: daily_returns.xlsx**
- Check that `project/data/daily_returns.xlsx` exists
- Run `python verify_setup.py` to diagnose

**Out of memory**
- Reduce population size: Edit `run_wilcoxon_test.py` line: `NP=50` (was 70)
- Reduce FEs: Edit line: `max_fes=5000` (was 10000)

**Very slow execution**
- Expected for 30 runs per algorithm
- Run `--quick` for preliminary results (use main Quant project script instead)

## Next Steps

After running the Wilcoxon test:

1. **View Results:**
   - Open CSV files in Excel/Pandas
   - View PNG plots in image viewer

2. **Run Full Suite:**
   ```bash
   python "Quant project.py" --all --runs 30
   ```

3. **Run Web Dashboard:**
   ```bash
   python webapp.py
   # Visit http://localhost:5000
   ```

---

**Created:** 2026-05-10  
**Project:** Multi-Objective Differential Evolution with PCA  
**Test Focus:** Wilcoxon Rank-Sum Significance (n=30)
