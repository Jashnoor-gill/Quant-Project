# Multi-Objective Differential Evolution with PCA

A research project implementing multi-objective differential evolution algorithms with Principal Component Analysis (PCA) applied to benchmark problems and real financial data.

## Project Structure

```
project/
├── project_code/          # Python implementation files
│   ├── Quant project.py   # Main quantitative research driver
│   ├── run_wilcoxon_test.py # Wilcoxon significance test runner (n=30)
│   ├── complete_project_code.py # Core DE/PCA-DE/PMODE/HECO-PDE algorithms
│   └── webapp.py          # Web application interface
│
├── data/                  # Input data files
│   ├── daily_returns.xlsx # Real stock return data
│   ├── *.csv              # Stock data (50 NIFTY constituents)
│   ├── *.xlsx             # Processed datasets
│   └── selected_stocks.json # Stock configuration
│
├── plots/                 # Generated visualizations
│   ├── *.png              # Algorithm comparison plots
│   ├── wilcoxon_*.png     # Statistical significance heatmaps
│   └── budget_*.png       # Budget sensitivity curves
│
├── website/               # Web application templates
│   ├── base.html          # Base template
│   ├── dashboard.html     # Main dashboard
│   ├── algorithm_tuner.html # Algorithm configuration
│   ├── budget_visualizer.html # Budget analysis UI
│   └── finance_simulator.html  # Portfolio simulator
│
└── results/               # Saved results and reports
    ├── *.pkl              # Pickled result objects
    ├── *.csv              # Summary tables
    ├── quant_project_report.md # Research report
    └── QuantProjectResults/    # Organized results directory
        ├── plots/         # Detailed plots
        ├── tables/        # Result tables by problem
        ├── pickles/       # Serialized objects
        └── reports/       # Text reports
```

## Setup & Installation

### System Requirements

- **Python Version:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 4 GB (8 GB+ recommended for large runs)
- **Disk Space:** 500 MB (1 GB+ for all results)

### Required Libraries & Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.19.0 | Numerical computations |
| pandas | ≥1.1.0 | Data manipulation |
| scipy | ≥1.5.0 | Scientific algorithms, statistics |
| matplotlib | ≥3.3.0 | Visualization and plotting |
| pymoo | ≥0.4.0 | DTLZ benchmark problems |
| scikit-learn | ≥0.23.0 | PCA, clustering |
| openpyxl | ≥3.0.0 | Excel file handling |
| flask | ≥2.0.0 | Web application framework |

### Installation Steps

#### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd c:\Users\Jashnoor\Desktop\sem 4\IQF\CFM4_final

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 2. Install Dependencies

```bash
# Install all required packages
pip install numpy pandas scipy matplotlib pymoo scikit-learn openpyxl flask

# Or install from requirements file (if present)
pip install -r webapp_requirements.txt
```

#### 3. Verify Installation

```bash
python project/project_code/verify_setup.py
```

This will check:
- ✓ All directories exist
- ✓ Data files are accessible
- ✓ Required modules can be imported
- ✓ Setup is ready to run

### Steps to Run the Code

#### **Option 1: Run Wilcoxon Test (Statistical Significance, n=30)**

Recommended starting point for reproducible research results.

```bash
cd project/project_code
python run_wilcoxon_test.py
```

**What it does:**
- Runs 30 iterations of each algorithm on 6 DTLZ variants
- Computes Wilcoxon rank-sum test p-values
- Generates statistical significance heatmaps
- Saves results to `project/results/QuantProjectResults/`

**Expected Runtime:** 30-60 minutes (hardware dependent)

**Output Files:**
- `wilcoxon_pvalues_*.csv` - P-value matrices
- `wilcoxon_*.png` - Heatmap visualizations
- `dtlz_results.pkl` - Serialized result data
- `dtlz_summary.csv` - Summary statistics table

---

#### **Option 2: Run Full Research Suite**

Comprehensive experiments with optional parameters.

```bash
cd project/project_code
python "Quant project.py"
```

**Available Flags:**

| Flag | Purpose | Example |
|------|---------|---------|
| `--all` | Run all experiments (default) | `python "Quant project.py" --all` |
| `--dtlz` | Run only DTLZ benchmarks | `python "Quant project.py" --dtlz` |
| `--finance` | Run only portfolio optimization | `python "Quant project.py" --finance` |
| `--budget` | Run budget sweep experiments | `python "Quant project.py" --budget` |
| `--quick` | Fast mode (1 run, 2K FEs) | `python "Quant project.py" --quick` |
| `--fast` | Faster mode (2 runs, 5K FEs) | `python "Quant project.py" --fast` |
| `--runs N` | Number of iterations (default: 30) | `python "Quant project.py" --runs 5` |
| `--fes N` | Function evaluations (default: 10000) | `python "Quant project.py" --fes 5000` |
| `--np N` | Population size (default: 70) | `python "Quant project.py" --np 50` |

**Examples:**

```bash
# Fast testing with 2 runs, 5K FEs
python "Quant project.py" --fast --runs 2

# DTLZ only with 20 runs
python "Quant project.py" --dtlz --runs 20

# Finance only with reduced FEs
python "Quant project.py" --finance --fes 5000
```

---

#### **Option 3: Launch Web Application**

Interactive dashboard for simulations and visualizations.

```bash
cd project/project_code
python webapp.py
```

**Access the web interface:**
```
http://localhost:5000
```

**Web Pages Available:**
- **Dashboard** - Overview and summary statistics
- **Finance Simulator** - Portfolio optimization with custom parameters
- **Algorithm Tuner** - Test algorithm parameters on DTLZ problems
- **Budget Visualizer** - Analyze convergence with budget constraints
- **Download** - Export results as CSV files

**Features:**
- Real-time algorithm execution
- Interactive parameter adjustment
- Plot visualization (scatter plots, heatmaps, curves)
- CSV export functionality
- Responsive web interface

---

### Dependency & Setup Details

#### Python Path Configuration

The project uses dynamic module loading with `importlib.util` to avoid path conflicts:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("quant_module", "path/to/Quant project.py")
quant_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant_module)
```

**Benefit:** No need to modify `sys.path` or install as package.

#### Data File Requirements

**Location:** `project/data/`

| File | Purpose | Format |
|------|---------|--------|
| `daily_returns.xlsx` | Real stock returns (50 NIFTY stocks) | Excel |
| `selected_stocks.json` | Stock configuration | JSON |
| `*.csv` | Stock price data | CSV |

These files are required for the **Finance Problem** only. DTLZ benchmarks do not require external data.

#### Result Storage

All outputs automatically saved to:
```
project/results/QuantProjectResults/
├── pickles/       # Serialized Python objects (.pkl)
├── tables/        # Summary statistics (.csv)
├── plots/         # Visualization images (.png)
└── reports/       # Text reports (.md)
```

---

## Algorithms Implemented

1. **Standard DE** - Baseline differential evolution
2. **PCA-DE** - PCA-guided differential evolution
3. **PCA-Escape DE** - PCA-DE with escape mechanism
4. **PMODE** - Portfolio multi-objective DE
5. **HECO-PDE** - Hybrid evolutionary + PCA-guided DE

## Test Problems

### DTLZ Benchmarks
- DTLZ1, DTLZ2, DTLZ7
- 2-objective and 3-objective variants
- Standard parameterization (D=10)

### Regulated Finance Problem
- 3-objective portfolio optimization
- Real daily returns from 50 NIFTY stocks
- Objectives: Return, Risk, Carbon intensity
- Carbon footprint by sector

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas scipy matplotlib pymoo scikit-learn openpyxl flask
```

### Run Wilcoxon Test (n=30)

```bash
cd project/project_code
python run_wilcoxon_test.py
```

**Expected Runtime:** 20-60 minutes (depending on hardware)

**Output:**
- Statistical significance tables (Wilcoxon p-values)
- Heatmap visualizations
- Pickled results in `project/results/QuantProjectResults/`
- Timing information printed to console

### Run Full Research Suite

```bash
cd project/project_code
python "Quant project.py" --all --runs 30
```

**Options:**
- `--dtlz` - Run only DTLZ experiments
- `--finance` - Run only finance problem
- `--budget` - Run budget sweep experiments
- `--all` - Run all (default)
- `--quick` - Fast mode (1 run, 2K FEs)
- `--fast` - Faster mode (2 runs, 5K FEs)
- `--runs N` - Number of runs per algorithm (default: 30)
- `--fes N` - Max function evaluations (default: 10000)
- `--np N` - Population size (default: 70)

### Run Web Application

```bash
cd project/project_code
python webapp.py
# Visit http://localhost:5000
```

---

## Architecture Overview

Complete system architecture showing how the Flask web server, browser, and quantitative engine interact:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ HTML Pages (dashboard, finance_simulator, etc.)          │  │
│  │ + JavaScript (makes fetch() calls to /api/*)             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↑ ↓                                   │
│                    HTTP Requests/Responses                      │
│                    (JSON, PNG base64, CSV)                      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                    localhost:5000
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                    FLASK WEB SERVER                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Route Handlers (@app.route):                              │  │
│  │ - GET  / -> render dashboard.html                         │  │
│  │ - POST /api/finance/simulate -> run algos, return JSON    │  │
│  │ - POST /api/algorithm/simulate -> tune params, return img │  │
│  │ - GET  /api/download/dtlz_csv -> send CSV file           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RESULTS_CACHE (in-memory dictionary):                     │  │
│  │ - dtlz_results: algorithm runs on DTLZ problems          │  │
│  │ - finance_results: portfolio optimization results         │  │
│  │ - dtlz_df, finance_df: summary DataFrames                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ DYNAMICALLY IMPORTED QUANT ENGINE:                        │  │
│  │ (via importlib from Quant project.py)                    │  │
│  │                                                            │  │
│  │ Algorithms:                                               │  │
│  │ - standard_de(problem, NP, max_fes, seed) → result      │  │
│  │ - pca_de(problem, ...) → result                          │  │
│  │ - pmode(problem, ...) → result                           │  │
│  │ - heco_pde(problem, ...) → result                        │  │
│  │                                                            │  │
│  │ Problems:                                                 │  │
│  │ - PortfolioFrontier(mu, cov, ret_df, tickers, sectors)  │  │
│  │   → 3 objectives: return, risk, carbon                   │  │
│  │ - DTLZ1, DTLZ2, DTLZ7 (from pymoo)                       │  │
│  │                                                            │  │
│  │ Metrics:                                                  │  │
│  │ - igd(pf, true_pf) → convergence score                   │  │
│  │ - hypervolume(pf, ref_point) → coverage                  │  │
│  │ - portfolio_metrics(w, mu, cov, ret_df) → sharpe, etc   │  │
│  │                                                            │  │
│  │ Utilities:                                                │  │
│  │ - load_daily_returns_xlsx_strict() → mu, cov, ret_df    │  │
│  │ - non_dominated_front(OBJ) → pareto front indices       │  │
│  │ - proj_weights(x, max_w) → project to simplex           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ DATA FILES (on disk):                                     │  │
│  │ - project/data/daily_returns.xlsx → real market data     │  │
│  │ - project/results/QuantProjectResults/pickles → saved    │  │
│  │ - project/results/QuantProjectResults/tables → CSVs      │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Example: Finance Simulator Request

1. **Browser** → User clicks "Run Simulation" with parameters
2. **JavaScript** → Sends POST request to `/api/finance/simulate` with JSON body
3. **Flask Route** → Receives request, extracts parameters
4. **Quant Engine** → Creates portfolio problem, runs 5 algorithms × 3 runs each
5. **Metrics** → Computes Sharpe ratio, volatility, max drawdown for each solution
6. **Visualization** → Generates scatter plot (matplotlib), converts to base64 PNG
7. **Response** → Returns JSON with plot + top portfolios
8. **Browser** → Displays scatter plot and table of results

---

## Wilcoxon Test Details

The Wilcoxon rank-sum test (also called Mann-Whitney U test) compares the statistical significance of differences between algorithms.

**Script:** `run_wilcoxon_test.py`
- **Number of runs:** 30 per algorithm
- **Comparison metric:** Inverted Generational Distance (IGD)
- **Output:** P-value matrices and significance heatmaps
- **Interpretation:** p < 0.05 indicates statistically significant difference

**Test Configuration:**
```
Runs per algorithm: 30
Function evaluations: 10,000
Population size: 70
Problems: DTLZ1, DTLZ2, DTLZ7 (2D and 3D)
```

## Performance Metrics

1. **IGD (Inverted Generational Distance)** - Convergence and spread
2. **Hypervolume (HV)** - Solution space coverage
3. **Generational Distance (GD)** - Convergence to reference
4. **Spread** - Diversity of solutions
5. **Sharpe Ratio** (Finance) - Risk-adjusted return
6. **Sortino Ratio** (Finance) - Downside risk-adjusted return
7. **Max Drawdown** (Finance) - Worst peak-to-trough decline

## Key Results

- **Algorithm Comparison**: HECO-PDE consistently outperforms baselines
- **Significance**: Wilcoxon test confirms statistical significance
- **Budget Efficiency**: PCA variants show better convergence curves
- **Finance**: Regulated optimization finds practical portfolio allocations

## File Formats

### Data Files
- `.xlsx` - Excel spreadsheets (stock data)
- `.csv` - Comma-separated values (summary tables)
- `.json` - JSON configuration files

### Results
- `.pkl` - Python pickle (serialized Python objects)
- `.png` - PNG images (plots and heatmaps)
- `.md` - Markdown reports

## Troubleshooting

**Issue:** "FileNotFoundError: No daily returns Excel file found"
- **Solution:** Ensure `daily_returns.xlsx` exists in `project/data/`

**Issue:** "ModuleNotFoundError: No module named 'pymoo'"
- **Solution:** Install with `pip install pymoo`

**Issue:** Slow execution
- **Solution:** Use `--quick` flag for faster runs, reduce `--fes` or `--runs`

**Issue:** Memory errors on large runs
- **Solution:** Reduce population size with `--np 50` or limit to 2-obj problems

## Citation

If using this code in research, please cite:

```bibtex
@misc{moea_pca_2024,
  title={Multi-Objective Evolutionary Algorithm with PCA-Guided Search},
  author={Your Name},
  year={2024}
}
```

## References

- DTLZ problems: Deb et al. (2005)
- Differential Evolution: Storn & Price (1997)
- PCA in optimization: Runarsson & Jiang (2020)
- Wilcoxon test: Wilcoxon (1945)

## License

Educational and research use only.

## Contact

For questions or issues, refer to the embedded documentation in source code.

---

**Last Updated:** 2026-05-10
**Python:** 3.8+
**Status:** Active research
