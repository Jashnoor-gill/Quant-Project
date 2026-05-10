"""
Quant Project Web Application
=============================
Interactive Flask web app with:
- Results dashboard (DTLZ, Finance, Wilcoxon stats)
- Finance portfolio simulator
- Algorithm parameter tuner
- Budget sweep visualizer
- Download functionality
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import pickle

# Import the Quant project core
import importlib.util
BASE_DIR = Path(__file__).resolve().parent
QUANT_SCRIPT = BASE_DIR / "Quant project.py"
RESULTS_DIR = BASE_DIR / "QuantProjectResults"

spec = importlib.util.spec_from_file_location("quant_project", QUANT_SCRIPT)
quant_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant_module)

# Import core functions
load_daily_returns_xlsx_strict = quant_module.load_daily_returns_xlsx_strict
PortfolioFrontier = quant_module.PortfolioFrontier
PCAEscapeDE = quant_module.PCAEscapeDE
get_algorithms = quant_module.get_algorithms
standard_de = quant_module.standard_de
pca_de = quant_module.pca_de
pmode = quant_module.pmode
heco_pde = quant_module.heco_pde
non_dominated_front = quant_module.non_dominated_front
igd = quant_module.igd
hypervolume = quant_module.hypervolume
proj_weights = quant_module.proj_weights
portfolio_metrics = quant_module.portfolio_metrics

# Setup Flask app
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# Global results cache
RESULTS_CACHE = {}


def load_saved_results():
    """Load previously saved results from pickle files."""
    global RESULTS_CACHE
    try:
        if (RESULTS_DIR / "pickles" / "dtlz_results.pkl").exists():
            with open(RESULTS_DIR / "pickles" / "dtlz_results.pkl", "rb") as f:
                RESULTS_CACHE["dtlz"] = pickle.load(f)
        if (RESULTS_DIR / "pickles" / "finance_results.pkl").exists():
            with open(RESULTS_DIR / "pickles" / "finance_results.pkl", "rb") as f:
                RESULTS_CACHE["finance"] = pickle.load(f)
        if (RESULTS_DIR / "tables" / "dtlz_summary.csv").exists():
            RESULTS_CACHE["dtlz_df"] = pd.read_csv(RESULTS_DIR / "tables" / "dtlz_summary.csv")
        if (RESULTS_DIR / "tables" / "finance_summary.csv").exists():
            RESULTS_CACHE["finance_df"] = pd.read_csv(RESULTS_DIR / "tables" / "finance_summary.csv")
    except Exception as e:
        print(f"Warning: Could not load saved results: {e}")


load_saved_results()


@app.route('/')
def home():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/results/summary')
def api_results_summary():
    """Get summary of all saved results."""
    summary = {
        "dtlz_available": "dtlz_df" in RESULTS_CACHE,
        "finance_available": "finance_df" in RESULTS_CACHE,
        "dtlz_problems": [],
        "finance_metrics": {},
    }
    
    if "dtlz_df" in RESULTS_CACHE:
        df = RESULTS_CACHE["dtlz_df"]
        summary["dtlz_problems"] = df["problem"].unique().tolist()
        summary["dtlz_count"] = len(df)
    
    if "finance_df" in RESULTS_CACHE:
        df = RESULTS_CACHE["finance_df"]
        summary["finance_count"] = len(df)
        if "sharpe" in df.columns:
            summary["finance_metrics"]["best_sharpe"] = float(df["sharpe"].max())
            summary["finance_metrics"]["best_return"] = float(df["return"].max())
    
    return jsonify(summary)


@app.route('/api/dtlz/summary')
def api_dtlz_summary():
    """Get DTLZ summary table."""
    if "dtlz_df" not in RESULTS_CACHE:
        return jsonify({"error": "DTLZ results not available"}), 404
    
    df = RESULTS_CACHE["dtlz_df"]
    return jsonify(df.to_dict(orient="records"))


@app.route('/api/finance/summary')
def api_finance_summary():
    """Get Finance summary table."""
    if "finance_df" not in RESULTS_CACHE:
        return jsonify({"error": "Finance results not available"}), 404
    
    df = RESULTS_CACHE["finance_df"]
    return jsonify(df.to_dict(orient="records"))


@app.route('/simulator/finance')
def simulator_finance():
    """Finance portfolio simulator page."""
    return render_template('finance_simulator.html')


@app.route('/simulator/algorithm')
def simulator_algorithm():
    """Algorithm parameter tuner page."""
    return render_template('algorithm_tuner.html')


@app.route('/simulator/budget')
def simulator_budget():
    """Budget sweep visualizer page."""
    return render_template('budget_visualizer.html')


@app.route('/api/finance/simulate', methods=['POST'])
def api_finance_simulate():
    """Run finance portfolio simulation with user parameters."""
    try:
        params = request.json
        n_assets = int(params.get("n_assets", 30))
        max_weight = float(params.get("max_weight", 0.40))
        n_runs = int(params.get("n_runs", 3))
        max_fes = int(params.get("max_fes", 5000))
        NP = int(params.get("population_size", 50))
        algorithm = params.get("algorithm", "Standard DE")
        
        # Load real data
        mu, cov, ret_df, tickers, data_source = load_daily_returns_xlsx_strict()
        
        # Limit to n_assets
        if len(tickers) > n_assets:
            idx = np.random.choice(len(tickers), n_assets, replace=False)
            mu = mu[idx]
            cov = cov[np.ix_(idx, idx)]
            ret_df = ret_df.iloc[:, idx]
            tickers = [tickers[i] for i in idx]
        
        # Get sector map
        base = quant_module.base
        sector_map = {t: s for t, s in zip(base.TICKERS, base.SECTORS)}
        sectors = [sector_map.get(t, "Other") for t in tickers]
        
        # Create problem
        problem = PortfolioFrontier(mu, cov, ret_df, tickers, sectors)
        
        # Select algorithm
        algo_dict = {name: (fn, kwargs) for name, fn, kwargs in get_algorithms()}
        if algorithm not in algo_dict:
            algorithm = "Standard DE"
        
        alg_fn, kwargs = algo_dict[algorithm]
        
        # Run optimization
        all_results = []
        for run in range(n_runs):
            seed = 42 + run * 13
            res = alg_fn(problem, NP=NP, max_fes=max_fes, seed=seed, **kwargs)
            pf = np.asarray(res.get("pf", np.empty((0, 3))))
            pop = np.asarray(res.get("pop", np.empty((0, problem.n_var))))
            all_results.append({"pf": pf, "pop": pop})
        
        # Select best representative solution
        best_pf = np.vstack([r["pf"] for r in all_results])
        best_pop = np.vstack([r["pop"] for r in all_results])
        nd_idx = non_dominated_front(best_pf)
        best_pf = best_pf[nd_idx]
        best_pop = best_pop[nd_idx]
        
        # Evaluate solutions
        portfolio_results = []
        for i, w_raw in enumerate(best_pop):
            w = proj_weights(w_raw, max_weight)
            metrics = portfolio_metrics(w, mu, cov, ret_df, rf=0.04)
            metrics["weights"] = w.tolist()
            metrics["tickers"] = tickers
            portfolio_results.append(metrics)
        
        # Create scatter plot (risk vs return)
        fig, ax = plt.subplots(figsize=(8, 6))
        returns = [r["return"] for r in portfolio_results]
        vols = [r["vol"] for r in portfolio_results]
        sharpes = [r["sharpe"] for r in portfolio_results]
        
        scatter = ax.scatter(vols, returns, c=sharpes, cmap="viridis", s=100, alpha=0.7)
        ax.set_xlabel("Volatility (Risk)", fontsize=11)
        ax.set_ylabel("Expected Return", fontsize=11)
        ax.set_title(f"Finance Simulation Results - {algorithm}", fontsize=12)
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe Ratio", fontsize=10)
        
        # Save plot
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=100, bbox_inches='tight')
        img_io.seek(0)
        img_base64 = __import__('base64').b64encode(img_io.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            "success": True,
            "algorithm": algorithm,
            "n_solutions": len(portfolio_results),
            "portfolios": portfolio_results[:10],  # Return top 10
            "plot": f"data:image/png;base64,{img_base64}",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/algorithm/simulate', methods=['POST'])
def api_algorithm_simulate():
    """Run algorithm tuning simulation."""
    try:
        params = request.json
        algorithm_name = params.get("algorithm", "Standard DE")
        F = float(params.get("F", 0.5))
        CR = float(params.get("CR", 0.9))
        NP = int(params.get("population_size", 70))
        max_fes = int(params.get("max_fes", 10000))
        n_runs = int(params.get("n_runs", 3))
        
        # Use DTLZ2_2obj as test problem
        if not getattr(quant_module.base, "HAS_PYMOO", False):
            return jsonify({"error": "pymoo not available"}), 400
        
        p = quant_module.base.pymoo_get_problem
        problem = p("dtlz2", n_var=12, n_obj=2)
        true_pf = problem.pareto_front()
        ref_point = np.array([2.5, 2.5])
        
        # Map algorithm
        algo_map = {
            "Standard DE": (quant_module.standard_de, {}),
            "PCA-DE": (quant_module.pca_de, {"alpha": 0.5}),
            "PMODE": (quant_module.pmode, {"alpha": 0.45}),
            "HECO-PDE": (quant_module.heco_pde, {"alpha": 0.60, "n_weights": 12}),
        }
        
        if algorithm_name not in algo_map:
            return jsonify({"error": f"Unknown algorithm: {algorithm_name}"}), 400
        
        alg_fn, kwargs = algo_map[algorithm_name]
        
        # Run multiple times
        igd_values = []
        hv_values = []
        convergence_history = []
        
        for run in range(n_runs):
            seed = 42 + run * 13
            res = alg_fn(problem, NP=NP, F=F, CR=CR, max_fes=max_fes, seed=seed, **kwargs)
            pf = np.asarray(res.get("pf", np.empty((0, 2))))
            
            igd_val = float(igd(pf, true_pf)) if len(pf) else float("inf")
            hv_val = float(hypervolume(pf, ref_point, n_samples=5000)) if len(pf) else 0.0
            igd_values.append(igd_val)
            hv_values.append(hv_val)
        
        # Create results visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # IGD bar chart
        runs = [f"Run {i+1}" for i in range(n_runs)]
        ax1.bar(runs, igd_values, color="#4c78a8", alpha=0.7)
        ax1.set_ylabel("IGD", fontsize=11)
        ax1.set_title(f"{algorithm_name}: Inverted Generational Distance", fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(np.mean(igd_values), color='red', linestyle='--', label=f'Mean: {np.mean(igd_values):.4f}')
        ax1.legend()
        
        # HV bar chart
        ax2.bar(runs, hv_values, color="#e45756", alpha=0.7)
        ax2.set_ylabel("Hypervolume", fontsize=11)
        ax2.set_title(f"{algorithm_name}: Hypervolume", fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(np.mean(hv_values), color='blue', linestyle='--', label=f'Mean: {np.mean(hv_values):.4f}')
        ax2.legend()
        
        plt.tight_layout()
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=100, bbox_inches='tight')
        img_io.seek(0)
        img_base64 = __import__('base64').b64encode(img_io.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            "success": True,
            "algorithm": algorithm_name,
            "parameters": {"F": F, "CR": CR, "NP": NP, "max_fes": max_fes},
            "metrics": {
                "igd_mean": float(np.mean(igd_values)),
                "igd_std": float(np.std(igd_values)),
                "hv_mean": float(np.mean(hv_values)),
                "hv_std": float(np.std(hv_values)),
            },
            "plot": f"data:image/png;base64,{img_base64}",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/budget/simulate', methods=['POST'])
def api_budget_simulate():
    """Run budget sweep simulation."""
    try:
        params = request.json
        budgets_str = params.get("budgets", "1000,5000,10000")
        budgets = [int(b.strip()) for b in budgets_str.split(",")]
        n_runs = int(params.get("n_runs", 2))
        algorithm = params.get("algorithm", "Standard DE")
        
        if not getattr(quant_module.base, "HAS_PYMOO", False):
            return jsonify({"error": "pymoo not available"}), 400
        
        # Use DTLZ2_2obj
        p = quant_module.base.pymoo_get_problem
        problem = p("dtlz2", n_var=12, n_obj=2)
        true_pf = problem.pareto_front()
        
        algo_map = {
            "Standard DE": (quant_module.standard_de, {}),
            "PCA-DE": (quant_module.pca_de, {"alpha": 0.5}),
            "PMODE": (quant_module.pmode, {"alpha": 0.45}),
            "HECO-PDE": (quant_module.heco_pde, {"alpha": 0.60, "n_weights": 12}),
        }
        
        if algorithm not in algo_map:
            algorithm = "Standard DE"
        
        alg_fn, kwargs = algo_map[algorithm]
        
        # Run budget sweep
        igd_means = []
        igd_stds = []
        
        for budget in budgets:
            vals = []
            for run in range(n_runs):
                seed = 42 + run * 13
                res = alg_fn(problem, NP=50, max_fes=budget, seed=seed, **kwargs)
                pf = np.asarray(res.get("pf", np.empty((0, 2))))
                igd_val = float(igd(pf, true_pf)) if len(pf) else float("inf")
                vals.append(igd_val)
            igd_means.append(np.mean(vals))
            igd_stds.append(np.std(vals))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(budgets, igd_means, yerr=igd_stds, marker='o', linewidth=2, markersize=8, capsize=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Function Evaluations', fontsize=11)
        ax.set_ylabel('IGD', fontsize=11)
        ax.set_title(f'Budget Sweep: {algorithm}', fontsize=12)
        ax.grid(alpha=0.3, which='both')
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=100, bbox_inches='tight')
        img_io.seek(0)
        img_base64 = __import__('base64').b64encode(img_io.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            "success": True,
            "algorithm": algorithm,
            "budgets": budgets,
            "igd_means": igd_means,
            "igd_stds": igd_stds,
            "plot": f"data:image/png;base64,{img_base64}",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/download/<filetype>')
def api_download(filetype):
    """Download results as CSV or JSON."""
    try:
        if filetype == "dtlz_csv":
            if "dtlz_df" not in RESULTS_CACHE:
                return jsonify({"error": "Not available"}), 404
            df = RESULTS_CACHE["dtlz_df"]
            return send_file(
                io.BytesIO(df.to_csv(index=False).encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="dtlz_summary.csv"
            )
        elif filetype == "finance_csv":
            if "finance_df" not in RESULTS_CACHE:
                return jsonify({"error": "Not available"}), 404
            df = RESULTS_CACHE["finance_df"]
            return send_file(
                io.BytesIO(df.to_csv(index=False).encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="finance_summary.csv"
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting Quant Project Web App at http://localhost:5000")
    print("Make sure DTLZ and Finance results exist in QuantProjectResults/")
    app.run(debug=True, port=5000, host='127.0.0.1')
