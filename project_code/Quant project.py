"""
Quant Project
=============
Research extension built on top of the existing assignment 4 codebase.

What this adds:
- DTLZ benchmark suite with detailed tables and saved plots
- A regulated financial multiobjective problem with disconnected-feasible regions
- Budget sweep experiments across FEs = 1K, 5K, 10K, 50K, 100K, 200K
- Wilcoxon rank-sum significance testing across algorithms
- Automatic saving of every figure, table, and pickle result

This file imports the original implementation from assignment 4/code/complete_project_code.py
and reuses its core DE / PCA-DE / PMODE / HECO-PDE algorithms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment issue
    raise RuntimeError("pandas is required for Quant project reporting") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment issue
    raise RuntimeError("matplotlib is required for Quant project plotting") from exc

try:
    from scipy.stats import ranksums
except ImportError as exc:  # pragma: no cover - environment issue
    raise RuntimeError("scipy is required for Quant project statistics") from exc

BASE_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = BASE_DIR / "complete_project_code.py"
PROJECT_ROOT = BASE_DIR.parent  # Parent is /project
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "QuantProjectResults"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
PICKLE_DIR = OUTPUT_DIR / "pickles"
REPORT_DIR = OUTPUT_DIR / "reports"


def load_base_module():
    spec = importlib.util.spec_from_file_location("complete_project_code", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

# Reuse the original implementations directly.
standard_de = base.standard_de
pca_de = base.pca_de
pmode = base.pmode
heco_pde = base.heco_pde
non_dominated_front = base.non_dominated_front
igd = base.igd
hypervolume = base.hypervolume
generational_distance = base.generational_distance
spread_metric = base.spread_metric
randomized_svd = base.randomized_svd
proj_weights = base.proj_weights
portfolio_metrics = base.portfolio_metrics
get_dtlz7_true_pf_2obj = base.get_dtlz7_true_pf_2obj
get_dtlz7_true_pf_3obj = base.get_dtlz7_true_pf_3obj


def load_daily_returns_xlsx_strict(returns_xlsx=None):
    """Load real daily returns from Excel and never fall back to synthetic data."""
    if returns_xlsx:
        candidates = [Path(returns_xlsx)]
    else:
        candidates = [
            DATA_DIR / "daily_returns.xlsx",
            BASE_DIR / "daily_returns.xlsx",
        ]

    file_path = None
    for cand in candidates:
        if cand.exists():
            file_path = cand
            break

    if file_path is None:
        raise FileNotFoundError(
            "No daily returns Excel file found. Expected one of: "
            + ", ".join(str(c) for c in candidates)
        )

    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {file_path}")

    # If values look like prices, convert to returns.
    if float(np.nanmax(np.abs(num.values))) > 2.0:
        num = num.pct_change().dropna(how="all")

    ret_df = num.dropna(how="all").ffill().bfill().dropna(axis=1, how="all")
    if ret_df.shape[0] < 20 or ret_df.shape[1] < 2:
        raise ValueError(f"Insufficient return data in {file_path}: shape={ret_df.shape}")

    mu_ann = ret_df.mean().values * 252
    cov_ann = ret_df.cov().values * 252
    tickers = [str(c) for c in ret_df.columns]
    return mu_ann, cov_ann, ret_df, tickers, str(file_path)


class PortfolioFrontier:
    """Three-objective regulated portfolio problem with a disconnected frontier."""

    def __init__(self, mu, cov, ret_df, tickers, sectors):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.ret_df = ret_df
        self.tickers = list(tickers)
        self.sectors = list(sectors)
        self.n_var = len(self.tickers)
        self.n_obj = 3
        self.xl = np.zeros(self.n_var)
        self.xu = np.full(self.n_var, 0.40)
        sector_carbon = {
            "Tech": 0.35,
            "Finance": 0.45,
            "Health": 0.22,
            "Energy": 1.00,
            "Staples": 0.18,
            "Utilities": 0.28,
            "ConsDisc": 0.55,
            "Commodities": 0.95,
        }
        self.carbon = np.array([sector_carbon.get(sec, 0.40) for sec in self.sectors], dtype=float)

    def _evaluate_one(self, x):
        w = proj_weights(x, 0.40)
        risk = float(w @ self.cov @ w)
        expected_return = float(self.mu @ w)
        carbon = float(self.carbon @ w)
        high_emission = float(np.sum(w[np.array(self.sectors) == "Energy"]) + np.sum(w[np.array(self.sectors) == "Commodities"]))
        gap_penalty = 0.0
        if 0.13 <= expected_return <= 0.16:
            gap_penalty += 0.35
        if high_emission > 0.28:
            gap_penalty += 1.5 * (high_emission - 0.28)
        if expected_return > 0.24 and carbon > 0.55:
            gap_penalty += 0.50
        return np.array([risk + gap_penalty, -expected_return, carbon + 0.40 * gap_penalty], dtype=float)

    def evaluate(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self._evaluate_one(X)
        return np.vstack([self._evaluate_one(row) for row in X])


class PCAEscapeDE:
    """A small research prototype that adds PC2 escape steps after stagnation."""

    def __init__(self, alpha=0.5, escape_every=12, escape_scale=0.18):
        self.alpha = alpha
        self.escape_every = escape_every
        self.escape_scale = escape_scale

    def __call__(self, problem, NP=80, F=0.5, CR=0.9, max_fes=10000, seed=42):
        np.random.seed(seed)
        lb, ub = problem.xl, problem.xu
        d = problem.n_var
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        OBJ = problem.evaluate(pop)
        fes = NP
        stall = 0
        best_score = float(np.sum(OBJ.min(axis=0)))

        while fes < max_fes:
            improved = False
            for i in range(NP):
                if fes >= max_fes:
                    break
                idxs = [j for j in range(NP) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                diff = randomized_svd(pop - pop.mean(axis=0), n_components=min(2, max(1, d - 1)), n_iter=2, random_state=0)[2]
                pc1 = diff[0] if len(diff) > 0 else np.ones(d) / np.sqrt(d)
                projected = np.dot(pop[r2] - pop[r3], pc1) * pc1
                mutant = np.clip(pop[r1] + F * ((1 - self.alpha) * (pop[r2] - pop[r3]) + self.alpha * projected), lb, ub)
                mask = np.random.rand(d) < CR
                mask[np.random.randint(d)] = True
                trial = np.where(mask, mutant, pop[i])
                f_trial = problem.evaluate(trial)
                fes += 1
                if np.all(f_trial <= OBJ[i]) and np.any(f_trial < OBJ[i]):
                    pop[i] = trial
                    OBJ[i] = f_trial
                    improved = True
                elif not np.all(OBJ[i] <= f_trial) and np.random.rand() < 0.30:
                    pop[i] = trial
                    OBJ[i] = f_trial
                    improved = True
            current_score = float(np.sum(OBJ.min(axis=0)))
            if current_score + 1e-12 < best_score:
                best_score = current_score
                stall = 0
            else:
                stall += 1
            if stall >= self.escape_every and d >= 2:
                centered = pop - pop.mean(axis=0)
                try:
                    _, _, Vt = randomized_svd(centered, n_components=min(2, d), n_iter=2, random_state=0)
                    pc2 = Vt[1] if Vt.shape[0] > 1 else Vt[0]
                    jitter = np.random.randn(NP, 1) * pc2
                    pop = np.clip(pop + self.escape_scale * jitter, lb, ub)
                    OBJ = problem.evaluate(pop)
                    fes += NP
                except Exception:
                    pass
                stall = 0
        nd_idx = non_dominated_front(OBJ)
        return {"pf": OBJ[nd_idx], "pop": pop[nd_idx], "OBJ": OBJ[nd_idx]}


def ensure_dirs():
    for path in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR, PICKLE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df, name):
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, name):
    path = PICKLE_DIR / name
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def serialise_finance_result(finance_result):
    problem = finance_result["problem"]
    return {
        "problem": {
            "name": problem.__class__.__name__,
            "tickers": problem.tickers,
            "sectors": problem.sectors,
            "mu": np.asarray(problem.mu).tolist(),
            "cov": np.asarray(problem.cov).tolist(),
            "carbon": np.asarray(problem.carbon).tolist(),
        },
        "true_pf": np.asarray(finance_result["true_pf"]).tolist(),
        "ref": np.asarray(finance_result["ref"]).tolist(),
        "summary": finance_result["summary"].to_dict(orient="records"),
        "detail": finance_result["detail"].to_dict(orient="records"),
        "result": {
            name: {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in alg_data.items()
                if key not in {"best_pop"}
            }
            for name, alg_data in finance_result["result"].items()
        },
    }


def safe_true_pf(problem_name, problem, n_obj):
    if "DTLZ7_2obj" in problem_name:
        return get_dtlz7_true_pf_2obj(600)
    if "DTLZ7_3obj" in problem_name:
        return get_dtlz7_true_pf_3obj(1200)
    if hasattr(problem, "pareto_front"):
        try:
            pf = problem.pareto_front()
            if pf is not None and len(pf) > 0:
                return np.asarray(pf)
        except Exception:
            pass
    return None


def approximate_reference_front(problem, n_samples=15000, seed=0):
    rng = np.random.RandomState(seed)
    xl, xu = np.asarray(problem.xl), np.asarray(problem.xu)
    X = xl + rng.rand(n_samples, problem.n_var) * (xu - xl)
    F = problem.evaluate(X)
    nd = non_dominated_front(F)
    return F[nd]


def get_dtlz_problems(include_3obj=True):
    if not getattr(base, "HAS_PYMOO", False):
        return {}
    p = base.pymoo_get_problem
    problems = {
        "DTLZ1_2obj": {"prob": p("dtlz1", n_var=7, n_obj=2), "nobj": 2, "ref": np.array([0.6, 0.6]), "tpf": safe_true_pf("DTLZ1_2obj", p("dtlz1", n_var=7, n_obj=2), 2)},
        "DTLZ2_2obj": {"prob": p("dtlz2", n_var=12, n_obj=2), "nobj": 2, "ref": np.array([2.5, 2.5]), "tpf": safe_true_pf("DTLZ2_2obj", p("dtlz2", n_var=12, n_obj=2), 2)},
        "DTLZ7_2obj": {"prob": p("dtlz7", n_var=12, n_obj=2), "nobj": 2, "ref": np.array([1.0, 5.5]), "tpf": safe_true_pf("DTLZ7_2obj", p("dtlz7", n_var=12, n_obj=2), 2)},
    }
    if include_3obj:
        problems.update({
            "DTLZ1_3obj": {"prob": p("dtlz1", n_var=7, n_obj=3), "nobj": 3, "ref": np.array([0.6, 0.6, 0.6]), "tpf": safe_true_pf("DTLZ1_3obj", p("dtlz1", n_var=7, n_obj=3), 3)},
            "DTLZ2_3obj": {"prob": p("dtlz2", n_var=12, n_obj=3), "nobj": 3, "ref": np.array([2.5, 2.5, 2.5]), "tpf": safe_true_pf("DTLZ2_3obj", p("dtlz2", n_var=12, n_obj=3), 3)},
            "DTLZ7_3obj": {"prob": p("dtlz7", n_var=22, n_obj=3), "nobj": 3, "ref": np.array([1.0, 1.0, 20.0]), "tpf": safe_true_pf("DTLZ7_3obj", p("dtlz7", n_var=22, n_obj=3), 3)},
        })
    return problems


def get_algorithms():
    return [
        ("Standard DE", standard_de, {}),
        ("PCA-DE", pca_de, {"alpha": 0.5}),
        ("PCA-Escape DE", PCAEscapeDE(alpha=0.5, escape_every=10, escape_scale=0.14), {}),
        ("PMODE", pmode, {"alpha": 0.45}),
        ("HECO-PDE", heco_pde, {"alpha": 0.60, "n_weights": 12}),
    ]


def compute_metric_bundle(pf, tpf, ref, nobj, problem_name):
    if tpf is None or len(tpf) == 0:
        tpf = pf
    metrics = {
        "igd": float(igd(pf, tpf)) if len(pf) else float("inf"),
        "gd": float(generational_distance(pf, tpf)) if len(pf) else float("inf"),
        "hv": float(hypervolume(pf, ref, n_samples=8000)) if len(pf) else 0.0,
        "spread": float(spread_metric(pf, tpf)) if (len(pf) and nobj == 2) else float("nan"),
        "segment_coverage": float(base.dtlz7_segment_coverage(pf)) if (len(pf) and nobj == 2 and "DTLZ7" in problem_name) else float("nan"),
    }
    return metrics


def run_algorithm_suite(problem_name, problem, true_pf, ref_point, nobj, algorithms, n_runs=30, max_fes=10000, NP=70, seed_step=13):
    rows = []
    problem_result = {}
    for alg_name, alg_fn, kwargs in algorithms:
        all_igd = []
        all_hv = []
        all_gd = []
        all_spread = []
        all_seg = []
        best_pf = None
        best_pop = None
        best_score = float("inf")
        for run in range(n_runs):
            seed = 7 + run * seed_step
            try:
                res = alg_fn(problem, NP=NP, max_fes=max_fes, seed=seed, **kwargs)
                pf = np.asarray(res.get("pf", np.empty((0, nobj))))
                pop = np.asarray(res.get("pop", np.empty((0, problem.n_var))))
                metrics = compute_metric_bundle(pf, true_pf, ref_point, nobj, problem_name)
                all_igd.append(metrics["igd"])
                all_hv.append(metrics["hv"])
                all_gd.append(metrics["gd"])
                if not np.isnan(metrics["spread"]):
                    all_spread.append(metrics["spread"])
                if not np.isnan(metrics["segment_coverage"]):
                    all_seg.append(metrics["segment_coverage"])
                if metrics["igd"] < best_score:
                    best_score = metrics["igd"]
                    best_pf = pf
                    best_pop = pop
            except Exception as exc:
                print(f"[{problem_name}] {alg_name} run {run} failed: {exc}")
        problem_result[alg_name] = {
            "igd_mean": float(np.nanmean(all_igd)) if all_igd else float("inf"),
            "igd_std": float(np.nanstd(all_igd)) if len(all_igd) > 1 else 0.0,
            "hv_mean": float(np.nanmean(all_hv)) if all_hv else 0.0,
            "gd_mean": float(np.nanmean(all_gd)) if all_gd else float("inf"),
            "spread_mean": float(np.nanmean(all_spread)) if all_spread else float("nan"),
            "segment_coverage": float(np.nanmean(all_seg)) if all_seg else float("nan"),
            "all_igd": all_igd,
            "all_hv": all_hv,
            "all_gd": all_gd,
            "best_pf": best_pf,
            "best_pop": best_pop,
        }
        rows.append({
            "problem": problem_name,
            "algorithm": alg_name,
            "igd_mean": problem_result[alg_name]["igd_mean"],
            "igd_std": problem_result[alg_name]["igd_std"],
            "hv_mean": problem_result[alg_name]["hv_mean"],
            "gd_mean": problem_result[alg_name]["gd_mean"],
            "spread_mean": problem_result[alg_name]["spread_mean"],
            "segment_coverage": problem_result[alg_name]["segment_coverage"],
        })
    return problem_result, pd.DataFrame(rows)


def ranksum_summary(problem_result):
    alg_names = list(problem_result.keys())
    pvals = pd.DataFrame(np.ones((len(alg_names), len(alg_names))), index=alg_names, columns=alg_names)
    for a, b in combinations(alg_names, 2):
        sa = problem_result[a]["all_igd"]
        sb = problem_result[b]["all_igd"]
        if len(sa) and len(sb):
            stat, p = ranksums(sa, sb)
        else:
            p = 1.0
        pvals.loc[a, b] = p
        pvals.loc[b, a] = p
    return pvals


def plot_front(problem_name, true_pf, problem_result, nobj):
    if true_pf is None or len(true_pf) == 0:
        return None
    fig = plt.figure(figsize=(8, 6))
    if nobj == 2:
        ax = fig.add_subplot(111)
        ax.scatter(true_pf[:, 0], true_pf[:, 1], s=10, c="#c9c9c9", alpha=0.55, label="Reference PF")
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(problem_result))))
        for idx, (alg_name, data) in enumerate(problem_result.items()):
            pf = data.get("best_pf")
            if pf is None or len(pf) == 0:
                continue
            ax.scatter(pf[:, 0], pf[:, 1], s=18, alpha=0.85, color=colors[idx], label=alg_name)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=8, c="#c9c9c9", alpha=0.35, label="Reference PF")
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(problem_result))))
        for idx, (alg_name, data) in enumerate(problem_result.items()):
            pf = data.get("best_pf")
            if pf is None or len(pf) == 0:
                continue
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], s=16, alpha=0.85, color=colors[idx], label=alg_name)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
    ax.set_title(problem_name)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_metric_bars(df, metric, title, filename):
    pivot = df.pivot(index="problem", columns="algorithm", values=metric)
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, rot=30)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, filename)


def plot_pvalue_heatmap(pvals, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pvals.values, cmap="viridis_r", vmin=0.0, vmax=0.1)
    ax.set_xticks(range(len(pvals.columns)))
    ax.set_yticks(range(len(pvals.index)))
    ax.set_xticklabels(pvals.columns, rotation=35, ha="right")
    ax.set_yticklabels(pvals.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="p-value")
    return save_figure(fig, filename)


def plot_budget_curves(budget_bundle, problem_name, filename):
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg_name, rows in budget_bundle.items():
        budgets = [r["fes"] for r in rows]
        values = [r["igd_mean"] for r in rows]
        ax.plot(budgets, values, marker="o", linewidth=2, label=alg_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("IGD")
    ax.set_title(f"Budget sweep - {problem_name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return save_figure(fig, filename)


def select_representative_solution(problem, alg_data):
    pf = alg_data.get("best_pf")
    pop = alg_data.get("best_pop")
    if pf is None or len(pf) == 0 or pop is None or len(pop) == 0:
        return None, None
    score = pf[:, 0].astype(float) + pf[:, 1].astype(float)
    if pf.shape[1] > 2:
        score = score + pf[:, 2].astype(float)
    idx = int(np.argmin(score))
    return pop[idx], pf[idx]


def evaluate_finance_solution(w, mu, cov, ret_df, carbon, rf=0.04):
    w = proj_weights(w)
    metrics = portfolio_metrics(w, mu, cov, ret_df, rf=rf)
    metrics["carbon"] = float(np.dot(carbon, w))
    metrics["weights"] = w.tolist()
    return metrics


def run_finance_problem(n_runs=30, max_fes=10000, NP=70, returns_xlsx=None):
    mu, cov, ret_df, tickers, data_source = load_daily_returns_xlsx_strict(returns_xlsx=returns_xlsx)
    sector_map = {t: s for t, s in zip(base.TICKERS, base.SECTORS)}
    sectors = [sector_map.get(t, "Other") for t in tickers]
    print(f"Using real returns from: {data_source}")
    problem = PortfolioFrontier(mu, cov, ret_df, tickers, sectors)
    true_pf = approximate_reference_front(problem, n_samples=18000, seed=11)
    ref_point = np.array([float(true_pf[:, 0].max() * 1.15), float(true_pf[:, 1].max() * 1.10), float(true_pf[:, 2].max() * 1.15)])
    problem_result, detail_df = run_algorithm_suite(
        "RegulatedFinance", problem, true_pf, ref_point, 3, get_algorithms(),
        n_runs=n_runs, max_fes=max_fes, NP=NP,
    )

    finance_rows = []
    for alg_name, alg_data in problem_result.items():
        w, _ = select_representative_solution(problem, alg_data)
        if w is None:
            continue
        metrics = evaluate_finance_solution(w, mu, cov, ret_df, problem.carbon)
        finance_rows.append({
            "algorithm": alg_name,
            "return": metrics["return"],
            "vol": metrics["vol"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "mdd": metrics["mdd"],
            "carbon": metrics["carbon"],
        })

    finance_df = pd.DataFrame(finance_rows)
    return problem, true_pf, ref_point, problem_result, detail_df, finance_df


def run_dtlz_suite(n_runs=30, max_fes=10000, NP=70, include_3obj=True):
    if not getattr(base, "HAS_PYMOO", False):
        return {}, pd.DataFrame(), {}
    problems = get_dtlz_problems(include_3obj=include_3obj)
    all_tables = []
    results = {}
    pvals = {}
    for pname, cfg in problems.items():
        problem = cfg["prob"]
        true_pf = cfg["tpf"]
        if true_pf is None or len(true_pf) == 0:
            true_pf = approximate_reference_front(problem, n_samples=14000, seed=3)
        problem_result, detail_df = run_algorithm_suite(
            pname, problem, true_pf, cfg["ref"], cfg["nobj"], get_algorithms(),
            n_runs=n_runs, max_fes=max_fes, NP=NP,
        )
        results[pname] = {"result": problem_result, "true_pf": true_pf, "nobj": cfg["nobj"]}
        all_tables.append(detail_df)
        pvals[pname] = ranksum_summary(problem_result)
    return results, pd.concat(all_tables, ignore_index=True), pvals


def run_budget_sweep(problem_bundle, budgets, n_runs=3, NP=70):
    sweep = {}
    # Filter problems to speed up: only include 2-objective problems
    filtered_bundle = {k: v for k, v in problem_bundle.items() if v.get("nobj", 3) == 2 or k == "RegulatedFinance"}
    
    for pname, cfg in filtered_bundle.items():
        problem = cfg["prob"] if pname != "RegulatedFinance" else cfg["problem"]
        true_pf = cfg["tpf"] if pname != "RegulatedFinance" else cfg["true_pf"]
        ref = cfg["ref"] if pname != "RegulatedFinance" else cfg["ref"]
        nobj = cfg["nobj"] if pname != "RegulatedFinance" else 3
        sweep[pname] = {}
        for alg_name, alg_fn, kwargs in get_algorithms():
            rows = []
            for fes in budgets:
                vals = []
                for run in range(n_runs):
                    seed = 17 + run * 19
                    res = alg_fn(problem, NP=NP, max_fes=fes, seed=seed, **kwargs)
                    pf = np.asarray(res.get("pf", np.empty((0, nobj))))
                    vals.append(float(igd(pf, true_pf)) if len(pf) else float("inf"))
                rows.append({"fes": int(fes), "igd_mean": float(np.mean(vals)), "igd_std": float(np.std(vals))})
            sweep[pname][alg_name] = rows
            budget_df = pd.DataFrame(rows)
            save_table(budget_df.assign(problem=pname, algorithm=alg_name), f"budget_{pname}_{alg_name}.csv")
    return sweep


def write_markdown_report(dtlz_df, finance_df):
    lines = ["# Quant Project Report", "", "## DTLZ Summary", ""]
    if len(dtlz_df):
        lines.append(dtlz_df.sort_values(["problem", "igd_mean"]).to_string(index=False))
        lines.append("")
    lines.extend(["## Finance Summary", ""])
    if len(finance_df):
        lines.append(finance_df.sort_values("sharpe", ascending=False).to_string(index=False))
        lines.append("")
    path = REPORT_DIR / "quant_project_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_section(title):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def print_dataframe(label, df, sort_cols=None):
    print_section(label)
    if df is None or len(df) == 0:
        print("No rows to display.")
        return
    out = df.copy()
    if sort_cols:
        out = out.sort_values(sort_cols)
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 240):
        print(out.to_string(index=False))


def print_saved_artifacts(artifact_paths):
    print_section("Saved Artifacts")
    for p in artifact_paths:
        print(str(p))


def summarize_budget_sweep(sweep):
    rows = []
    for problem, alg_data in sweep.items():
        for alg_name, records in alg_data.items():
            if not records:
                continue
            first = records[0]
            last = records[-1]
            rows.append({
                "problem": problem,
                "algorithm": alg_name,
                "fes_start": first["fes"],
                "igd_start": first["igd_mean"],
                "fes_end": last["fes"],
                "igd_end": last["igd_mean"],
                "delta_igd": last["igd_mean"] - first["igd_mean"],
            })
    return pd.DataFrame(rows)


def run_all(args):
    ensure_dirs()
    bundle = {"meta": {"args": vars(args)}}
    dtlz_results = {}
    dtlz_df = pd.DataFrame()
    dtlz_pvals = {}
    finance_result = None
    finance_df = pd.DataFrame()
    saved_artifacts = []

    if args.dtlz or args.all:
        if not getattr(base, "HAS_PYMOO", False):
            print("pymoo is not available, skipping DTLZ suite.")
        else:
            # Skip 3-objective problems in quick mode
            skip_3obj = args.quick
            include_3obj = not skip_3obj
            dtlz_results, dtlz_df, dtlz_pvals = run_dtlz_suite(n_runs=args.runs, max_fes=args.fes, NP=args.np)
            if skip_3obj:
                # Filter out 3-objective problems from results
                dtlz_df = dtlz_df[~dtlz_df["problem"].str.contains("3obj")]
                dtlz_results = {k: v for k, v in dtlz_results.items() if "3obj" not in k}
                dtlz_pvals = {k: v for k, v in dtlz_pvals.items() if "3obj" not in k}
            bundle["dtlz"] = dtlz_results
            bundle["dtlz_table"] = dtlz_df.to_dict(orient="records")
            bundle["dtlz_pvalues"] = {k: v.to_dict() for k, v in dtlz_pvals.items()}
            saved_artifacts.append(save_pickle(dtlz_results, "dtlz_results.pkl"))
            saved_artifacts.append(save_table(dtlz_df, "dtlz_summary.csv"))
            for pname, pdata in dtlz_results.items():
                fig = plot_front(pname, pdata["true_pf"], pdata["result"], pdata["nobj"])
                if fig is not None:
                    saved_artifacts.append(save_figure(fig, f"{pname}_front.png"))
                saved_artifacts.append(plot_pvalue_heatmap(dtlz_pvals[pname], f"Wilcoxon p-values - {pname}", f"{pname}_wilcoxon.png"))
            saved_artifacts.append(plot_metric_bars(dtlz_df, "igd_mean", "DTLZ IGD comparison", "dtlz_igd_bar.png"))
            saved_artifacts.append(plot_metric_bars(dtlz_df, "hv_mean", "DTLZ hypervolume comparison", "dtlz_hv_bar.png"))
            print_dataframe("DTLZ Summary (saved to dtlz_summary.csv)", dtlz_df, sort_cols=["problem", "igd_mean"])
            for pname, pval_df in dtlz_pvals.items():
                print_dataframe(f"Wilcoxon P-Values: {pname}", pval_df.reset_index().rename(columns={"index": "algorithm"}))

    if args.finance or args.all:
        finance_problem, finance_true_pf, finance_ref, finance_problem_result, finance_detail_df, finance_df = run_finance_problem(
            n_runs=args.runs,
            max_fes=args.fes,
            NP=args.np,
            returns_xlsx=args.returns_xlsx,
        )
        finance_result = {
            "problem": finance_problem,
            "true_pf": finance_true_pf,
            "ref": finance_ref,
            "result": finance_problem_result,
            "detail": finance_detail_df,
            "summary": finance_df,
        }
        bundle["finance"] = {
            "summary": finance_df.to_dict(orient="records"),
            "detail": finance_detail_df.to_dict(orient="records"),
        }
        saved_artifacts.append(save_pickle(serialise_finance_result(finance_result), "finance_results.pkl"))
        saved_artifacts.append(save_table(finance_df, "finance_summary.csv"))
        saved_artifacts.append(save_table(finance_detail_df, "finance_objective_summary.csv"))
        fig = plot_front("RegulatedFinance", finance_true_pf, finance_problem_result, 3)
        if fig is not None:
            saved_artifacts.append(save_figure(fig, "RegulatedFinance_front.png"))
        if len(finance_df):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(finance_df["vol"], finance_df["return"], c=finance_df["sharpe"], cmap="viridis", s=90)
            for _, row in finance_df.iterrows():
                ax.annotate(row["algorithm"], (row["vol"], row["return"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Return")
            ax.set_title("Finance efficient frontier proxy")
            fig.colorbar(ax.collections[0], ax=ax, label="Sharpe")
            saved_artifacts.append(save_figure(fig, "finance_frontier_proxy.png"))
        saved_artifacts.append(plot_metric_bars(finance_detail_df, "igd_mean", "Finance IGD comparison", "finance_igd_bar.png"))
        if len(finance_df):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(finance_df["algorithm"], finance_df["carbon"], color="#4c78a8")
            ax.set_ylabel("Carbon intensity")
            ax.set_title("Selected finance solutions: carbon intensity")
            ax.tick_params(axis="x", rotation=25)
            saved_artifacts.append(save_figure(fig, "finance_carbon_bar.png"))
        print_dataframe("Finance Objective Summary (saved to finance_objective_summary.csv)", finance_detail_df, sort_cols=["igd_mean"])
        if len(finance_df):
            print_dataframe("Finance Portfolio Summary (saved to finance_summary.csv)", finance_df.sort_values("sharpe", ascending=False))

    if args.budget or args.all:
        if args.quick:
            # Quick mode: minimal budget sweep (3 budgets, 1 run each)
            budget_budgets = [5000, 50000, 200000]
            budget_n_runs = 1
            print("[QUICK MODE] Budget sweep: 3 budgets, 1 run per budget, 2-obj problems only")
        elif args.fast:
            # Fast mode: reduced budget sweep (4 budgets, 1 run each)
            budget_budgets = [5000, 20000, 100000, 200000]
            budget_n_runs = 1
            print("[FAST MODE] Budget sweep: 4 budgets, 1 run per budget, 2-obj problems only")
        else:
            # Normal mode: full budget sweep
            budget_budgets = [1000, 5000, 10000, 50000, 100000, 200000]
            budget_n_runs = max(2, args.runs // 2)
        
        budget_problem_bundle = {}
        if getattr(base, "HAS_PYMOO", False):
            budget_problem_bundle.update(get_dtlz_problems())
        if finance_result is None and (args.finance or args.all or args.budget):
            finance_problem, finance_true_pf, finance_ref, finance_problem_result, finance_detail_df, finance_df = run_finance_problem(
                n_runs=max(2, args.runs),
                max_fes=args.fes,
                NP=args.np,
                returns_xlsx=args.returns_xlsx,
            )
            finance_result = {
                "problem": finance_problem,
                "true_pf": finance_true_pf,
                "ref": finance_ref,
                "result": finance_problem_result,
                "detail": finance_detail_df,
                "summary": finance_df,
            }
        if finance_result is not None:
            budget_problem_bundle["RegulatedFinance"] = {
                "problem": finance_result["problem"],
                "true_pf": finance_result["true_pf"],
                "ref": finance_result["ref"],
                "nobj": 3,
            }
        if budget_problem_bundle:
            sweep = run_budget_sweep(budget_problem_bundle, budget_budgets, n_runs=budget_n_runs, NP=args.np)
            bundle["budget_sweep"] = sweep
            saved_artifacts.append(save_pickle(sweep, "budget_sweep.pkl"))
            for pname, alg_rows in sweep.items():
                saved_artifacts.append(plot_budget_curves(alg_rows, pname, f"budget_{pname}.png"))
            budget_summary = summarize_budget_sweep(sweep)
            print_dataframe("Budget Sweep Summary", budget_summary, sort_cols=["problem", "delta_igd"])

    if len(dtlz_df) or len(finance_df):
        saved_artifacts.append(write_markdown_report(dtlz_df, finance_df))

    if dtlz_results:
        bundle["dtlz_summary"] = dtlz_df.to_dict(orient="records")
    if finance_result is not None:
        bundle["finance_summary"] = finance_df.to_dict(orient="records")

    saved_artifacts.append(save_pickle(bundle, "quant_project_bundle.pkl"))
    print_saved_artifacts(saved_artifacts)
    return bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quant project research driver")
    parser.add_argument("--dtlz", action="store_true", help="Run only DTLZ experiments")
    parser.add_argument("--finance", action="store_true", help="Run only the regulated finance experiment")
    parser.add_argument("--budget", action="store_true", help="Run the budget sweep")
    parser.add_argument("--all", action="store_true", help="Run all research experiments")
    parser.add_argument("--quick", action="store_true", help="Fast mode: 1 run, 2K FEs, skip 3-obj, minimal budget sweep")
    parser.add_argument("--fast", action="store_true", help="Fast-ish mode: 2 runs, 5K FEs, reduced budget sweep")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs per algorithm (default: 30)")
    parser.add_argument("--fes", type=int, default=10000, help="Maximum function evaluations")
    parser.add_argument("--np", type=int, default=70, help="Population size")
    parser.add_argument("--returns-xlsx", type=str, default=None, help="Path to daily returns Excel file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not (args.dtlz or args.finance or args.budget or args.all):
        args.all = True
    
    # Apply quick/fast mode overrides
    if args.quick:
        args.runs = 1
        args.fes = 2000
        args.np = 30
        print("\n" + "="*80)
        print("QUICK MODE ENABLED: runs=1, fes=2000, np=30, minimal budget sweep")
        print("Expected runtime: 5-15 minutes")
        print("="*80 + "\n")
    elif args.fast:
        args.runs = 2
        args.fes = 5000
        args.np = 50
        print("\n" + "="*80)
        print("FAST MODE ENABLED: runs=2, fes=5000, np=50, reduced budget sweep")
        print("Expected runtime: 20-40 minutes")
        print("="*80 + "\n")
    
    ensure_dirs()
    bundle = run_all(args)
    print(f"Saved outputs to {OUTPUT_DIR}")
    return bundle


if __name__ == "__main__":
    main()
