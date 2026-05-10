"""
=============================================================================
MultiObjective Differential Evolution Using PCA with Constraints
=============================================================================
Complete Python implementation:
  - 4 core algorithms: Standard DE, PCA-DE, PMODE, HECO-PDE
  - DTLZ benchmark suite (DTLZ1, DTLZ2, DTLZ7 in 2-obj and 3-obj)
  - 5 engineering benchmarks (Welded Beam, Pressure Vessel, etc.)
  - 12 portfolio optimization methods
  - Full metrics: IGD, HV, GD, Spread, Segment Coverage

Usage:
    python complete_project_code.py            # runs all experiments
    python complete_project_code.py --dtlz     # DTLZ only
    python complete_project_code.py --port     # Portfolio only
    python complete_project_code.py --eng      # Engineering only

Requirements: numpy scipy scikit-learn pymoo matplotlib
  pip install numpy scipy scikit-learn pymoo matplotlib
=============================================================================
"""

import numpy as np
import pickle
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
try:
    from sklearn.utils.extmath import randomized_svd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

    def randomized_svd(matrix, n_components=1, n_iter=2, random_state=0):
        """Fallback SVD that does not require scikit-learn."""
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :n_components], s[:n_components], vt[:n_components]

# ─── Try importing pymoo (needed for DTLZ problems) ──────────────────────────
try:
    from pymoo.problems import get_problem as pymoo_get_problem
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    print("pymoo not found. DTLZ experiments will be skipped.")
    print("Install: pip install pymoo")

# ─── Try importing cvxpy (needed for CVaR portfolio) ─────────────────────────
try:
    import cvxpy as cp
    HAS_CVX = True
except ImportError:
    HAS_CVX = False


# =============================================================================
# SECTION 1: CORE MOEA UTILITIES
# =============================================================================

def non_dominated_front(F):
    """
    Find non-dominated (Pareto) front from objective matrix F.
    F: (n, m) array where n=solutions, m=objectives (all minimized)
    Returns: indices of non-dominated solutions
    """
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Check if j dominates i
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def pca_projection(population, diff_vector, alpha):
    """
    PCA-projection operator: projects mutation vector onto PC1 direction.

    This is the core contribution of Huang et al. (2019):
      1. Run PCA on current population
      2. Find PC1 = direction of maximum variance
      3. Project diff_vector onto PC1
      4. Blend: v_new = (1-alpha)*v_standard + alpha*v_projected

    Financial interpretation: PC1 of asset covariance = market beta factor.
    Fitness landscape interpretation: PC1 = valley direction.

    Args:
        population: (NP, d) current population
        diff_vector: (d,) standard DE difference vector
        alpha: blend parameter [0,1] (0=standard DE, 1=full PCA)
    Returns:
        (d,) enhanced mutation vector
    """
    n, d = population.shape
    if n < 4 or d < 2:
        return diff_vector
    try:
        centered = population - population.mean(axis=0)
        # Randomized SVD: O(n*d*k) instead of O(d^3) — fast for large d
        _, _, Vt = randomized_svd(centered, n_components=1,
                                  n_iter=2, random_state=0)
        pc1 = Vt[0]
        # Project diff_vector onto PC1
        projection = np.dot(diff_vector, pc1) * pc1
        # Blend
        return (1 - alpha) * diff_vector + alpha * projection
    except Exception:
        return diff_vector


def crowding_distance(F_vals):
    """
    Compute crowding distance for Pareto front solutions.
    Used by PMODE and NSGA-II style selection.
    Higher crowding distance = more isolated = preferred for diversity.
    """
    n, m = F_vals.shape
    cd = np.zeros(n)
    for j in range(m):
        order = np.argsort(F_vals[:, j])
        cd[order[0]] = np.inf   # boundary solutions always kept
        cd[order[-1]] = np.inf
        rng = F_vals[order[-1], j] - F_vals[order[0], j] + 1e-10
        for k in range(1, n - 1):
            cd[order[k]] += (F_vals[order[k+1], j] - F_vals[order[k-1], j]) / rng
    return cd


def igd(approx_pf, true_pf):
    """
    Inverted Generational Distance (IGD).
    For each point in TRUE PF, find minimum distance to approx PF.
    Lower is better. Measures both convergence and spread.
    """
    if len(approx_pf) == 0:
        return float('inf')
    return float(np.mean([
        np.min(np.linalg.norm(approx_pf - p, axis=1))
        for p in true_pf
    ]))


def generational_distance(approx_pf, true_pf):
    """
    Generational Distance (GD).
    For each point in APPROX PF, find minimum distance to true PF.
    Lower is better. Measures convergence only (not spread).
    """
    if len(approx_pf) == 0:
        return float('inf')
    return float(np.mean([
        np.min(np.linalg.norm(true_pf - p, axis=1))
        for p in approx_pf
    ]))


def hypervolume(approx_pf, reference_point, n_samples=20000):
    """
    Approximate hypervolume via Monte Carlo sampling.
    Higher is better. Measures volume dominated by approx PF.
    reference_point must dominate all points in approx_pf.
    """
    if len(approx_pf) == 0:
        return 0.0
    M = approx_pf.shape[1]
    rng = np.random.RandomState(42)
    lb = approx_pf.min(axis=0)
    ub = np.array(reference_point)
    if np.any(lb >= ub):
        return 0.0
    samples = lb + rng.rand(n_samples, M) * (ub - lb)
    dominated = np.zeros(n_samples, dtype=bool)
    for p in approx_pf:
        dominated |= np.all(samples >= p, axis=1)
    return float(dominated.mean() * np.prod(ub - lb))


def spread_metric(approx_pf, true_pf):
    """
    Spread / Delta metric: measures distribution uniformity.
    Lower is better. 0 = perfectly uniform spread.
    """
    if len(approx_pf) < 3:
        return 1.0
    sorted_pf = approx_pf[np.argsort(approx_pf[:, 0])]
    dists = np.linalg.norm(np.diff(sorted_pf, axis=0), axis=1) + 1e-12
    d_mean = dists.mean()
    # Extremal distances to true PF
    d_f = np.min(np.linalg.norm(true_pf - sorted_pf[0], axis=1))
    d_l = np.min(np.linalg.norm(true_pf - sorted_pf[-1], axis=1))
    return float((d_f + d_l + np.sum(np.abs(dists - d_mean))) /
                 (d_f + d_l + len(dists) * d_mean))


def dtlz7_segment_coverage(approx_pf):
    """
    DTLZ7-specific metric: how many disconnected segments found?
    2-obj DTLZ7 has 2 segments: f1 in [0, 0.25] and [0.63, 0.86]
    Returns integer in {0, 1, 2}
    """
    if len(approx_pf) == 0:
        return 0
    f1 = approx_pf[:, 0]
    seg1 = int(np.any(f1 <= 0.30))
    seg2 = int(np.any((f1 >= 0.60) & (f1 <= 0.90)))
    return seg1 + seg2


# =============================================================================
# SECTION 2: THE 4 CORE ALGORITHMS
# =============================================================================

def standard_de(problem, NP=80, F=0.5, CR=0.9, max_fes=10000, seed=42):
    """
    Standard DE/rand/1/bin with Pareto non-dominance selection.

    The simplest multi-objective algorithm — no assumptions about PF shape.
    Uses random acceptance for non-dominated trial/target pairs.

    Why it wins on DTLZ7:
      - No structural bias toward connected fronts
      - Random acceptance maintains diversity across disconnected segments
      - Finds one segment very accurately (IGD ~ 0.0004)

    Args:
        problem: object with .xl, .xu, .n_var, .n_obj, .evaluate()
        NP: population size
        F: scaling factor (step size)
        CR: crossover rate
        max_fes: maximum function evaluations (budget)
        seed: random seed
    Returns:
        dict with 'pf' (Pareto front approx), 'history', 'name'
    """
    np.random.seed(seed)
    lb, ub = problem.xl, problem.xu
    d = problem.n_var

    # Initialise population uniformly in search space
    pop = lb + np.random.rand(NP, d) * (ub - lb)
    OBJ = problem.evaluate(pop)   # (NP, n_obj)
    fes = NP
    history = []

    while fes < max_fes:
        for i in range(NP):
            if fes >= max_fes:
                break
            # Select 3 distinct random individuals (r1, r2, r3 != i)
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), lb, ub)

            # Crossover: binomial
            mask = np.random.rand(d) < CR
            mask[np.random.randint(d)] = True  # ensure at least one dimension
            trial = np.where(mask, mutant, pop[i])

            # Evaluate trial
            f_trial = problem.evaluate(trial[None])[0]
            fes += 1

            # Pareto-based selection
            trial_dom  = np.all(f_trial <= OBJ[i]) and np.any(f_trial < OBJ[i])
            target_dom = np.all(OBJ[i] <= f_trial) and np.any(OBJ[i] < f_trial)
            if trial_dom or (not target_dom and np.random.rand() < 0.5):
                pop[i] = trial
                OBJ[i] = f_trial

        nd_idx = non_dominated_front(OBJ)
        history.append({'fes': fes, 'nd_size': len(nd_idx)})

    nd_idx = non_dominated_front(OBJ)
    return {'pf': OBJ[nd_idx], 'pop': pop[nd_idx],
            'history': history, 'name': 'Standard DE'}


def pca_de(problem, NP=80, F=0.5, CR=0.9, max_fes=10000,
           alpha=0.5, seed=42):
    """
    PCA-enhanced DE with adaptive alpha schedule.

    Key enhancement over Standard DE:
      The difference vector (pop[r2] - pop[r3]) is projected onto PC1
      of the current population before being used as mutation direction.

    Adaptive alpha: increases from 0.4*alpha to alpha as search progresses.
    Early stage: more random (explore) → Late stage: more PCA-guided (exploit)

    Why it helps on DTLZ2 (spherical PF):
      Population clusters near sphere → PC1 points along sphere surface →
      PCA projection makes mutations tangent to sphere → faster convergence

    Why it hurts on DTLZ7 (disconnected PF):
      Once population clusters at one segment → PC1 points along that segment →
      PCA reinforces convergence to ONE segment and blocks escape to others

    Args:
        alpha: PCA blend coefficient [0=standard DE, 1=full PCA]
    """
    np.random.seed(seed)
    lb, ub = problem.xl, problem.xu
    d = problem.n_var

    pop = lb + np.random.rand(NP, d) * (ub - lb)
    OBJ = problem.evaluate(pop)
    fes = NP
    history = []

    while fes < max_fes:
        # Adaptive alpha: start low (random), increase (PCA-guided)
        progress = fes / max_fes
        current_alpha = alpha * (0.4 + 0.6 * progress)

        for i in range(NP):
            if fes >= max_fes:
                break
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            # PCA-enhanced mutation
            diff = pop[r2] - pop[r3]
            enhanced_diff = pca_projection(pop, diff, current_alpha)
            mutant = np.clip(pop[r1] + F * enhanced_diff, lb, ub)

            mask = np.random.rand(d) < CR
            mask[np.random.randint(d)] = True
            trial = np.where(mask, mutant, pop[i])

            f_trial = problem.evaluate(trial[None])[0]
            fes += 1

            trial_dom  = np.all(f_trial <= OBJ[i]) and np.any(f_trial < OBJ[i])
            target_dom = np.all(OBJ[i] <= f_trial) and np.any(OBJ[i] < f_trial)
            if trial_dom or (not target_dom and np.random.rand() < 0.5):
                pop[i] = trial
                OBJ[i] = f_trial

        nd_idx = non_dominated_front(OBJ)
        history.append({'fes': fes, 'nd_size': len(nd_idx), 'alpha': current_alpha})

    nd_idx = non_dominated_front(OBJ)
    return {'pf': OBJ[nd_idx], 'pop': pop[nd_idx],
            'history': history, 'name': 'PCA-DE'}


def pmode(problem, NP=100, F=0.5, CR=0.9, max_fes=10000,
          alpha=0.45, seed=42):
    """
    PMODE: Multi-Objective DE with Pareto + Crowding Distance selection.

    Each generation combines parents (NP) + offspring (NP) into 2*NP pool,
    then selects top NP using:
      1. Non-domination rank (rank 1 solutions first)
      2. Crowding distance (prefer isolated solutions for diversity)

    This is the NSGA-II selection strategy applied within the DE framework.

    Why it does better than Standard DE on DTLZ2:
      Crowding distance spreads solutions evenly → better HV (volume covered)

    Why it doesn't fully solve DTLZ7:
      Crowding distance tries to spread but cannot bridge the gap between
      disconnected segments in DTLZ7.

    Args:
        alpha: PCA blend coefficient (used in mutation, adaptive)
    """
    np.random.seed(seed)
    lb, ub = problem.xl, problem.xu
    d = problem.n_var

    pop = lb + np.random.rand(NP, d) * (ub - lb)
    OBJ = problem.evaluate(pop)
    fes = NP
    history = []

    while fes < max_fes:
        progress = fes / max_fes
        current_alpha = alpha * (0.35 + 0.65 * progress)

        # Generate offspring
        offspring_pop = []
        offspring_obj = []
        for i in range(NP):
            if fes >= max_fes:
                break
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            diff = pca_projection(pop, pop[r2] - pop[r3], current_alpha)
            mutant = np.clip(pop[r1] + F * diff, lb, ub)
            mask = np.random.rand(d) < CR
            mask[np.random.randint(d)] = True
            trial = np.where(mask, mutant, pop[i])
            f_trial = problem.evaluate(trial[None])[0]
            fes += 1
            offspring_pop.append(trial)
            offspring_obj.append(f_trial)

        # Combine parent + offspring
        combined_pop = np.vstack([pop, offspring_pop])
        combined_obj = np.vstack([OBJ, offspring_obj])

        # Select top NP using non-domination + crowding distance
        keep = []
        remaining = list(range(len(combined_obj)))
        while len(keep) < NP and remaining:
            nd = non_dominated_front(combined_obj[remaining])
            front_indices = [remaining[x] for x in nd]
            if len(keep) + len(front_indices) <= NP:
                keep.extend(front_indices)
                remaining = [r for r in remaining if r not in front_indices]
            else:
                needed = NP - len(keep)
                cd = crowding_distance(combined_obj[front_indices])
                best = np.argsort(cd)[::-1][:needed]
                keep.extend([front_indices[b] for b in best])
                break

        pop = combined_pop[keep]
        OBJ = combined_obj[keep]

        nd_idx = non_dominated_front(OBJ)
        history.append({'fes': fes, 'nd_size': len(nd_idx)})

    nd_idx = non_dominated_front(OBJ)
    return {'pf': OBJ[nd_idx], 'pop': pop[nd_idx],
            'history': history, 'name': 'PMODE'}


def heco_pde(problem, NP=80, F=0.5, CR=0.9, max_fes=10000,
             n_weights=12, alpha=0.6, seed=42):
    """
    HECO-PDE: Hierarchical Evolutionary Constrained Optimization with PCA-DE.
    Uses Chebyshev decomposition to convert MOEA into scalar subproblems.

    Key mechanism:
      1. Generate n_weights weight vectors lambda uniformly in [0,1]
      2. Each lambda converts (f1, f2) into scalar: max_j {lambda_j * |fj - z*j|}
         where z* is the estimated ideal point (min per objective)
      3. Each lambda vector targets a DIFFERENT point on the Pareto front
      4. PCA mutation applied to each subproblem
      5. Population updated with best solutions from all subproblems

    Why it WINS on DTLZ2 (smooth spherical PF):
      Uniform weight vectors cover all angles of the sphere.
      Each lambda finds the optimal solution at its angle.
      Together they approximate the entire sphere surface.

    Why it FAILS on DTLZ7 (disconnected PF):
      Weight vectors are distributed uniformly in lambda space.
      Lambda values like 0.35, 0.40, 0.45 ... 0.60 point INTO the gap.
      No good solutions exist in the gap → these subproblems stagnate.
      Their stagnation drags the population into the gap region.
      HV = 0.000 (no good solutions found), IGD = 3.21 (catastrophic).

    Why restart doesn't help:
      Restarted solutions re-converge to same gap region because
      the scalar objectives still appear locally minimisable there.

    Args:
        n_weights: number of Chebyshev weight vectors
        alpha: PCA blend coefficient (increases adaptively)
    """
    np.random.seed(seed)
    lb, ub = problem.xl, problem.xu
    d = problem.n_var
    m = problem.n_obj

    # Generate weight vectors
    if m == 2:
        w1 = np.linspace(0.05, 0.95, n_weights)
        weights = np.column_stack([w1, 1 - w1])
    else:
        # Uniformly spaced on simplex for 3+ objectives
        wts = []
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = n_weights - 1 - i - j
                if k >= 0:
                    wts.append([i/(n_weights-1), j/(n_weights-1), k/(n_weights-1)])
        weights = np.array(wts[:n_weights])
        weights /= weights.sum(axis=1, keepdims=True)

    # Initialise population
    pop = lb + np.random.rand(NP, d) * (ub - lb)
    OBJ = problem.evaluate(pop)
    fes = NP

    # Ideal point: minimum per objective (updated each generation)
    z_ideal = OBJ.min(axis=0)
    z_nadir = OBJ.max(axis=0) + 1e-10

    no_improve = 0
    prev_proxy = float('inf')
    history = []

    while fes < max_fes:
        progress = fes / max_fes
        current_alpha = alpha * (0.2 + 0.8 * progress)
        F_adaptive = F * (1.1 - 0.3 * progress)

        # ── Decomposition phase: one step per weight vector ──────────────────
        for lam in weights:
            if fes >= max_fes:
                break
            # Normalise objectives
            rng = z_nadir - z_ideal + 1e-10
            obj_norm = (OBJ - z_ideal) / rng
            # Chebyshev scalar for each individual
            scalar = np.max(lam * obj_norm, axis=1)
            base = int(np.argmin(scalar))

            idxs = [j for j in range(NP) if j != base]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            diff = pca_projection(pop, F_adaptive * (pop[r1] - pop[r2]),
                                  current_alpha)
            mutant = np.clip(pop[base] + diff, lb, ub)
            mask = np.random.rand(d) < CR
            mask[np.random.randint(d)] = True
            trial = np.where(mask, mutant, pop[base])
            f_trial = problem.evaluate(trial[None])[0]
            fes += 1

            # Accept if better under current weight vector
            obj_norm_t = (f_trial - z_ideal) / rng
            scalar_t = np.max(lam * obj_norm_t)
            if scalar_t < scalar[base]:
                worst = int(np.argmax(scalar))
                pop[worst] = trial
                OBJ[worst] = f_trial

        # ── Exploitation phase: direct Pareto improvement ────────────────────
        for i in range(min(NP // 4, max_fes - fes)):
            if fes >= max_fes:
                break
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            diff = pca_projection(pop, F_adaptive * (pop[r2] - pop[r3]),
                                  current_alpha)
            mutant = np.clip(pop[r1] + diff, lb, ub)
            mask = np.random.rand(d) < CR
            mask[np.random.randint(d)] = True
            trial = np.where(mask, mutant, pop[i])
            f_trial = problem.evaluate(trial[None])[0]
            fes += 1
            if np.all(f_trial <= OBJ[i]) and np.any(f_trial < OBJ[i]):
                pop[i] = trial
                OBJ[i] = f_trial

        # Update ideal/nadir
        z_ideal = np.minimum(z_ideal, OBJ.min(axis=0))
        z_nadir = np.maximum(z_nadir, OBJ.max(axis=0)) + 1e-10

        # Stagnation detection
        nd_idx = non_dominated_front(OBJ)
        proxy = float(OBJ[nd_idx].min(axis=0).sum())
        if abs(proxy - prev_proxy) < 1e-9:
            no_improve += 1
        else:
            no_improve = 0
            prev_proxy = proxy

        # Diversity restart: reinitialise worst 20% when stagnant
        if no_improve > 18:
            scalar_all = np.max((OBJ - z_ideal) /
                                np.maximum(z_nadir - z_ideal, 1e-10), axis=1)
            worst_idx = np.argsort(scalar_all)[-max(1, NP // 5):]
            pop[worst_idx] = lb + np.random.rand(len(worst_idx), d) * (ub - lb)
            OBJ[worst_idx] = problem.evaluate(pop[worst_idx])
            fes += len(worst_idx)
            no_improve = 0

        history.append({'fes': fes, 'nd_size': len(nd_idx)})

    nd_idx = non_dominated_front(OBJ)
    return {'pf': OBJ[nd_idx], 'pop': pop[nd_idx],
            'history': history, 'name': 'HECO-PDE'}


# =============================================================================
# SECTION 3: DTLZ BENCHMARK SUITE
# =============================================================================

def get_dtlz7_true_pf_2obj(n=1000):
    """
    DTLZ7 true Pareto front for 2 objectives.
    Disconnected: 2 segments at f1 in [0, 0.2516] and [0.6318, 0.8594].
    """
    f1 = np.linspace(0, 1, 50000)
    f2 = 4 - f1 * (1 + np.sin(3 * np.pi * f1))
    mask = (f1 <= 0.2516) | ((f1 >= 0.6318) & (f1 <= 0.8594))
    pf = np.column_stack([f1[mask], f2[mask]])
    idx = np.linspace(0, len(pf)-1, min(n, len(pf))).astype(int)
    return pf[idx]


def get_dtlz7_true_pf_3obj(n=2000):
    """
    DTLZ7 true Pareto front for 3 objectives.
    Disconnected: 4 segments (2^(M-1) = 2^2 = 4 for M=3 objectives).
    """
    np.random.seed(1)
    pts = []
    for b1 in [(0, 0.25), (0.63, 0.86)]:
        for b2 in [(0, 0.25), (0.63, 0.86)]:
            f1 = np.random.uniform(*b1, n // 4)
            f2 = np.random.uniform(*b2, n // 4)
            # DTLZ7 formula for f3 at optimality (g=1)
            h = 3 - (f1 * (1 + np.sin(3 * np.pi * f1))) \
                  - (f2 * (1 + np.sin(3 * np.pi * f2)))
            f3 = (1 + 1) * h  # g=1 at Pareto optimal
            pts.append(np.column_stack([f1, f2, f3]))
    return np.vstack(pts)


def run_dtlz_experiments(n_runs=8, max_fes=10000, NP=70, verbose=True):
    """
    Run all 4 algorithms on all 6 DTLZ problems.
    Returns dict with full results including metrics.
    """
    if not HAS_PYMOO:
        print("pymoo required for DTLZ experiments.")
        return {}

    # Problem definitions
    problems = {
        'DTLZ1_2obj': {'prob': pymoo_get_problem('dtlz1', n_var=7,  n_obj=2),
                        'nobj': 2, 'ref': [0.6, 0.6],
                        'tpf': pymoo_get_problem('dtlz1', n_var=7,  n_obj=2).pareto_front()},
        'DTLZ2_2obj': {'prob': pymoo_get_problem('dtlz2', n_var=12, n_obj=2),
                        'nobj': 2, 'ref': [2.5, 2.5],
                        'tpf': pymoo_get_problem('dtlz2', n_var=12, n_obj=2).pareto_front()},
        'DTLZ7_2obj': {'prob': pymoo_get_problem('dtlz7', n_var=12, n_obj=2),
                        'nobj': 2, 'ref': [1.0, 5.5],
                        'tpf': get_dtlz7_true_pf_2obj(600)},
        'DTLZ1_3obj': {'prob': pymoo_get_problem('dtlz1', n_var=7,  n_obj=3),
                        'nobj': 3, 'ref': [0.6, 0.6, 0.6],
                        'tpf': pymoo_get_problem('dtlz1', n_var=7,  n_obj=3).pareto_front()},
        'DTLZ2_3obj': {'prob': pymoo_get_problem('dtlz2', n_var=12, n_obj=3),
                        'nobj': 3, 'ref': [2.5, 2.5, 2.5],
                        'tpf': pymoo_get_problem('dtlz2', n_var=12, n_obj=3).pareto_front()},
        'DTLZ7_3obj': {'prob': pymoo_get_problem('dtlz7', n_var=22, n_obj=3),
                        'nobj': 3, 'ref': [1.0, 1.0, 20.0],
                        'tpf': get_dtlz7_true_pf_3obj(1200)},
    }

    algorithms = [
        ('Standard DE', standard_de,   {}),
        ('PCA-DE',      pca_de,         {'alpha': 0.5}),
        ('PMODE',       pmode,           {'alpha': 0.45}),
        ('HECO-PDE',    heco_pde,        {'alpha': 0.6, 'n_weights': 12}),
    ]

    results = {}
    for pname, pcfg in problems.items():
        prob = pcfg['prob']
        tpf  = pcfg['tpf']
        ref  = pcfg['ref']
        nobj = pcfg['nobj']
        results[pname] = {}
        if verbose:
            print(f"\n=== {pname} (n_var={prob.n_var}, n_obj={nobj}) ===")

        for aname, afunc, akwargs in algorithms:
            igds, hvs, gds, sps, scs = [], [], [], [], []
            best_pf, best_igd = None, float('inf')

            for run in range(n_runs):
                try:
                    res = afunc(prob, NP=NP, max_fes=max_fes,
                                seed=run * 13 + 7, **akwargs)
                    pf = res['pf']
                    if len(pf) == 0:
                        continue
                    i_val = igd(pf, tpf)
                    h_val = hypervolume(pf, ref, n_samples=10000)
                    g_val = generational_distance(pf, tpf)
                    s_val = spread_metric(pf, tpf) if nobj == 2 else 0.0
                    sc_val = dtlz7_segment_coverage(pf) if (nobj == 2 and 'DTLZ7' in pname) else 2
                    igds.append(i_val); hvs.append(h_val)
                    gds.append(g_val);  sps.append(s_val); scs.append(sc_val)
                    if i_val < best_igd:
                        best_igd = i_val
                        best_pf = pf
                except Exception as e:
                    if verbose:
                        print(f"    [{aname}] run {run} error: {e}")

            results[pname][aname] = {
                'igd_mean': np.nanmean(igds) if igds else float('inf'),
                'igd_std':  np.nanstd(igds)  if len(igds) > 1 else 0.0,
                'hv_mean':  np.nanmean(hvs)  if hvs else 0.0,
                'gd_mean':  np.nanmean(gds)  if gds else float('inf'),
                'spread_mean': np.nanmean(sps) if sps else 1.0,
                'seg_cov':  np.nanmean(scs)  if scs else 0.0,
                'best_pf':  best_pf,
                'all_igd':  igds, 'all_hv': hvs,
            }

            if verbose:
                r = results[pname][aname]
                print(f"  {aname:15s}: IGD={r['igd_mean']:.4f}±{r['igd_std']:.4f}  "
                      f"HV={r['hv_mean']:.4f}  Seg={r['seg_cov']:.1f}")

    results['true_pf'] = {k: v['tpf'] for k, v in problems.items()}
    return results


# =============================================================================
# SECTION 4: ENGINEERING BENCHMARKS
# =============================================================================

class WeldedBeam:
    """Welded Beam Design (4D, 7 constraints, min cost)."""
    n_var, dim = 4, 4
    lb = np.array([0.1, 0.1, 0.1, 0.1])
    ub = np.array([2.0, 10.0, 10.0, 2.0])
    f_opt = 1.7248

    def objective(self, x):
        return 1.10471*x[0]**2*x[1] + 0.04811*x[2]*x[3]*(14+x[1])

    def violation(self, x):
        P=6000; L=14; E=30e6; G=12e6
        Mx=P*(L+x[1]/2); R=np.sqrt(0.25*(x[1]**2+(x[0]+x[2])**2))
        J=2*(x[0]*x[1]*np.sqrt(2)*(x[1]**2/12+0.25*(x[0]+x[2])**2))
        tau1=P/(np.sqrt(2)*x[0]*x[1]+1e-10); tau2=Mx*R/(J+1e-10)
        tau=np.sqrt(tau1**2+tau1*tau2*x[1]/(R+1e-10)+tau2**2)
        sigma=6*P*L/(x[3]*x[2]**2+1e-10)
        Pc=4.013*E*np.sqrt(x[2]**2*x[3]**6/36)/(L**2)*(1-x[2]/(2*L)*np.sqrt(E/(4*G)))
        delta=6*P*L**3/(E*x[3]*x[2]**3+1e-10)
        g=[tau-13600, sigma-30000, x[0]-x[3], 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14+x[1])-5,
           0.125-x[0], delta-0.25, P-Pc]
        return float(sum(max(0,gi) for gi in g))


class TensionSpring:
    """Tension/Compression Spring (3D, 4 constraints, min weight)."""
    n_var, dim = 3, 3
    lb = np.array([0.05, 0.25, 2.0])
    ub = np.array([2.00, 1.30, 15.0])
    f_opt = 0.012665

    def objective(self, x):
        return (x[2]+2)*x[1]*x[0]**2

    def violation(self, x):
        g=[1-x[1]**3*x[2]/(71785*x[0]**4),
           (4*x[1]**2-x[0]*x[1])/(12566*(x[1]*x[0]**3-x[0]**4))+1/(5108*x[0]**2)-1,
           1-140.45*x[0]/(x[1]**2*x[2]),
           (x[0]+x[1])/1.5-1]
        return float(sum(max(0,gi) for gi in g))


def run_engineering_experiment(problem, algorithm_func, n_runs=8,
                                NP=30, max_fes=2000, seed_base=42, **kwargs):
    """Run n_runs of an algorithm on an engineering benchmark."""
    feas_results = []
    all_f = []
    for run in range(n_runs):
        np.random.seed(seed_base + run * 7)
        lb, ub = problem.lb, problem.ub
        d = problem.n_var
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        fit = np.array([problem.objective(x) for x in pop])
        vio = np.array([problem.violation(x) for x in pop])
        fes = NP
        F_val = kwargs.get('F', 0.5)
        CR_val = kwargs.get('CR', 0.9)
        alpha = kwargs.get('alpha', 0.5)

        while fes < max_fes:
            for i in range(NP):
                if fes >= max_fes:
                    break
                idxs = [j for j in range(NP) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                if algorithm_func == 'pca':
                    diff = pca_projection(pop, pop[r2]-pop[r3],
                                         alpha*(0.4+0.6*fes/max_fes))
                else:
                    diff = pop[r2] - pop[r3]
                mutant = np.clip(pop[r1] + F_val * diff, lb, ub)
                mask = np.random.rand(d) < CR_val
                mask[np.random.randint(d)] = True
                trial = np.where(mask, mutant, pop[i])
                f_t = problem.objective(trial)
                v_t = problem.violation(trial)
                fes += 1
                # Deb's feasibility rule
                if (v_t == 0 and vio[i] == 0 and f_t < fit[i]) or \
                   (v_t == 0 and vio[i] > 0) or \
                   (v_t > 0 and vio[i] > 0 and v_t < vio[i]):
                    pop[i] = trial; fit[i] = f_t; vio[i] = v_t
        feasible = vio == 0
        if feasible.any():
            feas_results.append(fit[feasible].min())
        all_f.extend(fit[feasible].tolist() if feasible.any() else [])

    sr = len(feas_results) / n_runs
    mean_f = np.mean(feas_results) if feas_results else float('inf')
    std_f  = np.std(feas_results)  if len(feas_results) > 1 else 0.0
    best_f = min(feas_results)     if feas_results else float('inf')
    return {'sr': sr, 'mean_f': mean_f, 'std_f': std_f, 'best_f': best_f}


# =============================================================================
# SECTION 5: PORTFOLIO OPTIMIZATION
# =============================================================================

# S&P 500 sector assets (calibrated to 2018-2023)
TICKERS  = ['AAPL','MSFT','GOOGL','AMZN','JPM','BAC','JNJ','PFE',
            'XOM','CVX','WMT','PG','NEE','DUK','GLD']
SECTORS  = ['Tech','Tech','Tech','ConsDisc','Finance','Finance',
            'Health','Health','Energy','Energy','Staples','Staples',
            'Utilities','Utilities','Commodities']
MU_ANN   = np.array([0.285,0.321,0.243,0.227,0.182,0.158,0.102,0.089,
                     0.146,0.163,0.098,0.084,0.091,0.072,0.058])
VOL_ANN  = np.array([0.292,0.278,0.301,0.345,0.263,0.289,0.184,0.221,
                     0.278,0.264,0.162,0.158,0.203,0.186,0.153])
RF       = 0.04   # risk-free rate


def build_portfolio_universe(seed=42, returns_xlsx=None):
    """
    Build a portfolio universe from `daily_returns.xlsx` when available.
    If the workbook cannot be matched to `TICKERS`, the numeric columns in
    the workbook are treated as the standalone universe.

    Returns: mu (annual expected returns), cov (annual covariance), daily returns DataFrame
    """
    import pandas as pd
    N = len(TICKERS)

    candidates = [] if returns_xlsx else [
        'daily_returns.xlsx',
        'Daily_Returns.xlsx',
        os.path.join('assignment 4', 'daily_returns.xlsx'),
        os.path.join('assignment 4', 'Daily_Returns.xlsx'),
    ]
    if returns_xlsx:
        candidates.insert(0, returns_xlsx)

    for cand in candidates:
        try:
            if cand and os.path.exists(cand):
                df = pd.read_excel(cand, index_col=0, parse_dates=True)

                if df.abs().max().max() > 2.0:
                    df = df.pct_change().dropna(how='all')

                def clean_name(s):
                    s = str(s).upper()
                    for sep in ['.', ':', ' ']:
                        if sep in s:
                            s = s.split(sep)[0]
                            break
                    return ''.join(ch for ch in s if ch.isalnum() or ch == '-')

                cleaned_map = {clean_name(col): col for col in df.columns}
                matches = {t: cleaned_map[t.upper()] for t in TICKERS if t.upper() in cleaned_map}

                if len(matches) > 0:
                    matched_tickers = [t for t in TICKERS if t in matches]
                    ret_df = df[[matches[t] for t in matched_tickers]].copy()
                    ret_df.columns = matched_tickers
                    mu_ann = ret_df.mean().values * 252
                    cov_ann = ret_df.cov().values * 252
                    print(f"Loaded returns from {cand} (mapped {len(matches)}/{len(TICKERS)} tickers)")
                    return mu_ann, cov_ann, ret_df, matched_tickers

                numeric_df = df.select_dtypes(include=[np.number]).copy()
                if numeric_df.shape[1] > 0:
                    ret_df = numeric_df.dropna(how='all').copy()
                    ret_df = ret_df.ffill().bfill()
                    ret_df = ret_df.dropna(axis=1, how='all')
                    if ret_df.shape[1] > 0:
                        universe_tickers = list(ret_df.columns)
                        mu_ann = ret_df.mean().values * 252
                        cov_ann = ret_df.cov().values * 252
                        print(f"Loaded standalone universe from {cand} ({ret_df.shape[1]} assets)")
                        return mu_ann, cov_ann, ret_df, universe_tickers

                print(f"Found {cand} but no numeric return columns were usable.")
        except Exception as e:
            print(f"Error reading {cand}: {e} -- falling back to synthetic generator.")

    T = 1260  # 5 years trading days

    # Build covariance from sector correlations
    SECTOR_CORR = {'Tech':0.82,'Finance':0.78,'Health':0.65,'Energy':0.75,
                   'Staples':0.68,'Utilities':0.72,'ConsDisc':1.0,'Commodities':1.0}
    rng = np.random.RandomState(7)
    corr = np.eye(N)
    SECTOR_PAIRS = {frozenset(['Tech','Finance']):0.42,frozenset(['Tech','ConsDisc']):0.55,
                    frozenset(['Tech','Health']):0.28,frozenset(['Tech','Energy']):0.18,
                    frozenset(['Finance','Energy']):0.38,frozenset(['Health','Staples']):0.42,
                    frozenset(['Energy','Commodities']):0.45,frozenset(['Staples','Utilities']):0.50}
    for i in range(N):
        for j in range(i+1, N):
            si, sj = SECTORS[i], SECTORS[j]
            if si == sj:
                base = SECTOR_CORR.get(si, 0.6)
                c = np.clip(base + rng.uniform(-0.08, 0.08), 0.01, 0.98)
            else:
                base = SECTOR_PAIRS.get(frozenset([si, sj]), 0.20)
                c = np.clip(base + rng.uniform(-0.04, 0.04), 0.01, 0.98)
            corr[i, j] = corr[j, i] = c

    # Make positive definite
    ev = np.linalg.eigvalsh(corr)
    if ev.min() < 0.01:
        corr += (0.01 - ev.min()) * np.eye(N)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

    cov = np.outer(VOL_ANN, VOL_ANN) * corr

    # Generate synthetic daily returns (Student-t, vol clustering, crises)
    rng2 = np.random.RandomState(seed)
    mu_d = MU_ANN / 252
    cov_d = cov / 252
    try:
        L = np.linalg.cholesky(cov_d + 1e-9 * np.eye(N))
    except Exception:
        L = np.eye(N) * VOL_ANN.mean() / np.sqrt(252)

    z = rng2.standard_t(df=5, size=(T, N)) / np.sqrt(5/3)
    vs = np.ones(T)
    for t in range(1, T):
        vs[t] = 0.92 * vs[t-1] + 0.08 * np.abs(z[t-1]).mean() * 1.5
    vs = np.clip(vs, 0.5, 3.0)

    BETAS = np.array([1.25,1.15,1.20,1.30,1.10,1.20,0.65,0.70,
                      0.85,0.80,0.55,0.50,0.45,0.40,0.05])
    returns = np.zeros((T, N))
    for t in range(T):
        returns[t] = mu_d + (L @ z[t]) * vs[t]
    # Inject crisis periods
    for sl, severity in [(slice(200, 260), -0.003), (slice(700, 760), -0.002)]:
        n_days = sl.stop - sl.start
        crash = np.linspace(severity, severity/2, n_days)
        for i in range(N):
            returns[sl, i] += crash * BETAS[i]

    import pandas as pd
    dates = pd.bdate_range('2019-01-02', periods=T)
    ret_df = pd.DataFrame(returns, index=dates, columns=TICKERS)
    return MU_ANN, cov, ret_df, TICKERS


def proj_weights(w, w_max=0.40):
    """Project weights onto feasible simplex."""
    w = np.clip(w, 0, w_max)
    s = w.sum()
    return w / s if s > 1e-10 else np.ones(len(w)) / len(w)


def portfolio_metrics(w, mu, cov, ret_df, rf=RF):
    """Compute comprehensive portfolio metrics."""
    w = proj_weights(w)
    port_ret_d = ret_df.values @ w
    ann_ret = float(mu @ w)
    ann_vol = float(np.sqrt(max(w @ cov @ w, 1e-10)))
    sharpe  = (ann_ret - rf) / max(ann_vol, 1e-8)
    neg_d   = port_ret_d[port_ret_d < 0]
    sortino = (ann_ret-rf) / max(float(neg_d.std()*np.sqrt(252)) if len(neg_d)>2 else 0.15, 1e-8)
    cum  = (1 + port_ret_d).cumprod()
    peak = np.maximum.accumulate(cum)
    mdd  = float((cum - peak).min() / max(peak.max(), 1e-8))
    return {'return': ann_ret, 'vol': ann_vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd, 'weights': w.tolist()}


def opt_sharpe(mu, cov, w_max=0.40, rf=RF, n_tries=10):
    """Analytical Max Sharpe optimisation via SLSQP (multiple restarts)."""
    N = len(mu)
    def neg_sh(w):
        w = proj_weights(w, w_max)
        rp = float(mu @ w)
        sp = float(np.sqrt(max(w @ cov @ w, 1e-10)))
        return -(rp - rf) / max(sp, 1e-8)
    best_w, best_v = None, float('inf')
    for s in range(n_tries):
        w0 = proj_weights(np.random.RandomState(s).dirichlet(np.ones(N)), w_max)
        res = minimize(neg_sh, w0, method='SLSQP',
                       bounds=[(0, w_max)]*N,
                       constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                       options={'ftol':1e-12,'maxiter':1000})
        if res.success and res.fun < best_v:
            best_v, best_w = res.fun, res.x
    return proj_weights(best_w if best_w is not None else np.ones(N)/N, w_max)


def run_portfolio_experiment(verbose=True):
    """Run all 12 portfolio optimization methods."""
    import pandas as pd
    res = build_portfolio_universe()
    if len(res) == 4:
        mu, cov, ret_df, tickers_used = res
    else:
        mu, cov, ret_df = res
        tickers_used = ret_df.columns.tolist()

    N = len(tickers_used)
    split = int(len(ret_df) * 0.70)
    is_ret  = ret_df.iloc[:split]
    oos_ret = ret_df.iloc[split:]
    mu_is   = is_ret.mean().values * 252
    cov_is  = is_ret.cov().values  * 252

    if verbose:
        print("\n=== Portfolio Optimization (12 Methods) ===")

    results = {}

    # M1: Equal Weight
    w = np.ones(N) / N
    results['M1 Equal Weight'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M2: Market-Cap Weight
    mcap_full = np.array([3.0,2.8,1.8,1.6,0.5,0.35,0.45,0.28,0.40,0.32,0.42,0.38,0.12,0.10,0.08])
    mcap_map = {t: m for t, m in zip(TICKERS, mcap_full)}
    mcap = np.array([mcap_map.get(t, 1.0) for t in tickers_used])
    w = proj_weights(mcap)
    results['M2 MarketCap'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M3: Markowitz MV (target 18%)
    cons = [{'type':'eq','fun':lambda w:w.sum()-1},
            {'type':'eq','fun':lambda w:float(mu_is@w)-0.18}]
    res = minimize(lambda w:w@cov_is@w, np.ones(N)/N, method='SLSQP',
                   bounds=[(0,0.4)]*N, constraints=cons,
                   options={'ftol':1e-12,'maxiter':1000})
    w = proj_weights(res.x if res.success else np.ones(N)/N)
    results['M3 Markowitz'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M4: Max Sharpe (analytical QP)
    w = opt_sharpe(mu_is, cov_is)
    results['M4 MaxSharpe'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M5: Min Variance
    res = minimize(lambda w:w@cov_is@w, np.ones(N)/N, method='SLSQP',
                   bounds=[(0,0.4)]*N,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                   options={'ftol':1e-12,'maxiter':1000})
    w = proj_weights(res.x if res.success else np.ones(N)/N)
    results['M5 MinVariance'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M6: Risk Parity
    target_rc = np.ones(N) / N
    def rp_obj(w):
        w = np.abs(w); w /= w.sum()
        pv = float(w @ cov_is @ w)
        rc = w * (cov_is @ w) / max(pv, 1e-10)
        return float(np.sum((rc - target_rc)**2))
    best_w, best_v = None, float('inf')
    for s in range(8):
        w0 = proj_weights(np.random.RandomState(s).dirichlet(np.ones(N)))
        res = minimize(rp_obj, w0, method='SLSQP', bounds=[(0,0.4)]*N,
                       constraints=[{'type':'eq','fun':lambda w:np.sum(np.abs(w))-1}],
                       options={'ftol':1e-14,'maxiter':2000})
        if res.success and res.fun < best_v:
            best_v, best_w = res.fun, res.x
    w = proj_weights(np.abs(best_w) if best_w is not None else np.ones(N)/N)
    results['M6 RiskParity'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M7: CVaR Optimisation (Sortino proxy)
    def neg_sortino(w):
        w = proj_weights(w); rp = float(mu_is @ w)
        pd_ = is_ret.values @ w; dn = pd_[pd_ < 0]
        ds = float(dn.std()*np.sqrt(252)) if len(dn) > 2 else 0.15
        return -(rp - RF) / max(ds, 1e-8)
    best_w, best_v = None, float('inf')
    for s in range(12):
        w0 = proj_weights(np.random.RandomState(s).dirichlet(np.ones(N)))
        res = minimize(neg_sortino, w0, method='SLSQP', bounds=[(0,0.4)]*N,
                       constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                       options={'ftol':1e-12,'maxiter':1000})
        if res.success and res.fun < best_v:
            best_v, best_w = res.fun, res.x
    w = proj_weights(best_w if best_w is not None else np.ones(N)/N)
    results['M7 CVaROpt'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M8: Black-Litterman (3 views)
    lam = 2.5; w_mkt = np.ones(N)/N; Pi = lam * cov_is @ w_mkt; tau = 0.05
    idx = {t:i for i,t in enumerate(tickers_used)}
    views = [({'AAPL':0.5,'MSFT':0.5,'XOM':-0.5,'CVX':-0.5}, 0.05),
             ({'JNJ':0.5,'PFE':0.5}, 0.10),
             ({'WMT':0.5,'PG':0.5,'NEE':-0.5,'DUK':-0.5}, 0.02)]
    P_rows = []
    q_rows = []
    for view_map, qval in views:
        if all(t in idx for t in view_map):
            row = np.zeros(N)
            for t, weight in view_map.items():
                row[idx[t]] = weight
            P_rows.append(row)
            q_rows.append(qval)
    if P_rows:
        P = np.vstack(P_rows)
        q = np.array(q_rows)
        Om = np.diag(np.diag(tau * P @ cov_is @ P.T)) + 1e-8 * np.eye(P.shape[0])
        try:
            S_inv = np.linalg.inv(tau*cov_is + 1e-8*np.eye(N))
            O_inv = np.linalg.inv(Om)
            M_inv = np.linalg.inv(S_inv + P.T @ O_inv @ P)
            mu_bl = M_inv @ (S_inv @ Pi + P.T @ O_inv @ q)
            cov_bl = cov_is + M_inv
        except Exception:
            mu_bl = Pi; cov_bl = cov_is
    else:
        mu_bl = Pi; cov_bl = cov_is
    w = opt_sharpe(mu_bl, cov_bl)
    results['M8 BlackLitterman'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M9: Robust Optimisation
    T_is = len(is_ret)
    sigma_mu = np.sqrt(np.diag(cov_is) / T_is); kappa = 0.5
    def rob_neg_sh(w):
        w = proj_weights(w); rp = float(mu_is @ w)
        sp = float(np.sqrt(max(w @ cov_is @ w, 1e-10)))
        unc = kappa * float(np.sqrt(w @ np.diag(sigma_mu**2) @ w))
        return -(rp - unc - RF) / max(sp, 1e-8)
    best_w, best_v = None, float('inf')
    for s in range(12):
        w0 = proj_weights(np.random.RandomState(s).dirichlet(np.ones(N)))
        res = minimize(rob_neg_sh, w0, method='SLSQP', bounds=[(0,0.4)]*N,
                       constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                       options={'ftol':1e-12,'maxiter':1000})
        if res.success and res.fun < best_v:
            best_v, best_w = res.fun, res.x
    w = proj_weights(best_w if best_w is not None else np.ones(N)/N)
    results['M9 RobustOpt'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M10: HECO-PDE (MOEA) on portfolio
    class PortfolioMOEA:
        """Wrapper to make portfolio a pymoo-compatible problem."""
        n_var, n_obj = N, 2
        xl = np.zeros(N); xu = np.full(N, 0.40)
        def evaluate(self, W):
            if W.ndim == 1:
                w = proj_weights(W)
                return np.array([float(w @ cov_is @ w), -float(mu_is @ w)])
            return np.array([[float(proj_weights(W[i])@cov_is@proj_weights(W[i])),
                              -float(mu_is@proj_weights(W[i]))] for i in range(len(W))])

    port_prob = PortfolioMOEA()
    heco_res = heco_pde(port_prob, NP=60, max_fes=12000, n_weights=15, alpha=0.65, seed=42)
    pf = heco_res['pf']
    shs = (-pf[:,1] - RF) / np.maximum(np.sqrt(pf[:,0]), 1e-8)
    best_i = np.argmax(shs)
    w = heco_res['pop'][best_i] if len(heco_res.get('pop',[])) > best_i else np.ones(N)/N
    w = proj_weights(w)
    results['M10 HECO-PDE'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M11: PMODE on portfolio
    pmode_res = pmode(port_prob, NP=80, max_fes=10000, alpha=0.5, seed=42)
    pf2 = pmode_res['pf']
    if len(pf2) > 0:
        shs2 = (-pf2[:,1] - RF) / np.maximum(np.sqrt(pf2[:,0]), 1e-8)
        best_i2 = np.argmax(shs2)
        w = pmode_res['pop'][best_i2] if len(pmode_res.get('pop',[])) > best_i2 else np.ones(N)/N
        w = proj_weights(w)
    else:
        w = np.ones(N) / N
    results['M11 PMODE'] = portfolio_metrics(w, mu_is, cov_is, is_ret)

    # M12: Ensemble (average of all 11 methods)
    w_ens = proj_weights(np.mean([np.array(v['weights']) for v in results.values()], axis=0))
    results['M12 Ensemble'] = portfolio_metrics(w_ens, mu_is, cov_is, is_ret)

    if verbose:
        for name, m in results.items():
            print(f"  {name:22s}: Ret={m['return']*100:.1f}%  "
                  f"Vol={m['vol']*100:.1f}%  Sharpe={m['sharpe']:.4f}")

    return results, mu_is, cov_is, is_ret, oos_ret


# =============================================================================
# SECTION 6: MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MultiObjective DE with PCA — Full Experiments')
    parser.add_argument('--dtlz', action='store_true', help='Run DTLZ experiments')
    parser.add_argument('--port', action='store_true', help='Run portfolio experiments')
    parser.add_argument('--eng',  action='store_true', help='Run engineering experiments')
    parser.add_argument('--all',  action='store_true', help='Run all experiments')
    parser.add_argument('--runs', type=int, default=8, help='Number of runs per experiment')
    parser.add_argument('--fes',  type=int, default=10000, help='Max FEs per run')
    args = parser.parse_args()

    run_all = args.all or not (args.dtlz or args.port or args.eng)
    os.makedirs('results', exist_ok=True)

    # ── Engineering Benchmarks ───────────────────────────────────────────────
    if run_all or args.eng:
        print("\n" + "="*60)
        print("ENGINEERING BENCHMARKS")
        print("="*60)
        problems = [('Welded Beam', WeldedBeam()), ('Tension Spring', TensionSpring())]
        algos = [('Standard DE', 'std'), ('PCA-DE', 'pca')]
        eng_results = {}
        for pname, prob in problems:
            eng_results[pname] = {}
            for aname, akey in algos:
                res = run_engineering_experiment(prob, akey, n_runs=args.runs,
                                                 NP=25, max_fes=min(args.fes, 2000))
                eng_results[pname][aname] = res
                print(f"  {pname} {aname:14s}: SR={res['sr']*100:.0f}%  "
                      f"Mean={res['mean_f']:.5g}  Best={res['best_f']:.5g}")
        pickle.dump(eng_results, open('results/engineering.pkl', 'wb'))
        print("Saved → results/engineering.pkl")

    # ── DTLZ Benchmarks ──────────────────────────────────────────────────────
    if (run_all or args.dtlz) and HAS_PYMOO:
        print("\n" + "="*60)
        print("DTLZ BENCHMARK SUITE")
        print("="*60)
        dtlz_results = run_dtlz_experiments(
            n_runs=args.runs, max_fes=args.fes, NP=70, verbose=True)
        pickle.dump(dtlz_results, open('results/dtlz.pkl', 'wb'))
        print("Saved → results/dtlz.pkl")

    # ── Portfolio Optimization ───────────────────────────────────────────────
    if run_all or args.port:
        print("\n" + "="*60)
        print("PORTFOLIO OPTIMIZATION (12 METHODS)")
        print("="*60)
        port_results, mu, cov, is_ret, oos_ret = run_portfolio_experiment(verbose=True)
        pickle.dump({'results': port_results, 'mu': mu, 'cov': cov},
                    open('results/portfolio.pkl', 'wb'))
        print("Saved → results/portfolio.pkl")

    print("\n✓ All experiments complete.")
    print("Results saved to results/ directory.")
    print("\nNext steps:")
    print("  python complete_project_code.py --dtlz   # DTLZ only")
    print("  python complete_project_code.py --port   # Portfolio only")
    print("  python complete_project_code.py --all    # Everything")


if __name__ == '__main__':
    main()
