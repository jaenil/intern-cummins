import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import mean_absolute_error, accuracy_score

from scipy.spatial.distance import pdist, cdist
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from pycaret.regression import load_model as load_reg_model, predict_model as predict_reg_model
from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf_model

# ── Drift Injection Functions ──────────────────────────────────────────────────

def apply_mean_shift(df, col, s):
    delta = 0.5 * df[col].std()
    df[col] = df[col] + (s * delta)
    return df

def apply_median_shift(df, col, s):
    delta = 0.3 * df[col].std()
    m = df[col].median()
    m_new = m + (s * delta)
    orig_mean = df[col].mean()
    
    mask_lower = df[col] <= m
    mask_upper = df[col] > m
    
    df.loc[mask_lower, col] = df.loc[mask_lower, col] * (m_new / m) if m != 0 else df.loc[mask_lower, col] + (m_new - m)
    
    if orig_mean != m:
        df.loc[mask_upper, col] = m_new + (df.loc[mask_upper, col] - m) * ((orig_mean - m_new) / (orig_mean - m))
    else:
        df.loc[mask_upper, col] = m_new + (df.loc[mask_upper, col] - m)
        
    return df

def apply_variance_flatten(df, col, s):
    mean_val = df[col].mean()
    df[col] = mean_val + (df[col] - mean_val) * (1 + 0.15 * s)
    return df

def apply_variance_sharpen(df, col, s):
    mean_val = df[col].mean()
    scale = max(1 - 0.08 * s, 0.05)
    df[col] = mean_val + (df[col] - mean_val) * scale
    return df

def apply_skewness_drift(df, col, s):
    min_val = df[col].min()
    x_shifted = df[col] - min_val + 1
    exponent = 1 / (1 + 0.2 * s)
    x_prime = x_shifted ** exponent
    
    orig_mean = df[col].mean()
    orig_std = df[col].std()
    x_prime_std = x_prime.std()
    if x_prime_std > 0:
        df[col] = ((x_prime - x_prime.mean()) / x_prime_std) * orig_std + orig_mean
    else:
        df[col] = x_prime
    return df

def apply_kurtosis_drift(df, col, s):
    nu = max(30 - 3 * s, 2)
    np.random.seed(42)
    noise = scipy.stats.t.rvs(df=nu, size=len(df), random_state=42)
    df[col] = df[col] + noise * (df[col].std() * 0.1 * s)
    return df

def apply_range_drift(df, col, s):
    p95 = df[col].quantile(0.95)
    ext_factor = 1 + (0.08 * s)
    mask = df[col] > p95
    df.loc[mask, col] = p95 + (df.loc[mask, col] - p95) * ext_factor
    return df

def apply_quantile_drift(df, col, s):
    q75 = df[col].quantile(0.75)
    max_val = df[col].max()
    q_shift = 0.25 * s * df[col].std()
    mask = df[col] > q75
    
    if max_val > q75:
        df.loc[mask, col] = df.loc[mask, col] + q_shift * ((df.loc[mask, col] - q75) / (max_val - q75))
    else:
        df.loc[mask, col] = df.loc[mask, col] + q_shift
    return df


# ── Multivariate Drift Detectors ────────────────────────────────────────────────
def energy_distance(X, Y):
    """
    Compute energy distance between two samples.
    E(X,Y) = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||
    """
    exy = cdist(X, Y, metric='euclidean').mean()
    exx = cdist(X, X, metric='euclidean').mean()
    eyy = cdist(Y, Y, metric='euclidean').mean()
    return max(2 * exy - exx - eyy, 0.0)


def detect_drift(X_ref, X_new, alpha=0.05, n_permutations=200,
                 max_samples=2000, random_state=42, n_jobs=-1):
    """
    Detect multivariate drift between X_ref and X_new using Energy Distance.

    Parameters
    ----------
    X_ref : array-like (n_ref, n_features)
        Reference data.
    X_new : array-like (n_new, n_features)
        New data to test for drift.
    alpha : float
        Significance level.
    n_permutations : int
        Number of permutations for p-value estimation.
    max_samples : int
        Cap sample size per group (for speed).
    random_state : int
        Reproducibility seed.
    n_jobs : int
        Parallel jobs for permutation test.

    Returns
    -------
    dict with keys:
        energy_distance, p_value, drift_detected, threshold, n_ref, n_new
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    X_new = np.asarray(X_new, dtype=np.float64)

    rng = np.random.default_rng(random_state)

    # ---- Subsample for speed ----
    if len(X_ref) > max_samples:
        X_ref = X_ref[rng.choice(len(X_ref), max_samples, replace=False)]
    if len(X_new) > max_samples:
        X_new = X_new[rng.choice(len(X_new), max_samples, replace=False)]

    # ---- Observed statistic ----
    observed = energy_distance(X_ref, X_new)

    # ---- Permutation test ----
    combined = np.vstack([X_ref, X_new])
    n = len(X_ref)
    seeds = rng.integers(0, 1_000_000, size=n_permutations)

    def one_perm(seed):
        local_rng = np.random.default_rng(seed)
        perm = local_rng.permutation(len(combined))
        return energy_distance(combined[perm[:n]], combined[perm[n:]])

    perm_stats = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one_perm)(s) for s in seeds
    )
    perm_stats = np.array(perm_stats)

    p_value = (np.sum(perm_stats >= observed) + 1) / (n_permutations + 1)

    return {
        "energy_distance": float(observed),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
        "threshold": float(np.quantile(perm_stats, 1 - alpha)),
        "n_ref": len(X_ref),
        "n_new": len(X_new),
    }

class MMDDriftDetector:
    """
    Fully Automated MMD Drift Detector.
    
    Simply call:
        detector = MMDDriftDetector()
        result = detector.run(X_ref, X_new)
    
    Everything is handled internally:
      - Bandwidth selection (median heuristic)
      - Kernel matrix computation (cached)
      - Permutation testing (parallel)
      - Auto distance threshold from null distribution
      - Combined decision (p-value + distance + z-score)
      - Severity assessment
    """
    
    def __init__(
        self,
        kernel='rbf',
        bandwidth='median',
        n_permutations=1000,
        p_threshold=0.05,
        distance_threshold='auto',
        z_threshold=3.0,
        combine_rule='all',           # 'all' | 'majority' | 'any'
        subsample_size=2000,          # auto-subsamples large data
        n_jobs=-1,
        random_state=42,
        verbose=True,
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_permutations = n_permutations
        self.p_threshold = p_threshold
        self.distance_threshold = distance_threshold
        self.z_threshold = z_threshold
        self.combine_rule = combine_rule
        self.subsample_size = subsample_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
    
    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _prepare(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
    
    @staticmethod
    def _median_heuristic(X, Y):
        Z = np.vstack([X, Y])
        if Z.shape[0] > 1000:
            idx = np.random.choice(Z.shape[0], 1000, replace=False)
            Z = Z[idx]
        dists = pdist(Z, metric='sqeuclidean')
        med = np.median(dists)
        return np.sqrt(med / 2.0) if med > 0 else 1.0
    
    def _kernel_matrix(self, Z, sigma):
        gamma = 1.0 / (2.0 * sigma ** 2)
        return rbf_kernel(Z, Z, gamma=gamma)
    
    @staticmethod
    def _mmd2_unbiased(K, m):
        n = K.shape[0] - m
        K_xx, K_yy, K_xy = K[:m, :m], K[m:, m:], K[:m, m:]
        sum_xx = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
        sum_yy = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
        sum_xy = K_xy.sum() / (m * n)
        return sum_xx + sum_yy - 2 * sum_xy
    
    def _permutation_worker(self, K, m, n_perms, seed):
        rng = np.random.default_rng(seed)
        N = K.shape[0]
        stats = np.empty(n_perms)
        for i in range(n_perms):
            perm = rng.permutation(N)
            stats[i] = self._mmd2_unbiased(K[np.ix_(perm, perm)], m)
        return stats
    
    @staticmethod
    def _mean_distance(X, Y):
        mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
        pooled = np.sqrt((X.std(axis=0) ** 2 + Y.std(axis=0) ** 2) / 2) + 1e-8
        return float(np.linalg.norm((mu_x - mu_y) / pooled))
    
    @staticmethod
    def _covariance_distance(X, Y):
        cov_x, cov_y = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
        if cov_x.ndim == 0:  # 1-D case
            return float(abs(cov_x - cov_y) / (abs(cov_x) + 1e-8))
        return float(np.linalg.norm(cov_x - cov_y, 'fro') /
                     (np.linalg.norm(cov_x, 'fro') + 1e-8))
    
    # ------------------------------------------------------------
    # MAIN — ONE-CALL DETECTION
    # ------------------------------------------------------------
    def run(self, X_ref, X_new):
        """
        Perform full MMD drift detection pipeline.
        
        Parameters
        ----------
        X_ref : array-like  (n_ref, n_features)   Reference data
        X_new : array-like  (n_new, n_features)   New data to test
        
        Returns
        -------
        dict with full results and decision
        """
        # ---- 1. Prepare data ----
        X = self._prepare(X_ref)
        Y = self._prepare(X_new)
        
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_ref has {X.shape[1]} features, "
                f"X_new has {Y.shape[1]}"
            )
        
        rng = np.random.default_rng(self.random_state)
        
        # ---- 2. Auto-subsample if too large ----
        if self.subsample_size is not None:
            if X.shape[0] > self.subsample_size:
                X = X[rng.choice(X.shape[0], self.subsample_size, replace=False)]
            if Y.shape[0] > self.subsample_size:
                Y = Y[rng.choice(Y.shape[0], self.subsample_size, replace=False)]
        
        m = X.shape[0]
        
        # ---- 3. Bandwidth ----
        sigma = (self._median_heuristic(X, Y)
                 if self.bandwidth == 'median' else float(self.bandwidth))
        
        # ---- 4. Kernel matrix (cached) ----
        Z = np.vstack([X, Y])
        K = self._kernel_matrix(Z, sigma)
        
        # ---- 5. Observed MMD² ----
        mmd2_obs = self._mmd2_unbiased(K, m)
        
        # ---- 6. Permutation null distribution (parallel) ----
        if self.n_jobs == 1 or self.n_permutations < 50:
            perm_stats = self._permutation_worker(K, m, self.n_permutations,
                                                  self.random_state)
        else:
            n_workers = 8 if self.n_jobs == -1 else self.n_jobs
            per_w = self.n_permutations // n_workers
            rem = self.n_permutations % n_workers
            seeds = rng.integers(0, 2**31 - 1, size=n_workers)
            counts = [per_w + (1 if i < rem else 0) for i in range(n_workers)]
            results = Parallel(n_jobs=n_workers)(
                delayed(self._permutation_worker)(K, m, counts[i], seeds[i])
                for i in range(n_workers) if counts[i] > 0
            )
            perm_stats = np.concatenate(results)
        
        # ---- 7. Null distribution stats ----
        null_mean = float(np.mean(perm_stats))
        null_std = float(np.std(perm_stats)) + 1e-12
        
        # ---- 8. Compute all decision metrics ----
        p_value = (np.sum(perm_stats >= mmd2_obs) + 1) / (len(perm_stats) + 1)
        mmd_distance = float(np.sqrt(max(mmd2_obs, 0)))
        z_score = (mmd2_obs - null_mean) / null_std
        auto_dist_thr = float(np.sqrt(max(np.percentile(perm_stats, 95), 0)))
        dist_thr = (auto_dist_thr if self.distance_threshold == 'auto'
                    else float(self.distance_threshold))
        
        # ---- 9. Extra distance signals ----
        mean_dist = self._mean_distance(X, Y)
        cov_dist = self._covariance_distance(X, Y)
        
        # ---- 10. Combined decision ----
        c_p    = p_value < self.p_threshold
        c_dist = mmd_distance > dist_thr
        c_z    = z_score > self.z_threshold
        criteria = np.array([c_p, c_dist, c_z])
        
        if self.combine_rule == 'all':
            drift = bool(criteria.all())
        elif self.combine_rule == 'any':
            drift = bool(criteria.any())
        elif self.combine_rule == 'majority':
            drift = bool(criteria.sum() >= 2)
        else:
            raise ValueError(f"Unknown combine_rule: {self.combine_rule}")
        
        n_trig = int(criteria.sum())
        severity = ['no_drift', 'weak', 'moderate', 'strong'][n_trig]
        
        result = {
            'drift_detected': drift,
            'severity': severity,
            'n_criteria_triggered': n_trig,
            
            'mmd2': float(mmd2_obs),
            'mmd_distance': mmd_distance,
            'p_value': float(p_value),
            'z_score': float(z_score),
            
            'mean_distance': mean_dist,
            'covariance_distance': cov_dist,
            
            'thresholds': {
                'p_threshold': self.p_threshold,
                'distance_threshold': float(dist_thr),
                'z_threshold': self.z_threshold,
            },
            'criteria': {
                'pvalue_triggered':   bool(c_p),
                'distance_triggered': bool(c_dist),
                'zscore_triggered':   bool(c_z),
            },
            'null_distribution': {
                'mean': null_mean,
                'std': null_std,
                'auto_distance_threshold': auto_dist_thr,
            },
            'config': {
                'sigma': float(sigma),
                'n_permutations': len(perm_stats),
                'combine_rule': self.combine_rule,
                'n_ref_used': m,
                'n_new_used': Y.shape[0],
            },
        }
        
        if self.verbose:
            self._print_report(result)
        
        return result
    
    # Alias so both .run() and .detect() work
    def detect(self, X_ref, X_new):
        return self.run(X_ref, X_new)
    
    # ------------------------------------------------------------
    # Pretty report
    # ------------------------------------------------------------
    @staticmethod
    def _print_report(res):
        line = "=" * 60
        print(line)
        print(" MMD DRIFT DETECTION REPORT")
        print(line)
        print(f"  Drift detected     : {res['drift_detected']}")
        print(f"  Severity           : {res['severity'].upper()}")
        print(f"  Criteria triggered : {res['n_criteria_triggered']} / 3")
        print("-" * 60)
        print(f"  MMD² statistic     : {res['mmd2']:.6f}")
        print(f"  MMD distance       : {res['mmd_distance']:.6f}  "
              f"(thr: {res['thresholds']['distance_threshold']:.6f})  "
              f"{'✔' if res['criteria']['distance_triggered'] else '✘'}")
        print(f"  p-value            : {res['p_value']:.4f}       "
              f"(thr: {res['thresholds']['p_threshold']})       "
              f"{'✔' if res['criteria']['pvalue_triggered'] else '✘'}")
        print(f"  z-score            : {res['z_score']:.3f}       "
              f"(thr: {res['thresholds']['z_threshold']})       "
              f"{'✔' if res['criteria']['zscore_triggered'] else '✘'}")
        print("-" * 60)
        print(f"  Mean distance (std): {res['mean_distance']:.4f}")
        print(f"  Cov distance (rel) : {res['covariance_distance']:.4f}")
        print(f"  Bandwidth (sigma)  : {res['config']['sigma']:.4f}")
        print(f"  Samples used       : ref={res['config']['n_ref_used']}, "
              f"new={res['config']['n_new_used']}")
        print(f"  Permutations       : {res['config']['n_permutations']}")
        print(line)





# ── Configuration ──────────────────────────────────────────────────────────────

DRIFT_CONFIG = {
    'mean_shift':       {'col': 'coolant_temperature_mean', 'fn': apply_mean_shift, 'desc': 'Mean Shift'},
    'median_shift':     {'col': 'engine_load_mean',         'fn': apply_median_shift, 'desc': 'Median Shift'},
    'var_flatten':      {'col': 'exhaust_temperature_mean', 'fn': apply_variance_flatten, 'desc': 'Variance Flatten'},
    'var_sharpen':      {'col': 'fuel_rate_mean',           'fn': apply_variance_sharpen, 'desc': 'Variance Sharpen'},
    'skewness':         {'col': 'fuel_pressure_mean',       'fn': apply_skewness_drift, 'desc': 'Skewness'},
    'kurtosis':         {'col': 'boost_pressure_mean',      'fn': apply_kurtosis_drift, 'desc': 'Kurtosis'},
    'range_drift':      {'col': 'intake_pressure_mean',     'fn': apply_range_drift, 'desc': 'Range'},
    'quantile_drift':   {'col': 'rail_pressure_mean',       'fn': apply_quantile_drift, 'desc': 'Quantile'},
}

# ── Evaluator & Plotting ────────────────────────────────────────────────────────

def evaluate_single_drift(drift_fn, col, test_df, reg_model, clf_model, severities):
    mae_scores, acc_scores = [], []
    for s in severities:
        df = test_df.copy()
        if s > 0:
            df = drift_fn(df, col, s)
        
        preds_reg = predict_reg_model(reg_model, data=df)
        mae = mean_absolute_error(preds_reg['RUL'], preds_reg['prediction_label'])
        
        preds_clf = predict_clf_model(clf_model, data=df)
        acc = accuracy_score(preds_clf['RUL_class'], preds_clf['prediction_label'])
        
        mae_scores.append(mae)
        acc_scores.append(acc)
    return mae_scores, acc_scores


def evaluate_multivariate_drift(test_df, severities):
    print("Evaluating Multivariate Drift (Energy & MMD)...")
    detector = MMDDriftDetector(n_permutations=200, verbose=False)
    
    metadata_cols = ['Engine_number', 'trip_id', 'duration_sec', 'engine_runtime', 'Cumulative_odometer', 'vehicle_speed', 'RUL', 'RUL_class']
    sensor_cols = [c for c in test_df.columns if c not in metadata_cols and not c.startswith('fault_list')]
    
    X_ref = test_df[sensor_cols].values
    
    results = {'energy': [], 'mmd': []}
    
    for s in severities:
        print(f"  Severity: {s}")
        df_drifted = test_df.copy()
        if s > 0:
            for drift_name, cfg in DRIFT_CONFIG.items():
                df_drifted = cfg['fn'](df_drifted, cfg['col'], s)
                
        X_new = df_drifted[sensor_cols].values
        
        # Energy Distance
        energy_res = detect_drift(X_ref, X_new, max_samples=2000)
        results['energy'].append(energy_res['energy_distance'])
        
        # MMD
        mmd_res = detector.run(X_ref, X_new)
        results['mmd'].append(mmd_res['mmd_distance'])
        
    return results

def plot_multivariate_drifts(results, severities):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Severity Level', fontsize=12)
    ax1.set_ylabel('Energy Distance', color=color1, fontsize=12)
    ax1.plot(severities, results['energy'], marker='o', color=color1, linewidth=2, label='Energy Distance')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(severities)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('MMD Distance', color=color2, fontsize=12)  
    ax2.plot(severities, results['mmd'], marker='s', color=color2, linewidth=2, label='MMD Distance')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()  
    plt.title("Multivariate Drift Detection vs. Severity", fontsize=14, fontweight='bold')
    
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    fig.savefig("multivariate_drift_metrics.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_model_drifts_grid(all_results, severities, model_type):
    # model_type is 'mae' for Regressor, 'acc' for Classifier
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    is_reg = (model_type == 'mae')
    ylabel = "Mean Absolute Error (MAE)" if is_reg else "Accuracy"
    title_prefix = "Regressor" if is_reg else "Classifier"
    
    for i, ((drift_name, res), ax) in enumerate(zip(all_results.items(), axes)):
        desc = res['desc']
        c = colors[i % len(colors)]
        
        ax.plot(severities, res[model_type], marker='o', color=c, linewidth=2)
        ax.set_title(f"{title_prefix}: {desc}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Severity Level", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(severities)
        
    fig.tight_layout()
    filename = "regressor_drifts_grid.png" if is_reg else "classifier_drifts_grid.png"
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_summary_panel(all_results, severities):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors/markers for different drifts
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p']
    
    for i, (drift_name, res) in enumerate(all_results.items()):
        desc = res['desc']
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        
        # Regressor
        axes[0].plot(severities, res['mae'], marker=m, color=c, linewidth=2, label=desc)
        
        # Classifier
        axes[1].plot(severities, res['acc'], marker=m, color=c, linewidth=2, label=desc)
        
    axes[0].set_title("Regressor: Impact of All Drift Types", fontsize=14)
    axes[0].set_xlabel("Severity", fontsize=12)
    axes[0].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(severities)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Classifier: Impact of All Drift Types", fontsize=14)
    axes[1].set_xlabel("Severity", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(severities)
    axes[1].legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig("drift_impact_summary.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    severities = list(range(10))
    test_df = pd.read_csv('../data-generation/test_fleet_data.csv')
    
    print("Loading models...")
    reg_model = load_reg_model('../ml-models/best_rul_regressor_pipeline')
    clf_model = load_clf_model('../ml-models/best_rul_classifier_pipeline')

    all_results = {}

    for drift_name, cfg in DRIFT_CONFIG.items():
        print(f"Evaluating: {drift_name}")
        mae, acc = evaluate_single_drift(cfg['fn'], cfg['col'], test_df, reg_model, clf_model, severities)
        all_results[drift_name] = {'mae': mae, 'acc': acc, 'desc': cfg['desc']}

    
    # Multivariate Evaluation
    multi_severities = [0, 2, 4, 6, 9]
    multi_results = evaluate_multivariate_drift(test_df, multi_severities)
    print("Generating multivariate drift panel...")
    plot_multivariate_drifts(multi_results, multi_severities)

    print("Generating summary panel...")
    plot_summary_panel(all_results, severities)
    
    print("Generating individual grid panels...")
    plot_model_drifts_grid(all_results, severities, 'mae')
    plot_model_drifts_grid(all_results, severities, 'acc')
    print("Drift evaluation completed successfully!")

if __name__ == '__main__':
    main()
