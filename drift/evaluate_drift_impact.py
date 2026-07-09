import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import mean_absolute_error, accuracy_score
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

# ── Configuration ──────────────────────────────────────────────────────────────

DRIFT_CONFIG = {
    'mean_shift':       {'col': 'coolant_temperature_mean', 'fn': apply_mean_shift, 'desc': 'Mean Shift'},
    'median_shift':     {'col': 'engine_load_mean',         'fn': apply_median_shift, 'desc': 'Median Shift'},
    'var_flatten':      {'col': 'exhaust_temperature_mean', 'fn': apply_variance_flatten, 'desc': 'Variance Flatten'},
    'var_sharpen':      {'col': 'fuel_rate_mean',           'fn': apply_variance_sharpen, 'desc': 'Variance Sharpen'},
    'skewness':         {'col': 'fuel_pressure_mean',       'fn': apply_skewness_drift, 'desc': 'Skewness'},
    'kurtosis':         {'col': 'boost_pressure_mean',      'fn': apply_kurtosis_drift, 'desc': 'Kurtosis'},
    'range_drift':      {'col': 'engine_speed_mean',        'fn': apply_range_drift, 'desc': 'Range'},
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

    print("Generating summary panel...")
    plot_summary_panel(all_results, severities)
    
    print("Generating individual grid panels...")
    plot_model_drifts_grid(all_results, severities, 'mae')
    plot_model_drifts_grid(all_results, severities, 'acc')
    print("Drift evaluation completed successfully!")

if __name__ == '__main__':
    main()
