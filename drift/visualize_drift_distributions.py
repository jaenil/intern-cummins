import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from evaluate_drift_impact import DRIFT_CONFIG

# ── Configuration ────────────────────────────────────────────────────────────
SEVERITIES   = [0, 3, 5, 9]                # clean, low, moderate, severe
COLORS       = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']   # blue, green, orange, red
LABELS       = ['Severity 0 (Clean)', 'Severity 3 (Low)', 'Severity 5 (Moderate)', 'Severity 9 (Severe)']
ALPHA_HIST   = 0.30
ALPHA_KDE    = 0.90
OUTPUT_DIR   = '.'

# ── KDE helper ────────────────────────────────────────────────────────────────
def plot_kde_hist(ax, data, color, label, bins=60):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(data, bins=bins, color=color, alpha=ALPHA_HIST, density=True)
    try:
        kde = gaussian_kde(data, bw_method='scott')
        xs  = np.linspace(data.min(), data.max(), 500)
        ax.plot(xs, kde(xs), color=color, lw=2.2, alpha=ALPHA_KDE, label=label)
    except Exception:
        pass
    ax.axvline(data.mean(), color=color, lw=1.4, linestyle='--', alpha=0.75)

# ── Common figure styling ─────────────────────────────────────────────────────
def style_ax(ax, title, xlabel):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.spines[['top', 'right']].set_visible(False)

def plot_drift_type_on_ax(ax, snapshots, column_name, title_desc):
    for (sev, df_s), color, label in zip(snapshots.items(), COLORS, LABELS):
        plot_kde_hist(ax, df_s[column_name].copy(), color, label)
    style_ax(ax, f'{title_desc}\n{column_name}', column_name.replace('_', ' ').title())

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading test_fleet_data.csv ...')
    base_df = pd.read_csv('../data-generation/test_fleet_data.csv')

    print(f'Applying 8 drifts at severities: {SEVERITIES}')
    snapshots_per_drift = {} # drift_name -> {severity -> df}
    
    for drift_name, cfg in DRIFT_CONFIG.items():
        snapshots = {}
        for sev in SEVERITIES:
            df = base_df.copy()
            if sev > 0:
                df = cfg['fn'](df, cfg['col'], sev)
            snapshots[sev] = df
        snapshots_per_drift[drift_name] = snapshots

    print('Generating combined grid plot ...')
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    
    for ax, (drift_name, cfg) in zip(axes, DRIFT_CONFIG.items()):
        plot_drift_type_on_ax(ax, snapshots_per_drift[drift_name], cfg['col'], f"Drift: {cfg['desc']}")
        
    fig.tight_layout()
    output_path = f'{OUTPUT_DIR}/all_drift_distributions.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'All 8 distribution plots saved successfully to {output_path}!')

if __name__ == '__main__':
    main()
