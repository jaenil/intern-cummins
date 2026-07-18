# shared_config.py
# Ground-truth configuration to ensure mathematical consistency across RL Agent and Baseline

import pandas as pd
import numpy as np

# --- Ground Truth Cost Model (Must exactly match Cost Model Document) ---
UPC = 100              # Under-Prediction Cost (per engine with mean error <= -100)
DC = 200               # Degradation Cost (per engine with mean error >= 100)
TEC = 1500              # Traditional Engineering Cost (per manual PyCaret run)
CIC = 10                 # Cloud Infrastructure Cost ($10 per compute hour)

# --- RL Hyperparameters ---
ALPHA_RL = 0.1           # Learning rate
GAMMA = 0.95             # Discount factor
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 0.023    # Linear decay per episode
N_EPISODES = 20          # Total training episodes

# --- Environment Tuning ---
DAY_WINDOW = 10          # Number of days to process per step
MAX_BATCHES = 25         # Truncation limit (number of windows)
DRIFT_MAX = 48526.17     # 95th percentile from new day-window evaluation
MAE_MAX = 504.84         # 95th percentile from new day-window evaluation
AGE_MAX = 10             # Windows

# --- State Discretisation Thresholds (3-Bin: Low/Med/High) ---
DRIFT_BIN_THRESHOLD_1 = 0.18  
DRIFT_BIN_THRESHOLD_2 = 0.40  
ERROR_BIN_THRESHOLD_1 = 0.39  
ERROR_BIN_THRESHOLD_2 = 0.61  
AGE_BIN_THRESHOLD_1 = 0.2     # 2 batches
AGE_BIN_THRESHOLD_2 = 0.5     # 5 batches

# --- Reward Normalization (Updated: Engine-Wise + Day-Window) ---
# Worst-case cost = 20 engines * DC (200) = 4000
REWARD_SCALE = 4000
REWARD_SCALE_LEGACY = 400_000  # Old row-wise upper bound — kept for reference

# --- Static/Traditional Threshold Baseline ---
DRIFT_THRESHOLD_STATIC = 20.0
MAE_THRESHOLD_STATIC = 300.0

# --- Dataset Columns ---
METADATA_COLS = ['Engine_number', 'trip_id', 'RUL_class', 'day']
TARGET_COLS = ['RUL']

# --- Engine-Wise Cost Functions ---

def cat_upc_dc(mean_error: float) -> str:
    """Classify a single engine's mean RUL prediction error.
    error = prediction_label - RUL
    >= +100 → Over-prediction (DC: engine fails sooner than expected)
    <= -100 → Under-prediction (UPC: unnecessary early maintenance)
    """
    if mean_error >= 100:
        return 'dc'
    elif mean_error <= -100:
        return 'upc'
    else:
        return 'NA'

def compute_rul_error_cost(
    preds_df: pd.DataFrame,
    pred_col: str = 'prediction_label',
    actual_col: str = 'RUL',
    engine_col: str = 'Engine_number'
) -> tuple:
    """
    Compute per-engine RUL error cost.
    Groups predictions by engine, takes mean error per engine,
    then counts UPC and DC violations.

    Returns:
        cost (float): m*UPC + n*DC
        m    (int):   engines with mean error <= -100 (UPC)
        n    (int):   engines with mean error >= +100 (DC)
    """
    df = preds_df.copy()
    df['_error'] = df[pred_col] - df[actual_col]
    df_grp = df.groupby(engine_col)['_error'].mean().reset_index()
    df_grp['_type'] = df_grp['_error'].apply(cat_upc_dc)
    m = int((df_grp['_type'] == 'upc').sum())
    n = int((df_grp['_type'] == 'dc').sum())
    cost = float(m * UPC + n * DC)
    return cost, m, n

def append_dfs_with_days(
    dfs: list,
    engine_col: str = 'Engine_number',
    min_day: int = 1,
    max_day: int = 5
) -> pd.DataFrame:
    """
    Assign synthetic streaming 'day' labels across a list of DataFrames.
    A chunk spans 5 days. All engines in the same chunk share the same 5-day pool.
    Used by the streaming data generator to simulate temporal progression.
    """
    result = []
    day_counter = min_day
    current_max = max_day
    for df in dfs:
        df = df.copy()
        
        # All engines share the same day pool for this chunk
        days_array = np.random.randint(day_counter, current_max + 1, size=len(df))
        df['day'] = days_array
        
        day_counter = current_max + 1
        current_max = day_counter + 4
        result.append(df)
    return pd.concat(result, ignore_index=True)
