# shared_config.py
# Ground-truth configuration to ensure mathematical consistency across RL Agent and Baseline

# --- Ground Truth Cost Model (Must exactly match Cost Model Document) ---
UPC = 1                # Under-Prediction Cost per hour
DC = 2                 # Degradation Cost (over-prediction) per hour
TEC = 25000              # Traditional Engineering Cost (per manual PyCaret run)
CIC = 10                 # Cloud Infrastructure Cost ($10 per compute hour)

# --- RL Hyperparameters ---
ALPHA_RL = 0.1           # Learning rate
GAMMA = 0.95             # Discount factor
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 0.023    # Linear decay per episode (drops to 0.05 over 50 episodes)
N_EPISODES = 30          # Total training episodes

# --- Environment Tuning ---
BATCH_SIZE = 2000
MAX_BATCHES = 16         # Truncation limit
DRIFT_MAX = 5843.93      # 95th percentile from historical validation
MAE_MAX = 471.48         # 95th percentile from historical validation
AGE_MAX = 10             # Batches

# --- State Discretisation Thresholds (3-Bin: Low/Med/High) ---
DRIFT_BIN_THRESHOLD_1 = 0.18  # 33rd percentile (normalized 1058.87 / 5843.93)
DRIFT_BIN_THRESHOLD_2 = 0.40  # 66th percentile (normalized 2315.18 / 5843.93)
ERROR_BIN_THRESHOLD_1 = 0.39  # 33rd percentile (normalized 182.06 / 471.48)
ERROR_BIN_THRESHOLD_2 = 0.61  # 66th percentile (normalized 289.13 / 471.48)
AGE_BIN_THRESHOLD_1 = 0.2     # 2 batches
AGE_BIN_THRESHOLD_2 = 0.5     # 5 batches

# --- Reward Normalization ---
# Theoretical worst-case cost per batch: MAE_MAX * BATCH_SIZE * DC
# ~ 471.48 * 2000 * 2 ≈ 1,885,920
REWARD_SCALE = 1_885_920
