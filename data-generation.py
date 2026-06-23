import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# --- Fleet Configuration ---
NUM_ENGINES = 100
MIN_SPEED_MPH = 25
MAX_SPEED_MPH = 30
MAX_DRIFT = 20.0
MIN_TRIP_DURATION_SEC = 1800
MAX_TRIP_DURATION_SEC = 14400

# CMAPSS distributions (Mean and Standard Deviation)
SENSOR_STATS = {
    'sensor_2': {'mean': 597.36, 'std': 42.47},
    'sensor_3': {'mean': 1467.03, 'std': 118.17},
    'sensor_4': {'mean': 1260.95, 'std': 136.30},
    'sensor_7': {'mean': 359.72, 'std': 174.13},
    'sensor_8': {'mean': 2273.82, 'std': 142.42},
    'sensor_9': {'mean': 8677.55, 'std': 374.65},
    'sensor_11': {'mean': 44.21, 'std': 3.42},
    'sensor_12': {'mean': 338.78, 'std': 164.19},
    'sensor_13': {'mean': 2349.64, 'std': 111.16},
    'sensor_14': {'mean': 8088.95, 'std': 80.62},
}

# New Descriptive Sensor Names
SENSOR_NAMES = {
    'sensor_2': 'Total temperature at LPC outlet (in °R)',
    'sensor_3': 'Total temperature at HPC outlet (in °R)',
    'sensor_4': 'Total temperature at LPT outlet (in °R)',
    'sensor_7': 'Total pressure at HPC outlet (in psia)',
    'sensor_8': 'Physical fan speed (in rpm)',
    'sensor_9': 'Physical core speed (in rpm)',
    'sensor_11': 'Static pressure at HPC outlet (in psia)',
    'sensor_12': 'Ratio of fuel flow to Ps30 (in pps/psi)',
    'sensor_13': 'Corrected fan speed (in rpm)',
    'sensor_14': 'Corrected core speed (in rpm)'
}

# Assign a random drift direction (+1 or -1) to each sensor
SENSOR_DRIFT_DIR = {s: np.random.choice([-1, 1]) for s in SENSOR_STATS.keys()}

def compute_drift(current_hours, total_hours):
    wear = np.clip(current_hours / total_hours, 0, 1)
    return MAX_DRIFT * (wear ** 2)

records = []
print(f"Generating synthetic fleet dataset for {NUM_ENGINES} engines...")

for engine_id in range(1, NUM_ENGINES + 1):
    avg_speed_mph = np.random.uniform(MIN_SPEED_MPH, MAX_SPEED_MPH)
    total_miles = np.random.normal(loc=100000, scale=15000)
    total_miles = max(total_miles, 50000)
    total_hours = total_miles / avg_speed_mph
    
    cumulative_hours = 0.0
    trip_id = 1
    
    while cumulative_hours < total_hours:
        duration_sec = np.random.randint(MIN_TRIP_DURATION_SEC, MAX_TRIP_DURATION_SEC)
        duration_hours = duration_sec / 3600.0
        
        if cumulative_hours + duration_hours > total_hours:
            duration_hours = total_hours - cumulative_hours
            duration_sec = duration_hours * 3600.0
            
        cumulative_hours += duration_hours
        rul = total_hours - cumulative_hours
        drift_mag = compute_drift(cumulative_hours, total_hours)
        
        row = {
            'engine_id': engine_id,
            'trip_id': trip_id,
            'duration_sec': duration_sec,
            'cumulative_hours': cumulative_hours,
        }
        
        # Generate 4 features per sensor: mean, std, min, max
        for sensor_id, stats in SENSOR_STATS.items():
            title_name = SENSOR_NAMES[sensor_id]
            
            mu = stats['mean'] + (drift_mag * SENSOR_DRIFT_DIR[sensor_id])
            sigma = stats['std']
            
            chunk_mean = np.random.normal(loc=mu, scale=sigma / np.sqrt(duration_sec))
            chunk_std = sigma * np.random.uniform(0.95, 1.05)
            extreme_factor = np.random.uniform(3.0, 4.0)
            chunk_max = chunk_mean + (chunk_std * extreme_factor)
            chunk_min = chunk_mean - (chunk_std * extreme_factor)
            
            row[f'{title_name}_mean'] = chunk_mean
            row[f'{title_name}_std'] = chunk_std
            row[f'{title_name}_min'] = chunk_min
            row[f'{title_name}_max'] = chunk_max
            
        row['RUL'] = rul
        records.append(row)
        trip_id += 1

df_fleet = pd.DataFrame(records)

# Re-order columns neatly
metadata_cols = ['engine_id', 'trip_id', 'duration_sec', 'cumulative_hours']
sensor_cols = [c for c in df_fleet.columns if c not in metadata_cols and c != 'RUL']
target_col = ['RUL']

df_fleet = df_fleet[metadata_cols + sorted(sensor_cols) + target_col]

# --- Train / Test Split ---
train_df = df_fleet[df_fleet['engine_id'] <= 80]
test_df = df_fleet[df_fleet['engine_id'] > 80]

train_path = "train_fleet_data.csv"
test_path = "test_fleet_data.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Data generation complete!")
print(f"Saved Training Set (80 engines): {len(train_df)} trips -> {train_path}")
print(f"Saved Testing Set (20 engines): {len(test_df)} trips -> {test_path}")
