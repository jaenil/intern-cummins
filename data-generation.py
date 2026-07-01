import pandas as pd
import numpy as np

np.random.seed(42)

# --- Fleet Configuration ---
NUM_ENGINES = 100
MIN_SPEED_MPH = 25
MAX_SPEED_MPH = 30
MAX_DRIFT = 20.0
MIN_TRIP_DURATION_SEC = 1800
MAX_TRIP_DURATION_SEC = 14400

# CMAPSS distributions (Mean and Standard Deviation)
# Mapped based on available sensors in the original script or reasonable synthetic defaults
SENSOR_STATS = {
    'engine_speed': {'mean': 8677.55, 'std': 374.65},         # Mapped from Physical core speed (Nc, sensor_9)
    'exhaust_temperature': {'mean': 1260.95, 'std': 136.30},  # Mapped from Total temp at LPT outlet (T50, sensor_4)
    'boost_pressure': {'mean': 359.72, 'std': 174.13},        # Mapped from Total pressure at HPC outlet (P30, sensor_7)
    'turbo_speed': {'mean': 2273.82, 'std': 142.42},          # Mapped from Physical fan speed (Nf, sensor_8)
    'air_fuel_ratio': {'mean': 0.02, 'std': 0.005},           # Mapped from Burner fuel-air ratio (farB)
    'fuel_rate': {'mean': 150.0, 'std': 15.0},                # Mapped from Fuel flow (Wf) - Synthetic baseline
    'intake_pressure': {'mean': 14.6, 'std': 0.5},            # Mapped from Pressure at fan inlet (P2) - Synthetic baseline
    'ambient_temperature': {'mean': 518.67, 'std': 5.0},      # Mapped from Total temp at fan inlet (T2) - Synthetic baseline
}

#drift will always increase as the engine_runtime increases
SENSOR_DRIFT_DIR = 1

MAPPED_PARAMS_STATS = {
    'engine_load': {'mean': 31.8, 'std': 17.2, 'min': -5.2, 'max': 104.5},
    'fuel_pressure': {'mean': 548.2, 'std': 285.6, 'min': 0.0, 'max': 735.8},
    'fuel_level': {'mean': 72.4, 'std': 68.3, 'min': -105.0, 'max': 8970.2},
    'actuator_position': {'mean': 2.1, 'std': 6.9, 'min': -12.5, 'max': 105.3},
    'coolant_temperature': {'mean': 78.9, 'std': 19.2, 'min': -42.3, 'max': 176.1},
    'emission_index': {'mean': 1.02, 'std': 0.12, 'min': 0.65, 'max': 1.58},
    'filter_load': {'mean': 4.9, 'std': 5.7, 'min': -9.8, 'max': 118.4},
    'rail_pressure': {'mean': 1012.5, 'std': 301.1, 'min': -315.0, 'max': 3150.5},
    'ambient_pressure': {'mean': 24.3, 'std': 17.8, 'min': -65.0, 'max': 10250.0},
    'system_voltage': {'mean': 701.4, 'std': 38.4, 'min': 0.0, 'max': 780.5},
    'energy_storage_level': {'mean': 76.8, 'std': 19.5, 'min': 0.0, 'max': 101.8},
    'idle_duration': {'mean': 1425.0, 'std': 7125.0, 'min': 0.0, 'max': 1425000.0},
    'fluid_level': {'mean': 73.5, 'std': 23.7, 'min': 0.0, 'max': 104.2},
    'fluid_dosing_rate': {'mean': 33.1, 'std': 2.2, 'min': 4.5, 'max': 63.7},
}

# Unmapped parameters to be initialized or generated
MAPPED_PARAMS = [
    'engine_load',
    'fuel_pressure',
    'fuel_level',
    'actuator_position',
    'coolant_temperature',
    'emission_index',
    'filter_load',
    'rail_pressure',
    'ambient_pressure',
    'system_voltage',
    'energy_storage_level',
    'idle_duration',
    'fluid_level',
    'fluid_dosing_rate',
    'fault_list'
]

def compute_drift(current_hours, total_hours):
    wear = np.clip(current_hours / total_hours, 0, 1)
    return MAX_DRIFT * (wear ** 2)

records = []

for engine_id in range(1, NUM_ENGINES + 1):
    avg_speed_mph = np.random.uniform(MIN_SPEED_MPH, MAX_SPEED_MPH)
    total_miles = np.random.normal(loc=100000, scale=15000)
    total_miles = max(total_miles, 50000)
    total_hours = total_miles / avg_speed_mph
    
    cumulative_hours = 0.0
    cumulative_odometer = 0.0
    trip_id = 1
    
    while cumulative_hours < total_hours:
        duration_sec = np.random.randint(MIN_TRIP_DURATION_SEC, MAX_TRIP_DURATION_SEC)
        duration_hours = duration_sec / 3600.0
        
        if cumulative_hours + duration_hours > total_hours:
            duration_hours = total_hours - cumulative_hours
            duration_sec = duration_hours * 3600.0
            
        cumulative_hours += duration_hours
        cumulative_odometer += (avg_speed_mph * duration_hours)
        rul = total_hours - cumulative_hours
        drift_mag = compute_drift(cumulative_hours, total_hours)
        
        row = {
            'Engine_number': engine_id,
            'trip_id': trip_id,
            'duration_sec': duration_sec,
            'engine_runtime': cumulative_hours,
            'Cumulative_odometer': cumulative_odometer,
            'vehicle_speed': avg_speed_mph
        }
        
        # Generate 4 features per sensor: mean, std, min, max
        for sensor_name, stats in SENSOR_STATS.items():
            mu = stats['mean'] + (drift_mag * SENSOR_DRIFT_DIR)
            sigma = stats['std']
            
            chunk_mean = np.random.normal(loc=mu, scale=sigma / np.sqrt(duration_sec))
            chunk_std = sigma * np.random.uniform(0.95, 1.05)
            extreme_factor = np.random.uniform(3.0, 4.0)
            chunk_max = chunk_mean + (chunk_std * extreme_factor)
            chunk_min = chunk_mean - (chunk_std * extreme_factor)
            
            row[f'{sensor_name}_mean'] = chunk_mean
            row[f'{sensor_name}_std'] = chunk_std
            row[f'{sensor_name}_min'] = chunk_min
            row[f'{sensor_name}_max'] = chunk_max
            
        # Generate synthetic data for unmapped parameters using distribution stats
        for param in MAPPED_PARAMS:
            if param in MAPPED_PARAMS_STATS:
                stats = MAPPED_PARAMS_STATS[param]
                val = np.random.normal(loc=stats['mean'], scale=stats['std'])
                row[param] = np.clip(val, stats['min'], stats['max'])
            else:
                row[param] = np.nan
            
        row['RUL'] = rul
        if rul > 2000:
            row['RUL_class'] = 'perfect'
        elif rul > 1000:
            row['RUL_class'] = 'good'
        elif rul > 500:
            row['RUL_class'] = 'bad'
        else:
            row['RUL_class'] = 'worse'
            
        records.append(row)
        trip_id += 1

df_fleet = pd.DataFrame(records)

# Re-order columns
metadata_cols = ['Engine_number', 'trip_id', 'duration_sec', 'engine_runtime', 'Cumulative_odometer', 'vehicle_speed']
sensor_cols = [c for c in df_fleet.columns if c not in metadata_cols and c not in MAPPED_PARAMS and c != 'RUL']
target_col = ['RUL', 'RUL_class']

df_fleet = df_fleet[metadata_cols + sorted(sensor_cols) + MAPPED_PARAMS + target_col]

# --- Train / Test Split ---
train_df = df_fleet[df_fleet['Engine_number'] <= 80]
test_df = df_fleet[df_fleet['Engine_number'] > 80]

train_path = "train_fleet_data.csv"
test_path = "test_fleet_data.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Data generation complete!")
print(f"Saved Training Set (80 engines): {len(train_df)} trips -> {train_path}")
print(f"Saved Testing Set (20 engines): {len(test_df)} trips -> {test_path}")
