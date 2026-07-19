import pandas as pd
import numpy as np

np.random.seed(42)

# --- Fleet Configuration ---
NUM_ENGINES = 100
MIN_SPEED_MPH = 25
MAX_SPEED_MPH = 30
MAX_DRIFT_PERCENT = 0.15
MIN_TRIP_DURATION_SEC = 1800
MAX_TRIP_DURATION_SEC = 14400

MAPPED_PARAMS_STATS = {
    'engine_load': {'mean': 31.8, 'std': 17.2, 'min': -5.2, 'max': 104.5, 'drift_dir': 1},
    'fuel_pressure': {'mean': 548.2, 'std': 285.6, 'min': 0.0, 'max': 735.8, 'drift_dir': 0},
    'fuel_level': {'mean': 72.4, 'std': 68.3, 'min': -105.0, 'max': 8970.2, 'drift_dir': -1},
    'actuator_position': {'mean': 2.1, 'std': 6.9, 'min': -12.5, 'max': 105.3, 'drift_dir': 1},
    'coolant_temperature': {'mean': 78.9, 'std': 19.2, 'min': -42.3, 'max': 176.1, 'drift_dir': 0},
    'emission_index': {'mean': 1.02, 'std': 0.12, 'min': 0.65, 'max': 1.58, 'drift_dir': -1},
    'filter_load': {'mean': 4.9, 'std': 5.7, 'min': -9.8, 'max': 118.4, 'drift_dir': 1},
    'rail_pressure': {'mean': 1012.5, 'std': 301.1, 'min': -315.0, 'max': 3150.5, 'drift_dir': -1},
    'ambient_pressure': {'mean': 24.3, 'std': 17.8, 'min': -65.0, 'max': 10250.0, 'drift_dir': 0},
    'system_voltage': {'mean': 701.4, 'std': 38.4, 'min': 0.0, 'max': 780.5, 'drift_dir': 1},
    'energy_storage_level': {'mean': 76.8, 'std': 19.5, 'min': 0.0, 'max': 101.8, 'drift_dir': -1},
    'idle_duration': {'mean': 1425.0, 'std': 7125.0, 'min': 0.0, 'max': 1425000.0, 'drift_dir': 0},
    'fluid_level': {'mean': 73.5, 'std': 23.7, 'min': 0.0, 'max': 104.2, 'drift_dir': -1},
    'fluid_dosing_rate': {'mean': 33.1, 'std': 2.2, 'min': 4.5, 'max': 63.7, 'drift_dir': 1},
    #these next values are from cmapps
    'engine_speed': {'mean': 8677.5537, 'std': 374.6575, 'min': 7984.5100, 'max': 9244.5900, 'drift_dir': -1},
    'exhaust_temperature': {'mean': 1260.9564, 'std': 136.3001, 'min': 1023.7700, 'max': 1441.4900, 'drift_dir': 1},
    'boost_pressure': {'mean': 359.7300, 'std': 174.1338, 'min': 136.1700, 'max': 570.8100, 'drift_dir': -1},
    'turbo_speed': {'mean': 2273.8297, 'std': 142.4266, 'min': 1914.7200, 'max': 2388.6400, 'drift_dir': -1},
    'air_fuel_ratio': {'mean': 0.0252, 'std': 0.0050, 'min': 0.0200, 'max': 0.0300, 'drift_dir': 1},
    'fuel_rate': {'mean': 338.7898, 'std': 164.1935, 'min': 128.3100, 'max': 537.4900, 'drift_dir': 1},
    'intake_pressure': {'mean': 9.8950, 'std': 4.2656, 'min': 3.9100, 'max': 14.6200, 'drift_dir': -1},
    'ambient_temperature': {'mean': 485.8409, 'std': 30.4204, 'min': 445.0000, 'max': 518.6700, 'drift_dir': 1},
}

MAPPED_PARAMS = [
    'engine_speed',
    'exhaust_temperature',
    'boost_pressure',
    'turbo_speed',
    'air_fuel_ratio',
    'fuel_rate',
    'intake_pressure',
    'ambient_temperature',
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
    return MAX_DRIFT_PERCENT * (wear ** 2)

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
        
        for param in MAPPED_PARAMS:
            if param in MAPPED_PARAMS_STATS:
                stats = MAPPED_PARAMS_STATS[param]
                
                # Apply drift
                mu = stats['mean'] * (1 + (drift_mag * stats['drift_dir']))
                
                # Protect against divide-by-zero for constant sensors
                sigma = stats['std'] if stats['std'] > 0 else 1e-6
                
                # 1. Base trip mean
                raw_chunk_mean = np.random.normal(loc=mu, scale=sigma / np.sqrt(duration_sec))
                
                if 'min' in stats and 'max' in stats:
                    # Clip mean to bounds just in case
                    chunk_mean = np.clip(raw_chunk_mean, stats['min'], stats['max'])
                    
                    # 2. Trip std
                    chunk_std = sigma * np.random.uniform(0.95, 1.05)
                    
                    # 3. Dynamic Extreme Factors (Distance to bounds in standard deviations)
                    # We add a tiny epsilon to chunk_std to avoid division by zero
                    safe_std = chunk_std if chunk_std > 0 else 1e-6
                    
                    z_max = (stats['max'] - chunk_mean) / safe_std
                    z_min = (chunk_mean - stats['min']) / safe_std
                    
                    # Target extreme factor is around 4.0, but capped at what the bounds allow
                    max_ef = min(4.0, z_max)
                    min_ef = min(4.0, z_min)
                    
                    # Sample the actual extreme multipliers for this trip
                    ef_upper = np.random.uniform(0.75 * max_ef, max_ef) if max_ef > 0 else 0
                    ef_lower = np.random.uniform(0.75 * min_ef, min_ef) if min_ef > 0 else 0
                    
                    chunk_max = chunk_mean + (chunk_std * ef_upper)
                    chunk_min = chunk_mean - (chunk_std * ef_lower)
                    
                    # Final safety clip (should practically do nothing because of the math above, 
                    # but protects against floating point rounding issues)
                    row[f'{param}_mean'] = chunk_mean
                    row[f'{param}_std'] = chunk_std
                    row[f'{param}_min'] = np.clip(chunk_min, stats['min'], stats['max'])
                    row[f'{param}_max'] = np.clip(chunk_max, stats['min'], stats['max'])
                else:
                    # Fallback for unbounded
                    chunk_mean = raw_chunk_mean
                    chunk_std = sigma * np.random.uniform(0.95, 1.05)
                    ef = np.random.uniform(3.0, 4.0)
                    
                    row[f'{param}_mean'] = chunk_mean
                    row[f'{param}_std'] = chunk_std
                    row[f'{param}_min'] = chunk_mean - (chunk_std * ef)
                    row[f'{param}_max'] = chunk_mean + (chunk_std * ef)
            else:
                row[f'{param}_mean'] = np.nan
                row[f'{param}_std'] = np.nan
                row[f'{param}_min'] = np.nan
                row[f'{param}_max'] = np.nan
            
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


metadata_cols = ['Engine_number', 'trip_id', 'duration_sec', 'engine_runtime', 'Cumulative_odometer', 'vehicle_speed']
sensor_cols = [c for c in df_fleet.columns if c not in metadata_cols and c not in ['RUL', 'RUL_class']]
target_col = ['RUL', 'RUL_class']

df_fleet = df_fleet[metadata_cols + sorted(sensor_cols) + target_col]

train_df = df_fleet[df_fleet['Engine_number'] <= 60]
test_df = df_fleet[(df_fleet['Engine_number'] > 60) & (df_fleet['Engine_number'] <= 80)]
eval_df = df_fleet[df_fleet['Engine_number'] > 80]

train_path = "train_fleet_data.csv"
test_path = "test_fleet_data.csv"
eval_path = "eval_fleet_data.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
eval_df.to_csv(eval_path, index=False)

print(f"Data generation complete!")
print(f"Saved Training Set (60 engines): {len(train_df)} trips -> {train_path}")
print(f"Saved Testing Set (20 engines): {len(test_df)} trips -> {test_path}")
print(f"Saved Eval Set (20 engines): {len(eval_df)} trips -> {eval_path}")
