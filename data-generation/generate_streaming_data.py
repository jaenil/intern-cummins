import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'drift')))
from evaluate_drift_impact import DRIFT_CONFIG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml-models')))
from shared_config import append_dfs_with_days

def generate_streaming_dataset(test_df: pd.DataFrame, severities: np.ndarray, output_path: str, seed: int = 42) -> pd.DataFrame:
    n_chunks = len(severities)
    
    chunks = []
    for i, s in enumerate(severities):
        # We need to take the i-th fraction of rows for EVERY engine
        chunk_list = []
        for engine, group in test_df.groupby('Engine_number'):
            engine_chunk_size = len(group) // n_chunks
            start_idx = i * engine_chunk_size
            end_idx = (i + 1) * engine_chunk_size if i < n_chunks - 1 else len(group)
            chunk_list.append(group.iloc[start_idx:end_idx].copy())
            
        df_chunk = pd.concat(chunk_list, ignore_index=True)
        
        if s > 0:
            for drift_name, cfg in DRIFT_CONFIG.items():
                df_chunk = cfg['fn'](df_chunk, cfg['col'], s)
                
        chunks.append(df_chunk)
        
    streaming_df = append_dfs_with_days(chunks, engine_col='Engine_number')
    streaming_df = streaming_df.sort_values('day').reset_index(drop=True)
    streaming_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {streaming_df.shape}")
    return streaming_df

def main():
    print("Loading clean test fleet data...")
    test_df = pd.read_csv('../data-generation/test_fleet_data.csv')
    
    print("Generating training streaming dataset (fine drift)...")
    severities_fine = np.arange(0, 10.5, 0.5)
    generate_streaming_dataset(test_df, severities_fine, "../data-generation/streaming_test_data.csv")
    
    print("Generating evaluation streaming dataset (coarse drift)...")
    severities_coarse = np.arange(0, 11.0, 1.0)
    generate_streaming_dataset(test_df, severities_coarse, "../data-generation/streaming_eval_data.csv")

if __name__ == '__main__':
    main()
