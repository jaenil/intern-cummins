import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'drift')))
from evaluate_drift_impact import DRIFT_CONFIG

def main():
    print("Loading clean test fleet data...")
    test_df = pd.read_csv('../data-generation/test_fleet_data.csv')
    
    print("Generating chronological streaming drift dataset...")
    severities_fine = np.arange(0, 10.5, 0.5)
    n_chunks = len(severities_fine)
    chunk_size = len(test_df) // n_chunks
    
    chunks = []
    for i, s in enumerate(severities_fine):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(test_df)
        
        df_chunk = test_df.iloc[start_idx:end_idx].copy()
        
        if s > 0:
            for drift_name, cfg in DRIFT_CONFIG.items():
                df_chunk = cfg['fn'](df_chunk, cfg['col'], s)
                
        chunks.append(df_chunk)
        
    streaming_df = pd.concat(chunks, ignore_index=True)
    filename = "../data-generation/streaming_test_data.csv"
    streaming_df.to_csv(filename, index=False)
    print(f"Saved {filename} with shape {streaming_df.shape}")

if __name__ == '__main__':
    main()
