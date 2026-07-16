import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pycaret.regression import load_model, predict_model, setup, create_model
from sklearn.metrics import mean_absolute_error
import sys
import os

# Add parent directory to path to allow importing from shared_config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import shared_config as sc

# Add drift directory to path for evaluate_drift_impact
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'drift')))
from evaluate_drift_impact import energy_distance

class StaticRuleAgent:
    def __init__(self, drift_threshold=sc.DRIFT_THRESHOLD_STATIC,
                 mae_threshold=sc.MAE_THRESHOLD_STATIC):
        self.drift_threshold = drift_threshold
        self.mae_threshold = mae_threshold

    def should_retrain(self, mae: float, drift_score: float) -> bool:
        return (mae > self.mae_threshold) or (drift_score > self.drift_threshold)

@dataclass
class BenchmarkResult:
    total_cost_traditional: float
    n_retrains: int
    avg_mae: float
    total_error_cost: float
    total_retrain_cost: float
    total_M: int
    total_N: int
    batch_log: pd.DataFrame

class BenchmarkSimulator:
    def __init__(self, df, ref_df, reg_model, agent, batch_size=sc.BATCH_SIZE):
        self.df = df
        self.ref_df = ref_df
        self.reg_model = reg_model
        self.agent = agent
        self.batch_size = batch_size

    def run(self) -> BenchmarkResult:
        sensor_cols = [c for c in self.df.columns if c not in sc.METADATA_COLS and c not in sc.TARGET_COLS and c != 'prediction_label' and c != 'Label' and 'fault_list' not in c]
        
        X_ref = self.ref_df[sensor_cols].values.astype(np.float64)
        
        total_cost = 0.0
        n_retrains = 0
        total_error_cost = 0.0
        total_M = 0
        total_N = 0
        
        logs = []
        
        num_batches = len(self.df) // self.batch_size + (1 if len(self.df) % self.batch_size != 0 else 0)
        
        print(f"Running simulation over {num_batches} batches...")
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(self.df))
            batch = self.df.iloc[start_idx:end_idx].copy()
            
            # Predict RUL
            predictions = predict_model(self.reg_model, data=batch, verbose=False)
            pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else 'Label'
            y_pred = predictions[pred_col].values
            y_true = predictions['RUL'].values
            
            # 2. Evaluate Error
            mae = mean_absolute_error(y_true, y_pred)
            c_err, M, N = sc.compute_rul_error_cost(y_true, y_pred)
            
            # 3. Evaluate Drift
            X_batch = batch[sensor_cols].values.astype(np.float64)
            # using energy distance as a fast proxy instead of full MMD
            # Need a quick function. If energy_distance expects 1D arrays or something, we need to adapt.
            # evaluate_drift_impact.py has energy_distance(x, y). Let's use it per column and average, or flatten.
            # Actually energy_distance in scipy computes between two 1D arrays. 
            # We will flatten for a quick proxy.
            # Subsample X_ref for speed if it's too large
            if len(X_ref) > 500:
                idx = np.random.choice(len(X_ref), 500, replace=False)
                X_ref_sample = X_ref[idx]
            else:
                X_ref_sample = X_ref
                
            drift_score = energy_distance(X_ref_sample, X_batch)
            
            # 4. Agent Decision
            retrain_flag = self.agent.should_retrain(mae, drift_score)
            
            if retrain_flag:
                total_cost += sc.TEC
                n_retrains += 1
                
                print(f"  [Batch {i}] Retraining triggered! (MAE={mae:.2f}, Drift={drift_score:.4f})")
                print(f"  --> Combining historical data and retraining 'lr' model...")
                
                # Combine historical data + all test stream seen so far
                combined_train_df = pd.concat([self.ref_df, self.df.iloc[:end_idx]], ignore_index=True)
                
                # PyCaret setup and actual training
                ignore_cols = ['Engine_number', 'trip_id', 'RUL_class']
                _ = setup(
                    data=combined_train_df,
                    target='RUL',
                    ignore_features=ignore_cols,
                    numeric_imputation='mean',
                    session_id=42,
                    verbose=False,
                    index=False
                )
                self.reg_model = create_model('lr', verbose=False)
                
                # Update drift reference
                X_ref = X_batch
                
            total_cost += c_err
            total_error_cost += c_err
            total_M += M
            total_N += N
            
            logs.append({
                'batch_idx': i,
                'mae': mae,
                'drift_score': drift_score,
                'M': M,
                'N': N,
                'c_err': c_err,
                'retrain_flag': int(retrain_flag),
                'total_cost': total_cost
            })
            
        log_df = pd.DataFrame(logs)
        avg_mae = log_df['mae'].mean()
        
        return BenchmarkResult(
            total_cost_traditional=total_cost,
            n_retrains=n_retrains,
            avg_mae=avg_mae,
            total_error_cost=total_error_cost,
            total_retrain_cost=n_retrains * sc.TEC,
            total_M=total_M,
            total_N=total_N,
            batch_log=log_df
        )

    def plot_timeline(self, result: BenchmarkResult, filename='results/benchmark_timeline.png'):
        log_df = result.batch_log
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Panel 1: MAE
        axes[0].plot(log_df['batch_idx'], log_df['mae'], label='MAE')
        axes[0].axhline(y=self.agent.mae_threshold, color='r', linestyle='--', label='MAE Threshold')
        retrain_batches = log_df[log_df['retrain_flag'] == 1]['batch_idx']
        retrain_maes = log_df[log_df['retrain_flag'] == 1]['mae']
        axes[0].scatter(retrain_batches, retrain_maes, color='red', zorder=5, label='Retrain Triggered')
        axes[0].set_ylabel('MAE (hrs)')
        axes[0].legend()
        axes[0].set_title('MAE per Batch')
        
        # Panel 2: Drift
        axes[1].plot(log_df['batch_idx'], log_df['drift_score'], label='Energy Distance', color='orange')
        axes[1].axhline(y=self.agent.drift_threshold, color='r', linestyle='--', label='Drift Threshold')
        axes[1].set_ylabel('Drift Score')
        axes[1].legend()
        axes[1].set_title('Drift Distance per Batch')
        
        # Panel 3: Cumulative Cost
        axes[2].plot(log_df['batch_idx'], log_df['total_cost'], label='Cumulative Cost', color='green')
        axes[2].set_ylabel('Cost ($)')
        axes[2].set_xlabel('Batch Index')
        axes[2].legend()
        axes[2].set_title('Cumulative Operational Cost')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")

def main():
    print("Loading streaming test and train fleet data...")
    test_df = pd.read_csv('../data-generation/streaming_test_data.csv')
    train_df = pd.read_csv('../data-generation/train_fleet_data.csv')
    
    print("Loading RUL regressor model...")
    model = load_model('results/best_rul_regressor_pipeline')
    
    agent = StaticRuleAgent()
    
    simulator = BenchmarkSimulator(df=test_df, ref_df=train_df, reg_model=model, agent=agent)
    
    result = simulator.run()
    
    print("\n" + "="*60)
    print("  BASELINE BENCHMARK (Traditional Retraining)")
    print("="*60)
    print(f"  Total Operational Cost : ${result.total_cost_traditional:,.2f}")
    print(f"  ─── Error costs        : ${result.total_error_cost:,.2f}")
    print(f"      Early predictions M: {result.total_M} engines × ${sc.UPC:,.0f} = ${result.total_M * sc.UPC:,.2f}")
    print(f"      Late predictions  N: {result.total_N} engines × ${sc.DC:,.0f} = ${result.total_N * sc.DC:,.2f}")
    print(f"  ─── Retraining costs   : ${result.total_retrain_cost:,.2f}")
    print(f"      Retraining events  : {result.n_retrains} × TEC (${sc.TEC:,.0f}) = ${result.total_retrain_cost:,.2f}")
    print("="*60)
    
    simulator.plot_timeline(result)

if __name__ == '__main__':
    main()
