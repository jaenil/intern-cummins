# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
from pycaret.regression import setup, create_model, load_model, predict_model
import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from drift.evaluate_drift_impact import energy_distance
from shared_config import (
    UPC, DC, TEC, BATCH_SIZE, MAX_BATCHES,
    DRIFT_MAX, MAE_MAX, AGE_MAX, REWARD_SCALE
)

class RULRetrainEnv(gym.Env):
    """
    Gymnasium Environment for RUL Predictive Maintenance Retraining.
    State: 6 continuous features [drift, mae, Δmae, staleness, slope, cost]
    Actions: 0 (Do Nothing), 1 (Retrain PyCaret Pipeline)
    Reward: -(Error Cost + Retrain Cost)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,                        
        ref_df: pd.DataFrame,                    
        base_model_path: str,                     
        history_df: Optional[pd.DataFrame] = None, 
        batch_size: int = BATCH_SIZE,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.ref_df = ref_df.reset_index(drop=True)
        self.history_df = history_df.reset_index(drop=True) if history_df is not None else None
        self.base_model_path = base_model_path
        self.working_model_path = f"{self.base_model_path}_temp"
        self.batch_size = batch_size
        self.render_mode = render_mode
        
        # Action space: 0 = Do Nothing, 1 = Retrain
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation space: 6 normalised features
        # [drift, mae, Δmae, age, slope, cum_cost]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -1, 0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Identify non-target columns for drift calculation (excluding NaNs)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [
            c for c in numeric_cols 
            if c not in ['engine_id', 'cycle', 'RUL', 'severity'] 
            and self.df[c].dtype in [np.float64, np.int64]
            and self.df[c].isna().sum() == 0
            and self.ref_df[c].isna().sum() == 0
        ]
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)
        
        # 1. Reset physical state: Copy base model to working model
        if os.path.exists(f"{self.base_model_path}.pkl"):
            import shutil
            shutil.copyfile(f"{self.base_model_path}.pkl", f"{self.working_model_path}.pkl")
        self._reg_model = load_model(self.working_model_path)
        self._X_ref = self.ref_df[self.feature_cols].sample(n=min(500, len(self.ref_df)), random_state=42).astype(np.float32).values
        
        # 2. Reset tracking variables
        self._current_batch_idx = 0
        self._n_retrains = 0
        self._episode_cost = 0.0
        self._mae_history = []
        self._step_costs = []
        self._step_actions = []
        self._last_retrain_idx = 0
        
        # 3. Compute initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _get_obs(self) -> np.ndarray:
        # Terminal state protection
        total_batches = len(self.df) // self.batch_size
        if self._current_batch_idx >= total_batches:
            return np.zeros(6, dtype=np.float32)
            
        batch_start = self._current_batch_idx * self.batch_size
        batch_end = batch_start + self.batch_size
        X_batch_df = self.df.iloc[batch_start:batch_end]
        
        # Predict on current batch to get MAE
        preds = predict_model(self._reg_model, data=X_batch_df, verbose=False)
        raw_mae = np.abs(preds['RUL'] - preds['prediction_label']).mean()
        self._mae_history.append(raw_mae)
        
        # Compute drift
        X_batch_arr = X_batch_df[self.feature_cols].sample(n=min(500, len(X_batch_df)), random_state=42).astype(np.float32).values
        raw_drift = energy_distance(self._X_ref, X_batch_arr)
        
        # State vector computation
        f1_drift = np.clip(raw_drift / DRIFT_MAX, 0.0, 1.0)
        f2_mae   = np.clip(raw_mae / MAE_MAX, 0.0, 1.0)
        
        if len(self._mae_history) >= 2:
            prev_mae = self._mae_history[-2]
            pct_change = (raw_mae - prev_mae) / (prev_mae + 1e-6)
            f3_delta = np.clip(pct_change, -1.0, 1.0)
        else:
            f3_delta = 0.0
            
        age_batches = self._current_batch_idx - self._last_retrain_idx
        f4_age = np.clip(age_batches / AGE_MAX, 0.0, 1.0)
        
        if len(self._mae_history) >= 2:
            y = self._mae_history[-min(5, len(self._mae_history)):]
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            f5_slope = np.clip(slope / (MAE_MAX * 0.1), -1.0, 1.0)
        else:
            f5_slope = 0.0
            
        f6_cost = np.clip(self._episode_cost / 1_000_000.0, 0.0, 1.0)
        
        # Save raw values for info dictionary
        self._last_raw_mae = raw_mae
        self._last_raw_drift = raw_drift
        
        return np.array([f1_drift, f2_mae, f3_delta, f4_age, f5_slope, f6_cost], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            'batch_idx': getattr(self, '_current_batch_idx', 0),
            'raw_mae': getattr(self, '_last_raw_mae', 0.0),
            'raw_drift': getattr(self, '_last_raw_drift', 0.0),
            'n_retrains': getattr(self, '_n_retrains', 0),
            'episode_cost': getattr(self, '_episode_cost', 0.0),
            'step_costs': getattr(self, '_step_costs', []),
            'step_actions': getattr(self, '_step_actions', []),
            'step_maes': getattr(self, '_mae_history', [])
        }

    def _do_retrain(self) -> None:
        seen_rows = self._current_batch_idx * self.batch_size
        stream_so_far = self.df.iloc[:seen_rows]
        
        dfs_to_concat = [self.ref_df]
        if self.history_df is not None:
            dfs_to_concat.append(self.history_df)
        if len(stream_so_far) > 0:
            dfs_to_concat.append(stream_so_far)
        
        combined = pd.concat(dfs_to_concat, ignore_index=True)
        
        # PyCaret retraining
        ignore_cols = [c for c in ['Engine_number', 'trip_id', 'RUL_class', 'engine_id', 'cycle', 'severity'] if c in combined.columns]
        setup(combined, target="RUL", ignore_features=ignore_cols, verbose=False, session_id=42, n_jobs=1)
        best = create_model('lr', cross_validation=False, verbose=False)
        from pycaret.regression import save_model
        save_model(best, self.working_model_path)
        self._reg_model = load_model(self.working_model_path)
        
        # Update physical state
        self._last_retrain_idx = self._current_batch_idx
        self._X_ref = combined[self.feature_cols].sample(n=min(500, len(combined)), random_state=42).astype(np.float32).values

    def step(self, action: int) -> tuple:
        # Calculate cost BEFORE retrain using current model (Dynamic Baseline)
        batch_start = self._current_batch_idx * self.batch_size
        batch_end = batch_start + self.batch_size
        X_batch_df = self.df.iloc[batch_start:batch_end]
        
        pre_preds = predict_model(self._reg_model, data=X_batch_df, verbose=False)
        pre_errors = pre_preds['RUL'] - pre_preds['prediction_label']
        pre_action_cost = np.sum(np.where(pre_errors >= 0, pre_errors * UPC, -pre_errors * DC))
        
        retrain_cost = 0.0
        post_action_cost = pre_action_cost  # Defaults to same cost if we don't retrain
        
        if action == 1:
            self._do_retrain()
            self._n_retrains += 1
            retrain_cost = TEC
            
            # Predict AGAIN on the same batch with the NEW model to see the improvement
            post_preds = predict_model(self._reg_model, data=X_batch_df, verbose=False)
            post_errors = post_preds['RUL'] - post_preds['prediction_label']
            post_action_cost = np.sum(np.where(post_errors >= 0, post_errors * UPC, -post_errors * DC))
            
        step_cost = post_action_cost + retrain_cost
        self._episode_cost += step_cost
        self._step_costs.append(step_cost)
        self._step_actions.append(action)
        
        # Normalized Negative Cost Reward
        reward = float(-step_cost / REWARD_SCALE)
        
        # Advance stream
        self._current_batch_idx += 1
        
        # Termination conditions
        total_batches = len(self.df) // self.batch_size
        terminated = self._current_batch_idx >= total_batches
        truncated = self._current_batch_idx >= MAX_BATCHES
        
        observation = self._get_obs() if not (terminated or truncated) else np.zeros(6, dtype=np.float32)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Batch {self._current_batch_idx} | MAE: {getattr(self, '_last_raw_mae', 0):.2f} | Drift: {getattr(self, '_last_raw_drift', 0):.2f} | Retrains: {self._n_retrains} | Cost: ${self._episode_cost:,.2f}")

    def close(self):
        pass
