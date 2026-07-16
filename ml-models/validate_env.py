import os
import sys
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rul_retrain_env import RULRetrainEnv

def validate():
    print("Loading data...")
    df = pd.read_csv("../data-generation/streaming_test_data.csv")
    train_split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:train_split_idx].reset_index(drop=True)
    ref_df = pd.read_csv("../data-generation/train_fleet_data.csv")
    
    print("Initializing environment...")
    env = RULRetrainEnv(
        df=df_train,
        ref_df=ref_df,
        base_model_path="results/best_rul_regressor_pipeline"
    )
    
    print("Running env_checker...")
    check_env(env.unwrapped)
    print("env_checker passed!")
    
    print("Running sanity check episode...")
    obs, info = env.reset()
    
    actions = []
    maes = []
    drifts = []
    rewards = []
    costs = []
    
    done = False
    step_idx = 0
    while not done:
        # Alternating actions: 0, 1, 0, 1...
        action = step_idx % 2
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Only collect values if the step actually yielded real metrics (not terminal dummy step)
        # Actually in gymnasium, the terminal step does return info and reward for that step.
        maes.append(info.get('raw_mae', 0.0))
        drifts.append(info.get('raw_drift', 0.0))
        rewards.append(reward)
        costs.append(info.get('episode_cost', 0.0))
        
        done = terminated or truncated
        step_idx += 1
        
    print("Sanity check episode finished. Plotting...")
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    x = range(len(actions))
    
    axs[0].plot(x, actions, marker='o', drawstyle='steps-mid')
    axs[0].set_ylabel('Action (0/1)')
    axs[0].set_title('Actions over batches')
    
    axs[1].plot(x, maes, marker='o', color='orange')
    axs[1].set_ylabel('MAE')
    
    axs[2].plot(x, drifts, marker='o', color='green')
    axs[2].set_ylabel('Drift (Energy Distance)')
    
    axs[3].plot(x, rewards, marker='o', color='red')
    axs[3].set_ylabel('Reward (Negative Cost)')
    
    axs[4].plot(x, costs, marker='o', color='purple')
    axs[4].set_ylabel('Cumulative Cost')
    axs[4].set_xlabel('Batch')
    
    plt.tight_layout()
    plt.savefig('results/env_sanity_check.png')
    print("Saved results/env_sanity_check.png")
    
    print("Verifying State Isolation...")
    # Change some internal state manually or just check that it's reset correctly.
    env._episode_cost = 999999
    obs, info = env.reset()
    assert env._episode_cost == 0.0, "Episode cost was not reset!"
    assert info['episode_cost'] == 0.0, "Episode cost in info was not reset!"
    print("State isolation check passed!")
    
if __name__ == '__main__':
    validate()
