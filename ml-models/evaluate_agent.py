import numpy as np
import pandas as pd
import json
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from rul_retrain_env import RULRetrainEnv
from tabular_q_agent import TabularQAgent
import shared_config as cfg
import matplotlib.pyplot as plt
import os

def run_pass(agent, env, force_action_0=False) -> dict:
    """Run one evaluation episode."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0   # Pure exploitation
    
    obs, info = env.reset()
    state = agent.discretize(obs)
    
    step_info = None
    done = False
    
    while not done:
        if force_action_0:
            action = 0
        else:
            action = agent.get_action(state)
            
        obs, reward, terminated, truncated, info = env.step(action)
        state = agent.discretize(obs)
        step_info = info
        done = terminated or truncated
        
    agent.epsilon = old_epsilon  # Restore
    return step_info

def main():
    print("Loading data...")
    train_df = pd.read_csv('../data-generation/train_fleet_data.csv')
    test_df = pd.read_csv('../data-generation/streaming_eval_data.csv')
    
    env = RULRetrainEnv(
        df=test_df,
        ref_df=train_df,
        base_model_path="results/best_rul_regressor_pipeline",
        batch_size=cfg.BATCH_SIZE
    )
    env = RecordEpisodeStatistics(env)
    
    agent = TabularQAgent(
        env=env,
        learning_rate=cfg.ALPHA_RL,
        initial_epsilon=cfg.EPSILON_START,
        epsilon_decay=cfg.EPSILON_DECAY,
        final_epsilon=cfg.EPSILON_END,
        discount_factor=cfg.GAMMA,
    )
    
    print("Loading trained Q-table")
    from collections import defaultdict
    try:
        loaded_dict = np.load('results/q_table.npy', allow_pickle=True).item()
        # FIX: Load into agent.q_values, not agent.q_table
        agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), loaded_dict)
    except FileNotFoundError:
        print("Warning: q_table.npy not found in results/. Proceeding with empty Q-table.")
    
    print("\n--- Running Baseline (No Retraining) ---")
    baseline_info = run_pass(agent, env, force_action_0=True)
    baseline_total_cost = baseline_info['episode_cost']
    
    print("\n--- Running Trained RL Agent ---")
    rl_info = run_pass(agent, env, force_action_0=False)
    rl_total_cost = rl_info['episode_cost']
    
    savings = baseline_total_cost - rl_total_cost
    
    # Save the evaluation results
    results = {
        "baseline_cost": float(baseline_total_cost),
        "rl_agent_cost": float(rl_total_cost),
        "savings": float(savings),
        "rl_retrains": int(rl_info['n_retrains'])
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBaseline Cost: ${baseline_total_cost:,.0f}")
    print(f"RL Agent Cost: ${rl_total_cost:,.0f} ({rl_info['n_retrains']} retrains)")
    print(f"Cost Savings:  ${savings:,.0f}")
    print("Results saved to results/evaluation_results.json")
    
    # --- Generate Visualisations ---
    print("\nGenerating visualisations...")
    os.makedirs('results', exist_ok=True)
    
    # Graph 1: Step Detail
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top Panel: MAE
    x_steps = np.arange(len(rl_info['step_maes']))
    ax1.plot(x_steps, rl_info['step_maes'], marker='o', label='MAE', color='blue')
    
    # Add vertical lines for retrains
    retrain_steps = [i for i, a in enumerate(rl_info['step_actions']) if a == 1]
    for r in retrain_steps:
        ax1.axvline(x=r, color='red', linestyle='--', alpha=0.7)
    
    # Proxy artist for legend
    if retrain_steps:
        ax1.plot([], [], color='red', linestyle='--', label='Retrain Triggered')
        
    ax1.set_ylabel("Mean Absolute Error (hours)")
    ax1.set_title("RL Agent Performance per Batch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom Panel: Step Costs
    ax2.bar(x_steps, rl_info['step_costs'], color='orange', alpha=0.7, label='Cost Incurred')
    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("Cost ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("results/eval_step_detail.png")
    plt.close()
    
    # Graph 2: Cumulative Cost
    plt.figure(figsize=(10, 5))
    baseline_cumulative = np.cumsum(baseline_info['step_costs'])
    rl_cumulative = np.cumsum(rl_info['step_costs'])
    
    plt.plot(x_steps, baseline_cumulative, label='Baseline (No Retraining)', color='gray', linestyle='--')
    plt.plot(x_steps, rl_cumulative, label='RL Agent', color='green', linewidth=2)
    
    plt.xlabel("Batch Index")
    plt.ylabel("Cumulative Cost ($)")
    plt.title("Cumulative Cost: Baseline vs RL Agent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/eval_cumulative_cost.png")
    plt.close()
    
    print("Visualisations saved to results/")

if __name__ == '__main__':
    main()
