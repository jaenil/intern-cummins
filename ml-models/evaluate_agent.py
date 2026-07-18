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
from pycaret.regression import load_model
from baseline_threshold import StaticRuleAgent, ThresholdSimulator

def run_pass(agent, df, ref_df, base_model_path, force_action_0=False) -> dict:
    """Run one evaluation episode with a fresh environment."""
    env = RULRetrainEnv(
        df=df,
        ref_df=ref_df,
        base_model_path=base_model_path,
        max_retrain_rows=None   # No cap during evaluation — use full dataset for accurate costs
    )
    env = RecordEpisodeStatistics(env)
    
    agent.env = env
    
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0   # Pure exploitation
    
    obs, info = env.reset()
    state = agent.discretize(obs)
    
    done = False
    
    while not done:
        if force_action_0:
            action = 0
        else:
            action = agent.get_action(state)
            
        obs, reward, terminated, truncated, info = env.step(action)
        state = agent.discretize(obs)
        done = terminated or truncated
        
    agent.epsilon = old_epsilon  # Restore
    
    unwrapped_env = env.unwrapped
    return {
        'n_retrains': unwrapped_env._n_retrains,
        'episode_cost': unwrapped_env._episode_cost,
        'step_costs': unwrapped_env._step_costs.copy(),
        'step_actions': unwrapped_env._step_actions.copy(),
        'step_maes': unwrapped_env._mae_history.copy()
    }

def main():
    print("Loading data...")
    train_df = pd.read_csv('../data-generation/train_fleet_data.csv')
    test_df = pd.read_csv('../data-generation/streaming_eval_data.csv')
    
    # Create a dummy environment just to initialize the agent's action space
    dummy_env = RULRetrainEnv(
        df=test_df,
        ref_df=train_df,
        base_model_path="results/best_rul_regressor_pipeline"
    )
    
    agent = TabularQAgent(
        env=dummy_env,
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
        agent.q_values = defaultdict(lambda: np.zeros(dummy_env.action_space.n), loaded_dict)
    except FileNotFoundError:
        print("Warning: q_table.npy not found in results/. Proceeding with empty Q-table.")
    
    print("\n--- Running Baseline (No Retraining) ---")
    baseline_info = run_pass(agent, test_df, train_df, "results/best_rul_regressor_pipeline", force_action_0=True)
    baseline_total_cost = baseline_info['episode_cost']
    
    print("\n--- Running Threshold Model ---")
    reg_model = load_model('results/best_rul_regressor_pipeline')
    threshold_agent = StaticRuleAgent()
    simulator = ThresholdSimulator(df=test_df, ref_df=train_df, reg_model=reg_model, agent=threshold_agent)
    threshold_result = simulator.run()
    threshold_total_cost = threshold_result.total_cost_traditional
    
    print("\n--- Running Trained RL Agent ---")
    rl_info = run_pass(agent, test_df, train_df, "results/best_rul_regressor_pipeline", force_action_0=False)
    rl_total_cost = rl_info['episode_cost']
    
    savings = threshold_total_cost - rl_total_cost
    
    # Save the evaluation results
    results = {
        "baseline_no_retrain_cost": float(baseline_total_cost),
        "threshold_model_cost": float(threshold_total_cost),
        "rl_agent_cost": float(rl_total_cost),
        "savings_vs_threshold": float(savings),
        "rl_retrains": int(rl_info['n_retrains'])
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBaseline (No Retrain) Cost: ${baseline_total_cost:,.0f}")
    print(f"Threshold Model Cost:       ${threshold_total_cost:,.0f} ({threshold_result.n_retrains} retrains)")
    print(f"RL Agent Cost:              ${rl_total_cost:,.0f} ({rl_info['n_retrains']} retrains)")
    print(f"Cost Savings (vs Threshold):${savings:,.0f}")
    print("Results saved to results/evaluation_results.json")
    
    # --- Generate Visualisations ---
    print("\nGenerating visualisations...")
    os.makedirs('results', exist_ok=True)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
        
    # Graph 1: Step Detail
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Panel: MAE
    x_steps = np.arange(len(rl_info['step_maes']))
    ax1.plot(x_steps, rl_info['step_maes'], marker='o', label='RL Agent MAE', color='#1f77b4', linewidth=2)
    
    # Add vertical lines for retrains
    retrain_steps = [i for i, a in enumerate(rl_info['step_actions']) if a == 1]
    for r in retrain_steps:
        ax1.axvline(x=r, color='#d62728', linestyle='--', alpha=0.7)
    
    # Proxy artist for legend
    if retrain_steps:
        ax1.plot([], [], color='#d62728', linestyle='--', label='RL Retrain Triggered')
        
    ax1.set_ylabel("Mean Absolute Error (hours)", fontsize=12)
    ax1.set_title("RL Agent Performance per Step (Day Window)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    
    # Bottom Panel: Step Costs
    ax2.bar(x_steps, rl_info['step_costs'], color='#ff7f0e', alpha=0.8, label='Step Cost Incurred')
    ax2.set_xlabel("Step Index (Day Window)", fontsize=12)
    ax2.set_ylabel("Cost ($)", fontsize=12)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/eval_step_detail.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 2: Cumulative Cost
    plt.figure(figsize=(12, 6))
    baseline_cumulative = np.cumsum(baseline_info['step_costs'])
    rl_cumulative = np.cumsum(rl_info['step_costs'])
    threshold_cumulative = threshold_result.batch_log['total_cost'].values
    
    # Align x-axes in case lengths differ slightly due to early termination
    min_len = min(len(baseline_cumulative), len(rl_cumulative), len(threshold_cumulative))
    x_common = np.arange(min_len)
    
    plt.plot(x_common, baseline_cumulative[:min_len], label='Baseline (No Retraining)', color='#7f7f7f', linestyle='--', linewidth=2)
    plt.plot(x_common, threshold_cumulative[:min_len], label='Threshold Model', color='#ff7f0e', linestyle='-.', linewidth=2)
    plt.plot(x_common, rl_cumulative[:min_len], label='RL Agent', color='#2ca02c', linewidth=3)
    
    plt.xlabel("Step Index (Day Window)", fontsize=12)
    plt.ylabel("Cumulative Cost ($)", fontsize=12)
    plt.title("Cumulative Cost Comparison: Baseline vs Threshold vs RL Agent", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 3: Threshold Timeline
    print("Generating threshold timeline on eval data...")
    simulator.plot_timeline(threshold_result, filename="results/threshold_timeline.png")
    
    print("Visualisations saved to results/")

if __name__ == '__main__':
    main()
