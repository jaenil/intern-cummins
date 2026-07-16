import os
import sys
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm
import logging

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_config import (
    ALPHA_RL, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, N_EPISODES,
    DRIFT_BIN_THRESHOLD_1, DRIFT_BIN_THRESHOLD_2,
    ERROR_BIN_THRESHOLD_1, ERROR_BIN_THRESHOLD_2,
    AGE_BIN_THRESHOLD_1, AGE_BIN_THRESHOLD_2
)
from rul_retrain_env import RULRetrainEnv

class TabularQAgent:
    """
    Tabular Q-Learning agent for RULRetrainEnv.

    Maintains a Q-table over a discretised 3-bin state space (27 states)
    (drift × error × age) and 2 actions (Do Nothing / Retrain).
    Uses ε-greedy exploration with linear epsilon decay.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        self.env = env
        # Q-table: defaultdict creates zeros for unseen (state, action) pairs
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []  # TD errors — used for convergence diagnostics

    def discretize(self, obs: np.ndarray) -> tuple:
        """Map 6-feature continuous obs → 3-bin discrete state.
        
        Only drift (f1), normalised MAE (f2), and staleness (f4) are used.
        """
        f1_drift = obs[0]
        f2_mae = obs[1]
        f4_stale = obs[3]

        if f1_drift < DRIFT_BIN_THRESHOLD_1:
            drift_bin = 0
        elif f1_drift < DRIFT_BIN_THRESHOLD_2:
            drift_bin = 1
        else:
            drift_bin = 2
            
        if f2_mae < ERROR_BIN_THRESHOLD_1:
            error_bin = 0
        elif f2_mae < ERROR_BIN_THRESHOLD_2:
            error_bin = 1
        else:
            error_bin = 2

        if f4_stale < AGE_BIN_THRESHOLD_1:
            age_bin = 0
        elif f4_stale < AGE_BIN_THRESHOLD_2:
            age_bin = 1
        else:
            age_bin = 2

        return (drift_bin, error_bin, age_bin)

    def get_action(self, state: tuple) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()       # Explore
        return int(np.argmax(self.q_values[state]))     # Exploit

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple,
    ):
        """Bellman update — the core Q-learning step."""
        # If episode terminated, no future value exists
        future_q = (not terminated) * np.max(self.q_values[next_state])

        # TD target (what we now believe the value should be)
        target = reward + self.discount_factor * future_q

        # TD error (how wrong our current estimate was)
        td_error = target - self.q_values[state][action]

        # Move estimate toward target
        self.q_values[state][action] += self.lr * td_error

        self.training_error.append(td_error)

    def decay_epsilon(self):
        """Linear epsilon decay — call once per episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str):
        """Persist Q-table as numpy array."""
        np.save(path, dict(self.q_values))

    def load(self, path: str):
        """Restore Q-table from saved numpy array."""
        loaded = np.load(path, allow_pickle=True).item()
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n), loaded)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data-generation")
    df = pd.read_csv(os.path.join(data_dir, "streaming_test_data.csv"))
    train_split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:train_split_idx].reset_index(drop=True)
    ref_df = pd.read_csv(os.path.join(data_dir, "train_fleet_data.csv"))

    # Register environment if not already registered (to use with gym.make wrappers, though we can use it directly with RecordEpisodeStatistics)
    gym.register(
        id="RULRetrain-v0",
        entry_point="rul_retrain_env:RULRetrainEnv",
        max_episode_steps=20,
    )
    
    # Initialize environment with TRAIN split only
    # Note: to use custom args with gym.make we pass them as kwargs
    env = gym.make(
        "RULRetrain-v0", 
        df=df_train,
        ref_df=ref_df,
        base_model_path=os.path.join(os.path.dirname(__file__), "results/best_rul_regressor_pipeline")
    )

    # Wrap env to record episode statistics automatically
    env = RecordEpisodeStatistics(env)

    agent = TabularQAgent(
        env=env,
        learning_rate=ALPHA_RL,
        initial_epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=EPSILON_END,
        discount_factor=GAMMA,
    )

    logging.info(f"Starting Q-Learning training for {N_EPISODES} episodes...")
    for episode in tqdm(range(N_EPISODES)):
        obs, info = env.reset()
        state = agent.discretize(obs)
        done = False

        while not done:
            action = agent.get_action(state)                          # ε-greedy
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize(next_obs)

            agent.update(state, action, reward, terminated, next_state)

            state = next_state
            done = terminated or truncated

        agent.decay_epsilon()                                         # reduce ε after each episode
        
        # Log episode stats (available in info after episode ends)
        if "episode" in info:
            ep = info["episode"]
            logging.info(
                f"Ep {episode:03d} | reward={ep['r']:,.0f} | "
                f"len={ep['l']} | t={ep['t']:.1f}s | ε={agent.epsilon:.3f}"
            )

    # Save Q-table
    output_qtable_path = os.path.join(os.path.dirname(__file__), "results/q_table.npy")
    agent.save(output_qtable_path)
    logging.info(f"Q-table saved to {output_qtable_path}")
    
    # Save training stats
    stats = {
        "return_queue": [float(r) for r in env.return_queue],
        "length_queue": [int(l) for l in env.length_queue],
        "training_error": [float(e) for e in agent.training_error]
    }
    output_stats_path = os.path.join(os.path.dirname(__file__), "results/training_stats.json")
    with open(output_stats_path, "w") as f:
        json.dump(stats, f)
    logging.info(f"Training stats saved to {output_stats_path}")
    
    env.close()
