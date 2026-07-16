# RL Agent Configuration

## 1. Environment Details
| Parameter | Description |
|-----------|-------------|
| **Context** | Continuous data stream (31,454 trips) split 80% train / 20% test. |
| **Batch Size** | 2,000 trips |
| **Max Batches** | 16 per episode |

## 2. State Space
The environment calculates a 6-feature vector which the agent discretizes into 3 bins (Low/Medium/High) for the Q-Table. 

**Continuous Features:**
- **Drift**: Normalized energy distance.
- **MAE**: Normalized mean absolute error.
- **Staleness**: Batches since last retrain.
- **MAE % Change**: Direction of error.
- **MAE Slope**: Trend of error over last 5 values.
- **Cumulative Cost**: Normalized cost.

**Discretized Q-Table States (27 Total States):**
- **Drift (`D_drift`)**, **Error (`D_error`)**, **Staleness (`D_age`)**.

## 3. Action Space
| Action | Description | Cost Incurred |
|--------|-------------|---------------|
| **0 (Do Nothing)** | Continue using the existing model. | Operational Error Costs |
| **1 (Retrain)** | Trigger PyCaret retraining on all data seen so far. | Operational Error Costs + `TEC` ($25,000) |

## 4. Value Function & Reward
- **Reward**: Dynamic Difference Reward (Advantage): `R = pre_action_cost - post_action_cost - retrain_cost`.
- **Value Function**: Updated via Bellman Equation: `Q(s,a) <- Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]`.
- **Algorithm**: Tabular Q-Learning using an ε-greedy exploration strategy.

## 5. Configuration Values
| Category | Parameter | Value |
|----------|-----------|-------|
| **Cost Model** | Under-Prediction Cost (`UPC`) | $1 / hr |
| | Degradation Cost (`DC`) | $2 / hr |
| | Retrain Cost (`TEC`) | $25,000 |
| | Infrastructure Cost (`CIC`) | $10 / hr |
| **RL Hyperparameters** | Learning Rate (`ALPHA_RL`) | 0.1 |
| | Discount Factor (`GAMMA`) | 0.95 |
| | `EPSILON_START` / `END` | 0.5 / 0.05 |
| | `EPSILON_DECAY` | 0.023 (per episode) |
| | `N_EPISODES` | 30 |
| **Environment Limits** | Drift Limit (`DRIFT_MAX`) | 5843.93 |
| | MAE Limit (`MAE_MAX`) | 471.48 |
| **Discretization Bins<br>(Low / Med / High)**| **Drift** | < 0.18 / < 0.40 / ≥ 0.40 |
| | **Error** | < 0.39 / < 0.61 / ≥ 0.61 |
| | **Age** | < 0.20 / < 0.50 / ≥ 0.50 |
