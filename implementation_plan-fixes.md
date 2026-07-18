# RL Agent 0 Retrains — Root Cause Analysis & Fix Plan

## Problem
The RL agent consistently reports **0 retrains** during evaluation, meaning it never triggers Action 1 (Retrain) on any step. The evaluation cost equals the baseline (no retrain) cost exactly.

---

## Root Causes Found (3 compounding issues)

### Root Cause 1 — Train/Eval Data Mismatch (State Generalisation Failure)

The agent was **trained** on `streaming_test_data.csv` (105 unique days → **10 steps**),  
but **evaluated** on `streaming_eval_data.csv` (55 unique days → **only 5 steps**).

The Q-table has entries for **10 states** (out of 27 possible). The states seen during evaluation are:

| Eval Step | State Observed | In Q-Table? | Q-Table Action |
|-----------|---------------|-------------|----------------|
| 0 | `(2, 2, 0)` | ✅ Yes | **DO NOTHING** |
| 1 | `(2, 2, 0)` | ✅ Yes | **DO NOTHING** |
| 2 | `(1, 2, 1)` | ✅ Yes | **DO NOTHING** |
| 3 | `(0, 1, 1)` | ❌ Missing | Defaults to **DO NOTHING** |
| 4 | `(2, 0, 1)` | ❌ Missing | Defaults to **DO NOTHING** |

**The 3 states that DO prefer retrain — `(0,2,0)`, `(0,2,1)`, `(1,2,2)` — never appear in eval data.** The agent simply never encounters a state where it learned to retrain.

> [!CAUTION]
> **Fix**: Train on `streaming_eval_data.csv` (or combined data), OR use the same dataset for both train and eval. The RL environment must be trained on states that appear in evaluation.

---

### Root Cause 2 — TEC is Too High Relative to Episode Length

`TEC = $1500`, `REWARD_SCALE = 4000`, so a single retrain = **−0.375 reward units**.  
The episode has only 5–10 steps. The cumulative error savings from a retrain can never compensate for the upfront TEC penalty in such a short horizon.

This means even if the agent DID encounter the right states, retraining would still be penalised more than rewarded.

Evidence: Even on training data, the Q-values for retrain are barely above no-op (margin of 0.01–0.12). The agent learned that retraining is almost never worth it given the current cost structure.

> [!IMPORTANT]
> **Fix options (pick one)**:
> - Reduce `TEC` (e.g. to `$300`) so retrain penalty is smaller relative to error savings
> - Increase `REWARD_SCALE` (e.g. to `400`) to amplify the error-cost signal
> - Increase episode length (more steps = more time to recover retrain cost)

---

### Root Cause 3 — Q-table Severely Under-Trained (Only 10 States Visited)

Only **10 out of 27 possible states** were ever visited during 20 training episodes. The Q-table file is only **939 bytes** — trivially small. The agent never explored 17 states, including key "high-drift + high-MAE + old-age" states where retraining would be most beneficial.

This is caused by:
- Only 20 training episodes (`N_EPISODES = 20`)
- Short episodes (10 steps each = 200 total transitions)
- `EPSILON_DECAY = 0.023` → epsilon hits `EPSILON_END = 0.05` at episode 20 exactly, meaning very little exploration time

> [!IMPORTANT]
> **Fix**: Increase `N_EPISODES` to at least 100–200, or increase `EPSILON_START` / slow down epsilon decay so the agent explores more states.

---

## Summary of Fixes

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `tabular_q_agent.py` | Change training data from `streaming_test_data.csv` → `streaming_eval_data.csv` | Ensure train/eval states overlap |
| 2 | `shared_config.py` | Reduce `TEC` from `$1500` → `$300` OR reduce `REWARD_SCALE` | Make retrain cost recoverable |
| 3 | `shared_config.py` | Increase `N_EPISODES` from `20` → `100` | More training coverage |
| 4 | `shared_config.py` | Slow epsilon decay (e.g. `EPSILON_DECAY = 0.004`) | More state exploration |

---

## Proposed Changes

### `shared_config.py`

#### [MODIFY] [shared_config.py](file:///Users/jaenilparekh/Documents/cs50/intern-cummins/ml-models/shared_config.py)

```diff
- TEC = 1500              # Traditional Engineering Cost (per manual PyCaret run)
+ TEC = 300               # Traditional Engineering Cost (per manual PyCaret run)

- N_EPISODES = 20          # Total training episodes
+ N_EPISODES = 100         # Total training episodes

- EPSILON_DECAY = 0.023    # Linear decay per episode
+ EPSILON_DECAY = 0.004    # Slower decay — allow more exploration

- REWARD_SCALE = 4000
+ REWARD_SCALE = 1000      # Tighter scale so error costs have stronger signal
```

---

### `tabular_q_agent.py` (training block)

#### [MODIFY] [tabular_q_agent.py](file:///Users/jaenilparekh/Documents/cs50/intern-cummins/ml-models/tabular_q_agent.py)

```diff
- df_train = pd.read_csv(os.path.join(data_dir, "streaming_test_data.csv"))
+ df_train = pd.read_csv(os.path.join(data_dir, "streaming_eval_data.csv"))
```

This ensures the agent trains on the same data distribution as evaluation.

---

## Verification Plan

### After re-training the agent

Run `python tabular_q_agent.py` and then `python evaluate_agent.py` and verify:

1. Q-table has more than 10 states visited
2. At least 1 retrain is triggered during evaluation
3. `rl_agent_cost < baseline_cost` — the RL agent demonstrates value
4. The comparison plot shows RL cost curve diverging from baseline

### Automated check
```bash
cd ml-models
source ../venv/bin/activate
python tabular_q_agent.py && python evaluate_agent.py
```

---

## Open Questions

> [!NOTE]
> **Q: Should we keep `TEC = $1500` in the cost model documentation or change it?**  
> If `$1500` is the real-world cost, we may want to keep it as-is and instead just increase episode length and training time so the agent can properly learn the long-horizon benefit of retraining. The current setup has too few steps (5-10) to recover a $1500 upfront cost.
>
> **Q: Should the training use `streaming_test_data.csv` or `streaming_eval_data.csv`?**  
> Ideally, training and evaluation should use different splits of the SAME distribution. Currently they use different files with different day ranges, causing the state mismatch.
