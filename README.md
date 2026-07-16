# Sensor Prognostic Model Pipeline

This project builds a prognostic framework to predict the Remaining Useful Life (RUL) of engines based on high-frequency, synthetic sensor data with injected drift. 

It contains a data synthesis script that generates operational data and two PyCaret model training scripts (one for continuous RUL regression and one for discrete health-state classification).

## 🛠 Setup Instructions

This project requires a Python virtual environment to manage dependencies locally (preventing conflicts with global system packages).

1. **Open your terminal** and navigate to the project directory:
   ```bash
   cd /Users/jaenilparekh/Documents/cs50/intern-cummins
   ```

2. **Activate the Virtual Environment**:
   If you have already created the `venv` folder, activate it by running:
   ```bash
   source venv/bin/activate
   ```
   *(Note: You will need to run this `source` command every time you open a new terminal window to work on this project).*

3. **Install Dependencies** (if not already installed):
   With the virtual environment active, run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the Pipeline

The scripts must be executed in a specific order, as the models depend on the generated datasets. **Ensure your virtual environment is activated before running any scripts.**

### Step 1: Generate the Synthetic Dataset
Run the data generation script to synthesize irregular time-series data for 100 engines. This script injects an upward sensor drift bias over simulated operating hours and automatically maps continuous RUL into 4 distinct classes (`perfect`, `good`, `bad`, `worse`).

```bash
cd data-generation
python3 data-generation.py
cd ..
```
*Output: `train_fleet_data.csv` and `test_fleet_data.csv`*

---
**Note:** The following scripts (Steps 2-7) must be run from inside the `ml-models` directory:
```bash
cd ml-models
```

### Step 2: Train the Regressor Model (Continuous RUL)
This script uses PyCaret to automatically build, compare, and select the best regression model (e.g., Random Forest) for predicting exact Remaining Useful Life (RUL) hours.

```bash
python3 train_regressor.py
```
*Output: Evaluates model MAE and saves `best_rul_regressor_pipeline.pkl`*

### Step 3: Train the Classifier Model (Health Classes)
This script uses PyCaret to automatically build, compare, and select the best classification model for categorizing the engine's current state into discrete labels based on the RUL thresholds.

```bash
python3 train_classifier.py
```
*Output: Evaluates model Accuracy and saves `best_rul_classifier_pipeline.pkl`*

---

### Step 4: Validate the RL Environment
This script checks the custom Gymnasium environment for compliance with the standard API.

```bash
python3 validate_env.py
```

### Step 5: Run Baseline Benchmark
This script evaluates the default PyCaret model's performance on the continuous data stream without retraining.

```bash
python3 baseline_benchmark.py
```

### Step 6: Train the RL Agent
This script trains a Tabular Q-Learning agent to optimize retraining decisions over multiple episodes.

```bash
python3 tabular_q_agent.py
```

### Step 7: Evaluate the RL Agent
This script evaluates the trained agent's retraining policy against the test data stream.

```bash
python3 evaluate_agent.py
```

---