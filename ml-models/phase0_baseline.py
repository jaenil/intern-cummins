import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error
import os

def main():
    print("Loading test fleet data...")
    test_df = pd.read_csv('../data-generation/test_fleet_data.csv')
    
    print("Loading RUL regressor model...")
    model = load_model('results/best_rul_regressor_pipeline')
    
    print("Generating predictions...")
    predictions = predict_model(model, data=test_df)
    
    # PyCaret 3.x returns 'prediction_label', older versions return 'Label'
    pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else 'Label'
    
    y_true = predictions['RUL']
    y_pred = predictions[pred_col]
    
    baseline_mae = mean_absolute_error(y_true, y_pred)
    print(f"Baseline MAE: {baseline_mae:.2f}")
    
    # --- Generate Plots ---
    print("Generating results/model_baseline_performance.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Predicted vs True
    axes[0].scatter(y_true, y_pred, alpha=0.3)
    # diagonal line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[0].set_xlabel('True RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title('Predicted vs True RUL')
    
    # Panel 2: Error Distribution
    errors = y_pred - y_true
    sns.histplot(errors, bins=50, kde=True, ax=axes[1])
    axes[1].set_xlabel('Prediction Error (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution (Baseline)')
    
    plt.tight_layout()
    plt.savefig('results/model_baseline_performance.png', dpi=300)
    print("Saved results/model_baseline_performance.png")

if __name__ == "__main__":
    main()
