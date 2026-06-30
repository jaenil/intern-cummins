import pandas as pd
from pycaret.regression import setup, compare_models, predict_model, save_model

def main():
    train_df = pd.read_csv('train_fleet_data.csv')
    test_df = pd.read_csv('test_fleet_data.csv')
    
    ignore_cols = ['Engine_number', 'trip_id', 'RUL_class']
    
    print("Initializing PyCaret Regression Setup...")
    reg_setup = setup(
        data=train_df,
        target='RUL',
        test_data=test_df,
        ignore_features=ignore_cols,
        numeric_imputation='mean',
        session_id=42,
        verbose=True,
        index=False
    )
    
    print("Comparing models...")
    best_model = compare_models(sort='MAE')
    print(f"Best Regressor Model found: {best_model}")
    
    print("\nPredicting on test data...")
    predictions = predict_model(best_model)
    print(predictions.head())
    
    print("\nSaving regression model...")
    save_model(best_model, 'best_rul_regressor_pipeline')
    print("Saved to best_rul_regressor_pipeline.pkl")

if __name__ == '__main__':
    main()
