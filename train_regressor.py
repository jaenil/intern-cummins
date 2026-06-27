import pandas as pd
from pycaret.regression import setup, compare_models, predict_model, save_model

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('train_fleet_data.csv')
    test_df = pd.read_csv('test_fleet_data.csv')
    
    # Ignore identifiers and the classification target to prevent data leakage
    ignore_cols = ['Engine_number', 'trip_id', 'RUL_class']
    
    print("Initializing PyCaret Regression Setup...")
    # numeric_imputation='mean' handles missing values (including the fully empty columns)
    # They will be safely dropped or imputed now, but utilized correctly once you populate them later.
    reg_setup = setup(
        data=train_df,
        target='RUL',
        test_data=test_df,
        ignore_features=ignore_cols,
        numeric_imputation='mean',
        session_id=42,
        verbose=True
    )
    
    print("Comparing models to find the best regressor...")
    best_model = compare_models(sort='MAE') # Sorting by Mean Absolute Error
    print(f"Best Regressor Model found: {best_model}")
    
    print("\nPredicting on unseen test data...")
    predictions = predict_model(best_model)
    print(predictions.head())
    
    print("\nSaving best regression model...")
    save_model(best_model, 'best_rul_regressor_pipeline')
    print("Saved to best_rul_regressor_pipeline.pkl")

if __name__ == '__main__':
    main()
