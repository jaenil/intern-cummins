import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('train_fleet_data.csv')
    test_df = pd.read_csv('test_fleet_data.csv')
    
    # Ignore identifiers and the regression target to prevent data leakage
    ignore_cols = ['Engine_number', 'trip_id', 'RUL']
    
    print("Initializing PyCaret Classification Setup...")
    # numeric_imputation='mean' handles missing values (including the fully empty columns)
    # They will be safely dropped or imputed now, but utilized correctly once you populate them later.
    clf_setup = setup(
        data=train_df,
        target='RUL_class',
        test_data=test_df,
        ignore_features=ignore_cols,
        numeric_imputation='mean',
        session_id=42,
        verbose=True
    )
    
    print("Comparing models to find the best classifier...")
    best_model = compare_models(sort='Accuracy') # Sorting by Accuracy
    print(f"Best Classifier Model found: {best_model}")
    
    print("\nPredicting on unseen test data...")
    predictions = predict_model(best_model)
    print(predictions.head())
    
    print("\nSaving best classification model...")
    save_model(best_model, 'best_rul_classifier_pipeline')
    print("Saved to best_rul_classifier_pipeline.pkl")

if __name__ == '__main__':
    main()
