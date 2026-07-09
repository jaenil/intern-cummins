import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model

def main():
    train_df = pd.read_csv('../data-generation/train_fleet_data.csv')
    test_df = pd.read_csv('../data-generation/test_fleet_data.csv')
    
    ignore_cols = ['Engine_number', 'trip_id', 'RUL']
    clf_setup = setup(
        data=train_df,
        target='RUL_class',
        test_data=test_df,
        ignore_features=ignore_cols,
        numeric_imputation='mean',
        session_id=42,
        verbose=True,
        index=False
    )
    
    print("Comparing models")
    best_model = compare_models(include=['lightgbm', 'lr', 'dt'], sort='Accuracy') # Sorting by Accuracy
    print(f"Best Classifier Model : {best_model}")
    
    print("\nPredicting on test data")
    predictions = predict_model(best_model)
    print(predictions.head())
    
    print("\nSaving classification model...")
    save_model(best_model, 'best_rul_classifier_pipeline')
    print("Saved to best_rul_classifier_pipeline.pkl")

if __name__ == '__main__':
    main()