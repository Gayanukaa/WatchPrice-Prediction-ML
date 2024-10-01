from train import TrainModel
import config

def main():
    # Initialize the training process
    model_trainer = TrainModel()

    # Step 1: Load and split the data
    print("Reading and preprocessing the dataset...")
    model_trainer._read_df()
    print(f"Data loaded from {config.DATA_PATH}")

    model_trainer._split_data()
    print("Data split into training and testing sets.")

    # Step 2: Train a basic XGB model
    print("Training basic XGB model...")
    xgb_result = model_trainer._train_xgb()
    print(f"Basic XGB model trained with R2 scores: {xgb_result['score']}")
    print(f"Mean R2 score: {xgb_result['mean']:.4f}, Standard Deviation: {xgb_result['std']:.4f}")

    # Step 3: Perform hyperparameter tuning for XGB model
    print("Starting hyperparameter tuning for XGB...")
    r2_score, best_params, best_model = model_trainer._hyperparameter_train_xgb()
    print(f"Hyperparameter tuning completed. Best R2 score: {r2_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    # Step 4: Save the best model
    model_trainer._save_model(best_model)
    print(f"Best model saved successfully at {config.MODEL_SAVE_NAME}")

if __name__ == "__main__":
    main()