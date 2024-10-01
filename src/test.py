from train import TrainModel
import config

def main():
    model_trainer = TrainModel()

    print("Reading and preprocessing the dataset...")
    model_trainer._read_df()
    print(f"Data loaded from {config.DATA_PATH}")

    model_trainer._split_data()
    print("Data split into training and testing sets.")

    print("\n ------------------------------------ \n ")

    #Train a basic models
    #Decision Tree model
    print("Training basic Decision Tree model...")
    dt_result = model_trainer._train_decision_tree()
    print(f"Basic Decision Tree model trained with R2 scores: {dt_result['score']}")
    print(f"Mean R2 score: {dt_result['mean']:.4f}, Standard Deviation: {dt_result['std']:.4f}")

    print("\n ------------------------------------ \n ")

    #Random Forest model
    print("Training basic Random Forest model...")
    rf_result = model_trainer._train_random_forest()
    print(f"Basic Random Forest model trained with R2 scores: {rf_result['score']}")
    print(f"Mean R2 score: {rf_result['mean']:.4f}, Standard Deviation: {rf_result['std']:.4f}")

    print("\n ------------------------------------ \n ")

    #Decision XGB model
    print("Training basic XGB model...")
    xgb_result = model_trainer._train_xgb()
    print(f"Basic XGB model trained with R2 scores: {xgb_result['score']}")
    print(f"Mean R2 score: {xgb_result['mean']:.4f}, Standard Deviation: {xgb_result['std']:.4f}")

    print("\n ------------------------------------ \n ")

    #Hyperparameter tuning for XGB model
    print("Starting hyperparameter tuning for XGB...")
    r2_score, best_params, best_model = model_trainer._hyperparameter_train_xgb()
    print(f"Hyperparameter tuning completed. Best R2 score: {r2_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    print("\n ------------------------------------ \n ")

    #Save the best model
    model_trainer._save_model(best_model)
    print(f"Best model saved successfully at {config.MODEL_SAVE_NAME}")

if __name__ == "__main__":
    main()