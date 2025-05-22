import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Preprocessing
from feature_engine.selection import RecursiveFeatureElimination


from darts import TimeSeries
from darts.models import XGBModel


# Data processing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Model evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Hyperparameter tuning
import optuna

'''
Best value: 0.4524017188563648, 
Best params: {'lags': 54, 
'lags_past_covariates': 0, 
'n_estimators': 183, 
'max_depth': 5, 
'learning_rate': 0.002653610630726174, 
'subsample': 0.7542897410934524, 
'colsample_bytree': 0.6778649320879656, 
'gamma': 1.1274317214894483, 
'lambda': 0.2204539291884646, 
'alpha': 0.3135293351461979, 
'early_stopping_rounds': 25}
'''


# Predicting next step's failure status
OUTPUT_CHUNK_LENGTH = 20

# Define objective function
def objective(trial):
    lags = trial.suggest_int("lags", 5, 60)
    lags_past_covariates = trial.suggest_int("lags_past_covariates", 0, 60) 
    output_chunk_length = OUTPUT_CHUNK_LENGTH # Predicting next step's failure status

    xgb_specific_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "lambda": trial.suggest_float("lambda", 1e-2, 5.0, log=True), # L2
        "alpha": trial.suggest_float("alpha", 1e-2, 5.0, log=True),   # L1
        "random_state": 42,
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 25)
    }


    min_series_len_needed = lags + output_chunk_length
    min_past_cov_len_needed = 0

    # Determine if past covariates with lags will be used
    use_past_covs_with_lags = train_all_features_as_past_cov_xgb is not None and lags_past_covariates > 0

    if use_past_covs_with_lags:
        min_past_cov_len_needed = lags_past_covariates + output_chunk_length

    if len(train_target_ts_for_xgb) < min_series_len_needed or \
       (use_past_covs_with_lags and len(train_all_features_as_past_cov_xgb) < min_past_cov_len_needed) or \
       len(val_target_ts_for_xgb) < output_chunk_length:
        return float('inf') # Infeasible trial

    model = XGBModel(
        lags=lags,
        lags_past_covariates=lags_past_covariates if use_past_covs_with_lags else None,
        output_chunk_length=output_chunk_length,
        add_encoders=None,
        **xgb_specific_params
    )

    try:
        model.fit(
            series=train_target_ts_for_xgb,
            past_covariates=train_all_features_as_past_cov_xgb if use_past_covs_with_lags else None,
            val_series=val_target_ts_for_xgb,
            val_past_covariates=val_all_features_as_past_cov_xgb if use_past_covs_with_lags else None
        )

        predictions_val = model.predict(
            n=len(val_target_ts_for_xgb),
            series=train_target_ts_for_xgb,
            past_covariates=train_all_features_as_past_cov_xgb if use_past_covs_with_lags else None
        )

        pred_continuous_val = predictions_val.values(copy=True).flatten()
        actual_vals_val = val_target_ts_for_xgb.values(copy=True).flatten()
        min_eval_len = min(len(actual_vals_val), len(pred_continuous_val))

        if min_eval_len < 1: return float('inf')

        actual_vals_eval = actual_vals_val[:min_eval_len]
        pred_continuous_eval = pred_continuous_val[:min_eval_len]

        if len(np.unique(actual_vals_eval)) < 2: return 1.0 

        score = roc_auc_score(actual_vals_eval, pred_continuous_eval)
        objective_value = 1.0 - score

        return objective_value if not (np.isnan(objective_value) or np.isinf(objective_value)) else float('inf')

    except Exception as e: 
        print(f"Optuna trial {trial.number} failed: {e}")
        return float('inf')



# Print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


if __name__ == "__main__":

    load_dotenv()
    DATASET = os.getenv("DATASET")

    # load dataset
    dataset = pd.read_csv(DATASET)

    # Description of dataset
    print("Description of dataset: \n", dataset.describe())

    # the data types
    print("Data Types: \n", dataset.info())

    # Check for missing values
    print(" Missing Values: \n", dataset.isna().sum())

    # Check Type counts
    print("Check Type Counts: \n", dataset['Type'].value_counts())

    # Drop redundant columns
    dataset = dataset.drop(["UDI", "Product ID", "Failure Type"], axis=1)

    # --- Feature Creation ---
    # Temperature difference between process and air
    dataset['temperature_difference'] = dataset['Process temperature [K]'] - dataset['Air temperature [K]']
    # Mechanical power using torque and rotational speed
    dataset['Mechanical Power [W]'] = np.round((dataset['Torque [Nm]']*dataset['Rotational speed [rpm]']* 2 * np.pi) / 60, 4)

    # Statistical Description
    print("Description of dataset: \n", dataset.describe().T)

    # Label encoding categorical variables
    dataset['Type'] = LabelEncoder().fit_transform(dataset['Type'])

    # --- Feature Elimination ---
    rfr_model = RandomForestRegressor()
    rfe = RecursiveFeatureElimination(estimator=rfr_model, scoring="r2", cv=3)

    # Prepare dataset
    X = dataset.drop(["Target"], axis=1)
    y = dataset["Target"]

    x_transformed = rfe.fit_transform(X, y)
    print("Dataset After Elimination: \n", x_transformed.head(5))

    # --- Data Preparation for Fine Tuning ---
    column_names_list = x_transformed.columns.tolist()

    main_feature_cols = [col for col in column_names_list if col != 'Type']

    # Convert DataFrame into Darts TimeSeries format
    all_features_as_past_cov_ts_full = TimeSeries.from_dataframe(x_transformed, value_cols=main_feature_cols)

    # The 'Target' column (0 or 1) is what we want to predict.
    target_ts_full = TimeSeries.from_series(y)

    # Define split proportions
    train_frac = 0.7  # 70% for training
    val_frac = 0.15   # 15% for validation / 15% for test

    temp_target_ts, test_target_ts = target_ts_full.split_before(train_frac + val_frac)
    train_target_ts_for_xgb, val_target_ts_for_xgb = temp_target_ts.split_before(train_frac / (train_frac + val_frac))

    temp_all_features_ts, test_all_features_as_past_cov_xgb = all_features_as_past_cov_ts_full.split_before(train_frac + val_frac)
    train_all_features_as_past_cov_xgb, val_all_features_as_past_cov_xgb = temp_all_features_ts.split_before(train_frac / (train_frac + val_frac))

    history_target_for_test_pred_xgb = temp_target_ts
    history_all_features_for_test_pred_xgb = temp_all_features_ts

    print(f"\n--- Data Splitting for XGBModel Optuna ---")
    print(f"Train target (XGB): {len(train_target_ts_for_xgb)}")
    print(f"Val target (XGB): {len(val_target_ts_for_xgb)}")
    print(f"Test target (XGB): {len(test_target_ts)}")
    print(f"Train past_cov (XGB): {len(train_all_features_as_past_cov_xgb)}, Components: {train_all_features_as_past_cov_xgb.n_components if train_all_features_as_past_cov_xgb else 'None'}")
    print(f"Val past_cov (XGB): {len(val_all_features_as_past_cov_xgb)}, Components: {val_all_features_as_past_cov_xgb.n_components if val_all_features_as_past_cov_xgb else 'None'}")

    # --- Start Optuna Hyperparameter Tuning for XGBModel ---
    print("\n--- Starting Optuna Hyperparameter Tuning for XGBModel ---")
    study_xgb = optuna.create_study(direction="minimize", study_name="XGB_FailurePrediction_Optuna")
    study_xgb.optimize(objective, n_trials=100, callbacks=[print_callback]) # Adjust n_trials as needed

    print("\n--- XGBModel Optuna Hyperparameter Tuning Complete ---")
    best_xgb_params = study_xgb.best_params
    best_xgb_value = study_xgb.best_value
    print(f"Best Score from Optuna (1 - ROC_AUC): {best_xgb_value:.4f} (ROC_AUC: {1.0 - best_xgb_value:.4f if best_xgb_value != float('inf') else 'N/A'})")
    print(f"Best Hyperparameters for XGBModel: {best_xgb_params}")

    # --- FINAL XGBMODEL TRAINING with Best Optuna Hyperparameters ---
    print("\n--- Retraining Final XGBModel with Best Optuna Hyperparameters on Combined Train + Validation Data ---")

    final_train_target_xgb = train_target_ts_for_xgb.append(val_target_ts_for_xgb)
    final_train_past_covariates_xgb = train_all_features_as_past_cov_xgb.append(val_all_features_as_past_cov_xgb)

    # Prepare final parameters
    final_model_darts_config_xgb = {
        "lags": best_xgb_params.pop("lags"),
        "lags_past_covariates": best_xgb_params.pop("lags_past_covariates"),
        "output_chunk_length": OUTPUT_CHUNK_LENGTH,
        "add_encoders": None
    }

    # best_xgb_params
    final_model_xgb_native_config_xgb = best_xgb_params.copy()

    # Remove for final fit on combined data
    final_model_xgb_native_config_xgb.pop("early_stopping_rounds", None)

    # Add the final model parameters
    final_xgb_model = XGBModel(**final_model_darts_config_xgb, **final_model_xgb_native_config_xgb)

    print("Fitting final tuned XGBModel on combined train+val data...")

    # Determine if past_covariates should be used for the final fit
    use_past_covs_final_fit = final_train_past_covariates_xgb is not None and \
                              final_model_darts_config_xgb.get('lags_past_covariates') is not None and \
                              final_model_darts_config_xgb['lags_past_covariates'] > 0

    final_xgb_model.fit(
        series=final_train_target_xgb,
        past_covariates=final_train_past_covariates_xgb if use_past_covs_final_fit else None
    )
    print("Final tuned XGBModel training complete.")

    # --- PREDICTION with Final Tuned XGBModel on Test Set ---
    if len(test_target_ts) > 0:
        print("\n--- Making Predictions on Test Set with Final Tuned XGBModel ---")
        try:
            use_past_covs_test_pred = history_all_features_for_test_pred_xgb is not None and \
                                      final_model_darts_config_xgb.get('lags_past_covariates') is not None and \
                                      final_model_darts_config_xgb['lags_past_covariates'] > 0
            
            xgb_test_predictions = final_xgb_model.predict(
                n=len(test_target_ts),
                series=history_target_for_test_pred_xgb, # History for target lags
                past_covariates=history_all_features_for_test_pred_xgb if use_past_covs_test_pred else None # History for covariate lags
            )

            xgb_test_pred_continuous = xgb_test_predictions.values(copy=True).flatten()
            actual_test_vals = test_target_ts.values(copy=True).flatten()

            min_len_test_eval = min(len(actual_test_vals), len(xgb_test_pred_continuous))
            actual_test_labels_eval = actual_test_vals[:min_len_test_eval]
            xgb_test_pred_continuous_eval = xgb_test_pred_continuous[:min_len_test_eval]

            if min_len_test_eval > 0:
                threshold = 0.5
                xgb_test_predicted_labels = (xgb_test_pred_continuous_eval >= threshold).astype(int)

                print(f"\n--- Evaluation of Final Tuned XGBModel on Test Set (Threshold = {threshold}) ---")
                test_accuracy_xgb = accuracy_score(actual_test_labels_eval, xgb_test_predicted_labels)
                test_roc_auc_xgb = roc_auc_score(actual_test_labels_eval, xgb_test_pred_continuous_eval) if len(np.unique(actual_test_labels_eval)) > 1 else 0.5
                
                print(f"Test Accuracy (XGB): {test_accuracy_xgb:.4f}")
                print(f"Test ROC AUC (XGB): {test_roc_auc_xgb:.4f}")
                print("\nTest Classification Report (XGB):")
                print(classification_report(actual_test_labels_eval, xgb_test_predicted_labels, zero_division=0))
                print("\nTest Confusion Matrix (XGB):")
                cm_xgb_test = confusion_matrix(actual_test_labels_eval, xgb_test_predicted_labels, labels=[0,1])
                print(cm_xgb_test)
                
            else:
                print("Not enough data in test set for XGBoost evaluation.")
        except Exception as e:
            print(f"Error during XGBModel prediction or evaluation on test set: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Test set for XGBoost is empty. Skipping prediction.")

    print("\n--- XGBoost Script End ---")
  