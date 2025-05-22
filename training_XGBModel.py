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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Weight and Biases
import wandb

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

    train_val_target_ts, test_target_ts = target_ts_full.split_before(train_frac + val_frac)
    train_val_all_features_ts, test_all_features_as_past_cov_xgb = all_features_as_past_cov_ts_full.split_before(train_frac + val_frac)
    
    # For Optuna, you would have split train_val_target_ts and train_val_all_features_ts further into train and val.
    # For this final training script, we use the combined train_val sets.

    print(f"\n--- Data for Final XGBModel Training and Evaluation ---")
    print(f"Combined Train+Validation target (XGB): {len(train_val_target_ts)}")
    print(f"Test target (XGB): {len(test_target_ts)}")
    print(f"Combined Train+Validation past_cov (XGB): {len(train_val_all_features_ts)}, Components: {train_val_all_features_ts.n_components if train_val_all_features_ts else 'None'}")
    print(f"Test past_cov (XGB): {len(test_all_features_as_past_cov_xgb)}, Components: {test_all_features_as_past_cov_xgb.n_components if test_all_features_as_past_cov_xgb else 'None'}")

    # --- Best Hyperparameters from Optuna (Paste your actual best params here) ---
    best_xgb_params = {
        'lags': 54,
        'lags_past_covariates': 54,
        'n_estimators': 183,
        'max_depth': 5,
        'learning_rate': 0.002653610630726174,
        'subsample': 0.7542897410934524,
        'colsample_bytree': 0.6778649320879656,
        'gamma': 1.1274317214894483,
        'lambda': 0.2204539291884646,
        'alpha': 0.3135293351461979,
        # 'early_stopping_rounds' is not needed for the final fit on combined data
    }
    print(f"\nUsing Best Hyperparameters for Final XGBModel: {best_xgb_params}")

    # --- Initialize W&B Run ---
    wandb_project_name = "predictive-maintenance" # Specific project for these runs
    wandb_run_name = f"XGB_Final_{best_xgb_params['lags']}_est{best_xgb_params['n_estimators']}"
    
    run = wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        config=best_xgb_params, # Log Optuna's best hyperparameters
        job_type="final-xgb-training"
    )
    print(f"W&B Run Initialized: {run.url}")


    # --- FINAL XGBMODEL TRAINING with Best Optuna Hyperparameters & W&B Logging ---
    print("\n--- Training Final XGBModel with Best Hyperparameters (Tracked by W&B) ---")

    # Prepare final model Darts-specific parameters
    final_model_darts_config_xgb = {
        "lags": best_xgb_params.pop("lags"), # Remove as it's passed to constructor
        "lags_past_covariates": best_xgb_params.pop("lags_past_covariates"), # Remove
        "output_chunk_length": 20,
        "add_encoders": None
    }
    # The rest of best_xgb_params are native XGBoost parameters
    final_model_xgb_native_config_xgb = best_xgb_params.copy()
    if 'random_state' not in final_model_xgb_native_config_xgb:
        final_model_xgb_native_config_xgb['random_state'] = 42

    # Add Darts specific params back to the config that will be logged if they were popped
    # Or, log the full final_model_params dictionary created below
    final_model_full_params_for_log = {**final_model_darts_config_xgb, **final_model_xgb_native_config_xgb}
    wandb.config.update({"final_model_darts_config": final_model_darts_config_xgb,
                         "final_model_xgb_native_config": final_model_xgb_native_config_xgb})


    final_xgb_model = XGBModel(**final_model_darts_config_xgb, **final_model_xgb_native_config_xgb)

    print("Fitting final tuned XGBModel on combined train+val data...")
    use_past_covs_final_fit = train_val_all_features_ts is not None and \
                              final_model_darts_config_xgb.get('lags_past_covariates') is not None and \
                              final_model_darts_config_xgb['lags_past_covariates'] > 0
    if use_past_covs_final_fit and len(train_val_all_features_ts) == 0:
        print("Warning: `lags_past_covariates` is set, but combined past covariate series is empty. Fitting without.")
        use_past_covs_final_fit = False


    final_xgb_model.fit(
        series=train_val_target_ts,
        past_covariates=train_val_all_features_ts if use_past_covs_final_fit else None
    )
    print("Final tuned XGBModel training complete.")

    # --- Save the Trained Final Model and Log as W&B Artifact ---
    model_save_dir = "./darts_models" # Directory to save the model
    final_model_filename = f"final_xgb_model_{run.id}.pt" # Include run ID for uniqueness
    final_model_path = os.path.join(model_save_dir, final_model_filename)
    
    final_xgb_model.save(final_model_path)
    print(f"Final XGBModel saved to {final_model_path}")

    model_artifact = wandb.Artifact(
        name=f"xgb-failure-predictor-{run.id}", # Unique artifact name
        type="model",
        description="Final XGBModel trained with best Optuna hyperparameters for failure prediction.",
        metadata=final_model_full_params_for_log # Log the full parameters used
    )
    model_artifact.add_file(final_model_path)
    run.log_artifact(model_artifact)
    print(f"Logged XGBModel artifact to W&B: {model_artifact.name}")


    # --- PREDICTION with Final Tuned XGBModel on Test Set ---
    if len(test_target_ts) > 0:
        print("\n--- Making Predictions on Test Set with Final Tuned XGBModel ---")
        try:
            # Determine if past covariates were used during training and should be used for prediction
            use_past_covs_test_pred = final_model_darts_config_xgb.get('lags_past_covariates') is not None and \
                                      final_model_darts_config_xgb['lags_past_covariates'] > 0

            print(f"Predicting n={len(test_target_ts)} steps.")
            print(f"History series for lags (train_val_target_ts) length: {len(train_val_target_ts)}")
            if use_past_covs_test_pred:
                # This is the key: use the full feature set for past_covariates in predict
                past_covariates_for_prediction_call = all_features_as_past_cov_ts_full
                print(f"Past covariates for predict (all_features_as_past_cov_ts_full) length: {len(past_covariates_for_prediction_call)}")
            else:
                past_covariates_for_prediction_call = None
                print("Not using past_covariates for prediction as lags_past_covariates was 0 or None.")


            xgb_test_predictions = final_xgb_model.predict(
                n=len(test_target_ts),
                series=train_val_target_ts,                 # History of TARGET for its lags
                past_covariates=past_covariates_for_prediction_call # <<<< CORRECTED: Full history AND future known values of COVARIATES
            )

            xgb_test_pred_continuous = xgb_test_predictions.values(copy=True).flatten()
            actual_test_vals = test_target_ts.values(copy=True).flatten()

            min_len_test_eval = min(len(actual_test_vals), len(xgb_test_pred_continuous))
            actual_test_labels_for_eval = actual_test_vals[:min_len_test_eval]
            xgb_pred_continuous_for_eval = xgb_test_pred_continuous[:min_len_test_eval]

            if min_len_test_eval > 0:
                threshold = 0.5
                xgb_test_predicted_labels_for_eval = (xgb_pred_continuous_for_eval >= threshold).astype(int)

                print(f"\n--- Evaluation of Final Tuned XGBModel on Test Set (Threshold = {threshold}) ---")
                test_accuracy_xgb = accuracy_score(actual_test_labels_for_eval, xgb_test_predicted_labels_for_eval)
                test_roc_auc_xgb = roc_auc_score(actual_test_labels_for_eval, xgb_pred_continuous_for_eval) if len(np.unique(actual_test_labels_for_eval)) > 1 else 0.5
                
                print(f"Test Accuracy (XGB): {test_accuracy_xgb:.4f}")
                print(f"Test ROC AUC (XGB): {test_roc_auc_xgb:.4f}")
                
                report_dict_xgb = classification_report(
                    actual_test_labels_for_eval,
                    xgb_test_predicted_labels_for_eval,
                    zero_division=0,
                    output_dict=True,
                    labels=[0, 1], # Assuming 0 and 1 are your class labels
                    target_names=["Good", "Failure"] # Optional: if you want named keys
                )
                print("\nTest Classification Report (XGB):")
                # Print the string version for console
                print(classification_report(
                    actual_test_labels_for_eval,
                    xgb_test_predicted_labels_for_eval,
                    zero_division=0,
                    labels=[0,1],
                    target_names=["Good", "Failure"]
                ))
                
                print("\nTest Confusion Matrix (XGB):")
                cm_xgb_test = confusion_matrix(
                    actual_test_labels_for_eval,
                    xgb_test_predicted_labels_for_eval,
                    labels=[0,1] # Ensure consistent labeling with report
                )
                print(cm_xgb_test)
                
                # --- Log Evaluation Metrics and Plots to W&B ---
                if wandb.run:
                    metrics_to_log_xgb = {
                        "xgb_test_accuracy": test_accuracy_xgb,
                        "xgb_test_roc_auc": test_roc_auc_xgb,
                        "xgb_threshold": threshold,
                        "xgb_test_precision_Failure": report_dict_xgb.get("Failure", {}).get("precision", 0.0),
                        "xgb_test_recall_Failure": report_dict_xgb.get("Failure", {}).get("recall", 0.0),
                        "xgb_test_f1_Failure": report_dict_xgb.get("Failure", {}).get("f1-score", 0.0),
                        "xgb_test_support_Failure": report_dict_xgb.get("Failure", {}).get("support", 0),
                        "xgb_test_precision_Good": report_dict_xgb.get("Good", {}).get("precision", 0.0),
                        "xgb_test_recall_Good": report_dict_xgb.get("Good", {}).get("recall", 0.0),
                        "xgb_test_f1_Good": report_dict_xgb.get("Good", {}).get("f1-score", 0.0),
                        "xgb_test_support_Good": report_dict_xgb.get("Good", {}).get("support", 0),
                        "xgb_test_macro_avg_f1": report_dict_xgb.get("macro avg", {}).get("f1-score", 0.0)
                    }
                    wandb.log(metrics_to_log_xgb)

                    fig_cm_xgb, ax_cm_xgb = plt.subplots(figsize=(6, 5))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb_test, display_labels=["Good", "Failure"]) # Use names if defined
                    disp.plot(ax=ax_cm_xgb, cmap=plt.cm.Blues); ax_cm_xgb.set_title("XGBoost Test Confusion Matrix")
                    wandb.log({"confusion_matrix": wandb.Image(fig_cm_xgb)})
                    plt.close(fig_cm_xgb)


                    # Plotting the actual vs predicted labels
                    x_min_display, x_max_display = 9000, 9050 
                    fig_preds, ax_preds = plt.subplots(figsize=(15, 7))
                    time_index_full_test = test_target_ts.time_index
                    actual_labels_np = np.array(actual_test_labels_for_eval)
                    predicted_labels_np = np.array(xgb_test_predicted_labels_for_eval)
                    time_index_np = time_index_full_test[:min_len_test_eval].to_numpy()
                    display_mask = (time_index_np >= x_min_display) & (time_index_np <= x_max_display)
                    time_index_display, actual_labels_display, predicted_labels_display = time_index_np[display_mask], actual_labels_np[display_mask], predicted_labels_np[display_mask]
                    if len(time_index_display) > 0:
                        ax_preds.plot(time_index_display, actual_labels_display, label='Actual Failures', marker='o', linestyle='-', color='blue', alpha=0.7, markersize=8)
                        ax_preds.plot(time_index_display, predicted_labels_display, label='Predicted Failures', marker='x', linestyle='--', color='red', alpha=0.7, markersize=8)
                        ax_preds.set_title(f'Final N-HiTS: Actual vs. Predicted (Index {x_min_display}-{x_max_display})')
                        ax_preds.set_xlabel('Time Step / Index'); ax_preds.set_ylabel('Failure (1) / No Failure (0)'); ax_preds.set_yticks([0, 1])
                        ax_preds.set_xlim(x_min_display -1, x_max_display +1); ax_preds.legend()
                        plt.tight_layout()
                        wandb.log({"actual_vs_predicted_zoom": wandb.Image(fig_preds)})
                        plt.close(fig_preds)

                else: # Local display if not using W&B
                    disp_local = ConfusionMatrixDisplay(confusion_matrix=cm_xgb_test, display_labels=["Good", "Failure"])
                    disp_local.plot(cmap=plt.cm.Blues); plt.title("XGBoost Test Confusion Matrix (Local)"); plt.show()

            else:
                print("Not enough data in test set for XGBoost evaluation after slicing.")
        except Exception as e:
            print(f"Error during XGBModel prediction or evaluation on test set: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Test set for XGBoost is empty. Skipping prediction.")

    # --- Finish W&B Run ---
    if wandb.run:
        wandb.finish()
    print("\n--- XGBoost Script with W&B Tracking End ---")