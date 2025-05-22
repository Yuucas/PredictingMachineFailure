import os
import numpy as np
import pandas as pd
import traceback
from dotenv import load_dotenv

# Preprocessing
from feature_engine.selection import RecursiveFeatureElimination

# Training
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mae, mse
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

# Data processing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Model evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

# Tracking
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

    # initialize feature selector
    rfe = RecursiveFeatureElimination(estimator=rfr_model, scoring="r2", cv=3)

    # Prepare dataset
    X = dataset.drop(["Target"], axis=1)
    y = dataset["Target"]

    x_transformed = rfe.fit_transform(X, y)
    print("Dataset After Elimination: \n", x_transformed.head(5))

    # --- Data Preparation for Fine Tuning ---
    column_names_list = x_transformed.columns.tolist()

    covariate_cols = [col for col in column_names_list if col == 'Type']
    main_feature_cols = [col for col in column_names_list if col != 'Type']

    # Convert DataFrame into Darts TimeSeries format
    main_ts = TimeSeries.from_dataframe(x_transformed, value_cols=main_feature_cols)
    covariates_ts = TimeSeries.from_dataframe(x_transformed, value_cols=covariate_cols)

    target_ts = TimeSeries.from_series(y)

    # Define split proportions
    train_frac = 0.7  # 70% for training
    val_frac = 0.15   # 15% for validation / test

    # Calculate split points
    temp_main_ts, test_main_ts = main_ts.split_before(1.0 - val_frac) 
    temp_target_ts, test_target_ts = target_ts.split_before(1.0 - val_frac)
    if covariates_ts:
        temp_covariates_ts, test_covariates_ts = covariates_ts.split_before(1.0 - val_frac)
    else:
        temp_covariates_ts, test_covariates_ts = None, None

    # Split the remaining data into training and validation
    train_main_ts, val_main_ts = temp_main_ts.split_before(train_frac / (train_frac + val_frac))
    train_target_ts, val_target_ts = temp_target_ts.split_before(train_frac / (train_frac + val_frac))
    if temp_covariates_ts:
        train_covariates_ts, val_covariates_ts = temp_covariates_ts.split_before(train_frac / (train_frac + val_frac))
    else:
        train_covariates_ts, val_covariates_ts = None, None

    print(f"\n--- Data Splitting (Train, Validation, Test) ---")
    print(f"Training main series length: {len(train_main_ts)}")
    print(f"Validation main series length: {len(val_main_ts)}")
    print(f"Test main series length: {len(test_main_ts)}")

    if train_covariates_ts:
        print(f"Training covariates series length: {len(train_covariates_ts)}")
        print(f"Validation covariates series length: {len(val_covariates_ts)}")
        print(f"Test covariates series length: {len(test_covariates_ts)}")

    # --- Scaling ---
    scaler = Scaler(StandardScaler())
    scaler.fit(train_main_ts)
    train_main_ts_scaled = scaler.transform(train_main_ts)
    val_main_ts_scaled = scaler.transform(val_main_ts)
    test_main_ts_scaled = scaler.transform(test_main_ts)

    # --- Prepare Past Covariates for Model Fitting ---
    train_past_covariates_for_fit = train_main_ts_scaled.stack(train_covariates_ts) if train_covariates_ts else train_main_ts_scaled
    val_past_covariates_for_fit = val_main_ts_scaled.stack(val_covariates_ts) if val_covariates_ts else val_main_ts_scaled

    # --- Prepare Full Past Covariates for Prediction on Test Set ---
    full_main_ts_original_scaled = scaler.transform(main_ts)
    full_past_covariates_for_prediction = full_main_ts_original_scaled.stack(covariates_ts) if covariates_ts else full_main_ts_original_scaled


    # Best parameters from Optuna
    best_hyperparams = {
        'input_chunk_length': 197,
        'output_chunk_length': 20,
        'hidden_size': 113,
        'ff_size': 55,
        'num_blocks': 3,
        'dropout': 0.05427613479681044,
        'optimizer_kwargs': {'lr': 0.009620626803863358},
        'activation': 'Tanh'
        # 'MaxPool1d': False 
    }

    # --- Initialize W&B Logger for PyTorch Lightning ---
    model_name = "TSMixer"
    wandb_project_name = "predictive-maintenance" # Define your project name
    wandb_run_name = f"{model_name}_Final_in{best_hyperparams['input_chunk_length']}_lr{best_hyperparams['optimizer_kwargs']['lr']:.0e}"

    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=wandb_run_name,
        job_type="final-model-training"
    )


    # --- Final Model Parameters ---
    final_model_params = {
        'output_chunk_length': best_hyperparams['output_chunk_length'], # Keep consistent with Optuna's out_len_model_config if applicable
        'batch_size': 16,
        'n_epochs': 300,
        'nr_epochs_val_period': 1,
        'likelihood': QuantileRegression(),
        'model_name': f"{model_name}_Final_Trained_Model", # For local checkpoints
        'work_dir': './darts_models', # Specify a directory for Darts models/checkpoints
        'force_reset': True,
        'save_checkpoints': True, # Critical for loading best model based on val_loss
        'random_state': 42
    }

    # Merge Optuna best params with other fixed params
    final_model_params.update(best_hyperparams)

    # Log final model configuration to W&B (WandbLogger does this if config is passed to it,
    wandb_logger.experiment.config.update(final_model_params)


    # --- Callbacks and Trainer Arguments ---
    final_early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=20, verbose=True, mode="min") # Increased patience
    final_callbacks = [final_early_stopper]

    final_pl_trainer_kwargs = {
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "callbacks": final_callbacks,
        "enable_progress_bar": True,
        "logger": wandb_logger # Integrate WandbLogger
    }
    final_model_params['pl_trainer_kwargs'] = final_pl_trainer_kwargs

    # Create and train the final model
    final_model = TSMixerModel(**final_model_params)

    print(f"\n--- Training Final {model_name} Model (Tracked by W&B) ---")

    if len(train_target_ts) < final_model_params['input_chunk_length'] + final_model_params['output_chunk_length'] or \
       (val_target_ts is not None and len(val_target_ts) < final_model_params['output_chunk_length']):
        print(f"Warning: Training or Validation target series is too short. Adjust parameters or data.")
        if wandb.run: wandb.finish(exit_code=1) # Finish W&B run with an error code
    else:
        final_model.fit(
            series=train_target_ts,
            val_series=val_target_ts,
            past_covariates=train_past_covariates_for_fit,
            val_past_covariates=val_past_covariates_for_fit,
            verbose=True
        )
        print(f"\n--- Final {model_name} Model Training Complete ---")

        # --- Save the Final Model ---
        final_model_save_path = os.path.join(final_model_params['work_dir'], final_model_params['model_name'], "final_best_model")
        final_model.save(final_model_save_path)
        print(f"Final model explicitly saved to: {final_model_save_path}")

        # --- Log Model as W&B Artifact ---
        model_artifact = wandb.Artifact(
            name=final_model_params['model_name'], # Artifact name
            type="model",
            description=f"Final trained {model_name} model after hyperparameter tuning. Best val_loss.",
            metadata=final_model_params # Log model config with artifact
        )
        model_artifact.add_file(final_model_save_path) # Add the saved model file
        wandb_logger.experiment.log_artifact(model_artifact)
        print(f"Logged model artifact '{final_model_params['model_name']}' to W&B.")

        
    # --- PREDICTION (using the final W&B tracked model) ---
    model_to_predict_with = final_model

    if len(test_target_ts) > 0:
        print("\n--- Making Predictions on Test Set with Final W&B-Tracked Model ---")
        try:
            history_for_prediction_target = train_target_ts.append(val_target_ts)

            probabilistic_predictions = model_to_predict_with.predict(
                n=len(test_target_ts),
                series=history_for_prediction_target, # Conditioning series (train + val target)
                past_covariates=full_past_covariates_for_prediction, # Full history of covariates
                num_samples=100
            )
            predictions = probabilistic_predictions.quantile_timeseries(quantile=0.5)

            # --- Evaluation ---
            actual_values = test_target_ts.values(copy=True).flatten()
            predicted_continuous_values = predictions.values(copy=True).flatten()
            threshold = 0.5
            predicted_labels = (predicted_continuous_values >= threshold).astype(int)
            min_len = min(len(actual_values), len(predicted_labels))
            actual_labels_eval, predicted_labels_eval = actual_values[:min_len], predicted_labels[:min_len]

            if min_len > 0:
                print(f"\n--- Evaluation (Threshold = {threshold}) ---")
                accuracy = accuracy_score(actual_labels_eval, predicted_labels_eval)
                roc_auc = roc_auc_score(actual_labels_eval, predicted_continuous_values[:min_len]) if len(np.unique(actual_labels_eval)) > 1 else 0.5
                print(f"Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

                wandb_logger.experiment.log({
                    "test_accuracy": accuracy,
                    "test_roc_auc": roc_auc,
                    "threshold": threshold
                })
                report = classification_report(actual_labels_eval, predicted_labels_eval, 
                                               zero_division=0, output_dict=True, labels=[0, 1], 
                                               target_names=["Good", "Failure"])
                
                print("\n--- Classification Report ---")
                print(report)

                # --- Logging of Classification Report Metrics ---
                metrics_to_log = {}

                # Log metrics for class 'Good' (No Failure)
                if "Good" in report:
                    metrics_to_log["test_precision_class_Good"] = report["Good"]["precision"]
                    metrics_to_log["test_recall_class_Good"] = report["Good"]["recall"]
                    metrics_to_log["test_f1_class_Good"] = report["Good"]["f1-score"]
                    metrics_to_log["test_support_class_Good"] = report["Good"]["support"]
                else: # Log default values if class 'Good' is missing 
                    metrics_to_log["test_precision_class_Good"] = 0.0
                    metrics_to_log["test_recall_class_Good"] = 0.0
                    metrics_to_log["test_f1_class_Good"] = 0.0
                    metrics_to_log["test_support_class_Good"] = 0

                # Log metrics for class 'Failure' 
                if "Failure" in report:
                    metrics_to_log["test_precision_class_Failure"] = report["Failure"]["precision"]
                    metrics_to_log["test_recall_class_Failure"] = report["Failure"]["recall"]
                    metrics_to_log["test_f1_class_Failure"] = report["Failure"]["f1-score"]
                    metrics_to_log["test_support_class_Failure"] = report["Failure"]["support"]
                else: # Log default values if class 'Failure' is missing
                    print("Warning: Class '1' (Failure) not found in classification report. Logging default values for its metrics.")
                    metrics_to_log["test_precision_class_Failure"] = 0.0
                    metrics_to_log["test_recall_class_Failure"] = 0.0
                    metrics_to_log["test_f1_class_Failure"] = 0.0
                    metrics_to_log["test_support_class_Failure"] = 0 # No support means it wasn't in actuals or preds

                # Log macro averages (these should generally always be present if report is generated)
                if "macro avg" in report:
                    metrics_to_log["test_macro_avg_precision"] = report["macro avg"]["precision"]
                    metrics_to_log["test_macro_avg_recall"] = report["macro avg"]["recall"]
                    metrics_to_log["test_macro_avg_f1"] = report["macro avg"]["f1-score"]
                if "weighted avg" in report: # Also good to log
                    metrics_to_log["test_weighted_avg_precision"] = report["weighted avg"]["precision"]
                    metrics_to_log["test_weighted_avg_recall"] = report["weighted avg"]["recall"]
                    metrics_to_log["test_weighted_avg_f1"] = report["weighted avg"]["f1-score"]


                wandb_logger.experiment.log(metrics_to_log)

                cm = confusion_matrix(actual_labels_eval, predicted_labels_eval)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
                disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
                ax_cm.set_title(f'Final {model_name} Model - Confusion Matrix')
                wandb_logger.experiment.log({"confusion_matrix": wandb.Image(fig_cm)})
                plt.close(fig_cm)

                # Plotting the actual vs predicted values
                x_min_display, x_max_display = 9500, 9600
                fig_preds, ax_preds = plt.subplots(figsize=(15, 7))

                time_index_full_test = test_target_ts.time_index
                actual_labels_np = np.array(actual_labels_eval)
                predicted_labels_np = np.array(predicted_labels_eval)
                time_index_np = time_index_full_test[:min_len].to_numpy()
                display_mask = (time_index_np >= x_min_display) & (time_index_np <= x_max_display)
                time_index_display, actual_labels_display, predicted_labels_display = time_index_np[display_mask], actual_labels_np[display_mask], predicted_labels_np[display_mask]
                if len(time_index_display) > 0:
                    ax_preds.plot(time_index_display, actual_labels_display, label='Actual Failures', marker='o', linestyle='-', color='blue', alpha=0.7, markersize=8)
                    ax_preds.plot(time_index_display, predicted_labels_display, label='Predicted Failures', marker='x', linestyle='--', color='red', alpha=0.7, markersize=8)
                    ax_preds.set_title(f'Final {model_name}: Actual vs. Predicted (Index {x_min_display}-{x_max_display})')
                    ax_preds.set_xlabel('Time Step / Index'); ax_preds.set_ylabel('Failure (1) / No Failure (0)'); ax_preds.set_yticks([0, 1])
                    ax_preds.set_xlim(x_min_display -1, x_max_display +1); ax_preds.legend(); ax_preds.grid(False, which='both', linestyle='--', linewidth=0.5)
                    plt.tight_layout()
                    wandb_logger.experiment.log({"actual_vs_predicted_zoom": wandb.Image(fig_preds)})
                    plt.close(fig_preds)

            else: print("Not enough data for evaluation.")
        except Exception as e:
            print(f"Error during prediction or evaluation: {e}")
            traceback.print_exc()
    else: print("\nTest set is empty.")

    if wandb.run: # Check if a W&B run is active
         wandb.finish()
    print("\n--- W&B Run Finished (if active) ---")