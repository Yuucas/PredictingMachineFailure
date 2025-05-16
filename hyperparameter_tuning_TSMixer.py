import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Preprocessing
from feature_engine.selection import RecursiveFeatureElimination

# Training
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

# Data processing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Model evaluation
from sklearn.metrics import roc_auc_score

# Hyperparameter tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 150, 300)
    out_len = 20
    # Other hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 64, 128)
    ff_size = trial.suggest_int("ff_size", 48, 96)
    num_blocks = trial.suggest_int("num_blocks", 2, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "Tanh", "RReLU"])
    maxpool = trial.suggest_categorical("MaxPool1d", [True, False])

    # Ensure train/val series are long enough for the suggested input_chunk_length and the model's fixed output_chunk_length
    if len(train_target_ts) < in_len + out_len or \
       len(val_target_ts) < out_len : 
        trial.report(float('inf'), step=0) # Report a high value
        if optuna.trial.TrialState.PRUNED != trial.state: # Check if not already pruned
             raise optuna.exceptions.TrialPruned() # Prune if infeasible due to data length
        return float('inf') # pruner or returned if not

    # Throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=20, verbose=True)
    callbacks = [pruner, early_stopper]

    pl_trainer_kwargs = {
        "accelerator": "auto",
        "callbacks": callbacks,
    }

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model = TSMixerModel(
        input_chunk_length = in_len,
        output_chunk_length = out_len,
        hidden_size = hidden_size,
        ff_size = ff_size,
        num_blocks= num_blocks,
        batch_size=16,
        activation = activation,
        n_epochs=250,
        nr_epochs_val_period=1,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        likelihood=QuantileRegression(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="NHits",
        force_reset=True,
        save_checkpoints=True,
    )

    try:
        # train the model
        model.fit(series=train_target_ts, val_series=val_target_ts, 
                past_covariates=train_past_covariates_for_tuning, 
                val_past_covariates=val_past_covariates_for_tuning,
                verbose=False)

        # --- Corrected Evaluation Section ---
        # Predict on the entire validation set to get a score for Optuna
        # n = len(val_target_ts) means we predict for the whole validation period.
        # The model internally uses its output_chunk_length (30) to make these predictions.
        probabilistic_preds_val = model.predict(
            n=len(val_target_ts),
            series=train_target_ts, # History of target to condition on
            past_covariates=full_past_covariates_for_prediction # Provide all covariates, Darts will slice
        )

        # Extract median for point forecast since QuantileRegression is used
        preds_val_median = probabilistic_preds_val.quantile_timeseries(quantile=0.5)

        actual_vals_val = val_target_ts.values(copy=True).flatten()
        pred_continuous_val = preds_val_median.values(copy=True).flatten()

        # Ensure lengths match for evaluation
        min_eval_len = min(len(actual_vals_val), len(pred_continuous_val))

        if min_eval_len < 1: # Need at least one point to evaluate
            print(f"Warning: Prediction length ({min_eval_len}) on validation set is too short for trial {trial.number}.")
            return float('inf')

        actual_vals_eval = actual_vals_val[:min_eval_len]
        pred_continuous_eval = pred_continuous_val[:min_eval_len]

        # Calculate ROC AUC score
        score = roc_auc_score(actual_vals_eval, pred_continuous_eval)
        objective_value = 1.0 - score

        if np.isnan(objective_value) or np.isinf(objective_value):
            print(f"Warning: Objective value is NaN/inf for trial {trial.number}. ROC AUC: {score}")
            return float('inf') 

        return objective_value

    except optuna.exceptions.TrialPruned:
        # If pruner callback prunes the trial, re-raise to inform Optuna
        raise
    except RuntimeError as e:
        # Catch specific errors like CUDA OOM or issues during model.fit/predict
        print(f"Optuna trial {trial.number} failed with RuntimeError: {e}")
        # Check if trial was already pruned, otherwise report a high value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return float('inf') # Indicate failure for this trial
    except ValueError as e:
        # Catch ValueErrors (e.g., from roc_auc_score if only one class in validation slice)
        print(f"Optuna trial {trial.number} - ROC AUC calculation error or other ValueError: {e}")
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return 1.0 # Return a "bad" score (max possible for 1-AUC if AUC is >=0)
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Optuna trial {trial.number} failed with an unexpected error: {e}")
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return float('inf') # Indicate failure


# for convenience, print some optimization trials information
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
    # initialize linear regresion estimator
    rfr_model = RandomForestRegressor()

    # initialize feature selector
    # neg_mean_squared_error
    rfe = RecursiveFeatureElimination(estimator=rfr_model, scoring="r2", cv=3)

    # Prepare dataset
    X = dataset.drop(["Target"], axis=1)
    y = dataset["Target"]

    x_transformed = rfe.fit_transform(X, y)
    print("Dataset After Elimination: \n", x_transformed.head(5))

    # --- Data Preparation for Fine Tuning ---
    # Define which columns are your main features and which are covariates
    column_names_list = x_transformed.columns.tolist()

    covariate_cols = [col for col in column_names_list if col == 'Type']
    main_feature_cols = [col for col in column_names_list if col != 'Type']

    # Convert DataFrame into Darts TimeSeries format
    main_ts = TimeSeries.from_dataframe(x_transformed, value_cols=main_feature_cols)
    covariates_ts = TimeSeries.from_dataframe(x_transformed, value_cols=covariate_cols)

    # The 'Target' column (0 or 1) is what we want to predict.
    target_ts = TimeSeries.from_series(y) # y is dataset["Target"]

    # Define split proportions
    train_frac = 0.7  # 70% for training
    val_frac = 0.15   # 15% for validation

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

    # --- Scale the main feature data ---
    scaler = Scaler(StandardScaler())

    # Fit the scaler ONLY on the new, smaller training part of the main features
    scaler.fit(train_main_ts)

    # Transform the training, validation, and test parts of the main features
    train_main_ts_scaled = scaler.transform(train_main_ts)
    val_main_ts_scaled = scaler.transform(val_main_ts) 
    test_main_ts_scaled = scaler.transform(test_main_ts)

    # --- Prepare past_covariates for Training (used by gridsearch) ---
    if train_covariates_ts:
        train_past_covariates_for_tuning = train_main_ts_scaled.stack(train_covariates_ts)
    else:
        train_past_covariates_for_tuning = train_main_ts_scaled

    # --- Prepare past_covariates for Validation (used by gridsearch) ---
    if val_covariates_ts:
        val_past_covariates_for_tuning = val_main_ts_scaled.stack(val_covariates_ts)
    else:
        val_past_covariates_for_tuning = val_main_ts_scaled

    # --- Prepare full_past_covariates for final Prediction on the test set ---
    full_main_ts_scaled = scaler.transform(main_ts)
    if covariates_ts:
        full_past_covariates_for_prediction = full_main_ts_scaled.stack(covariates_ts)
    else:
        full_past_covariates_for_prediction = full_main_ts_scaled

    # --- Start Hyperparameter Tuning ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=300, callbacks=[print_callback])
  