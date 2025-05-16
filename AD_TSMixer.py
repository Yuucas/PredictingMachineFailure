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
from darts.ad import ForecastingAnomalyModel
from darts.ad.scorers import NormScorer, DifferenceScorer
from darts.ad.detectors import ThresholdDetector, QuantileDetector

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
    print(f"train_main_ts length: {len(train_main_ts)}")
    print(f"val_main_ts length: {len(val_main_ts)}")
    print(f"test_main_ts length: {len(test_main_ts)}")

    if train_covariates_ts:
        print(f"train_covariates_ts length: {len(train_covariates_ts)}")
        print(f"val_covariates_ts length: {len(val_covariates_ts)}")
        print(f"test_covariates_ts length: {len(test_covariates_ts)}")

    # --- Scaling ---
    scaler = Scaler(StandardScaler())
    scaler.fit(train_main_ts) # Fit ONLY on training main features
    train_main_ts_scaled = scaler.transform(train_main_ts)
    val_main_ts_scaled = scaler.transform(val_main_ts)
    test_main_ts_scaled = scaler.transform(test_main_ts)

    # --- Prepare Past Covariates for Model Fitting ---
    train_past_covariates_for_fit = train_main_ts_scaled.stack(train_covariates_ts) if train_covariates_ts else train_main_ts_scaled
    val_past_covariates_for_fit = val_main_ts_scaled.stack(val_covariates_ts) if val_covariates_ts else val_main_ts_scaled

    # --- Prepare Full Past Covariates for Prediction on Test Set ---
    # Scale the entire original main_ts (before any splits) using the fitted scaler
    full_main_ts_original_scaled = scaler.transform(main_ts)
    full_past_covariates_for_prediction = full_main_ts_original_scaled.stack(covariates_ts) if covariates_ts else full_main_ts_original_scaled

    # --- 1. Define and Train the TSMixerModel ---
    input_chunk_length = 120 # How many past steps TSMixer sees. Adjust based on data.
    output_chunk_length = 20  # Predict one step ahead.
    n_epochs = 250           

    # Ensure training data is sufficient for the TSMixer model
    if len(train_main_ts_scaled) < input_chunk_length + output_chunk_length:
        raise ValueError(f"Training main series (length {len(train_main_ts_scaled)}) is too short for TSMixer "
                        f"input_chunk_length {input_chunk_length} and output_chunk_length {output_chunk_length}.")
    if val_main_ts_scaled is not None and len(val_main_ts_scaled) < output_chunk_length:
        print(f"Warning: Validation series is shorter than TSMixer's output_chunk_length. May affect validation.")

    # Define TSMixer model parameters
    tsmixer_forecaster_params_config = { # Params that define the model structure and training
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'n_epochs': n_epochs,
        'batch_size': 16,
        'likelihood_quantiles': [0.1, 0.5, 0.9], # Log specific details
        'random_state': 42,
        'model_name': "TSMixer_for_Anomaly",
        'work_dir': './darts_models',
        'save_checkpoints': True,
        'force_reset': True, # Be careful with this in repeated runs if you want to resume
        # pl_trainer_kwargs will be added below
    }

    # --- W&B Initialization for Anomaly Detection Experiment ---
    wandb_ad_project_name = "predictive-maintenance"
    wandb_ad_run_name = f"TSMixer_AD_in{120}_out{20}" 

    # Initialize WandbLogger for the TSMixerModel training
    # This will also call wandb.init()
    wandb_ad_logger = WandbLogger(
        project=wandb_ad_project_name,
        name=wandb_ad_run_name,
        job_type="tsmixer-anomaly-detection-training"
    )


    print("\n--- Loading Best TSMixer Model Checkpoint ---")

    tsmixer_model = TSMixerModel.load_from_checkpoint(
        work_dir='darts_models',
        model_name="TSMixer_for_Anomaly",
        best=True                            
    )

    # --- 2. Define the Anomaly Scorer ---
    '''
    NormScorer: computes z-scores of the residuals (actual - forecast).
    component_wise=True gives a separate score for each feature in main_ts.
    window=1 means it looks at each residual individually. A larger window would smooth scores.
    '''
    anomaly_scorer = NormScorer(component_wise=True)

    # Log scorer info
    if wandb.run: wandb.config.update({"anomaly_scorer": anomaly_scorer.__class__.__name__})

    # --- 3. Define the ForecastingAnomalyModel ---
    '''
    This model wraps the TSMixerModel and the NormScorer.
    It will use TSMixer to predict main_ts_scaled and then use NormScorer
    to compare these predictions against the actual main_ts_scaled.
    '''
    forecasting_anomaly_model = ForecastingAnomalyModel(
        model=tsmixer_model,  
        scorer=anomaly_scorer
    )

    # --- 4. Fit the ForecastingAnomalyModel ---
    '''
    This will train the underlying TSMixerModel.
    The `series` is what TSMixer will learn to predict main features.
    'past_covariates` are used by TSMixer if provided.
    '''
    print("\n--- Fitting ForecastingAnomalyModel (Training TSMixerModel) ---")
    print(f"train_main_ts_scaled length: {len(train_main_ts_scaled)}")
    print(f"val_main_ts_scaled length: {len(val_main_ts_scaled)}")
    print(f"train_covariates_ts length: {len(train_covariates_ts)}")
    print(f"val_covariates_ts length: {len(val_covariates_ts)}")
    forecasting_anomaly_model.fit(
        series=train_main_ts_scaled,
        past_covariates=train_covariates_ts, # Pass 'Type' as past covariate
        val_series=val_main_ts_scaled,
        val_past_covariates=val_covariates_ts
    )
    print("\n--- ForecastingAnomalyModel Fitting Complete ---")

    # --- 5. Generate Anomaly Scores on the Test Set ---
    '''
    The `score` method will:
      1. Use the trained TSMixerModel to generate forecasts for `test_main_ts_scaled`.
        It needs `full_type_covariate_for_prediction` to make these forecasts,
        and `history_for_prediction_main` as the conditioning series.
      2. Use the NormScorer to compute anomaly scores between actuals and forecasts.
    '''

    print("\n--- Generating Anomaly Scores on Test Set ---")

    if covariates_ts is None:
        print("Warning: `covariates_ts` (full 'Type' series) is None. Scoring without past_covariates.")

    history_for_prediction_target = train_target_ts.append(val_target_ts)
    print(f"history_for_prediction_target length: {len(history_for_prediction_target)}")
    print(f"full_past_covariates_for_prediction length: {len(full_past_covariates_for_prediction)}")
    
    anomaly_scores_ts, model_forecasting = forecasting_anomaly_model.score(
    series=test_main_ts_scaled,      
    past_covariates=covariates_ts
    )

    # --- 6. Visualize Anomaly Scores ---
    fig_scores, ax_scores = plt.subplots(figsize=(20, 11)) 
    anomaly_scores_ts.plot(ax=ax_scores, label="Anomaly Scores (Component-wise)")
    ax_scores.set_title("Anomaly Scores from TSMixer + NormScorer on Test Set")
    ax_scores.set_xlabel("Time Step / Index")
    ax_scores.set_ylabel("Anomaly Score (e.g., Z-score)")
    ax_scores.legend()
    ax_scores.grid(False) 


    # --- 7. Detect Anomalies from Scores ---
    anomaly_detector = QuantileDetector(high_quantile=0.90)
    binary_anomalies_ts = anomaly_detector.fit_detect(anomaly_scores_ts)

    # Prepare binary anomalies plot for W&B
    fig_binary, ax_binary = plt.subplots(figsize=(14, 6)) 
    binary_anomalies_ts.plot(ax=ax_binary, label="Binary Anomalies (Component-wise)") 
    ax_binary.set_title("Detected Binary Anomalies on Test Set (QuantileDetector)")
    ax_binary.set_xlabel("Time Step / Index")
    ax_binary.set_ylabel("Anomaly (1) / Normal (0)")
    ax_binary.set_yticks([0,1])
    ax_binary.legend()
    ax_binary.grid(False) 
    # plt.show() 

    # --- Log to W&B ---
    if wandb.run: # Check if a W&B run is active
        print("\n--- Logging Anomaly Detection Results to W&B ---")
        wandb.log({"tsmixer_anomaly_scores_plot": wandb.Image(fig_scores)})
        wandb.log({"tsmixer_binary_anomalies_plot": wandb.Image(fig_binary)})

        # Log some summary statistics of anomaly scores if desired
        anomaly_scores_summary = anomaly_scores_ts.pd_dataframe().describe()
        for col in anomaly_scores_summary.columns:
            wandb.log({f"anomaly_score_{col}_mean": anomaly_scores_summary[col]["mean"],
                    f"anomaly_score_{col}_std": anomaly_scores_summary[col]["std"],
                    f"anomaly_score_{col}_max": anomaly_scores_summary[col]["max"]})

        # Log detector parameters
        wandb.config.update({"anomaly_detector": anomaly_detector.__class__.__name__,
                            "detector_high_quantile": 0.90})
        plt.close(fig_scores) 
        plt.close(fig_binary)
    else:
        print("No active W&B run to log anomaly plots.")
        # If no W&B run, plot locally
        plt.show(fig_scores)
        plt.show(fig_binary)

    # Check if a W&B run is active
    if wandb.run:
         wandb.finish()
    print("\n--- W&B Run Finished (if active) ---")