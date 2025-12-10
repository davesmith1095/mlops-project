import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import shutil
import os
import argparse

# Set up parser to test new parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# 1. Data Ingestion (Simulated)
print("Loading data...")
# Using Wine Quality dataset directly from URL for speed
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Split Data
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting experiment name to keep everything together
mlflow.set_experiment("Wine_Quality_Experiments")

## Disabling autolog because argparse should log this manually
# # Enable Auto-Logging
# mlflow.autolog()
###

with mlflow.start_run():
    # Log the parameters so you see them in the UI
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    print(f"Training with n_estimators={args.n_estimators} and max_depth={args.max_depth}...")
    
    # --- 3. Use the Arguments in the Model ---
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 3. Evaluation
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"MSE: {mse}, MAE: {mae}")
    
    # Manually log specific metrics if needed (autolog does most of this)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # 4. Interpretability (SHAP)
    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Plot summary and save
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    
    # Log the artifact (the image) to MLflow
    mlflow.log_artifact("shap_summary.png")

    print("Training Complete. Run 'mlflow ui' to view results.")

# This creates a folder named "manual_model" in your project root
    print("Saving model for Docker...")
    model_path = "manual_model"
    
    # Clean up old model if it exists
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        
    # Save the model locally
    mlflow.sklearn.save_model(rf, model_path)
    print(f"✅ Model saved to: {os.path.abspath(model_path)}")