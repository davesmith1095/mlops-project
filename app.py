import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import mlflow.sklearn
import os

app = FastAPI()

### This block was causing issues building the docker when loading directly from mlruns. Commented out for now.
# run_id = "adcd160c105a4d0ab2107d6343997cdd" 
# logged_model = f'mlruns/0/{run_id}/artifacts'
# model = mlflow.sklearn.load_model(logged_model)
###

## Direct model load strategy
current_dir = os.getcwd()
logged_model_path = os.path.join(current_dir, "manual_model")

try:
    model = mlflow.sklearn.load_model(logged_model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(fixed_acidity: float, volatile_acidity: float, citric_acid: float, 
            residual_sugar: float, chlorides: float, free_sulfur_dioxide: float, 
            total_sulfur_dioxide: float, density: float, pH: float, 
            sulphates: float, alcohol: float):
    
    # Prepare data for model
    data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
             chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
             pH, sulphates, alcohol]]
    
    df = pd.DataFrame(data, columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ])
    
    prediction = model.predict(df)
    
    # Simple drift check (monitoring)
    if alcohol < 8.0:
        print("WARNING: Input drift detected (Alcohol unusually low)")

    return {"prediction": prediction[0]}
