from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import ChurnModel
import pandas as pd

app = FastAPI()
model = ChurnModel()

# Load the model if it exists, or train it if not
try:
    model.load_model()
except FileNotFoundError:
    df = pd.read_csv(r'C:\Users\ADMIN\OneDrive\Desktop\TASK\survey_sparrow\data\churn_data\Bank_churn.csv')  

class PredictionInput(BaseModel):
    features: dict

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        return model.predict(input_data.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/")
def explain(input_data: PredictionInput, explanation_type: str):
    try:
        return model.explain(input_data.features, explanation_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
